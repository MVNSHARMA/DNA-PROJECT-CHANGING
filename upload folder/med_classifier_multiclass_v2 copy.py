import os
import argparse
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import models, transforms

class ChestXrayDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label

def compute_class_weights(train_df, class_names):
    counts = Counter(train_df["label"].tolist())
    total = sum(counts.values())
    weights = []
    for c in class_names:
        # inverse frequency
        w = total / (len(class_names) * counts.get(c, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Early stop training when monitored metric has not improved for patience epochs."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, output_dir, class_names):
    best_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "model_multiclass.pth")
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects, total = 0.0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

            epoch_loss = running_loss / max(total, 1)
            epoch_acc = running_corrects / max(total, 1)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                # step LR scheduler per epoch
                if scheduler is not None:
                    scheduler.step()
            else:
                # validation phase
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names
                    }, ckpt_path)
                    print("✅ Best model saved")
                # early stopping check on validation accuracy
                if early_stopper.step(epoch_acc):
                    print("🛑 Early stopping triggered.")
                    print(f"Training complete. Best val Acc: {best_acc:.4f}")
                    print(f"Model checkpoint: {ckpt_path}")
                    return

    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")
    print(f"Model checkpoint: {ckpt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--output_dir", default="outputs_multi")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    # class names sorted for reproducibility
    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"🔹 Classes ({len(class_names)}): {class_names}")

    transforms_map = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    datasets = {
        split: ChestXrayDataset(df[df["split"] == split], class_to_idx, transform=transforms_map[split])
        for split in ["train", "val"]
        if (df["split"] == split).any()
    }
    dataloaders = {}
    for split in datasets.keys():
        if split == "train":
            # Balanced sampling to mitigate class imbalance
            labels = [class_to_idx[row_label] for row_label in df[df["split"] == "train"]["label"].tolist()]
            class_sample_counts = [labels.count(i) for i in range(len(class_names))]
            # inverse frequency per class
            weights_per_class = [0 if c == 0 else 1.0 / c for c in class_sample_counts]
            weights = [weights_per_class[y] for y in labels]
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            dataloaders[split] = DataLoader(
                datasets[split], batch_size=args.batch_size, sampler=sampler, num_workers=0
            )
        else:
            dataloaders[split] = DataLoader(
                datasets[split], batch_size=args.batch_size, shuffle=False, num_workers=0
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    # class weights for imbalance (optional but helpful)
    if "train" in datasets:
        class_weights = compute_class_weights(df[df["split"] == "train"], class_names).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=args.epochs, output_dir=args.output_dir, class_names=class_names
    )

if __name__ == "__main__":
    main()
