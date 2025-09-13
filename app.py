import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ------------------------------
# Flask setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# Load Model
# ------------------------------
def load_model():
    model_path = "outputs_multi/model_multiclass.pth"
    checkpoint = torch.load(model_path, map_location="cpu")

    # ✅ Load class names from checkpoint
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]  # fallback

    # ✅ Define model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, class_names


model, class_names = load_model()

# ------------------------------
# Image Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).convert("RGB")
    img = transform(img).unsqueeze(0)

    # Model prediction with TB-priority detection + Unknown fallback
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)

    confidence_value = float(confidence.item())

    # Threshold for Unknown classification (can be tuned or set via env)
    threshold_str = os.getenv("PREDICTION_THRESHOLD", "0.50")
    try:
        threshold = float(threshold_str)
    except ValueError:
        threshold = 0.50

    # Separate TB presence threshold (prioritize TB detection)
    tb_threshold_str = os.getenv("TB_PRESENCE_THRESHOLD", "0.40")
    try:
        tb_threshold = float(tb_threshold_str)
    except ValueError:
        tb_threshold = 0.40

    # Identify TB-related labels present in the model
    tb_label_indices = [i for i, name in enumerate(class_names)
                        if name.upper().startswith("TB") or name.upper().startswith("TUBERCULOSIS")]

    # If any TB-related probability exceeds tb_threshold, predict TB (subtype if available)
    prediction = None
    if tb_label_indices:
        tb_probs = [(i, float(probabilities[i].item())) for i in tb_label_indices]
        top_tb_idx, top_tb_prob = max(tb_probs, key=lambda x: x[1])
        tb_total_prob = sum(p for _, p in tb_probs)
        if top_tb_prob >= tb_threshold or tb_total_prob >= tb_threshold:
            prediction = class_names[top_tb_idx]

    # Otherwise use top-1 with Unknown fallback
    if prediction is None:
        if confidence_value < threshold:
            prediction = "UNKNOWN"
        else:
            prediction = class_names[int(pred_idx.item())]

    # Build human-friendly display label (supports TB subtypes like TB_PULMONARY)
    def format_label(label: str) -> str:
        if label == "UNKNOWN":
            return label
        # Normalize common TB subtype naming schemes
        if label.upper().startswith("TB_"):
            subtype = label[3:].replace("_", " ").title()
            return f"Tuberculosis — {subtype}"
        if label.upper().startswith("TUBERCULOSIS_"):
            subtype = label.split("_", 1)[1].replace("_", " ").title()
            return f"Tuberculosis — {subtype}"
        if label.upper() in {"TB", "TUBERCULOSIS"}:
            return "Tuberculosis"
        # Generic prettifier
        return label.replace("_", " ").title()

    display_label = format_label(prediction)
    # If it's base TB without subtype, show Subtype: N/A in the UI
    show_subtype_na = (display_label == "Tuberculosis")

    # ✅ Send result to result.html
    return render_template(
        "result.html",
        filename=file.filename,
        prediction=prediction,
        display_label=display_label,
        show_subtype_na=show_subtype_na,
        confidence=confidence_value,
        threshold=threshold,
    )


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
