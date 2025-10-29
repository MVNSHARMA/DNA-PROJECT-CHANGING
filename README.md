# Chest X-Ray Classification System

A deep learning-based web application for automated detection and classification of respiratory diseases from chest X-ray images. The system can identify Normal chest X-rays, Pneumonia, and various types of Tuberculosis (TB) with high accuracy.

## 🏥 Features

- **Multi-class Classification**: Detects Normal, Pneumonia, and Tuberculosis (including subtypes)
- **Web Interface**: User-friendly Flask web application for image upload and prediction
- **High Accuracy**: Advanced deep learning model with ResNet18 architecture
- **Confidence Scoring**: Provides prediction confidence scores for clinical decision support
- **Real-time Processing**: Fast inference with optimized preprocessing pipeline

## 📊 Supported Classes

The system can classify chest X-rays into the following categories:

### Primary Classes
- **NORMAL**: Healthy chest X-rays without any abnormalities
- **PNEUMONIA**: Bacterial or viral pneumonia detection
- **TUBERCULOSIS**: Active tuberculosis infection

### TB Subtypes (Advanced Classification)
- **TB_ACTIVE**: Active tuberculosis lesions
- **TB_OBSOLETE_PULMONARY**: Healed/scarred tuberculosis
- **NON_TB**: Non-tuberculosis conditions

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd chest-xray-classification
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask web server**
   ```bash
   export FLASK_APP=app.py
   flask run --port 5002 --debug
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://127.0.0.1:5002`
   - Upload a chest X-ray image (JPEG, PNG supported)
   - View the prediction results with confidence scores

## 🧠 Model Architecture

### Training Pipeline
- **Model**: ResNet18 pre-trained on ImageNet
- **Framework**: PyTorch with torchvision
- **Augmentation**: Advanced data augmentation including:
  - Random resized crops
  - Horizontal flips
  - Rotation (±10°)
  - Color jittering
  - Normalization with ImageNet statistics

### Training Features
- **Class Balancing**: WeightedRandomSampler for handling imbalanced datasets
- **Loss Function**: CrossEntropyLoss with class weights
- **Optimization**: Adam optimizer with CosineAnnealingLR scheduler
- **Early Stopping**: Validation accuracy-based stopping
- **Label Smoothing**: Reduces overfitting (smoothing factor: 0.05)

## 📁 Project Structure

```
├── app.py                          # Main Flask web application
├── med_classifier_multiclass_v2.py # Advanced training script
├── labels_all.csv                  # Combined dataset labels (6,670 samples)
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── index.html                 # Upload interface
│   └── result.html                # Results display
├── static/uploads/                # Uploaded images storage
├── outputs_multi/                 # Model checkpoints
│   └── model_multiclass.pth       # Trained model weights
├── chest_xray/                    # Original pneumonia dataset
├── cxr_multi/                     # Multi-class dataset
└── TB_Chest_Radiography_Database/ # TB subtype dataset
```

## 📈 Dataset Information

### Combined Dataset Statistics
- **Total Samples**: 6,670 chest X-ray images
- **Classes**: 6 different disease categories
- **Splits**: Train/Validation/Test sets
- **Sources**: 
  - Pneumonia dataset (Normal vs Pneumonia)
  - TB Chest Radiography Database (TB subtypes)
  - Multi-class combined dataset

### Class Distribution
The model is trained on a balanced dataset with proper class weighting to handle imbalanced classes effectively.

## 🔧 Configuration

### Environment Variables
You can customize the prediction behavior using environment variables:

```bash
export PREDICTION_THRESHOLD=0.45      # General confidence threshold
export TB_PRESENCE_THRESHOLD=0.60     # TB detection threshold
export TB_MARGIN_MIN=0.25             # TB margin requirement
export NORMAL_MIN=0.55                # Normal classification threshold
export PNEUMONIA_MIN=0.55             # Pneumonia classification threshold
```

### Prediction Logic
The system uses a sophisticated multi-threshold approach:
1. **Priority Classification**: NORMAL and PNEUMONIA are prioritized when their probabilities exceed minimum thresholds
2. **TB Detection**: Special handling for tuberculosis with higher confidence requirements
3. **Unknown Fallback**: Low-confidence predictions are marked as "UNKNOWN"

## 🎯 Model Performance

### Training Metrics
- **Architecture**: ResNet18 (pre-trained)
- **Input Size**: 224x224 pixels
- **Batch Size**: 32
- **Learning Rate**: 3e-4 with cosine annealing
- **Epochs**: 35 (with early stopping)

### Accuracy Features
- **Validation Monitoring**: Real-time validation accuracy tracking
- **Early Stopping**: Prevents overfitting
- **Class Weighting**: Handles dataset imbalance
- **Robust Augmentation**: Improves generalization

## 🚀 Training the Model

To retrain the model with your own data:

```bash
python med_classifier_multiclass_v2.py \
    --csv_path labels_all.csv \
    --output_dir outputs_multi \
    --epochs 35 \
    --batch_size 32 \
    --lr 3e-4
```

### Training Parameters
- `--csv_path`: Path to your dataset CSV file
- `--output_dir`: Directory to save model checkpoints
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate

## 📝 API Usage

### Upload and Predict
```bash
curl -X POST -F "file=@chest_xray.jpg" http://127.0.0.1:5002/predict
```

### Response Format
```json
{
    "prediction": "NORMAL",
    "confidence": 0.85,
    "display_label": "Normal"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   flask run --port 5003  # Use different port
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Loading Errors**
   - Ensure `outputs_multi/model_multiclass.pth` exists
   - Check file permissions

### Performance Optimization
- Use GPU acceleration if available (CUDA-compatible PyTorch)
- Increase batch size for faster training
- Adjust confidence thresholds based on your requirements

## 📚 Dependencies

- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **Pillow**: Image processing
- **Pandas**: Data manipulation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations when using in clinical settings.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the training logs in `outputs_multi/training.log`
- Ensure all dependencies are correctly installed

---

**Note**: This system requires significant computational resources for training. For production deployment, consider using GPU acceleration and proper model optimization techniques.

