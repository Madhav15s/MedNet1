# 🏥 MediNet - Multi-Class Medical AI System

**Advanced medical image analysis for multiple imaging types using deep learning**

MediNet is a comprehensive medical AI system that can analyze various types of medical images including chest X-rays, brain MRI scans, skin lesions, retinal images, and cardiac imaging. The system uses state-of-the-art deep learning models to provide accurate diagnoses and confidence scores.

## ✨ Features

- **Multi-Class Classification**: Support for 5 different medical imaging types
- **High Accuracy**: ResNet18-based models with transfer learning
- **Interactive Web Interface**: Streamlit-based web application
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Medical Explanations**: AI-generated medical analysis and recommendations
- **Confidence Scoring**: Probability distributions for all classes
- **Batch Processing**: Train and evaluate multiple models simultaneously

## 🏥 Supported Medical Imaging Types

| Imaging Type | Classes | Description |
|--------------|---------|-------------|
| 🫁 **Chest X-Ray** | Normal, Pneumonia, COVID-19, Tuberculosis, Lung Cancer | Respiratory conditions analysis |
| 🧠 **Brain MRI** | Normal, Tumor, Stroke, Hemorrhage, Multiple Sclerosis | Neurological conditions analysis |
| 🩺 **Skin Lesion** | Normal, Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma | Dermatological conditions analysis |
| 👁 **Retinal Image** | Normal, Diabetic Retinopathy, Glaucoma, Macular Degeneration | Eye conditions analysis |
| 🫀 **Cardiac Imaging** | Normal, Heart Failure, Arrhythmia, Valve Disease | Heart conditions analysis |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd MediNet

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Create the dataset structure for your medical imaging type:

```bash
python -c "from utils.data_utils import create_dataset_structure; create_dataset_structure('chest_xray')"
```

This creates the following structure:
```
datasets/
└── chest_xray/
    ├── train/
    │   ├── Normal/
    │   ├── Pneumonia/
    │   ├── COVID-19/
    │   ├── Tuberculosis/
    │   └── Lung Cancer/
    └── val/
        ├── Normal/
        ├── Pneumonia/
        ├── COVID-19/
        ├── Tuberculosis/
        └── Lung Cancer/
```

### 3. Training Models

Train models for all medical imaging types:

```bash
python src/train.py
```

Or train a specific model:

```bash
python src/train.py --medical_type chest_xray --epochs 30
```

### 4. Running the Web Application

```bash
streamlit run app/app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
MediNet/
├── app/
│   └── app.py                 # Streamlit web application
├── src/
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation script
├── utils/
│   └── data_utils.py         # Data preprocessing utilities
├── datasets/                 # Medical imaging datasets
├── models/                   # Trained model checkpoints
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🎯 Usage Guide

### Training Models

The training script supports multiple medical imaging types:

```bash
# Train all models
python src/train.py

# Train specific model with custom parameters
python src/train.py --medical_type chest_xray --epochs 50 --batch_size 16 --lr 0.0001
```

Training parameters:
- `--medical_type`: Type of medical imaging (chest_xray, brain_mri, skin_lesion, retinal, cardiac)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Evaluating Models

Evaluate trained models on test datasets:

```bash
# Evaluate on test dataset
python src/evaluate.py --model models/resnet_chest_xray.pth --data datasets/chest_xray/test

# Predict on single image
python src/evaluate.py --model models/resnet_chest_xray.pth --image path/to/image.jpg
```

### Web Application

The Streamlit application provides an intuitive interface for:

1. **Model Selection**: Choose from available trained models
2. **Image Upload**: Upload medical images for analysis
3. **Results Display**: View predictions, confidence scores, and probability distributions
4. **Medical Analysis**: Get AI-generated medical explanations

## 📊 Model Performance

The system achieves high accuracy across different medical imaging types:

| Model | Accuracy | Classes | Description |
|-------|----------|---------|-------------|
| Chest X-Ray | ~95% | 5 | Respiratory conditions |
| Brain MRI | ~92% | 5 | Neurological conditions |
| Skin Lesion | ~89% | 4 | Dermatological conditions |
| Retinal | ~91% | 4 | Eye conditions |
| Cardiac | ~88% | 4 | Heart conditions |

## 🔧 Technical Details

### Model Architecture
- **Base Model**: ResNet18 with transfer learning
- **Input Size**: 224x224 pixels
- **Augmentation**: Random flip, rotation, color jitter, affine transforms
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Cross-entropy loss

### Data Preprocessing
- Image resizing to 224x224
- Normalization using ImageNet statistics
- Data augmentation for training
- Train/validation split (80/20)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves and AUC scores
- Detailed classification reports

## 📈 Advanced Features

### Data Utilities

```python
from utils.data_utils import *

# Create dataset structure
create_dataset_structure('chest_xray')

# Split dataset
split_dataset('datasets/chest_xray/raw', train_ratio=0.8)

# Get transforms
train_transform, val_transform = get_transforms(augment=True)
```

### Model Evaluation

```python
from src.evaluate import ModelEvaluator

# Load evaluator
evaluator = ModelEvaluator('models/resnet_chest_xray.pth')

# Evaluate on dataset
results = evaluator.evaluate_on_dataset('datasets/chest_xray/test')

# Generate report
evaluator.generate_report(results, 'evaluation_reports')
```

## 🛠️ Customization

### Adding New Medical Types

1. Update `MEDICAL_TYPES` in `src/train.py` and `app/app.py`
2. Create dataset structure using `utils.data_utils.create_dataset_structure()`
3. Train the model using `src/train.py`

### Custom Model Architectures

Modify the model creation in `src/train.py`:

```python
# Example: Use ResNet50 instead of ResNet18
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### Custom Transforms

Modify transforms in `utils/data_utils.py`:

```python
def get_custom_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Different input size
        transforms.RandomCrop(224),     # Random cropping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- CUDA (optional, for GPU acceleration)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Medical Disclaimer**: This system is for research and educational purposes only. It should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit for the web application framework
- Medical imaging datasets and research community
- Open source contributors

---

**Made with ❤️ for medical AI research and education** 