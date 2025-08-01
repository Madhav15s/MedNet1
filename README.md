# 🏥 MedNet1 — Multi-Class Medical AI System

[![Demo](https://img.shields.io/badge/Demo-YouTube-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=l_lQYxWWg9Y)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)](https://reactjs.org)

MedNet1 is an AI-powered system for analyzing various medical images across different imaging modalities. It supports multi-class classification and provides accurate predictions, confidence scores, and medically relevant explanations via a modern React frontend.

## 📋 Table of Contents

- [Features](#-features)
- [Supported Imaging Types](#-supported-imaging-types--classes)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#-backend-setup)
  - [Frontend Setup](#-frontend-setup)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Customization](#-customization)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## ✨ Features

- **🔬 Multi-Modal Medical Imaging** — Supports Chest X-rays, Brain MRIs, Skin Lesions, Retinal Scans, and Cardiac Imaging
- **🧠 Deep Learning Models** — Uses ResNet18 with transfer learning for accurate multi-class classification
- **⚡ React Frontend (Vite)** — Fast and responsive UI for easy image uploads and predictions
- **📈 Performance Metrics** — Visualizations like confusion matrices, ROC curves, and accuracy scores
- **📊 Confidence Scores** — Prediction probabilities for each class to improve trustworthiness
- **📁 Dataset Setup Utility** — One-command setup for standardized dataset directories
- **🧾 AI-Generated Medical Explanations** — Textual explanation for each diagnosis (prototype stage)

## 🧠 Supported Imaging Types & Classes

| Imaging Type     | Classes                                                        |
|------------------|----------------------------------------------------------------|
| **Chest X‑Ray**     | Normal, Pneumonia, COVID‑19, Tuberculosis, Lung Cancer        |
| **Brain MRI**       | Normal, Tumor, Stroke, Hemorrhage, Multiple Sclerosis         |
| **Skin Lesion**     | Normal, Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma |
| **Retinal Image**   | Normal, Diabetic Retinopathy, Glaucoma, Macular Degeneration  |
| **Cardiac Imaging** | Normal, Heart Failure, Arrhythmia, Valve Disease              |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn package manager
- CUDA-compatible GPU (recommended for training)

### 📦 Backend Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Madhav15s/MedNet1.git
cd MedNet1
```

#### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Prepare Dataset
Use the helper function to create a structured dataset directory:

```bash
python -c "from utils.data_utils import create_dataset_structure; create_dataset_structure('chest_xray')"
```

This creates the following structure:
```
datasets/
└── chest_xray/
    ├── train/
    └── val/
```

**Available dataset types:** `chest_xray`, `brain_mri`, `skin_lesion`, `retinal`, `cardiac`

#### 4. Train Model
```bash
python src/train.py --medical_type chest_xray --epochs 30 --batch_size 32 --lr 0.001
```

#### 5. Evaluate Model
```bash
# Evaluate on test dataset
python src/evaluate.py --model models/resnet_chest_xray.pth --data datasets/chest_xray/test

# Predict on single image
python src/evaluate.py --model models/resnet_chest_xray.pth --image path/to/image.jpg
```

### 🖥️ Frontend Setup

#### 1. Navigate to Frontend Directory
```bash
cd frontend
```

#### 2. Install Dependencies
```bash
npm install
```

#### 3. Run Development Server
```bash
npm run dev
```
Visit: [http://localhost:5173](http://localhost:5173)

#### 4. Build for Production
```bash
npm run build
```
Production files will be output to the `dist/` folder.

## 🗂 Project Structure

```
MedNet1/
├── frontend/             # React frontend (Vite)
│   ├── src/              # Components and pages
│   ├── public/           # Static assets
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Vite configuration
├── src/                  # Model training and evaluation scripts
│   ├── train.py          # Model training script
│   └── evaluate.py       # Model evaluation script
├── utils/                # Dataset setup helpers
│   └── data_utils.py     # Dataset utility functions
├── datasets/             # Dataset storage (user generated)
├── models/               # Saved model checkpoints
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

## 📖 Usage

### Training a Model
1. Prepare your dataset using the data utility
2. Run the training script with appropriate parameters
3. Monitor training progress and metrics

### Making Predictions
1. Load a trained model
2. Use the evaluation script for batch predictions or single image analysis
3. View results with confidence scores

### Using the Web Interface
1. Start the React development server
2. Upload medical images through the web interface
3. View predictions with confidence scores and explanations

## 🔧 Customization

### Model Architecture
- **Switch Models:** Replace ResNet18 with DenseNet, EfficientNet, or other architectures in `train.py`
- **Hyperparameter Tuning:** Adjust learning rates, batch sizes, and epochs for optimal performance

### Dataset Extension
- **Add New Modalities:** Extend dataset creation and training logic to support additional medical image types
- **Custom Classes:** Modify class definitions for specific use cases

### Frontend Enhancement
- **UI Improvements:** Add feedback forms, zoom tools, or heatmap overlays
- **Visualization:** Integrate advanced visualization libraries for better result presentation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/React code
- Include tests for new features
- Update documentation as needed

## ⚠️ Disclaimer

**IMPORTANT:** This project is for research and educational purposes only. It is not intended for use in actual medical diagnosis or treatment. Always consult a licensed medical professional for medical advice and diagnosis.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [React](https://reactjs.org/) - Frontend library
- [Vite](https://vitejs.dev/) - Build tool
- Public medical image datasets from [Kaggle](https://kaggle.com) and other open repositories
- [ResNet Architecture](https://arxiv.org/abs/1512.03385) - Original paper

---

<div align="center">
  Made with ❤️ for the medical AI community
</div>