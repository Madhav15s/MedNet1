# 🏥 MedNet1 — Multi-Class Medical AI System

MedNet1 is an AI-powered system for analyzing various medical images across different imaging modalities. It supports multi-class classification and provides accurate predictions, confidence scores, and medically relevant explanations via a modern React frontend.

---

## ✨ Features

- 🔬 **Multi-Modal Medical Imaging**  
  Supports Chest X-rays, Brain MRIs, Skin Lesions, Retinal Scans, and Cardiac Imaging.

- 🧠 **Deep Learning Models**  
  Uses ResNet18 with transfer learning for accurate multi-class classification.

- ⚡ **React Frontend (Vite)**  
  Fast and responsive UI for easy image uploads and predictions.

- 📈 **Performance Metrics**  
  Visualizations like confusion matrices, ROC curves, and accuracy scores.

- 📊 **Confidence Scores**  
  Prediction probabilities for each class to improve trustworthiness.

- 📁 **Dataset Setup Utility**  
  One-command setup for standardized dataset directories.

- 🧾 **AI-Generated Medical Explanations**  
  Textual explanation for each diagnosis (prototype stage).

---

## 🧠 Supported Imaging Types & Classes

| Imaging Type     | Classes                                                                                   |
|------------------|--------------------------------------------------------------------------------------------|
| Chest X‑Ray      | Normal, Pneumonia, COVID‑19, Tuberculosis, Lung Cancer                                    |
| Brain MRI        | Normal, Tumor, Stroke, Hemorrhage, Multiple Sclerosis                                     |
| Skin Lesion      | Normal, Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma                           |
| Retinal Image    | Normal, Diabetic Retinopathy, Glaucoma, Macular Degeneration                              |
| Cardiac Imaging  | Normal, Heart Failure, Arrhythmia, Valve Disease                                          |

---

## 🚀 Getting Started

### 📦 Backend Setup (Training & Evaluation)

#### 1. Clone the Repository
```bash
git clone https://github.com/Madhav15s/MedNet1.git
cd MedNet1
2. Set Up Python Environment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Prepare Dataset

Use the helper function to create a structured dataset directory:

python -c "from utils.data_utils import create_dataset_structure; create_dataset_structure('chest_xray')"
This creates:

datasets/
└── chest_xray/
    ├── train/
    └── val/
Replace 'chest_xray' with other types like 'brain_mri', 'skin_lesion', 'retinal', 'cardiac'.

4. Train Model

python src/train.py --medical_type chest_xray --epochs 30 --batch_size 32 --lr 0.001
5. Evaluate Model

python src/evaluate.py --model models/resnet_chest_xray.pth --data datasets/chest_xray/test
6. Predict on Single Image

python src/evaluate.py --model models/resnet_chest_xray.pth --image path/to/image.jpg
🖥️ Frontend Setup (React + Vite)
1. Navigate to Frontend

cd frontend
2. Install Dependencies

npm install
3. Run Development Server

npm run dev
Visit: http://localhost:5173

4. Build for Production

npm run build
This outputs production files to the dist/ folder.

🗂 Project Structure

MedNet1/
├── frontend/             # React frontend (Vite)
│   ├── src/              # Components and pages
│   └── public/           # Static assets
├── src/                  # Model training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── utils/                # Dataset setup helpers
│   └── data_utils.py
├── datasets/             # Placeholder for datasets (user generated)
├── models/               # Saved model checkpoints
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License
🔧 Customization Tips

🔄 Switch Models: Swap ResNet18 with DenseNet, EfficientNet, etc. in train.py.
➕ Add Modalities: Extend dataset creation and training logic to support new image types.
🌍 Frontend Extension: Add feedback forms, zoom tools, or heatmap overlays to improve the React UI.
⚠️ Disclaimer

This project is for research and educational purposes only.
It is not intended for use in actual medical diagnosis or treatment. Always consult a licensed medical professional.
🤝 Contributing

We welcome contributions!

Fork the repository
Create a new branch (git checkout -b feature-name)
Commit changes (git commit -m 'Add feature')
Push (git push origin feature-name)
Open a Pull Request
📝 License

This project is licensed under the MIT License.
See the LICENSE file for details.

🙌 Acknowledgements

PyTorch
React
Vite
Public medical image datasets from Kaggle and other open repositories
ResNet Architecture (paper)