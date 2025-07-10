
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from pathlib import Path

# Ensure the src directory is in the Python path
import sys
sys.path.append("src")

# Define the medical types and classes
MEDICAL_TYPES = {
    'chest_xray': {
        'classes': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer'],
    },
    'brain_mri': {
        'classes': ['Normal', 'Tumor', 'Stroke', 'Hemorrhage', 'Multiple Sclerosis'],
    },
    'skin_lesion': {
        'classes': ['Normal', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
    },
    'retinal': {
        'classes': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Macular Degeneration'],
    },
    'cardiac': {
        'classes': ['Normal', 'Heart Failure', 'Arrhythmia', 'Valve Disease'],
    }
}

class MedicalPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.medical_type = checkpoint['medical_type']

        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        return {
            'prediction': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'classes': self.classes,
            'medical_type': self.medical_type
        }

# FastAPI app
app = FastAPI()

# Enable CORS for local dev with React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all available models
def load_models():
    possible_paths = [
        Path("models"),
        Path("../models"),
        Path("app/models"),
        Path(".") / "models"
    ]
    models_dir = next((p for p in possible_paths if p.exists()), None)
    predictors = {}
    if models_dir:
        for model_file in models_dir.glob("resnet_*.pth"):
            medical_type = model_file.stem.replace("resnet_", "")
            try:
                predictors[medical_type] = MedicalPredictor(str(model_file))
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
    return predictors

# Load models at startup
predictors = load_models()

@app.post("/predict/{medical_type}")
async def predict(medical_type: str, image: UploadFile = File(...)):
    if medical_type not in predictors:
        return {"error": "Invalid medical type"}

    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = predictors[medical_type].predict(img)
    return result
