import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

# Medical imaging types and their classes
MEDICAL_TYPES = {
    'chest_xray': {
        'name': '🫁 Chest X-Ray',
        'classes': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer'],
        'description': 'Analyze chest X-rays for respiratory conditions'
    },
    'brain_mri': {
        'name': '🧠 Brain MRI',
        'classes': ['Normal', 'Tumor', 'Stroke', 'Hemorrhage', 'Multiple Sclerosis'],
        'description': 'Analyze brain MRI scans for neurological conditions'
    },
    'skin_lesion': {
        'name': '🩺 Skin Lesion',
        'classes': ['Normal', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
        'description': 'Analyze skin lesions for dermatological conditions'
    },
    'retinal': {
        'name': '👁 Retinal Image',
        'classes': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Macular Degeneration'],
        'description': 'Analyze retinal images for eye conditions'
    },
    'cardiac': {
        'name': '🫀 Cardiac Imaging',
        'classes': ['Normal', 'Heart Failure', 'Arrhythmia', 'Valve Disease'],
        'description': 'Analyze cardiac images for heart conditions'
    }
}

class MedicalPredictor:
    def __init__(self, model_path):
        """Initialize the medical predictor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.medical_type = checkpoint['medical_type']
        
        # Create model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Predict medical condition from image"""
        # Set deterministic behavior for consistent results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get results
        prediction = self.classes[predicted_class]
        all_probabilities = probabilities[0].cpu().numpy()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': all_probabilities,
            'classes': self.classes,
            'medical_type': self.medical_type
        }

@st.cache_resource
def load_models():
    """Load all available medical models"""
    # Try different possible paths for models
    possible_paths = [
        Path("models"),
        Path("../models"),
        Path("app/models"),
        Path(".") / "models"
    ]
    
    models_dir = None
    for path in possible_paths:
        if path.exists():
            models_dir = path
            break
    
    predictors = {}
    
    if models_dir and models_dir.exists():
        st.info(f"🔍 Looking for models in: {models_dir.absolute()}")
        model_files = list(models_dir.glob("resnet_*.pth"))
        
        if not model_files:
            st.error(f"❌ No model files found in {models_dir}")
            return {}
        
        for model_file in model_files:
            medical_type = model_file.stem.replace("resnet_", "")
            try:
                predictors[medical_type] = MedicalPredictor(str(model_file))
                st.success(f"✅ Loaded {medical_type} model")
            except Exception as e:
                st.error(f"❌ Failed to load {medical_type} model: {e}")
    else:
        st.error("❌ Models directory not found!")
        st.info("💡 Make sure models are in the 'models/' directory")
    
    return predictors

def main():
    """Main Streamlit app"""
    
    # Page config
    st.set_page_config(
        page_title="MediNet - Multi-Class Medical AI",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    # 🏥 MediNet - Multi-Class Medical AI System
    
    **Advanced medical image analysis for multiple imaging types**
    """)
    
    # Load models
    with st.spinner("Loading medical models..."):
        predictors = load_models()
    
    if not predictors:
        st.error("❌ No trained models found!")
        st.info("💡 Train models first using: `python src/train.py`")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Select medical type
    available_types = list(predictors.keys())
    selected_type = st.sidebar.selectbox(
        "Select Medical Imaging Type:",
        available_types,
        format_func=lambda x: MEDICAL_TYPES[x]['name']
    )
    
    # Display selected type info
    type_info = MEDICAL_TYPES[selected_type]
    st.sidebar.markdown(f"**{type_info['name']}**")
    st.sidebar.markdown(f"*{type_info['description']}*")
    st.sidebar.markdown(f"**Classes:** {', '.join(type_info['classes'])}")
    
    # Main content
    st.header(f"{type_info['name']} Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a medical image...",
        type=['png', 'jpg', 'jpeg'],
        help=f"Upload a {type_info['name'].lower()} image for analysis"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Basic image validation
                    img_array = np.array(image)
                    
                    # Check if image is grayscale (common for medical images)
                    if len(img_array.shape) == 2:
                        st.warning("⚠️ **Note:** This appears to be a grayscale image. Make sure you've selected the correct medical imaging type.")
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                        st.warning("⚠️ **Note:** This appears to be a grayscale image. Make sure you've selected the correct medical imaging type.")
                    
                    # Get prediction
                    predictor = predictors[selected_type]
                    result = predictor.predict(image)
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Add warning about model-image mismatch
                    if selected_type == 'brain_mri' and 'chest' in uploaded_file.name.lower():
                        st.error("🚨 **Warning:** You selected Brain MRI but uploaded what appears to be a chest X-ray. Please select the correct imaging type for accurate results.")
                    elif selected_type == 'chest_xray' and 'brain' in uploaded_file.name.lower():
                        st.error("🚨 **Warning:** You selected Chest X-Ray but uploaded what appears to be a brain image. Please select the correct imaging type for accurate results.")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Prediction", result['prediction'])
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    with col3:
                        st.metric("Model", "ResNet18")
                    
                    # Probability chart
                    st.markdown("### 📈 Probability Distribution")
                    
                    prob_df = {
                        'Class': result['classes'],
                        'Probability': result['probabilities']
                    }
                    
                    fig = px.bar(
                        x=prob_df['Class'], 
                        y=prob_df['Probability'],
                        color=prob_df['Probability'],
                        color_continuous_scale='RdYlBu_r',
                        title="Class Probabilities"
                    )
                    fig.update_layout(xaxis_title="Class", yaxis_title="Probability")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.markdown("### 📋 Detailed Probabilities")
                    for i, (class_name, prob) in enumerate(zip(result['classes'], result['probabilities'])):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"**{class_name}:**")
                        with col2:
                            # Convert to float to avoid progress bar issues
                            prob_float = float(prob)
                            st.progress(prob_float)
                            st.write(f"{prob_float:.2%}")
                    
                    # Medical explanation
                    st.markdown("### 💬 Medical Analysis")
                    
                    if result['prediction'] == 'Normal':
                        explanation = f"""
                        **AI Analysis:** This {type_info['name'].lower()} appears normal. 
                        The model did not detect any significant abnormalities that would 
                        indicate a medical condition requiring attention.
                        """
                    else:
                        explanation = f"""
                        **AI Analysis:** This {type_info['name'].lower()} shows signs of **{result['prediction']}**. 
                        The model detected abnormal patterns that are characteristic of this condition. 
                        Please consult with a healthcare professional for proper diagnosis and treatment.
                        """
                    
                    st.info(explanation)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    # Information section
    with st.expander("ℹ️ About MediNet"):
        st.markdown("""
        **MediNet** is a comprehensive multi-class medical AI system that can analyze various types of medical images:
        
        - **🫁 Chest X-rays** - Respiratory conditions
        - **🧠 Brain MRI** - Neurological conditions  
        - **🩺 Skin lesions** - Dermatological conditions
        - **👁 Retinal images** - Eye conditions
        - **🫀 Cardiac imaging** - Heart conditions
        
        **Features:**
        - Multi-class classification
        - High accuracy predictions
        - Confidence scoring
        - Probability distributions
        - Medical explanations
        
        **Built with:** PyTorch, ResNet18, Streamlit
        """)

if __name__ == "__main__":
    main() 