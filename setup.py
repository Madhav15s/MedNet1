#!/usr/bin/env python3
"""
MediNet Setup Script
Initialize the MediNet project structure and install dependencies
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_project_structure():
    """Create the basic project structure"""
    print("📁 Creating project structure...")
    
    directories = [
        "datasets",
        "models", 
        "evaluation_reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    # Create dataset structures for all medical types
    medical_types = ['chest_xray', 'brain_mri', 'skin_lesion', 'retinal', 'cardiac']
    
    for medical_type in medical_types:
        try:
            # Import and create dataset structure
            sys.path.append('utils')
            from data_utils import create_dataset_structure
            create_dataset_structure(medical_type)
        except ImportError:
            print(f"   ⚠️ Could not create dataset structure for {medical_type}")
    
    print("✅ Project structure created")

def create_sample_scripts():
    """Create sample scripts for common tasks"""
    print("📝 Creating sample scripts...")
    
    # Sample training script
    sample_train = '''#!/usr/bin/env python3
"""
Sample training script for MediNet
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from train import main as train_main

if __name__ == "__main__":
    print("🏥 Starting MediNet training...")
    train_main()
'''
    
    # Sample evaluation script
    sample_eval = '''#!/usr/bin/env python3
"""
Sample evaluation script for MediNet
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from evaluate import ModelEvaluator

def main():
    # Example: Evaluate chest X-ray model
    model_path = "models/resnet_chest_xray.pth"
    test_data = "datasets/chest_xray/val"
    
    if Path(model_path).exists():
        evaluator = ModelEvaluator(model_path)
        results = evaluator.evaluate_on_dataset(test_data)
        evaluator.generate_report(results)
        print(f"📊 Evaluation completed with {results['accuracy']:.2%} accuracy")
    else:
        print(f"❌ Model not found: {model_path}")
        print("💡 Train a model first using: python src/train.py")

if __name__ == "__main__":
    main()
'''
    
    # Write sample scripts
    with open("sample_train.py", "w") as f:
        f.write(sample_train)
    
    with open("sample_eval.py", "w") as f:
        f.write(sample_eval)
    
    # Make scripts executable
    os.chmod("sample_train.py", 0o755)
    os.chmod("sample_eval.py", 0o755)
    
    print("   ✅ Created sample_train.py")
    print("   ✅ Created sample_eval.py")

def create_config_file():
    """Create a configuration file"""
    print("⚙️ Creating configuration file...")
    
    config_content = '''# MediNet Configuration
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

# Training parameters
TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'model_architecture': 'resnet18',
    'input_size': (224, 224)
}

# Data augmentation settings
AUGMENTATION_CONFIG = {
    'random_flip': True,
    'random_rotation': 10,
    'color_jitter': True,
    'random_affine': True
}
'''
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("   ✅ Created config.py")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ CUDA is not available - training will use CPU")
        return True
    except ImportError:
        print("⚠️ PyTorch not installed - CUDA check skipped")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*50)
    print("🎉 MediNet setup completed successfully!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. 📁 Add your medical images to the datasets/ directory")
    print("   Example: datasets/chest_xray/train/Normal/image1.jpg")
    print("\n2. 🏋️ Train models:")
    print("   python src/train.py")
    print("\n3. 🌐 Run the web application:")
    print("   streamlit run app/app.py")
    print("\n4. 📊 Evaluate models:")
    print("   python src/evaluate.py --model models/resnet_chest_xray.pth --data datasets/chest_xray/val")
    print("\n📚 For more information, see README.md")
    print("="*50)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup MediNet project')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-structure', action='store_true', help='Skip project structure creation')
    
    args = parser.parse_args()
    
    print("🏥 MediNet Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            sys.exit(1)
    else:
        print("⏭️ Skipping dependency installation")
    
    # Create project structure
    if not args.skip_structure:
        create_project_structure()
        create_sample_scripts()
        create_config_file()
    else:
        print("⏭️ Skipping project structure creation")
    
    # Check CUDA
    check_cuda()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 