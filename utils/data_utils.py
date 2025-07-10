import os
import shutil
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_dataset_structure(medical_type, base_path="datasets"):
    """Create the directory structure for a medical imaging dataset"""
    base_dir = Path(base_path) / medical_type
    
    # Create main directories
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Medical imaging types and their classes
    MEDICAL_TYPES = {
        'chest_xray': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer'],
        'brain_mri': ['Normal', 'Tumor', 'Stroke', 'Hemorrhage', 'Multiple Sclerosis'],
        'skin_lesion': ['Normal', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
        'retinal': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Macular Degeneration'],
        'cardiac': ['Normal', 'Heart Failure', 'Arrhythmia', 'Valve Disease']
    }
    
    classes = MEDICAL_TYPES.get(medical_type, [])
    
    # Create class directories
    for class_name in classes:
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
    
    print(f"✅ Created dataset structure for {medical_type}")
    print(f"📁 Train directory: {train_dir}")
    print(f"📁 Validation directory: {val_dir}")
    print(f"📊 Classes: {', '.join(classes)}")
    
    return base_dir

def split_dataset(source_dir, train_ratio=0.8, random_seed=42):
    """Split a dataset into train and validation sets"""
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory {source_dir} does not exist")
        return
    
    # Find all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(class_dir.glob(ext))
        
        if not image_files:
            print(f"⚠️ No images found in {class_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create destination directories
        train_dest = source_path.parent / "train" / class_name
        val_dest = source_path.parent / "val" / class_name
        
        train_dest.mkdir(parents=True, exist_ok=True)
        val_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file in train_files:
            shutil.copy2(file, train_dest / file.name)
        
        for file in val_files:
            shutil.copy2(file, val_dest / file.name)
        
        print(f"📊 {class_name}: {len(train_files)} train, {len(val_files)} val")

def get_transforms(augment=True):
    """Get training and validation transforms"""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def visualize_batch(images, labels, class_names, num_images=8):
    """Visualize a batch of images with their labels"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classification_report': report,
        'accuracy': report['accuracy']
    }

def save_model_info(model_path, medical_type, classes, accuracy, epochs, batch_size, lr):
    """Save model information to a JSON file"""
    import json
    
    model_info = {
        'medical_type': medical_type,
        'classes': classes,
        'accuracy': accuracy,
        'training_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr
        },
        'model_architecture': 'ResNet18',
        'input_size': (224, 224),
        'created_at': str(Path().cwd())
    }
    
    info_path = Path(model_path).with_suffix('.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"💾 Model info saved to {info_path}")

def load_model_info(model_path):
    """Load model information from JSON file"""
    import json
    
    info_path = Path(model_path).with_suffix('.json')
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return None 