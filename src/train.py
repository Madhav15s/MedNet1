import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import json

# Medical imaging types and their classes
MEDICAL_TYPES = {
    'chest_xray': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer'],
    'brain_mri': ['Normal', 'Tumor', 'Stroke', 'Hemorrhage', 'Multiple Sclerosis'],
    'skin_lesion': ['Normal', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
    'retinal': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Macular Degeneration'],
    'cardiac': ['Normal', 'Heart Failure', 'Arrhythmia', 'Valve Disease']
}

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get class directories
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all images
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
            for img_path in class_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
            for img_path in class_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(medical_type, epochs=20, batch_size=32, learning_rate=0.001):
    """Train a model for a specific medical imaging type"""
    
    print(f"🏥 Training model for {medical_type}...")
    
    # Setup paths
    data_dir = Path(f"datasets/{medical_type}")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Check if dataset exists
    if not data_dir.exists():
        print(f"❌ Dataset not found at {data_dir}")
        print(f"📁 Please create the following structure:")
        print(f"   datasets/{medical_type}/")
        for class_name in MEDICAL_TYPES[medical_type]:
            print(f"   ├── {class_name}/")
            print(f"   │   ├── image1.jpg")
            print(f"   │   ├── image2.jpg")
            print(f"   │   └── ...")
        return None
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = MedicalDataset(data_dir / "train", transform=train_transform)
    val_dataset = MedicalDataset(data_dir / "val", transform=val_transform)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = models.resnet18(pretrained=True)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"🚀 Training on {device}")
    print(f"🎯 Model: ResNet18 with {num_classes} classes")
    
    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = model_dir / f"resnet_{medical_type}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_dataset.classes,
                'medical_type': medical_type,
                'val_accuracy': val_acc
            }, model_path)
            print(f"💾 Saved best model with {val_acc:.2f}% accuracy")
    
    print(f"✅ Training completed! Best accuracy: {best_acc:.2f}%")
    return model_path

def main():
    """Main training function"""
    print("🏥 MediNet - Multi-Class Medical AI Training")
    print("=" * 50)
    
    # Available medical types
    print("Available medical imaging types:")
    for i, (medical_type, classes) in enumerate(MEDICAL_TYPES.items(), 1):
        print(f"{i}. {medical_type.replace('_', ' ').title()}")
        print(f"   Classes: {', '.join(classes)}")
    
    print("\n" + "=" * 50)
    
    # Train for each medical type
    for medical_type in MEDICAL_TYPES.keys():
        print(f"\n🎯 Training {medical_type} model...")
        model_path = train_model(medical_type)
        
        if model_path:
            print(f"✅ {medical_type} model saved to {model_path}")
        else:
            print(f"❌ Failed to train {medical_type} model")
    
    print("\n🎉 All training completed!")

if __name__ == "__main__":
    main() 