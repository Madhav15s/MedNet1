#!/usr/bin/env python3
"""
MediNet Demo Script
Demonstrates the key features of the MediNet medical AI system
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import streamlit as st

# Add project paths
sys.path.append('src')
sys.path.append('utils')

def create_sample_image(size=(224, 224), text="Sample", color=(100, 150, 200)):
    """Create a sample medical image for demonstration"""
    # Create a simple image with text
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Add some medical-looking elements
    # Draw a simple cross symbol
    center_x, center_y = size[0] // 2, size[1] // 2
    cross_size = 30
    
    # Vertical line
    draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)], 
              fill=(255, 255, 255), width=3)
    # Horizontal line
    draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)], 
              fill=(255, 255, 255), width=3)
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (size[0] - text_width) // 2
    text_y = center_y + cross_size + 10
    
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return img

def create_sample_dataset():
    """Create sample dataset structure with dummy images"""
    print("📁 Creating sample dataset...")
    
    # Medical types and their classes
    medical_types = {
        'chest_xray': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer'],
        'brain_mri': ['Normal', 'Tumor', 'Stroke', 'Hemorrhage', 'Multiple Sclerosis'],
        'skin_lesion': ['Normal', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'],
        'retinal': ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Macular Degeneration'],
        'cardiac': ['Normal', 'Heart Failure', 'Arrhythmia', 'Valve Disease']
    }
    
    # Colors for different medical types
    colors = {
        'chest_xray': (150, 200, 255),  # Blue
        'brain_mri': (255, 200, 150),    # Orange
        'skin_lesion': (200, 255, 150),  # Green
        'retinal': (255, 150, 200),      # Pink
        'cardiac': (200, 150, 255)       # Purple
    }
    
    for medical_type, classes in medical_types.items():
        print(f"   Creating {medical_type} dataset...")
        
        # Create directories
        train_dir = Path(f"datasets/{medical_type}/train")
        val_dir = Path(f"datasets/{medical_type}/val")
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        color = colors[medical_type]
        
        for class_name in classes:
            # Create train and val directories for each class
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            
            # Create sample images
            for i in range(5):  # 5 sample images per class
                # Train images
                img = create_sample_image(text=f"{class_name}", color=color)
                img.save(train_dir / class_name / f"sample_{i+1}.jpg")
                
                # Val images (fewer)
                if i < 2:
                    img = create_sample_image(text=f"{class_name}", color=color)
                    img.save(val_dir / class_name / f"sample_{i+1}.jpg")
    
    print("✅ Sample dataset created")

def demo_training():
    """Demonstrate training functionality"""
    print("\n🏋️ Demo: Training Process")
    print("="*40)
    
    # Check if we have sample data
    if not Path("datasets/chest_xray/train").exists():
        print("❌ No sample dataset found. Creating sample dataset...")
        create_sample_dataset()
    
    print("📊 Training a chest X-ray model...")
    print("   This would normally take several minutes with real data.")
    print("   For demo purposes, we'll simulate the training process.")
    
    # Simulate training progress
    epochs = 5  # Reduced for demo
    for epoch in range(epochs):
        train_loss = 2.0 - (epoch * 0.3) + np.random.normal(0, 0.1)
        val_loss = 2.1 - (epoch * 0.25) + np.random.normal(0, 0.1)
        train_acc = 0.3 + (epoch * 0.15) + np.random.normal(0, 0.05)
        val_acc = 0.25 + (epoch * 0.12) + np.random.normal(0, 0.05)
        
        print(f"   Epoch {epoch+1}/{epochs}")
        print(f"     Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"     Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
    
    print("✅ Training simulation completed!")

def demo_prediction():
    """Demonstrate prediction functionality"""
    print("\n🔍 Demo: Prediction Process")
    print("="*40)
    
    # Create a sample medical image
    sample_img = create_sample_image(text="Chest X-Ray", color=(150, 200, 255))
    
    # Simulate model prediction
    classes = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer']
    
    # Generate realistic probabilities
    base_probs = [0.4, 0.25, 0.15, 0.12, 0.08]  # Normal is most likely
    noise = np.random.normal(0, 0.05, len(classes))
    probabilities = np.array(base_probs) + noise
    probabilities = np.clip(probabilities, 0, 1)
    probabilities = probabilities / probabilities.sum()  # Normalize
    
    predicted_class = classes[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    
    print(f"📸 Analyzing sample chest X-ray image...")
    print(f"🎯 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"📈 Class Probabilities:")
    
    for class_name, prob in zip(classes, probabilities):
        bar = "█" * int(prob * 20)
        print(f"   {class_name:15} {prob:.3f} {bar}")
    
    # Save sample image
    sample_img.save("demo_sample_image.jpg")
    print(f"💾 Sample image saved as 'demo_sample_image.jpg'")
    
    return sample_img, classes, probabilities

def demo_web_interface():
    """Demonstrate web interface functionality"""
    print("\n🌐 Demo: Web Interface")
    print("="*40)
    
    print("The MediNet web interface provides:")
    print("   ✅ Interactive model selection")
    print("   ✅ Drag-and-drop image upload")
    print("   ✅ Real-time prediction results")
    print("   ✅ Probability visualizations")
    print("   ✅ Medical explanations")
    print("   ✅ Confidence scoring")
    
    print("\nTo launch the web interface:")
    print("   streamlit run app/app.py")
    
    print("\nFeatures available in the web app:")
    print("   🫁 Chest X-Ray Analysis")
    print("   🧠 Brain MRI Analysis") 
    print("   🩺 Skin Lesion Analysis")
    print("   👁 Retinal Image Analysis")
    print("   🫀 Cardiac Imaging Analysis")

def demo_evaluation():
    """Demonstrate evaluation functionality"""
    print("\n📊 Demo: Model Evaluation")
    print("="*40)
    
    # Simulate evaluation metrics
    classes = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung Cancer']
    
    # Generate sample confusion matrix
    confusion_matrix = np.array([
        [45, 2, 1, 1, 1],   # Normal predictions
        [3, 42, 2, 2, 1],   # Pneumonia predictions
        [1, 2, 38, 3, 1],   # COVID-19 predictions
        [1, 1, 2, 41, 2],   # Tuberculosis predictions
        [1, 1, 1, 2, 40]    # Lung Cancer predictions
    ])
    
    print("📈 Confusion Matrix:")
    print("      Predicted →")
    print("Actual ↓")
    
    # Print header
    print("      ", end="")
    for class_name in classes:
        print(f"{class_name[:8]:>8}", end="")
    print()
    
    # Print matrix
    for i, class_name in enumerate(classes):
        print(f"{class_name[:8]:>8}", end="")
        for j in range(len(classes)):
            print(f"{confusion_matrix[i, j]:>8}", end="")
        print()
    
    # Calculate metrics
    total = confusion_matrix.sum()
    accuracy = np.trace(confusion_matrix) / total
    
    print(f"\n📊 Overall Accuracy: {accuracy:.2%}")
    
    # Per-class metrics
    print("\n📋 Per-Class Metrics:")
    for i, class_name in enumerate(classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {class_name:15} Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

def create_demo_visualization():
    """Create a demo visualization"""
    print("\n📈 Demo: Visualization")
    print("="*40)
    
    # Create sample training history
    epochs = range(1, 21)
    train_loss = [2.0 - 0.08 * i + np.random.normal(0, 0.05) for i in epochs]
    val_loss = [2.1 - 0.07 * i + np.random.normal(0, 0.05) for i in epochs]
    train_acc = [0.3 + 0.035 * i + np.random.normal(0, 0.02) for i in epochs]
    val_acc = [0.25 + 0.032 * i + np.random.normal(0, 0.02) for i in epochs]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Training history visualization saved as 'demo_training_history.png'")

def main():
    """Main demo function"""
    print("🏥 MediNet Demo")
    print("="*50)
    print("This demo showcases the key features of the MediNet medical AI system.")
    print("="*50)
    
    # Run all demos
    demo_training()
    demo_prediction()
    demo_evaluation()
    demo_web_interface()
    create_demo_visualization()
    
    print("\n" + "="*50)
    print("🎉 Demo completed successfully!")
    print("="*50)
    print("\n📋 Generated files:")
    print("   📸 demo_sample_image.jpg - Sample medical image")
    print("   📊 demo_training_history.png - Training visualization")
    print("\n🚀 Next steps:")
    print("   1. Add real medical images to datasets/")
    print("   2. Train models: python src/train.py")
    print("   3. Run web app: streamlit run app/app.py")
    print("   4. Evaluate models: python src/evaluate.py")
    print("="*50)

if __name__ == "__main__":
    main() 