import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import argparse
import sys

# Add utils to path
sys.path.append('utils')

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

class ModelEvaluator:
    def __init__(self, model_path):
        """Initialize the model evaluator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.medical_type = checkpoint['medical_type']
        
        # Create model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_on_dataset(self, data_dir):
        """Evaluate model on a dataset"""
        # Create dataset
        dataset = MedicalDataset(data_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        print(f"📊 Evaluating on {len(dataset)} samples")
        print(f"🎯 Classes: {self.classes}")
        
        # Evaluate
        results = evaluate_model(self.model, dataloader, self.device, self.classes)
        
        return results
    
    def generate_report(self, results, output_dir="evaluation_reports"):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create report directory for this model
        model_name = f"{self.medical_type}_evaluation"
        report_dir = output_path / model_name
        report_dir.mkdir(exist_ok=True)
        
        # Save results
        results_path = report_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'medical_type': self.medical_type,
                'classes': self.classes,
                'accuracy': results['accuracy'],
                'classification_report': results['classification_report']
            }, f, indent=2)
        
        # Plot confusion matrix
        cm_path = report_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            results['labels'], 
            results['predictions'], 
            self.classes,
            f"Confusion Matrix - {self.medical_type.replace('_', ' ').title()}"
        )
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curves
        self.plot_roc_curves(results, report_dir)
        
        # Generate detailed metrics
        self.generate_metrics_table(results, report_dir)
        
        print(f"📊 Evaluation report saved to {report_dir}")
        return report_dir
    
    def plot_roc_curves(self, results, report_dir):
        """Plot ROC curves for each class"""
        probabilities = np.array(results['probabilities'])
        labels = np.array(results['labels'])
        
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.classes):
            # Create binary labels for this class
            binary_labels = (labels == i).astype(int)
            binary_probs = probabilities[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(binary_labels, binary_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.medical_type.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True)
        
        roc_path = report_dir / "roc_curves.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_metrics_table(self, results, report_dir):
        """Generate detailed metrics table"""
        report = results['classification_report']
        
        # Create metrics DataFrame
        metrics_data = []
        for class_name in self.classes:
            if class_name in report:
                metrics_data.append({
                    'Class': class_name,
                    'Precision': report[class_name]['precision'],
                    'Recall': report[class_name]['recall'],
                    'F1-Score': report[class_name]['f1-score'],
                    'Support': report[class_name]['support']
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Save as CSV
        csv_path = report_dir / "detailed_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>MediNet Evaluation Report - {self.medical_type}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .accuracy {{ font-size: 24px; color: #4CAF50; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 MediNet Evaluation Report</h1>
                <h2>{self.medical_type.replace('_', ' ').title()}</h2>
                <p><strong>Model:</strong> ResNet18</p>
                <p><strong>Classes:</strong> {', '.join(self.classes)}</p>
                <p class="accuracy">Overall Accuracy: {results['accuracy']:.2%}</p>
            </div>
            
            <div class="metrics">
                <h3>📊 Detailed Metrics</h3>
                {df.to_html(index=False)}
            </div>
            
            <div class="metrics">
                <h3>📈 Visualizations</h3>
                <p>Confusion Matrix: <a href="confusion_matrix.png">confusion_matrix.png</a></p>
                <p>ROC Curves: <a href="roc_curves.png">roc_curves.png</a></p>
            </div>
        </body>
        </html>
        """
        
        html_path = report_dir / "report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def predict_single_image(self, image_path):
        """Predict on a single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return {
            'prediction': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'classes': self.classes
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate MediNet models')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, help='Path to test dataset')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--output', type=str, default='evaluation_reports', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    print(f"🏥 Evaluating {evaluator.medical_type} model")
    print(f"🎯 Classes: {evaluator.classes}")
    
    # Evaluate on dataset if provided
    if args.data:
        if not Path(args.data).exists():
            print(f"❌ Dataset not found: {args.data}")
            return
        
        results = evaluator.evaluate_on_dataset(args.data)
        report_dir = evaluator.generate_report(results, args.output)
        
        print(f"✅ Evaluation completed!")
        print(f"📊 Overall Accuracy: {results['accuracy']:.2%}")
        print(f"📁 Report saved to: {report_dir}")
    
    # Predict on single image if provided
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        result = evaluator.predict_single_image(args.image)
        
        print(f"\n🔍 Single Image Prediction:")
        print(f"📸 Image: {args.image}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"📈 Probabilities:")
        for class_name, prob in zip(result['classes'], result['probabilities']):
            print(f"   {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main() 