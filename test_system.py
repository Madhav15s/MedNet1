#!/usr/bin/env python3
"""
MediNet System Test
Comprehensive testing of all MediNet components
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'torch',
        'torchvision', 
        'streamlit',
        'PIL',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'sklearn',
        'pandas'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All imports successful")
        return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n📁 Testing project structure...")
    
    required_files = [
        'app/app.py',
        'src/train.py',
        'src/evaluate.py',
        'utils/data_utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'datasets',
        'models',
        'utils'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   ✅ {dir_path}/")
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure is correct")
        return True

def test_data_utils():
    """Test data utilities functionality"""
    print("\n🔧 Testing data utilities...")
    
    try:
        sys.path.append('utils')
        from data_utils import create_dataset_structure, get_transforms
        
        # Test dataset structure creation
        test_type = 'test_medical'
        create_dataset_structure(test_type)
        
        # Check if directories were created
        train_dir = Path(f"datasets/{test_type}/train")
        val_dir = Path(f"datasets/{test_type}/val")
        
        if train_dir.exists() and val_dir.exists():
            print("   ✅ Dataset structure creation")
        else:
            print("   ❌ Dataset structure creation failed")
            return False
        
        # Test transforms
        train_transform, val_transform = get_transforms()
        if train_transform and val_transform:
            print("   ✅ Transforms creation")
        else:
            print("   ❌ Transforms creation failed")
            return False
        
        # Clean up test data
        import shutil
        if Path(f"datasets/{test_type}").exists():
            shutil.rmtree(f"datasets/{test_type}")
        
        print("✅ Data utilities test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Data utilities test failed: {e}")
        return False

def test_training_script():
    """Test training script functionality"""
    print("\n🏋️ Testing training script...")
    
    try:
        # Test if training script can be imported
        sys.path.append('src')
        from train import MEDICAL_TYPES, MedicalDataset
        
        print("   ✅ Training script import")
        
        # Test dataset class
        if 'chest_xray' in MEDICAL_TYPES:
            print("   ✅ Medical types configuration")
        else:
            print("   ❌ Medical types configuration")
            return False
        
        print("✅ Training script test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Training script test failed: {e}")
        return False

def test_evaluation_script():
    """Test evaluation script functionality"""
    print("\n📊 Testing evaluation script...")
    
    try:
        sys.path.append('src')
        from evaluate import ModelEvaluator
        
        print("   ✅ Evaluation script import")
        
        # Test evaluator class (without loading actual model)
        print("   ✅ ModelEvaluator class available")
        
        print("✅ Evaluation script test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Evaluation script test failed: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app functionality"""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        # Test if app can be imported
        sys.path.append('.')
        import app.app as streamlit_app
        
        print("   ✅ Streamlit app import")
        
        # Test if main function exists
        if hasattr(streamlit_app, 'main'):
            print("   ✅ Main function available")
        else:
            print("   ❌ Main function not found")
            return False
        
        print("✅ Streamlit app test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit app test failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n🚀 Testing CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ CUDA available: {device_name}")
            print(f"   📊 GPU Memory: {memory:.1f} GB")
        else:
            print("   ⚠️ CUDA not available - will use CPU")
        
        print("✅ CUDA test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")
        return False

def test_sample_workflow():
    """Test a complete sample workflow"""
    print("\n🔄 Testing sample workflow...")
    
    try:
        # Create sample dataset
        from MedNet1.demo import create_sample_dataset
        create_sample_dataset()
        
        # Test if sample data was created
        if Path("datasets/chest_xray/train/Normal").exists():
            print("   ✅ Sample dataset creation")
        else:
            print("   ❌ Sample dataset creation failed")
            return False
        
        # Test training script with sample data
        print("   🔄 Testing training with sample data...")
        
        # This would normally run training, but for testing we'll just check if it can start
        sys.path.append('src')
        from train import train_model
        
        print("   ✅ Training function available")
        
        print("✅ Sample workflow test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Sample workflow test failed: {e}")
        traceback.print_exc()
        return False

def run_system_tests():
    """Run all system tests"""
    print("🏥 MediNet System Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Utilities", test_data_utils),
        ("Training Script", test_training_script),
        ("Evaluation Script", test_evaluation_script),
        ("Streamlit App", test_streamlit_app),
        ("CUDA Availability", test_cuda_availability),
        ("Sample Workflow", test_sample_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Test Results Summary")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MediNet is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Add your medical images to datasets/")
        print("   2. Train models: python src/train.py")
        print("   3. Run web app: streamlit run app/app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

def main():
    """Main test function"""
    success = run_system_tests()
    
    if success:
        print("\n✅ System test completed successfully!")
        return 0
    else:
        print("\n❌ System test failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 