"""
Quick test to verify VFM setup is working.
Run this after setup to ensure everything is configured correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all major dependencies can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ CUDA Available: {torch.cuda.is_available()}")
        
        import pytorch_lightning as pl
        print(f"   ✓ PyTorch Lightning: {pl.__version__}")
        
        import hydra
        print("   ✓ Hydra configuration")
        
        import numpy as np
        print(f"   ✓ NumPy: {np.__version__}")
        
        import matplotlib
        print(f"   ✓ Matplotlib: {matplotlib.__version__}")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_project_structure():
    """Test that project structure is created correctly."""
    print("🧪 Testing project structure...")
    
    required_dirs = [
        "src/vfm",
        "configs",
        "scripts", 
        "tests",
        "data",
        "models",
        "outputs"
    ]
    
    base_path = Path(__file__).parent
    all_good = True
    
    for directory in required_dirs:
        dir_path = base_path / directory
        if dir_path.exists():
            print(f"   ✓ {directory}")
        else:
            print(f"   ❌ {directory} - missing")
            all_good = False
    
    if all_good:
        print("✅ Project structure is correct!")
    else:
        print("❌ Some directories are missing")
    
    return all_good


def main():
    """Run all setup tests."""
    print("🚀 VFM Setup Verification")
    print("=" * 50)
    
    imports_ok = test_imports()
    print()
    structure_ok = test_project_structure()
    print()
    
    if imports_ok and structure_ok:
        print("🎉 VFM setup is complete and working!")
        print("Ready for weekend MVP development! 🚀")
    else:
        print("⚠️ Setup issues detected. Please check the errors above.")
    
    return imports_ok and structure_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
