#!/usr/bin/env python3
"""
VFM Project Setup Script
Creates the complete project structure for Vision Foundation Model
Designed for Michael's Windows + PyCharm + Anaconda environment

Project: C:/Users/micha/PycharmProjects/VFM
Environment: VFM
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def create_directory_structure(base_path: Path):
    """Create the complete directory structure."""
    directories = [
        "src/vfm/models",
        "src/vfm/data",
        "src/vfm/training",
        "src/vfm/evaluation",
        "src/vfm/utils",
        "configs/models",
        "configs/data",
        "configs/experiments",
        "scripts",
        "tests/integration",
        "notebooks",
        "data/raw",
        "data/processed",
        "data/synthetic",
        "data/external",
        "models/checkpoints",
        "models/pretrained",
        "models/exports",
        "outputs/logs",
        "outputs/runs",
        "outputs/results",
        "docs/api",
        "docs/tutorials",
        "docs/examples",
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")


def create_init_files(base_path: Path):
    """Create __init__.py files for Python packages."""
    print("📝 Creating __init__.py files...")

    init_files = [
        "src/vfm/__init__.py",
        "src/vfm/models/__init__.py",
        "src/vfm/data/__init__.py",
        "src/vfm/training/__init__.py",
        "src/vfm/evaluation/__init__.py",
        "src/vfm/utils/__init__.py",
        "tests/__init__.py",
        "tests/integration/__init__.py",
    ]

    init_contents = {
        "src/vfm/__init__.py": '''"""VFM - Vision Foundation Model for Multi-Domain Object Detection."""

__version__ = "0.1.0"
__author__ = "Michael Rushin"
__email__ = "michael.r.rushin@gmail.com"

from .models import UnifiedVisionFM
from .data import SyntheticMultiDomainDataset
from .training import VFMTrainer
from .evaluation import VFMEvaluator

__all__ = [
    "UnifiedVisionFM",
    "SyntheticMultiDomainDataset",
    "VFMTrainer", 
    "VFMEvaluator",
]
''',
        "src/vfm/models/__init__.py": '''"""Model architectures for VFM."""

from .backbone import UnifiedVisionFM
from .heads import MultiDomainDetectionHead, ClassificationHead
from .fusion import CrossDomainFusion

__all__ = [
    "UnifiedVisionFM",
    "MultiDomainDetectionHead",
    "ClassificationHead", 
    "CrossDomainFusion",
]
''',
        "src/vfm/data/__init__.py": '''"""Data loading and processing."""

from .synthetic import SyntheticMultiDomainDataset
from .loaders import create_dataloader
from .transforms import get_transforms

__all__ = [
    "SyntheticMultiDomainDataset",
    "create_dataloader",
    "get_transforms",
]
''',
        "src/vfm/training/__init__.py": '''"""Training utilities."""

from .trainer import VFMTrainer
from .losses import MultiDomainLoss

__all__ = [
    "VFMTrainer", 
    "MultiDomainLoss",
]
''',
        "src/vfm/evaluation/__init__.py": '''"""Evaluation metrics and benchmarks."""

from .metrics import VFMMetrics
from .evaluator import VFMEvaluator

__all__ = [
    "VFMMetrics",
    "VFMEvaluator",
]
''',
    }

    for init_file in init_files:
        file_path = base_path / init_file
        content = init_contents.get(init_file, "# Package initialization\n")
        file_path.write_text(content, encoding='utf-8')
        print(f"   ✓ {init_file}")


def create_config_files(base_path: Path):
    """Create configuration files."""
    print("⚙️ Creating configuration files...")

    # pyproject.toml
    pyproject_content = """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vfm"
version = "0.1.0"
description = "Vision Foundation Model for Multi-Domain Object Detection"
authors = [{name = "Michael Rushin", email = "michael.r.rushin@gmail.com"}]
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pytorch-lightning>=2.0.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "rich>=13.0.0",
    "einops>=0.6.0",
    "timm>=0.9.0",
    "opencv-python>=4.8.0",
    "pillow>=9.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
]

[project.scripts]
vfm-train = "vfm.scripts.train:main"
vfm-evaluate = "vfm.scripts.evaluate:main"
vfm-demo = "vfm.scripts.demo:main"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src/vfm --cov-report=term-missing"
"""

    # environment.yml for conda
    environment_content = """name: VFM
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - pytorch-cuda=11.8
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - tqdm>=4.64.0
  - opencv
  - jupyter
  - pytest
  - pip
  - pip:
    - pytorch-lightning>=2.0.0
    - hydra-core>=1.3.0
    - omegaconf>=2.3.0
    - wandb>=0.15.0
    - rich>=13.0.0
    - einops>=0.6.0
    - timm>=0.9.0
    - black>=22.0.0
    - isort>=5.10.0
    - flake8>=5.0.0
    - pytest-cov>=4.0.0
    - pillow>=9.0.0
"""

    # Default configuration
    default_config = """# VFM Default Configuration
# Optimized for Windows development environment

model:
  name: "UnifiedVisionFM"
  domains: ["land", "air", "sea"]
  modalities: ["optical", "radar", "infrared"]
  embed_dim: 384  # Start smaller for development
  depth: 6
  num_heads: 6
  patch_size: 16
  img_size: 224
  dropout: 0.1
  use_cross_domain_fusion: true

data:
  batch_size: 4  # Small for development
  num_workers: 0  # Windows compatibility
  pin_memory: false  # Windows compatibility
  synthetic: true  # Use synthetic data for MVP
  img_size: 224
  domains: ["land", "air", "sea"]
  num_samples: 1000

training:
  max_epochs: 10
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_epochs: 2
  scheduler: "cosine"
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2

evaluation:
  metrics: ["accuracy", "f1_score", "precision", "recall"]
  save_predictions: true
  compute_confusion_matrix: true

logging:
  project: "vfm-development"
  entity: "michael-rushin"
  name: "vfm-experiment"
  log_every_n_steps: 10
  save_top_k: 3
  monitor: "val_accuracy"
  mode: "max"

hardware:
  accelerator: "auto"
  devices: 1
  precision: 16  # Mixed precision for speed
  deterministic: false

paths:
  data_dir: "data"
  output_dir: "outputs"
  model_dir: "models"
  checkpoint_dir: "models/checkpoints"
  log_dir: "outputs/logs"
"""

    # MVP demo configuration
    mvp_demo_config = """# @package _global_
defaults:
  - /default
  - _self_

# MVP Demo Configuration - Ultra-fast for development
model:
  embed_dim: 256  # Even smaller
  depth: 4
  num_heads: 4
  img_size: 128  # Smaller images

data:
  batch_size: 2  # Very small batches
  synthetic: true
  num_samples: 100  # Limited samples for speed

training:
  max_epochs: 3  # Quick training
  learning_rate: 5e-4
  warmup_epochs: 1

logging:
  name: "mvp-demo-${now:%Y%m%d_%H%M%S}"
  log_every_n_steps: 5
"""

    # Quick test configuration
    quick_test_config = """# @package _global_
defaults:
  - /default
  - _self_

# Quick Test - Minimal config for debugging
model:
  embed_dim: 128
  depth: 2
  num_heads: 2
  img_size: 64

data:
  batch_size: 2
  num_samples: 20

training:
  max_epochs: 1
  learning_rate: 1e-3

logging:
  name: "quick-test"
"""

    configs = {
        "pyproject.toml": pyproject_content,
        "environment.yml": environment_content,
        "configs/default.yaml": default_config,
        "configs/experiments/mvp_demo.yaml": mvp_demo_config,
        "configs/experiments/quick_test.yaml": quick_test_config,
    }

    for config_file, content in configs.items():
        file_path = base_path / config_file
        file_path.write_text(content, encoding='utf-8')
        print(f"   ✓ {config_file}")


def create_batch_scripts(base_path: Path):
    """Create Windows batch scripts for easy execution."""
    print("🖥️ Creating batch scripts...")

    # Use raw strings to avoid escape sequence issues
    project_path = r"C:\Users\micha\PycharmProjects\VFM"

    scripts = {
        "train.bat": f"""@echo off
echo 🚀 Training VFM Model...
call conda activate VFM
cd /d "{project_path}"
python scripts\\train.py experiment=mvp_demo
if %ERRORLEVEL% EQU 0 (
    echo ✅ Training completed successfully!
) else (
    echo ❌ Training failed with error %ERRORLEVEL%
)
pause
""",
        "evaluate.bat": f"""@echo off
echo 📊 Evaluating VFM Model...
call conda activate VFM
cd /d "{project_path}"
python scripts\\evaluate.py experiment=mvp_demo
if %ERRORLEVEL% EQU 0 (
    echo ✅ Evaluation completed successfully!
) else (
    echo ❌ Evaluation failed with error %ERRORLEVEL%
)
pause
""",
        "demo.bat": f"""@echo off
echo 🎯 Running VFM Demo...
call conda activate VFM
cd /d "{project_path}"
python scripts\\demo.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Demo completed successfully!
) else (
    echo ❌ Demo failed with error %ERRORLEVEL%
)
pause
""",
        "test.bat": f"""@echo off
echo 🧪 Running Tests...
call conda activate VFM
cd /d "{project_path}"
python -m pytest tests/ -v --cov=src/vfm
pause
""",
        "jupyter.bat": f"""@echo off
echo 📚 Starting Jupyter Lab...
call conda activate VFM
cd /d "{project_path}"
jupyter lab --no-browser --port=8888
pause
""",
    }

    for script_name, content in scripts.items():
        file_path = base_path / script_name
        file_path.write_text(content, encoding='utf-8')
        print(f"   ✓ {script_name}")


def create_gitignore(base_path: Path):
    """Create .gitignore file."""
    print("📄 Creating .gitignore...")

    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# PyTorch model files
*.pth
*.pt
*.ckpt

# Data directories
data/raw/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep

# Model artifacts
models/checkpoints/*
models/pretrained/*
models/exports/*
!models/checkpoints/.gitkeep
!models/pretrained/.gitkeep
!models/exports/.gitkeep

# Output directories
outputs/logs/*
outputs/runs/*
outputs/results/*
!outputs/logs/.gitkeep
!outputs/runs/.gitkeep
!outputs/results/.gitkeep

# Environment and IDE
.env
.vscode/
.idea/
*.swp
*.swo

# OS specific
Thumbs.db
.DS_Store

# Jupyter
.ipynb_checkpoints

# Weights & Biases
wandb/

# Coverage reports
htmlcov/
.coverage
.pytest_cache/

# MyPy
.mypy_cache/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
"""

    (base_path / ".gitignore").write_text(gitignore_content, encoding='utf-8')
    print("   ✓ .gitignore")


def create_placeholder_files(base_path: Path):
    """Create placeholder .gitkeep files for empty directories."""
    print("📄 Creating placeholder files...")

    placeholder_dirs = [
        "data/raw",
        "data/processed",
        "data/synthetic",
        "data/external",
        "models/checkpoints",
        "models/pretrained",
        "models/exports",
        "outputs/logs",
        "outputs/runs",
        "outputs/results",
    ]

    for directory in placeholder_dirs:
        gitkeep_path = base_path / directory / ".gitkeep"
        gitkeep_path.write_text("", encoding='utf-8')
        print(f"   ✓ {directory}/.gitkeep")


def create_readme(base_path: Path):
    """Create README.md file."""
    print("📖 Creating README.md...")

    readme_content = f"""# VFM - Vision Foundation Model

Multi-Domain Vision Foundation Model for Land, Air, and Sea Object Detection

**Project Location:** `C:\\Users\\micha\\PycharmProjects\\VFM`  
**Conda Environment:** `VFM`  
**Developer:** Michael Rushin  
**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Quick Start

### 1. Environment Setup
```bash
# In Anaconda Prompt:
conda env update -f environment.yml
conda activate VFM
pip install -e .
```

### 2. Quick Test
```bash
# Test installation
python -c "import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA Available: {{torch.cuda.is_available()}}')"

# Run quick demo  
python scripts\\demo.py
```

### 3. Training
```bash
# Quick MVP training
train.bat

# Or manually:
python scripts\\train.py experiment=mvp_demo
```

### 4. Evaluation
```bash
evaluate.bat
```

## PyCharm Configuration

1. **Python Interpreter:** Point to VFM conda environment
2. **Working Directory:** `C:\\Users\\micha\\PycharmProjects\\VFM`
3. **Run Configurations:**
   - Script: `scripts\\train.py`
   - Parameters: `experiment=mvp_demo`
   - Environment: `VFM`

## Project Structure

```
VFM/
├── src/vfm/           # Main package
├── configs/           # Hydra configurations
├── scripts/           # Executable scripts
├── tests/             # Test suite
├── notebooks/         # Jupyter notebooks
├── data/              # Data storage
├── models/            # Model artifacts
└── outputs/           # Training outputs
```

## Weekend MVP Goals

- [x] Project structure
- [ ] Unified backbone model
- [ ] Synthetic data generation
- [ ] Multi-domain detection heads
- [ ] Training pipeline
- [ ] Evaluation framework
- [ ] Basic tests

## Development Workflow

1. **Code in PyCharm** with VFM environment
2. **Test quickly** with `quick_test` experiment
3. **Train MVP** with `mvp_demo` experiment
4. **Debug** using PyCharm debugger
5. **Iterate** on model architecture

## Batch Scripts

- `train.bat` - Quick training
- `evaluate.bat` - Model evaluation
- `demo.bat` - Run demo
- `test.bat` - Run tests
- `jupyter.bat` - Start Jupyter Lab

Happy coding! 🚀
"""

    (base_path / "README.md").write_text(readme_content, encoding='utf-8')
    print("   ✓ README.md")


def create_basic_python_files(base_path: Path):
    """Create basic Python files with starter code."""
    print("🐍 Creating basic Python files...")

    # Basic test setup file
    test_setup_content = '''"""
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
'''

    (base_path / "test_setup.py").write_text(test_setup_content, encoding='utf-8')
    print("   ✓ test_setup.py")


def main():
    """Main setup function."""
    print("🚀 VFM Project Setup")
    print("=" * 50)
    print("Setting up Vision Foundation Model project...")
    print("Target: C:/Users/micha/PycharmProjects/VFM")
    print("Environment: VFM")
    print()

    # Determine base path
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path.cwd()

    print(f"📍 Working in: {base_path.absolute()}")
    print()

    try:
        # Create all components
        create_directory_structure(base_path)
        print()

        create_init_files(base_path)
        print()

        create_config_files(base_path)
        print()

        create_batch_scripts(base_path)
        print()

        create_gitignore(base_path)
        print()

        create_placeholder_files(base_path)
        print()

        create_readme(base_path)
        print()

        create_basic_python_files(base_path)
        print()

        print("✅ VFM project structure created successfully!")
        print()
        print("📋 Next Steps:")
        print("1. Open Anaconda Prompt")
        print("2. Run: conda env update -f environment.yml")
        print("3. Run: conda activate VFM")
        print("4. Run: pip install -e .")
        print("5. Test: python test_setup.py")
        print()
        print("🎯 PyCharm Setup:")
        print("1. File → Settings → Project → Python Interpreter")
        print("2. Select VFM conda environment")
        print("3. Set working directory to project root")
        print()
        print("🚀 Ready for weekend MVP development!")

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()