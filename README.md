# VFM - Vision Foundation Model

Multi-Domain Vision Foundation Model for Land, Air, and Sea Object Detection

**Project Location:** `C:\Users\micha\PycharmProjects\VFM`  
**Conda Environment:** `VFM`  
**Developer:** Michael Rushin  
**Created:** 2025-08-02 01:07:13

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
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run quick demo  
python scripts\demo.py
```

### 3. Training
```bash
# Quick MVP training
train.bat

# Or manually:
python scripts\train.py experiment=mvp_demo
```

### 4. Evaluation
```bash
evaluate.bat
```

## PyCharm Configuration

1. **Python Interpreter:** Point to VFM conda environment
2. **Working Directory:** `C:\Users\micha\PycharmProjects\VFM`
3. **Run Configurations:**
   - Script: `scripts\train.py`
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
