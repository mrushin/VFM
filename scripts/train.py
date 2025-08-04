# scripts/train.py
"""
Main training script for Vision Foundation Model.
Usage: python scripts/train.py experiment=mvp_demo
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import os

from vfm.training.trainer import VFMTrainer, create_trainer_from_config


def setup_callbacks(config: DictConfig) -> list:
    """Setup training callbacks."""

    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename='vfm-{epoch:02d}-{val_loss:.3f}',
        monitor=config.logging.get('monitor', 'val_loss'),
        mode=config.logging.get('mode', 'min'),
        save_top_k=config.logging.get('save_top_k', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.training.get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor=config.logging.get('monitor', 'val_loss'),
            patience=config.training.get('patience', 5),
            mode=config.logging.get('mode', 'min'),
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(config: DictConfig):
    """Setup experiment logger."""

    logger = TensorBoardLogger(
        save_dir=config.paths.log_dir,
        name=config.logging.get('name', 'vfm_experiment'),
        version=None  # Auto-generate version
    )

    return logger


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    """Main training function."""

    print("🚀 Starting VFM Training")
    print("=" * 50)

    # Print configuration
    print("📋 Configuration:")
    print(OmegaConf.to_yaml(config))
    print()

    # Set random seed for reproducibility
    if hasattr(config, 'seed'):
        pl.seed_everything(config.seed, workers=True)
    else:
        pl.seed_everything(42, workers=True)

    # Create output directories
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)

    # Initialize trainer
    print("🏗️ Building model...")
    model = create_trainer_from_config(OmegaConf.to_container(config, resolve=True))
    print(f"   ✓ Model: {config.model.name}")
    print(f"   ✓ Domains: {config.model.domains}")
    print(f"   ✓ Embed dim: {config.model.embed_dim}")
    print(f"   ✓ Cross-domain fusion: {config.model.get('use_cross_domain_fusion', False)}")
    print()

    # Setup callbacks and logger
    print("⚙️ Setting up training...")
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)

    # Configure trainer
    trainer_config = {
        'max_epochs': config.training.max_epochs,
        'accelerator': config.hardware.accelerator,
        'devices': config.hardware.devices,
        'precision': config.hardware.get('precision', 32),
        'gradient_clip_val': config.training.get('gradient_clip_val', 1.0),
        'accumulate_grad_batches': config.training.get('accumulate_grad_batches', 1),
        'callbacks': callbacks,
        'logger': logger,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': config.hardware.get('deterministic', False),
        'log_every_n_steps': config.logging.get('log_every_n_steps', 10),
    }

    # Add fast dev run for debugging
    if config.get('debug', {}).get('fast_dev_run', False):
        trainer_config['fast_dev_run'] = True
        print("🐛 Debug mode: Fast dev run enabled")

    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(**trainer_config)

    print(f"   ✓ Max epochs: {config.training.max_epochs}")
    print(f"   ✓ Learning rate: {config.training.learning_rate}")
    print(f"   ✓ Batch size: {config.data.batch_size}")
    print(f"   ✓ Hardware: {config.hardware.accelerator}")
    print()

    # Check for resume from checkpoint
    resume_checkpoint = None
    if config.get('resume_from_checkpoint'):
        checkpoint_path = Path(config.resume_from_checkpoint)
        if checkpoint_path.exists():
            resume_checkpoint = str(checkpoint_path)
            print(f"📁 Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")

    try:
        # Start training
        print("🎯 Starting training...")
        trainer.fit(model, ckpt_path=resume_checkpoint)

        # Training completed
        print("\n✅ Training completed successfully!")

        # Print best model info
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            print(f"🏆 Best model saved at: {trainer.checkpoint_callback.best_model_path}")
            print(f"🏆 Best score: {trainer.checkpoint_callback.best_model_score:.4f}")

        # Save final model
        final_model_path = Path(config.paths.checkpoint_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"💾 Final model saved at: {final_model_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

        # Save current state
        interrupt_path = Path(config.paths.checkpoint_dir) / "interrupted_model.ckpt"
        trainer.save_checkpoint(interrupt_path)
        print(f"💾 Interrupted model saved at: {interrupt_path}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise e

    print("\n🎉 Training session complete!")


if __name__ == "__main__":
    main()

# scripts/demo.py
"""
Quick demo script to test VFM functionality.
Usage: python scripts/demo.py
"""


def run_demo():
    """Run a quick demo of the VFM model."""
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent / "src"))

    import torch
    from vfm.models.backbone import create_vfm_model
    from vfm.models.heads import MultiTaskHead
    from vfm.data.synthetic import SyntheticMultiDomainDataset

    print("🎯 VFM Quick Demo")
    print("=" * 40)

    # Create model
    print("🏗️ Creating VFM model...")
    model = create_vfm_model('small', domains=['land', 'air', 'sea'])

    # Create multi-task head
    head = MultiTaskHead(
        embed_dim=model.embed_dim,
        domains=['land', 'air', 'sea'],
        enable_detection=True,
        enable_classification=True
    )

    print(f"   ✓ Model created with embed_dim: {model.embed_dim}")
    print(f"   ✓ Domains: {model.domains}")

    # Test with synthetic data
    print("\n📊 Testing with synthetic data...")

    for domain in ['land', 'air', 'sea']:
        print(f"\n🧪 Testing {domain} domain...")

        # Create synthetic dataset
        dataset = SyntheticMultiDomainDataset(domain=domain, num_samples=3, img_size=224)
        sample = dataset[0]

        print(f"   ✓ Image shape: {sample['image'].shape}")
        print(f"   ✓ Objects detected: {len(sample['annotations'])}")

        # Test model forward pass
        with torch.no_grad():
            # Single domain input
            features = model(sample['image'].unsqueeze(0), domains=domain)

            # Apply task heads
            outputs = head({domain: features[domain]}, task='both')

            if 'classification' in outputs[domain]:
                cls_logits = outputs[domain]['classification']
                pred_class = torch.argmax(cls_logits, dim=1).item()
                confidence = torch.softmax(cls_logits, dim=1).max().item()

                print(f"   ✓ Classification - Predicted class: {pred_class}, Confidence: {confidence:.3f}")

            if 'detection' in outputs[domain]:
                det_output = outputs[domain]['detection']
                if 'class_logits' in det_output:
                    print(f"   ✓ Detection - Class logits shape: {det_output['class_logits'].shape}")

    # Test multi-domain
    print(f"\n🌍 Testing multi-domain processing...")

    # Create multi-domain input
    multi_input = {
        'land': torch.randn(1, 13, 224, 224),  # Sentinel-2
        'air': torch.randn(1, 3, 224, 224),  # RGB
        'sea': torch.randn(1, 3, 224, 224)  # RGB
    }

    with torch.no_grad():
        features = model(multi_input)
        outputs = head(features, task='both')

        for domain in ['land', 'air', 'sea']:
            if domain in outputs and 'classification' in outputs[domain]:
                cls_logits = outputs[domain]['classification']
                pred_class = torch.argmax(cls_logits, dim=1).item()
                print(f"   ✓ {domain} - Predicted class: {pred_class}")

    print(f"\n✅ Demo completed successfully!")
    print(f"🚀 Ready for training with: python scripts/train.py experiment=mvp_demo")


if __name__ == "__main__":
    run_demo()

# scripts/evaluate.py
"""
Evaluation script for VFM model.
Usage: python scripts/evaluate.py experiment=mvp_demo
"""


def main():
    print("📊 VFM Evaluation")
    print("=" * 40)
    print("🚧 Evaluation script will be implemented next!")
    print("For now, use the demo script: python scripts/demo.py")


if __name__ == "__main__":
    main()