# tests/test_integration.py
"""
Integration tests for the complete VFM pipeline.
Tests the end-to-end workflow from data generation to training.
"""

import sys
from pathlib import Path
import pytest
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vfm.data.synthetic import SyntheticMultiDomainDataset, create_multi_domain_dataloader
from vfm.models.backbone import create_vfm_model
from vfm.models.heads import MultiTaskHead
from vfm.training.trainer import VFMTrainer, create_trainer_from_config


class TestVFMIntegration:
    """Integration test suite for VFM."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation for all domains."""
        print("\n🧪 Testing synthetic data generation...")

        domains = ['land', 'air', 'sea']

        for domain in domains:
            dataset = SyntheticMultiDomainDataset(
                domain=domain,
                num_samples=5,
                img_size=128
            )

            assert len(dataset) == 5

            sample = dataset[0]

            # Check basic structure
            assert 'image' in sample
            assert 'label' in sample
            assert 'domain' in sample
            assert 'boxes' in sample
            assert 'annotations' in sample

            # Check image properties
            assert sample['domain'] == domain
            assert isinstance(sample['label'], (int, torch.Tensor))
            assert sample['image'].dim() == 3  # (C, H, W)

            print(f"   ✓ {domain}: image {sample['image'].shape}, {len(sample['annotations'])} objects")

    def test_multi_domain_dataloader(self):
        """Test multi-domain dataloader."""
        print("\n🧪 Testing multi-domain dataloader...")

        dataloader = create_multi_domain_dataloader(
            domains=['land', 'air'],
            batch_size=2,
            num_samples=4,
            img_size=64,
            shuffle=False
        )

        batch = next(iter(dataloader))

        assert 'image' in batch
        assert 'domain' in batch
        assert 'label' in batch

        # Check batch dimensions
        assert batch['image'].shape[0] == 2  # batch size
        assert batch['image'].dim() == 4  # (B, C, H, W)

        print(f"   ✓ Batch shape: {batch['image'].shape}")
        print(f"   ✓ Domains in batch: {batch['domain']}")

    def test_model_creation(self):
        """Test VFM model creation and forward pass."""
        print("\n🧪 Testing model creation...")

        # Test different model sizes
        for size in ['tiny', 'small']:
            model = create_vfm_model(
                model_size=size,
                domains=['land', 'air', 'sea']
            )

            assert model.embed_dim > 0
            assert len(model.domains) == 3

            print(f"   ✓ {size} model: embed_dim={model.embed_dim}")

    def test_single_domain_forward(self):
        """Test single domain forward pass."""
        print("\n🧪 Testing single domain forward pass...")

        model = create_vfm_model('tiny')

        # Test each domain
        test_inputs = {
            'land': torch.randn(1, 13, 128, 128),  # Sentinel-2
            'air': torch.randn(1, 3, 128, 128),  # RGB
            'sea': torch.randn(1, 3, 128, 128)  # RGB
        }

        for domain, x in test_inputs.items():
            with torch.no_grad():
                features = model(x, domains=domain)

                assert domain in features
                assert 'cls_token' in features[domain]
                assert 'patch_tokens' in features[domain]

                cls_shape = features[domain]['cls_token'].shape
                patch_shape = features[domain]['patch_tokens'].shape

                assert cls_shape[0] == 1  # batch size
                assert cls_shape[1] == model.embed_dim
                assert patch_shape[0] == 1  # batch size
                assert patch_shape[2] == model.embed_dim

                print(f"   ✓ {domain}: cls {cls_shape}, patches {patch_shape}")

    def test_multi_domain_forward(self):
        """Test multi-domain forward pass with fusion."""
        print("\n🧪 Testing multi-domain forward pass...")

        model = create_vfm_model('tiny', use_cross_domain_fusion=True)

        # Multi-domain input
        inputs = {
            'land': torch.randn(2, 13, 128, 128),
            'air': torch.randn(2, 3, 128, 128),
            'sea': torch.randn(2, 3, 128, 128)
        }

        with torch.no_grad():
            features = model(inputs)

            for domain in ['land', 'air', 'sea']:
                assert domain in features
                assert features[domain]['cls_token'].shape[0] == 2  # batch size

                print(f"   ✓ {domain}: {features[domain]['cls_token'].shape}")

    def test_task_heads(self):
        """Test multi-task heads."""
        print("\n🧪 Testing task heads...")

        model = create_vfm_model('tiny')
        head = MultiTaskHead(
            embed_dim=model.embed_dim,
            domains=['land', 'air', 'sea'],
            enable_detection=True,
            enable_classification=True
        )

        # Test input
        inputs = {
            'land': torch.randn(2, 13, 128, 128),
            'air': torch.randn(2, 3, 128, 128)
        }

        with torch.no_grad():
            features = model(inputs)
            outputs = head(features, task='both')

            for domain in ['land', 'air']:
                assert domain in outputs
                assert 'classification' in outputs[domain]
                assert 'detection' in outputs[domain]

                cls_shape = outputs[domain]['classification'].shape
                assert cls_shape[0] == 2  # batch size

                print(f"   ✓ {domain}: classification {cls_shape}")

    def test_trainer_creation(self):
        """Test trainer creation and basic functionality."""
        print("\n🧪 Testing trainer creation...")

        config = {
            'model': {
                'model_size': 'tiny',
                'domains': ['land', 'air'],
                'embed_dim': 192,
                'use_cross_domain_fusion': True
            },
            'training': {
                'learning_rate': 1e-3,
                'max_epochs': 1,
                'warmup_epochs': 0
            },
            'data': {
                'batch_size': 2,
                'num_samples': 4,
                'img_size': 64,
                'domains': ['land', 'air']
            }
        }

        trainer = create_trainer_from_config(config)

        assert trainer.backbone.embed_dim == 192
        assert len(trainer.model_config['domains']) == 2

        print(f"   ✓ Trainer created with embed_dim: {trainer.backbone.embed_dim}")

    def test_training_step(self):
        """Test a single training step."""
        print("\n🧪 Testing training step...")

        config = {
            'model': {
                'model_size': 'tiny',
                'domains': ['land'],
                'embed_dim': 128,
                'use_cross_domain_fusion': False
            },
            'training': {
                'learning_rate': 1e-3,
                'max_epochs': 1
            },
            'data': {
                'batch_size': 2,
                'num_samples': 4,
                'img_size': 64,
                'domains': ['land']
            }
        }

        trainer = create_trainer_from_config(config)
        trainer.train()

        # Create sample batch
        batch = {
            'image': torch.randn(2, 13, 64, 64),
            'domain': ['land', 'land'],
            'label': torch.tensor([1, 2]),
            'boxes': [[], []]
        }

        # Test training step
        loss = trainer.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

        print(f"   ✓ Training step completed, loss: {loss.item():.4f}")

    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        print("\n🧪 Testing end-to-end pipeline...")

        # 1. Create synthetic data
        dataset = SyntheticMultiDomainDataset(
            domain='land',
            num_samples=8,
            img_size=64
        )

        # 2. Create dataloader
        dataloader = create_multi_domain_dataloader(
            domains=['land'],
            batch_size=2,
            num_samples=8,
            img_size=64
        )

        # 3. Create model and trainer
        config = {
            'model': {
                'model_size': 'tiny',
                'domains': ['land'],
                'embed_dim': 128
            },
            'training': {
                'learning_rate': 1e-3,
                'max_epochs': 1
            },
            'data': {
                'batch_size': 2,
                'num_samples': 8,
                'img_size': 64,
                'domains': ['land']
            }
        }

        trainer = create_trainer_from_config(config)

        # 4. Test forward pass on real data
        batch = next(iter(dataloader))

        with torch.no_grad():
            outputs = trainer(batch)
            assert 'land' in outputs

        # 5. Test training step
        trainer.train()
        loss = trainer.training_step(batch, 0)

        print(f"   ✓ End-to-end pipeline completed successfully!")
        print(f"   ✓ Final loss: {loss.item():.4f}")


def run_all_tests():
    """Run all integration tests."""
    print("🚀 VFM Integration Tests")
    print("=" * 50)

    test_suite = TestVFMIntegration()

    try:
        # Run all tests
        test_suite.test_synthetic_data_generation()
        test_suite.test_multi_domain_dataloader()
        test_suite.test_model_creation()
        test_suite.test_single_domain_forward()
        test_suite.test_multi_domain_forward()
        test_suite.test_task_heads()
        test_suite.test_trainer_creation()
        test_suite.test_training_step()
        test_suite.test_end_to_end_pipeline()

        print("\n✅ All integration tests passed!")
        print("🎉 VFM is ready for training!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)