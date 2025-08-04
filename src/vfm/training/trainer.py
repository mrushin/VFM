# src/vfm/training/trainer.py
"""
VFM Training module using PyTorch Lightning.
Handles multi-domain training with classification and detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path

# Import our components
from ..models.backbone import UnifiedVisionFM, create_vfm_model
from ..models.heads import MultiTaskHead
from ..data.synthetic import create_multi_domain_dataloader


class VFMTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for Vision Foundation Model.
    Supports multi-domain training with both classification and detection.
    """

    def __init__(
            self,
            model_config: Dict = None,
            training_config: Dict = None,
            data_config: Dict = None,
            **kwargs
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Default configurations
        self.model_config = model_config or {
            'model_size': 'small',
            'domains': ['land', 'air', 'sea'],
            'embed_dim': 384,
            'use_cross_domain_fusion': True
        }

        self.training_config = training_config or {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 2,
            'max_epochs': 10,
            'gradient_clip_val': 1.0
        }

        self.data_config = data_config or {
            'batch_size': 4,
            'num_samples': 1000,
            'img_size': 224,
            'domains': ['land', 'air', 'sea']
        }

        # Domain configurations
        self.domain_configs = {
            'land': {'num_classes': 10},
            'air': {'num_classes': 5},
            'sea': {'num_classes': 8}
        }

        # Build model
        self._build_model()

        # Loss functions
        self._setup_losses()

        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}

    def _build_model(self):
        """Build the VFM model with backbone and heads."""

        # Create backbone
        self.backbone = create_vfm_model(
            model_size=self.model_config.get('model_size', 'small'),
            domains=self.model_config.get('domains', ['land', 'air', 'sea']),
            embed_dim=self.model_config.get('embed_dim', 384),
            use_cross_domain_fusion=self.model_config.get('use_cross_domain_fusion', True)
        )

        # Create multi-task head
        self.head = MultiTaskHead(
            embed_dim=self.backbone.embed_dim,
            domains=self.model_config.get('domains', ['land', 'air', 'sea']),
            num_classes_per_domain=self.domain_configs,
            enable_detection=True,
            enable_classification=True,
            img_size=self.data_config.get('img_size', 224)
        )

    def _setup_losses(self):
        """Setup loss functions for different tasks."""

        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

        # Detection losses
        self.objectness_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.SmoothL1Loss()

        # Domain classification loss
        self.domain_loss = nn.CrossEntropyLoss()

        # Loss weights
        self.loss_weights = {
            'classification': 1.0,
            'objectness': 1.0,
            'bbox': 2.0,
            'domain': 0.5
        }

    def forward(self, batch: Dict) -> Dict:
        """Forward pass through the model."""

        # Prepare inputs based on batch structure
        if 'image' in batch:
            # Single image format
            images = batch['image']
            domains = batch['domain']

            # Group by domain
            domain_inputs = {}
            for i, domain in enumerate(domains):
                if domain not in domain_inputs:
                    domain_inputs[domain] = []
                domain_inputs[domain].append(images[i])

            # Stack images per domain
            for domain in domain_inputs:
                domain_inputs[domain] = torch.stack(domain_inputs[domain])
        else:
            # Dictionary format
            domain_inputs = {k: v for k, v in batch.items() if k in self.model_config['domains']}

        # Extract features using backbone
        features = self.backbone(domain_inputs)

        # Apply multi-task head
        outputs = self.head(features, task='both')

        return outputs

    def _compute_classification_loss(self, outputs: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute classification loss."""

        total_loss = 0.0
        metrics = {}

        # Get labels
        if isinstance(batch['domain'], list):
            domains = batch['domain']
            labels = batch['label']
        else:
            domains = [batch['domain'][i] for i in range(len(batch['domain']))]
            labels = batch['label']

        # Group labels by domain
        domain_labels = {}
        for i, domain in enumerate(domains):
            if domain not in domain_labels:
                domain_labels[domain] = []
            domain_labels[domain].append(labels[i] if isinstance(labels, (list, tuple)) else labels[i].item())

        # Compute loss per domain
        for domain in outputs.keys():
            if 'classification' in outputs[domain] and domain in domain_labels:
                pred_logits = outputs[domain]['classification']
                true_labels = torch.tensor(domain_labels[domain], device=pred_logits.device, dtype=torch.long)

                loss = self.classification_loss(pred_logits, true_labels)
                total_loss += loss

                # Compute accuracy
                pred_classes = torch.argmax(pred_logits, dim=1)
                accuracy = (pred_classes == true_labels).float().mean()

                metrics[f'{domain}_cls_loss'] = loss.item()
                metrics[f'{domain}_cls_acc'] = accuracy.item()

        return total_loss, metrics

    def _compute_detection_loss(self, outputs: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute detection loss (simplified for MVP)."""

        total_loss = 0.0
        metrics = {}

        for domain in outputs.keys():
            if 'detection' in outputs[domain]:
                detection_out = outputs[domain]['detection']

                # Simplified detection loss (just objectness for MVP)
                if 'objectness' in detection_out:
                    objectness_logits = detection_out['objectness']

                    # Create dummy targets (positive if any boxes exist)
                    batch_size = objectness_logits.shape[0]
                    has_objects = torch.zeros(batch_size, device=objectness_logits.device)

                    # Check if batch has boxes
                    if 'boxes' in batch:
                        boxes = batch['boxes']
                        if isinstance(boxes, list):
                            for i, box_list in enumerate(boxes):
                                if len(box_list) > 0:
                                    has_objects[i] = 1.0
                        else:
                            has_objects = (boxes.sum(dim=-1) > 0).float()

                    # Simplified objectness loss
                    obj_targets = has_objects.unsqueeze(1).unsqueeze(1).expand_as(objectness_logits.squeeze(-1))
                    obj_loss = self.objectness_loss(objectness_logits.squeeze(-1), obj_targets)

                    total_loss += obj_loss
                    metrics[f'{domain}_obj_loss'] = obj_loss.item()

        return total_loss, metrics

    def _compute_domain_loss(self, outputs: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute domain classification loss."""

        total_loss = 0.0
        metrics = {}

        domain_names = list(self.model_config['domains'])

        for domain in outputs.keys():
            if 'detection' in outputs[domain] and 'domain_logits' in outputs[domain]['detection']:
                domain_logits = outputs[domain]['detection']['domain_logits']

                # Create domain labels
                domain_idx = domain_names.index(domain) if domain in domain_names else 0
                domain_labels = torch.full(
                    (domain_logits.shape[0],),
                    domain_idx,
                    device=domain_logits.device,
                    dtype=torch.long
                )

                loss = self.domain_loss(domain_logits, domain_labels)
                total_loss += loss

                # Compute accuracy
                pred_domains = torch.argmax(domain_logits, dim=1)
                accuracy = (pred_domains == domain_labels).float().mean()

                metrics[f'{domain}_domain_loss'] = loss.item()
                metrics[f'{domain}_domain_acc'] = accuracy.item()

        return total_loss, metrics

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""

        # Forward pass
        outputs = self(batch)

        # Compute losses
        cls_loss, cls_metrics = self._compute_classification_loss(outputs, batch)
        det_loss, det_metrics = self._compute_detection_loss(outputs, batch)
        domain_loss, domain_metrics = self._compute_domain_loss(outputs, batch)

        # Total loss
        total_loss = (
                self.loss_weights['classification'] * cls_loss +
                self.loss_weights['objectness'] * det_loss +
                self.loss_weights['domain'] * domain_loss
        )

        # Log metrics
        metrics = {
            'train_loss': total_loss.item(),
            'train_cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'train_det_loss': det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss,
            'train_domain_loss': domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss,
            **{f'train_{k}': v for k, v in cls_metrics.items()},
            **{f'train_{k}': v for k, v in det_metrics.items()},
            **{f'train_{k}': v for k, v in domain_metrics.items()}
        }

        # Log to tensorboard/wandb
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""

        # Forward pass
        outputs = self(batch)

        # Compute losses
        cls_loss, cls_metrics = self._compute_classification_loss(outputs, batch)
        det_loss, det_metrics = self._compute_detection_loss(outputs, batch)
        domain_loss, domain_metrics = self._compute_domain_loss(outputs, batch)

        # Total loss
        total_loss = (
                self.loss_weights['classification'] * cls_loss +
                self.loss_weights['objectness'] * det_loss +
                self.loss_weights['domain'] * domain_loss
        )

        # Log metrics
        metrics = {
            'val_loss': total_loss.item(),
            'val_cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'val_det_loss': det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss,
            'val_domain_loss': domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss,
            **{f'val_{k}': v for k, v in cls_metrics.items()},
            **{f'val_{k}': v for k, v in det_metrics.items()},
            **{f'val_{k}': v for k, v in domain_metrics.items()}
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""

        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )

        # Scheduler
        warmup_epochs = self.training_config.get('warmup_epochs', 2)
        max_epochs = self.training_config.get('max_epochs', 10)

        # Linear warmup followed by cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def train_dataloader(self):
        """Create training dataloader."""
        return create_multi_domain_dataloader(
            domains=self.data_config['domains'],
            batch_size=self.data_config['batch_size'],
            num_samples=self.data_config['num_samples'],
            img_size=self.data_config['img_size'],
            split='train',
            shuffle=True
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return create_multi_domain_dataloader(
            domains=self.data_config['domains'],
            batch_size=self.data_config['batch_size'],
            num_samples=self.data_config['num_samples'] // 4,  # Smaller val set
            img_size=self.data_config['img_size'],
            split='val',
            shuffle=False
        )


def create_trainer_from_config(config: Dict) -> VFMTrainer:
    """Create trainer from configuration dictionary."""

    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})

    return VFMTrainer(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config
    )


def test_trainer():
    """Test the VFM trainer."""
    print("🧪 Testing VFM Trainer...")

    # Create small config for testing
    config = {
        'model': {
            'model_size': 'tiny',
            'domains': ['land', 'air'],
            'embed_dim': 192,
            'use_cross_domain_fusion': True
        },
        'training': {
            'learning_rate': 1e-3,
            'max_epochs': 2,
            'warmup_epochs': 1
        },
        'data': {
            'batch_size': 2,
            'num_samples': 20,
            'img_size': 128,
            'domains': ['land', 'air']
        }
    }

    # Create trainer
    trainer = create_trainer_from_config(config)
    print(f"   ✓ Trainer created")
    print(f"   ✓ Backbone embed_dim: {trainer.backbone.embed_dim}")
    print(f"   ✓ Domains: {trainer.model_config['domains']}")

    # Test forward pass
    print("\n🧪 Testing forward pass...")
    trainer.eval()

    # Create sample batch
    sample_batch = {
        'image': torch.randn(2, 13, 128, 128),  # 2 land images
        'domain': ['land', 'land'],
        'label': torch.tensor([1, 3]),
        'boxes': [[], []]  # Empty boxes for simplicity
    }

    with torch.no_grad():
        outputs = trainer(sample_batch)
        print(f"   ✓ Outputs for domains: {list(outputs.keys())}")

        for domain in outputs:
            if 'classification' in outputs[domain]:
                print(f"   ✓ {domain} classification: {outputs[domain]['classification'].shape}")
            if 'detection' in outputs[domain]:
                print(f"   ✓ {domain} detection available")

    # Test training step
    print("\n🧪 Testing training step...")
    trainer.train()
    loss = trainer.training_step(sample_batch, 0)
    print(f"   ✓ Training loss: {loss.item():.4f}")

    print("\n✅ All trainer tests passed!")


if __name__ == "__main__":
    test_trainer()