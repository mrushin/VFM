# src/vfm/models/heads.py
"""
Multi-domain detection and classification heads for VFM.
Supports both object detection and classification tasks across domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ClassificationHead(nn.Module):
    """
    Classification head for scene classification tasks.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_classes: int = 10,
            dropout: float = 0.1,
            use_layernorm: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Optional layer normalization
        self.norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            cls_token: Class token features (B, embed_dim)

        Returns:
            Classification logits (B, num_classes)
        """
        x = self.norm(cls_token)
        logits = self.classifier(x)
        return logits


class DetectionHead(nn.Module):
    """
    Object detection head using patch tokens.
    Simplified detection approach for MVP.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_classes: int = 10,
            num_anchors: int = 3,
            patch_size: int = 16,
            img_size: int = 224,
            dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Detection heads
        self.classification_head = nn.Linear(embed_dim, num_classes * num_anchors)
        self.regression_head = nn.Linear(embed_dim, 4 * num_anchors)  # x, y, w, h
        self.objectness_head = nn.Linear(embed_dim, 1 * num_anchors)

        # Anchor generation
        self.register_buffer('anchors', self._generate_anchors())

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes for each grid cell."""
        # Simple anchor sizes (small, medium, large)
        anchor_sizes = [0.2, 0.5, 0.8]
        anchors = []

        for size in anchor_sizes:
            # Square anchors for simplicity
            anchors.append([size, size])

        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for object detection.

        Args:
            patch_tokens: Patch token features (B, num_patches, embed_dim)

        Returns:
            Dictionary containing detection outputs
        """
        B, N, D = patch_tokens.shape

        # Project features
        features = self.feature_proj(patch_tokens)  # (B, N, embed_dim)

        # Detection predictions
        class_logits = self.classification_head(features)  # (B, N, num_classes * num_anchors)
        bbox_preds = self.regression_head(features)  # (B, N, 4 * num_anchors)
        objectness = self.objectness_head(features)  # (B, N, 1 * num_anchors)

        # Reshape predictions
        class_logits = class_logits.view(B, N, self.num_anchors, self.num_classes)
        bbox_preds = bbox_preds.view(B, N, self.num_anchors, 4)
        objectness = objectness.view(B, N, self.num_anchors, 1)

        return {
            'class_logits': class_logits,
            'bbox_preds': bbox_preds,
            'objectness': objectness,
            'anchors': self.anchors
        }


class MultiDomainDetectionHead(nn.Module):
    """
    Multi-domain detection head that handles different domains.
    Each domain can have different number of classes and characteristics.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            domains: List[str] = ['land', 'air', 'sea'],
            num_classes_per_domain: Optional[Dict[str, int]] = None,
            num_anchors: int = 3,
            patch_size: int = 16,
            img_size: int = 224,
            dropout: float = 0.1,
            shared_features: bool = True
    ):
        super().__init__()

        self.domains = domains
        self.embed_dim = embed_dim
        self.shared_features = shared_features

        # Default class counts per domain
        if num_classes_per_domain is None:
            num_classes_per_domain = {
                'land': 10,  # vehicles, buildings, etc.
                'air': 5,  # aircraft types
                'sea': 8  # ship types
            }
        self.num_classes_per_domain = num_classes_per_domain

        # Shared feature processor (if enabled)
        if shared_features:
            self.shared_processor = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Domain-specific detection heads
        self.domain_heads = nn.ModuleDict()
        for domain in domains:
            num_classes = num_classes_per_domain.get(domain, 10)

            self.domain_heads[domain] = DetectionHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_anchors=num_anchors,
                patch_size=patch_size,
                img_size=img_size,
                dropout=dropout
            )

        # Domain classifier (for automatic domain detection)
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, len(domains))
        )

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            target_domains: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass for multi-domain detection.

        Args:
            features: Dictionary of domain features
            target_domains: Specific domains to process (optional)

        Returns:
            Dictionary of detection outputs per domain
        """
        outputs = {}

        # Process each domain
        domains_to_process = target_domains if target_domains else features.keys()

        for domain in domains_to_process:
            if domain not in features:
                continue

            domain_features = features[domain]

            # Extract different feature types
            if isinstance(domain_features, dict):
                cls_token = domain_features.get('cls_token')
                patch_tokens = domain_features.get('patch_tokens')
            else:
                # Assume it's the full token sequence
                cls_token = domain_features[:, 0]
                patch_tokens = domain_features[:, 1:]

            # Apply shared processing if enabled
            if self.shared_features:
                patch_tokens = self.shared_processor(patch_tokens)

            # Domain-specific detection
            if domain in self.domain_heads:
                detection_output = self.domain_heads[domain](patch_tokens)
            else:
                # Use a default domain head
                detection_output = self.domain_heads[self.domains[0]](patch_tokens)

            # Add domain classification
            domain_logits = self.domain_classifier(cls_token)

            outputs[domain] = {
                **detection_output,
                'domain_logits': domain_logits,
                'cls_features': cls_token
            }

        return outputs


class MultiTaskHead(nn.Module):
    """
    Multi-task head that combines classification and detection.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            domains: List[str] = ['land', 'air', 'sea'],
            num_classes_per_domain: Optional[Dict[str, int]] = None,
            enable_detection: bool = True,
            enable_classification: bool = True,
            **kwargs
    ):
        super().__init__()

        self.domains = domains
        self.enable_detection = enable_detection
        self.enable_classification = enable_classification

        # Default class counts
        if num_classes_per_domain is None:
            num_classes_per_domain = {
                'land': 10,
                'air': 5,
                'sea': 8
            }

        # Classification heads
        if enable_classification:
            self.classification_heads = nn.ModuleDict()
            for domain in domains:
                num_classes = num_classes_per_domain.get(domain, 10)
                self.classification_heads[domain] = ClassificationHead(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    **kwargs
                )

        # Detection head
        if enable_detection:
            self.detection_head = MultiDomainDetectionHead(
                embed_dim=embed_dim,
                domains=domains,
                num_classes_per_domain=num_classes_per_domain,
                **kwargs
            )

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            task: str = 'both'  # 'classification', 'detection', or 'both'
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass for multi-task learning.

        Args:
            features: Dictionary of domain features
            task: Which task(s) to perform

        Returns:
            Dictionary of task outputs per domain
        """
        outputs = {}

        for domain, domain_features in features.items():
            domain_outputs = {}

            # Extract features
            if isinstance(domain_features, dict):
                cls_token = domain_features.get('cls_token')
                patch_tokens = domain_features.get('patch_tokens')
            else:
                cls_token = domain_features[:, 0]
                patch_tokens = domain_features[:, 1:]

            # Classification task
            if (task in ['classification', 'both'] and
                    self.enable_classification and
                    domain in self.classification_heads):
                class_logits = self.classification_heads[domain](cls_token)
                domain_outputs['classification'] = class_logits

            # Detection task
            if (task in ['detection', 'both'] and
                    self.enable_detection):
                detection_outputs = self.detection_head({domain: domain_features})
                domain_outputs['detection'] = detection_outputs[domain]

            outputs[domain] = domain_outputs

        return outputs


def test_heads():
    """Test the detection and classification heads."""
    print("🧪 Testing Multi-Domain Heads...")

    # Test classification head
    print("\n1. Testing Classification Head...")
    cls_head = ClassificationHead(embed_dim=384, num_classes=10)
    cls_token = torch.randn(4, 384)  # Batch of 4

    class_logits = cls_head(cls_token)
    print(f"   ✓ Classification logits: {class_logits.shape}")

    # Test detection head
    print("\n2. Testing Detection Head...")
    det_head = DetectionHead(embed_dim=384, num_classes=10, img_size=224)
    patch_tokens = torch.randn(4, 196, 384)  # 14x14 patches

    det_outputs = det_head(patch_tokens)
    print(f"   ✓ Class logits: {det_outputs['class_logits'].shape}")
    print(f"   ✓ Bbox predictions: {det_outputs['bbox_preds'].shape}")
    print(f"   ✓ Objectness: {det_outputs['objectness'].shape}")

    # Test multi-domain detection head
    print("\n3. Testing Multi-Domain Detection Head...")
    multi_det_head = MultiDomainDetectionHead(
        embed_dim=384,
        domains=['land', 'air', 'sea']
    )

    features = {
        'land': {
            'cls_token': torch.randn(4, 384),
            'patch_tokens': torch.randn(4, 196, 384)
        },
        'air': {
            'cls_token': torch.randn(4, 384),
            'patch_tokens': torch.randn(4, 196, 384)
        }
    }

    multi_outputs = multi_det_head(features)
    for domain in features.keys():
        print(f"   ✓ {domain} - class logits: {multi_outputs[domain]['class_logits'].shape}")
        print(f"   ✓ {domain} - domain logits: {multi_outputs[domain]['domain_logits'].shape}")

    # Test multi-task head
    print("\n4. Testing Multi-Task Head...")
    multi_task_head = MultiTaskHead(
        embed_dim=384,
        domains=['land', 'air', 'sea'],
        enable_detection=True,
        enable_classification=True
    )

    task_outputs = multi_task_head(features, task='both')
    for domain in features.keys():
        print(f"   ✓ {domain} - classification: {task_outputs[domain]['classification'].shape}")
        print(f"   ✓ {domain} - detection available: {'detection' in task_outputs[domain]}")

    print("\n✅ All head tests passed!")


if __name__ == "__main__":
    test_heads()