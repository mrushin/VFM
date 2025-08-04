# src/vfm/data/synthetic.py
"""
Synthetic data generation for rapid MVP development.
Creates realistic multi-domain datasets for land, air, and sea domains.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SyntheticMultiDomainDataset(Dataset):
    """
    Synthetic dataset for multi-domain vision foundation model.
    Generates realistic synthetic data for land, air, and sea domains.
    """

    def __init__(
            self,
            domain: str = 'land',
            num_samples: int = 1000,
            img_size: int = 224,
            split: str = 'train',
            seed: int = 42
    ):
        """
        Initialize synthetic dataset.

        Args:
            domain: One of ['land', 'air', 'sea']
            num_samples: Number of synthetic samples to generate
            img_size: Image size (square images)
            split: Dataset split ('train', 'val', 'test')
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.num_samples = num_samples
        self.img_size = img_size
        self.split = split

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Domain-specific configurations
        self.domain_configs = {
            'land': {
                'channels': 13,  # Sentinel-2 multispectral
                'num_classes': 10,
                'class_names': ['vehicle', 'building', 'road', 'vegetation', 'water',
                                'bare_soil', 'forest', 'agricultural', 'urban', 'other'],
                'background_color': [0.3, 0.5, 0.2],  # Greenish
                'object_patterns': ['rectangular', 'circular', 'linear']
            },
            'air': {
                'channels': 3,  # RGB
                'num_classes': 5,
                'class_names': ['aircraft', 'helicopter', 'drone', 'missile', 'background'],
                'background_color': [0.7, 0.8, 0.9],  # Sky blue
                'object_patterns': ['aircraft_shape', 'rotor', 'small_dot']
            },
            'sea': {
                'channels': 3,  # RGB
                'num_classes': 8,
                'class_names': ['cargo_ship', 'tanker', 'fishing_boat', 'yacht',
                                'submarine', 'patrol_boat', 'carrier', 'background'],
                'background_color': [0.1, 0.3, 0.6],  # Ocean blue
                'object_patterns': ['ship_shape', 'wake', 'small_boat']
            }
        }

        self.config = self.domain_configs[domain]

        # Pre-generate object templates for consistency
        self._generate_object_templates()

    def _generate_object_templates(self):
        """Generate object templates for each class."""
        self.object_templates = {}

        for i, class_name in enumerate(self.config['class_names']):
            if class_name == 'background':
                continue

            # Create simple geometric templates
            template_size = 32
            template = torch.zeros(template_size, template_size)

            if 'vehicle' in class_name or 'aircraft' in class_name:
                # Rectangular with some noise
                template[8:24, 4:28] = 1.0
                template[10:22, 6:26] = 1.5
            elif 'building' in class_name:
                # Square/rectangular buildings
                template[4:28, 8:24] = 1.0
                template[6:26, 10:22] = 1.2
            elif 'ship' in class_name or 'boat' in class_name:
                # Ship-like shape
                template[12:20, 4:28] = 1.0
                template[8:24, 10:22] = 0.8
            else:
                # Generic circular/blob shape
                center = template_size // 2
                y, x = torch.meshgrid(torch.arange(template_size), torch.arange(template_size))
                dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
                template[dist < 8] = 1.0
                template[dist < 6] = 1.2

            # Add some noise for realism
            template += torch.randn_like(template) * 0.1
            template = torch.clamp(template, 0, 2)

            self.object_templates[i] = template

    def _generate_background(self) -> torch.Tensor:
        """Generate domain-specific background."""
        bg_color = self.config['background_color']
        channels = self.config['channels']

        # Create base background
        background = torch.zeros(channels, self.img_size, self.img_size)

        for c in range(min(3, channels)):  # RGB channels
            background[c] = bg_color[c] + torch.randn(self.img_size, self.img_size) * 0.1

        # For multispectral (land domain), fill additional channels
        if channels > 3:
            for c in range(3, channels):
                # Simulate NIR, SWIR channels with some correlation to RGB
                background[c] = (background[0] + background[1] + background[2]) / 3
                background[c] += torch.randn(self.img_size, self.img_size) * 0.05

        # Add domain-specific textures
        if self.domain == 'land':
            self._add_land_textures(background)
        elif self.domain == 'sea':
            self._add_sea_textures(background)
        elif self.domain == 'air':
            self._add_air_textures(background)

        return torch.clamp(background, 0, 1)

    def _add_land_textures(self, background: torch.Tensor):
        """Add land-specific textures (vegetation, terrain)."""
        # Add some vegetation patches
        for _ in range(np.random.randint(3, 8)):
            x = np.random.randint(0, self.img_size - 20)
            y = np.random.randint(0, self.img_size - 20)
            size = np.random.randint(10, 30)

            # Green vegetation patch
            background[1, y:y + size, x:x + size] += 0.2  # More green
            background[0, y:y + size, x:x + size] -= 0.1  # Less red

    def _add_sea_textures(self, background: torch.Tensor):
        """Add sea-specific textures (waves, foam)."""
        # Add wave patterns
        x = torch.linspace(0, 4 * np.pi, self.img_size)
        y = torch.linspace(0, 4 * np.pi, self.img_size)
        X, Y = torch.meshgrid(x, y)

        waves = 0.05 * torch.sin(X) * torch.cos(Y)
        background[2] += waves  # Add to blue channel

    def _add_air_textures(self, background: torch.Tensor):
        """Add air-specific textures (clouds, atmosphere)."""
        # Add some cloud-like patches
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(0, self.img_size - 40)
            y = np.random.randint(0, self.img_size - 40)
            size = np.random.randint(20, 60)

            # White/gray clouds
            cloud_intensity = np.random.uniform(0.1, 0.3)
            background[:3, y:y + size, x:x + size] += cloud_intensity

    def _place_objects(self, background: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """Place objects in the scene and return bounding boxes."""
        image = background.clone()
        annotations = []

        # Number of objects to place
        num_objects = np.random.randint(1, 6)

        for _ in range(num_objects):
            # Choose random class (excluding background)
            class_id = np.random.randint(0, self.config['num_classes'] - 1)

            # Choose random position
            obj_size = np.random.randint(20, 60)
            x = np.random.randint(0, max(1, self.img_size - obj_size))
            y = np.random.randint(0, max(1, self.img_size - obj_size))

            # Get object template and resize
            template = self.object_templates[class_id]
            template_resized = torch.nn.functional.interpolate(
                template.unsqueeze(0).unsqueeze(0),
                size=(obj_size, obj_size),
                mode='bilinear'
            ).squeeze()

            # Place object on image
            end_x = min(x + obj_size, self.img_size)
            end_y = min(y + obj_size, self.img_size)
            actual_size_x = end_x - x
            actual_size_y = end_y - y

            # Place on all channels with some variation
            for c in range(image.shape[0]):
                channel_variation = np.random.uniform(0.8, 1.2)
                image[c, y:end_y, x:end_x] += (
                        template_resized[:actual_size_y, :actual_size_x] *
                        channel_variation * 0.3
                )

            # Create annotation
            annotations.append({
                'bbox': [x, y, end_x, end_y],  # [x1, y1, x2, y2]
                'class_id': class_id,
                'class_name': self.config['class_names'][class_id],
                'area': actual_size_x * actual_size_y
            })

        return torch.clamp(image, 0, 1), annotations

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        # Set seed for reproducible samples
        torch.manual_seed(idx + hash(self.domain) + hash(self.split))
        np.random.seed(idx + hash(self.domain) + hash(self.split))

        # Generate background
        background = self._generate_background()

        # Place objects and get annotations
        image, annotations = self._place_objects(background)

        # Create labels for classification (most prominent object)
        if annotations:
            # Use the largest object as the main class
            main_annotation = max(annotations, key=lambda x: x['area'])
            main_class = main_annotation['class_id']
        else:
            main_class = self.config['num_classes'] - 1  # Background class

        # Convert bounding boxes to tensors
        if annotations:
            boxes = torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32)
            box_labels = torch.tensor([ann['class_id'] for ann in annotations], dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            box_labels = torch.zeros(0, dtype=torch.long)

        return {
            'image': image,
            'label': main_class,
            'domain': self.domain,
            'boxes': boxes,
            'box_labels': box_labels,
            'annotations': annotations,
            'sample_id': f"{self.domain}_{self.split}_{idx}"
        }


def create_multi_domain_dataloader(
        domains: List[str] = ['land', 'air', 'sea'],
        batch_size: int = 4,
        num_samples: int = 1000,
        img_size: int = 224,
        split: str = 'train',
        num_workers: int = 0,
        shuffle: bool = True
) -> DataLoader:
    """
    Create a multi-domain dataloader that samples from all domains.

    Args:
        domains: List of domains to include
        batch_size: Batch size
        num_samples: Number of samples per domain
        img_size: Image size
        split: Dataset split
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data

    Returns:
        DataLoader that yields mixed-domain batches
    """

    class MultiDomainDataset(Dataset):
        def __init__(self):
            self.datasets = {}
            self.domain_lengths = {}

            for domain in domains:
                self.datasets[domain] = SyntheticMultiDomainDataset(
                    domain=domain,
                    num_samples=num_samples,
                    img_size=img_size,
                    split=split
                )
                self.domain_lengths[domain] = len(self.datasets[domain])

            self.total_length = sum(self.domain_lengths.values())

        def __len__(self):
            return self.total_length

        def __getitem__(self, idx):
            # Determine which domain this index belongs to
            current_idx = idx
            for domain in domains:
                if current_idx < self.domain_lengths[domain]:
                    return self.datasets[domain][current_idx]
                current_idx -= self.domain_lengths[domain]

            # Fallback to first domain
            return self.datasets[domains[0]][0]

    dataset = MultiDomainDataset()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Windows compatibility
        drop_last=True
    )


def test_synthetic_data():
    """Test function to verify synthetic data generation."""
    print("🧪 Testing synthetic data generation...")

    for domain in ['land', 'air', 'sea']:
        print(f"\n Testing {domain} domain...")

        dataset = SyntheticMultiDomainDataset(
            domain=domain,
            num_samples=10,
            img_size=128
        )

        sample = dataset[0]

        print(f"   ✓ Image shape: {sample['image'].shape}")
        print(f"   ✓ Domain: {sample['domain']}")
        print(f"   ✓ Label: {sample['label']}")
        print(f"   ✓ Boxes: {sample['boxes'].shape}")
        print(f"   ✓ Box labels: {sample['box_labels'].shape}")
        print(f"   ✓ Annotations: {len(sample['annotations'])} objects")

    # Test multi-domain dataloader
    print(f"\n🧪 Testing multi-domain dataloader...")
    dataloader = create_multi_domain_dataloader(
        batch_size=2,
        num_samples=5,
        img_size=64
    )

    batch = next(iter(dataloader))
    print(f"   ✓ Batch image shape: {batch['image'].shape}")
    print(f"   ✓ Batch domains: {batch['domain']}")
    print(f"   ✓ Batch labels: {batch['label']}")

    print("\n✅ All synthetic data tests passed!")


if __name__ == "__main__":
    test_synthetic_data()