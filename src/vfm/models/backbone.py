# src/vfm/models/backbone.py
"""
Unified Vision Foundation Model backbone.
Multi-domain architecture inspired by TerraFM and foundation model best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple, Union
import math


class DomainSpecificEmbedding(nn.Module):
    """
    Domain and modality-specific patch embedding.
    Inspired by TerraFM's modality-specific approach.
    """

    def __init__(
            self,
            domains: List[str] = ['land', 'air', 'sea'],
            modalities: List[str] = ['optical', 'radar', 'infrared'],
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768
    ):
        super().__init__()

        self.domains = domains
        self.modalities = modalities
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Channel configurations for different domain-modality combinations
        self.channel_configs = {
            ('land', 'optical'): 13,  # Sentinel-2 multispectral
            ('land', 'radar'): 2,  # Sentinel-1 VV, VH
            ('air', 'optical'): 3,  # RGB
            ('air', 'infrared'): 1,  # Thermal
            ('sea', 'optical'): 3,  # RGB
            ('sea', 'radar'): 2,  # SAR
        }

        # Domain-modality specific patch projections
        self.patch_projections = nn.ModuleDict()
        for domain in domains:
            for modality in modalities:
                key = f"{domain}_{modality}"
                in_channels = self.channel_configs.get((domain, modality), 3)

                self.patch_projections[key] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size
                )

        # Domain-specific positional encodings
        self.domain_pos_encodings = nn.ParameterDict({
            domain: nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
            for domain in domains
        })

        # Domain-specific class tokens
        self.domain_cls_tokens = nn.ParameterDict({
            domain: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for domain in domains
        })

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, domain: str, modality: str = 'optical') -> torch.Tensor:
        """
        Forward pass for domain-specific embedding.

        Args:
            x: Input tensor (B, C, H, W)
            domain: Domain name
            modality: Modality name

        Returns:
            Embedded patches (B, num_patches + 1, embed_dim)
        """
        B = x.shape[0]
        key = f"{domain}_{modality}"

        # Apply domain-modality specific patch projection
        if key not in self.patch_projections:
            # Fallback to a default projection
            key = f"{domain}_optical"
            if key not in self.patch_projections:
                key = "land_optical"

        x = self.patch_projections[key](x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, num_patches, embed_dim)

        # Add domain-specific class token
        if domain in self.domain_cls_tokens:
            cls_token = self.domain_cls_tokens[domain].expand(B, -1, -1)
        else:
            cls_token = self.domain_cls_tokens['land'].expand(B, -1, -1)

        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add domain-specific positional encoding
        if domain in self.domain_pos_encodings:
            x = x + self.domain_pos_encodings[domain]
        else:
            x = x + self.domain_pos_encodings['land']

        return self.dropout(x)


class CrossDomainFusion(nn.Module):
    """
    Cross-domain fusion using learnable queries.
    Inspired by TerraFM's cross-attention fusion approach.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_queries: int = 5,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Learnable fusion queries (similar to TerraFM's spatial queries)
        self.fusion_queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Query scoring and aggregation
        self.query_scorer = nn.Linear(embed_dim, 1)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            domain_features: Dict[str, torch.Tensor],
            return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fuse features from multiple domains.

        Args:
            domain_features: Dict of domain features {domain: (B, seq_len, embed_dim)}
            return_attention: Whether to return attention weights

        Returns:
            Fused features (B, embed_dim) or (features, attention_weights)
        """
        if len(domain_features) == 1:
            # Single domain - just return cls token
            domain_feat = next(iter(domain_features.values()))
            return domain_feat[:, 0]  # Return cls token

        # Stack features from all domains
        domain_names = list(domain_features.keys())
        stacked_features = []

        for domain in domain_names:
            feat = domain_features[domain]
            stacked_features.append(feat)

        # Concatenate all domain features
        all_features = torch.cat(stacked_features, dim=1)  # (B, total_seq_len, embed_dim)
        B = all_features.shape[0]

        # Expand queries for batch
        queries = self.fusion_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, embed_dim)

        # Cross-attention: queries attend to all domain features
        fused_queries, attention_weights = self.cross_attention(
            query=queries,
            key=all_features,
            value=all_features
        )

        # Score each query and compute weighted average
        query_scores = self.query_scorer(fused_queries)  # (B, num_queries, 1)
        query_weights = F.softmax(query_scores, dim=1)  # (B, num_queries, 1)

        # Weighted aggregation of queries
        fused_output = torch.sum(fused_queries * query_weights, dim=1)  # (B, embed_dim)

        # Apply output projection and normalization
        fused_output = self.output_projection(fused_output)
        fused_output = self.norm(fused_output)
        fused_output = self.dropout(fused_output)

        if return_attention:
            return fused_output, attention_weights
        return fused_output


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative position encoding."""

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            dropout: float = 0.1,
            bias: bool = True
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            dropout: float = 0.1
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            drop_path: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

        # Drop path for stochastic depth
        self.drop_path = nn.Identity()  # Simplified for MVP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class UnifiedVisionFM(nn.Module):
    """
    Unified Vision Foundation Model for multi-domain object detection and classification.
    Supports land, air, and sea domains with cross-domain fusion capabilities.
    """

    def __init__(
            self,
            domains: List[str] = ['land', 'air', 'sea'],
            modalities: List[str] = ['optical', 'radar', 'infrared'],
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            use_cross_domain_fusion: bool = True,
            fusion_layers: List[int] = None
    ):
        super().__init__()

        self.domains = domains
        self.modalities = modalities
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cross_domain_fusion = use_cross_domain_fusion

        # Default fusion layers (apply fusion at 1/3, 2/3, and end)
        if fusion_layers is None:
            fusion_layers = [depth // 3, 2 * depth // 3, depth - 1]
        self.fusion_layers = set(fusion_layers)

        # Domain-specific embeddings
        self.domain_embeddings = DomainSpecificEmbedding(
            domains=domains,
            modalities=modalities,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Cross-domain fusion (applied at specific layers)
        if use_cross_domain_fusion:
            self.cross_domain_fusion = CrossDomainFusion(
                embed_dim=embed_dim,
                num_queries=5,
                num_heads=8,
                dropout=dropout
            )

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following best practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward_features(
            self,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            domains: Union[str, List[str]] = None,
            modalities: Union[str, List[str]] = 'optical'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the backbone to extract features.

        Args:
            inputs: Either single tensor or dict of domain tensors
            domains: Domain name(s) - required if inputs is a tensor
            modalities: Modality name(s)

        Returns:
            Dictionary of domain features
        """
        # Handle different input formats
        if isinstance(inputs, torch.Tensor):
            # Single tensor input
            if isinstance(domains, str):
                domain_inputs = {domains: inputs}
                modality_inputs = {domains: modalities}
            else:
                raise ValueError("domains must be specified when inputs is a tensor")
        else:
            # Dictionary input
            domain_inputs = inputs
            if isinstance(modalities, str):
                modality_inputs = {domain: modalities for domain in domain_inputs.keys()}
            else:
                modality_inputs = modalities

        # Apply domain-specific embeddings
        domain_features = {}
        for domain, x in domain_inputs.items():
            modality = modality_inputs.get(domain, 'optical')
            domain_features[domain] = self.domain_embeddings(x, domain, modality)

        # Apply transformer blocks with periodic cross-domain fusion
        for i, block in enumerate(self.blocks):
            # Apply transformer block to each domain
            for domain in domain_features:
                domain_features[domain] = block(domain_features[domain])

            # Apply cross-domain fusion at specific layers
            if (self.use_cross_domain_fusion and
                    i in self.fusion_layers and
                    len(domain_features) > 1):

                fused_features = self.cross_domain_fusion(domain_features)

                # Broadcast fused features back to all domains (residual connection)
                for domain in domain_features:
                    # Add fused features to cls token
                    domain_features[domain][:, 0] += fused_features

        # Final normalization
        for domain in domain_features:
            domain_features[domain] = self.norm(domain_features[domain])

        return domain_features

    def forward(
            self,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            domains: Union[str, List[str]] = None,
            modalities: Union[str, List[str]] = 'optical'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning features for downstream tasks.

        Returns:
            Dictionary with cls_token, patch_tokens, and all_tokens for each domain
        """
        domain_features = self.forward_features(inputs, domains, modalities)

        output = {}
        for domain, features in domain_features.items():
            output[domain] = {
                'cls_token': features[:, 0],  # For classification
                'patch_tokens': features[:, 1:],  # For detection/segmentation
                'all_tokens': features  # Full feature representation
            }

        return output


def create_vfm_model(
        model_size: str = 'base',
        domains: List[str] = ['land', 'air', 'sea'],
        **kwargs
) -> UnifiedVisionFM:
    """
    Create a VFM model with predefined configurations.

    Args:
        model_size: One of ['tiny', 'small', 'base', 'large']
        domains: List of domains to support
        **kwargs: Additional arguments

    Returns:
        UnifiedVisionFM model
    """

    model_configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    }

    if model_size not in model_configs:
        raise ValueError(f"Invalid model size: {model_size}")

    config = model_configs[model_size]
    config.update(kwargs)

    return UnifiedVisionFM(domains=domains, **config)


def test_unified_vfm():
    """Test the unified VFM model."""
    print("🧪 Testing Unified VFM...")

    # Test single domain
    print("\n1. Testing single domain (land)...")
    model = create_vfm_model('small')
    x = torch.randn(2, 13, 224, 224)  # Batch of 2, Sentinel-2 data

    features = model(x, domains='land', modalities='optical')
    print(f"   ✓ Land features - cls: {features['land']['cls_token'].shape}")
    print(f"   ✓ Land features - patches: {features['land']['patch_tokens'].shape}")

    # Test multi-domain
    print("\n2. Testing multi-domain...")
    inputs = {
        'land': torch.randn(2, 13, 224, 224),
        'air': torch.randn(2, 3, 224, 224),
        'sea': torch.randn(2, 3, 224, 224)
    }

    features = model(inputs)
    for domain in ['land', 'air', 'sea']:
        print(f"   ✓ {domain} - cls: {features[domain]['cls_token'].shape}")

    # Test cross-domain fusion
    print("\n3. Testing cross-domain fusion...")
    model_with_fusion = create_vfm_model('small', use_cross_domain_fusion=True)
    features_fused = model_with_fusion(inputs)
    print(f"   ✓ Fusion enabled, features extracted successfully")

    # Test different model sizes
    print("\n4. Testing different model sizes...")
    for size in ['tiny', 'small', 'base']:
        test_model = create_vfm_model(size)
        test_features = test_model(x, domains='land')
        print(f"   ✓ {size} model - cls: {test_features['land']['cls_token'].shape}")

    print("\n✅ All VFM tests passed!")


if __name__ == "__main__":
    test_unified_vfm()