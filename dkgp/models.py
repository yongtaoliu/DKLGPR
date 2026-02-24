"""
Core model classes and feature extractors for Deep Kernel GP.
"""
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Feature Extractors
# ============================================================================
class FCFeatureExtractor(nn.Module):
    """
    Simple fully-connected feature extractor.
    Lightweight, fast, good for prototyping.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    """
    
    def __init__(self, input_dim, feature_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim)
        )
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        return self.network(x)


class FCBNFeatureExtractor(nn.Module):
    """
    Fully-connected feature extractor with BatchNorm and Dropout.
    More robust, prevents overfitting. Recommended for general use.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int
        Hidden layer dimensions
    dropout : float
        Dropout rate
    """
    
    def __init__(self, input_dim, feature_dim=16, hidden_dims=None, dropout=0.2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.network = nn.Sequential(*layers)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.network(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-style feature extractor with skip connections.
    Better gradient flow for deeper networks.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dim : int
        Hidden layer dimension
    num_blocks : int
        Number of residual blocks
    dropout : float
        Dropout rate
    """
    
    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.fc1(x)))
            out = self.dropout(out)
            out = self.bn2(self.fc2(out))
            out += residual  # Skip connection
            return self.relu(out)
    
    def __init__(self, input_dim, feature_dim=16, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.res_blocks = nn.Sequential(
            *[self.ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.res_blocks(x)
        x = self.output_projection(x)
        return x


class AttentionFeatureExtractor(nn.Module):
    """
    Self-attention based feature extractor.
    Good for learning feature interactions.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    """
    
    class AttentionBlock(nn.Module):
        def __init__(self, dim, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            
            assert dim % num_heads == 0, "dim must be divisible by num_heads"
            
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)
            self.last_attention_weights = None # Store last attention weights
            
        def forward(self, x, return_attention=False):
            batch_size = x.shape[0]
            
            # Linear projections
            Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
            K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
            V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention = torch.softmax(scores, dim=-1)

            # Store for later retrieval
            self.last_attention_weights = attention.detach()

            # Apply attention
            out = torch.matmul(attention, V)
            out = out.view(batch_size, -1)
            out = self.out(out)
            # Optional return attention
            if return_attention:
                return out, attention
            return out
        
        def get_attention_weights(self):
            """Get the last computed attention weights."""
            return self.last_attention_weights
    
    def __init__(self, input_dim, feature_dim=16, hidden_dim=128, num_heads=4):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention = self.AttentionBlock(hidden_dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.input_projection(x)
        
        # Attention block with residual
        attn_out = self.attention(x)
        x = self.layer_norm(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm(x + ffn_out)
        
        x = self.output_projection(x)
        return x

    def get_attention_maps(self, x):
        """
        Get attention maps for input x.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim)
            
        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights, shape (batch, num_heads, num_heads)
        """
        self.eval()
        with torch.no_grad():
            x_proj = self.input_projection(x)
            _, attention = self.attention(x_proj, return_attention=True)
        return attention

# ============================================================================
# Feature Extractor Factory
# ============================================================================

def get_feature_extractor(
    extractor_type='fcbn',
    input_dim=None,
    feature_dim=16,
    hidden_dims=None,
    dropout=0.2,
    **kwargs
):
    """
    Factory function to create feature extractors.
    
    Parameters
    ----------
    extractor_type : str
        Type of feature extractor:
        - 'fc': Simple fully-connected
        - 'fcbn': FC + BatchNorm + Dropout
        - 'resnet': ResNet with skip connections
        - 'attention': Self-attention based
        - 'custom': User-provided nn.Module
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions (for 'fcbn' and 'wide_deep')
    dropout : float
        Dropout rate
    **kwargs : additional arguments
        - custom_extractor: nn.Module for 'custom' type
        - hidden_dim: for 'resnet' and 'attention'
        - num_blocks: for 'resnet'
        - num_heads: for 'attention'
        - deep_dims: for 'wide_deep'
    
    Returns
    -------
    feature_extractor : nn.Module
        Instantiated feature extractor
        
    Examples
    --------
    >>> # Simple FC
    >>> extractor = get_feature_extractor('fc', input_dim=100, feature_dim=16)
    
    >>> # FC + BatchNorm (recommended)
    >>> extractor = get_feature_extractor('fcbn', input_dim=100, feature_dim=16,
    ...                                   hidden_dims=[512, 256, 128])
    
    >>> # ResNet
    >>> extractor = get_feature_extractor('resnet', input_dim=100, feature_dim=16,
    ...                                   hidden_dim=128, num_blocks=3)
    
    >>> # Custom
    >>> my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
    >>> extractor = get_feature_extractor('custom', custom_extractor=my_net)
    """
    if extractor_type == 'fc':
        return FCFeatureExtractor(input_dim, feature_dim)
    
    elif extractor_type == 'fcbn':
        return FCBNFeatureExtractor(input_dim, feature_dim, hidden_dims, dropout)
    
    elif extractor_type == 'resnet':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_blocks = kwargs.get('num_blocks', 2)
        return ResNetFeatureExtractor(input_dim, feature_dim, hidden_dim, num_blocks, dropout)
    
    elif extractor_type == 'attention':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_heads = kwargs.get('num_heads', 4)
        return AttentionFeatureExtractor(input_dim, feature_dim, hidden_dim, num_heads)
    
    elif extractor_type == 'custom':
        custom_extractor = kwargs.get('custom_extractor')
        if custom_extractor is None:
            raise ValueError("Must provide 'custom_extractor' for type='custom'")
        return custom_extractor
    
    else:
        raise ValueError(f"Unknown extractor_type: {extractor_type}. "
                        f"Choose from: 'fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom'")


# ============================================================================
# Backward Compatibility
# ============================================================================

ImageFeatureExtractor = FCBNFeatureExtractor
