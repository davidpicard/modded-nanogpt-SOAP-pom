import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from typing import Optional, Tuple, Dict, Any

torch._dynamo.config.suppress_errors = True

# =============================================================================
# Core Polynomial Functions
# =============================================================================

@torch.compile
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation function with torch.compile optimization."""
    return F.gelu(x)

@torch.compile
def po2(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Second-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 2*dim) with polynomial interactions
    """
    h = gelu(x).unsqueeze(-1)
    h2 = h * h
    h = torch.cat([h, h2], dim=-1)
    return (h * coeff).sum(-1)

@torch.compile
def po3(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Third-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 3*dim) with polynomial interactions
    """
    h = gelu(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h = torch.cat([h, h2, h3], dim=-1)
    return (h * coeff).sum(-1)

@torch.compile
def po4(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Fourth-order polynomial expansion.
    
    Args:
        x: Input tensor of shape (..., dim)
        
    Returns:
        Tensor of shape (..., 4*dim) with polynomial interactions
    """
    h = gelu(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h4 = h3 * h
    h = torch.cat([h, h2, h3, h4], dim=-1)
    return (h * coeff).sum(-1)

# =============================================================================
# Masking and Aggregation Functions
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D mask mixing for attention.
    
    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, seq_len)
        
    Returns:
        Masked and aggregated tensor of shape (batch, 1, dim)
    """
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True) / (1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))

def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply 3D mask mixing for cross-attention.
    
    Args:
        h: Hidden states tensor of shape (batch, seq_len, dim)
        mask: Attention mask of shape (batch, query_len, seq_len)
        
    Returns:
        Masked and aggregated tensor of shape (batch, query_len, dim)
    """
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h

# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(x: torch.Tensor, coeff: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply polynomial aggregation with optional masking.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Polynomial coefficients of shape (dim, degree)
        k: Polynomial order (2, 3, 4, or higher)
        mask: Optional attention mask
        
    Returns:
        Aggregated tensor with polynomial interactions
    """
    # Use optimized functions for common cases
    if k == 2:
        h = po2(x, coeff)
    elif k == 3:
        h = po3(x, coeff)
    elif k == 4:
        h = po4(x, coeff)
    else:
        # Generic case for k > 4
        h = gelu(x).unsqueeze(-1)
        h = torch.cat([h ** i for i in range(k)], dim=-1)
        h = (h * coeff).sum(-1)
    
    # Apply masking if provided
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim() == 2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() == 3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise ValueError(f'Unsupported mask dimension: {mask.dim()}. Expected 2, 3, or None.')
    return h.expand(x.shape)

# =============================================================================
# Main PoM Function
# =============================================================================

def pom(xc: torch.Tensor, coeff: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Polynomial Mixer (PoM) operation.
    
    This function implements the polynomial mixer operation which combines
    polynomial aggregation of context with selection from queries.
    
    Args:
        xc: Context input tensor of shape (batch, context_len, dim)
        coeff: Polynomial coefficients of shape (dim, degree)
        k: Polynomial order (degree of interactions to capture)
        mask: Optional attention mask for masking specific positions
        
    Returns:
        Output tensor after polynomial mixing
    """
    return polynomial_aggregation_(xc, coeff, k, mask)

# =============================================================================
# ComPoM Module Class
# =============================================================================

class ComPoM(nn.Module):
    """
    More compact Polynomial Mixer (PoM) Module.
    
    A custom neural network layer designed for capturing higher-order interactions 
    between input features through polynomial expansions. This module consists of
    three linear projections and a custom PoM operation.
    
    Attributes:
        dim (int): The dimensionality of the input features
        order (int): The order of the polynomial interactions to capture
        order_expand (int): The expansion factor for the polynomial order
        po_proj (nn.Linear): Linear projection for polynomial computation
        po_coeff (nn.Parameter): Coefficients for polynomial computation
        ag_proj (nn.Linear): Linear projection for output aggregation
        pom (callable): The polynomial mixer operation function
    """
    
    def __init__(self, dim: int, degree: int, expand: int, bias: bool = True):
        """
        Initialize the PoM module.
        
        Args:
            dim: The dimensionality of the input features
            degree: The degree of the polynomial to capture
            expand: The expansion factor for the polynomial order
            bias: Whether to include bias terms in linear projections
        """
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand

        # Linear projections
        self.po_proj = nn.Linear(dim, expand * dim, bias=bias)
        self.po_coeff = nn.Parameter(0.02 * torch.randn(dim * expand, degree))
        self.ag_proj = nn.Linear(expand * dim, dim, bias=bias)
        self.pom = pom

    def forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the PoM module.
        
        Args:
            xq: Query input tensor of shape (batch, n_tokens, dim)
            xc: Context input tensor. If None, self-attention is performed
            mask: Optional attention mask tensor
            
        Returns:
            Output tensor after applying the PoM operation
        """
        if xc is None:
            xc = xq  # self-attention

        h = self.po_proj(xc)
        sh = self.pom(h, self.po_coeff, self.order, mask)
        return self.ag_proj(sh)

    def state_forward(self, xq: torch.Tensor, xc: Optional[torch.Tensor] = None, 
                     state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with state management for incremental processing.
        
        Args:
            xq: Query input tensor
            xc: Context input tensor. If None, self-attention is performed
            state: Optional state dictionary from previous forward pass
            
        Returns:
            Tuple of (output_tensor, new_state)
        """
        if xc is None:
            xc = xq  # self-attention

        xc = self.po_proj(xc)
        h_current = polynomial_aggregation_(xc, self.po_coeff, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            h_past = state['h']
            n_past = state['n']
            h = (n_past * h_past + n_current * h_current) / (n_past + n_current)
        else:
            h = h_current
            n_past = 0

        new_state = {'h': h, 'n': n_past + n_current}
        return self.ag_proj(h), new_state