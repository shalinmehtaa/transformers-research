"""Various attention implementations"""
from typing import Optional, Union
from jaxtyping import Float
import torch
from torch import nn, Tensor
from torch.nn.functional import softmax


def scaled_dot_product_attention(
        q: Float[Tensor, "... seq_len head_dim"],
        k: Float[Tensor, "... seq_len head_dim"],
        v: Float[Tensor, "... seq_len head_dim"],
        mask: Optional[Float[Tensor]] = None
    ):
    """Scaled dot product attention implementation.
    
    Args:
        q: Query projections
        k: Key projections
        v: Value projections
        mask: Causal mask

    Returns:
        Tensor with attention scores or weights
    """
    d_k = k.shape[-1]
    scale= d_k ** -0.5
    # get attention logits
    logits: Float[Tensor, "... seq_len seq_len"] = \
        q @ k.transpose(-2, -1)
    # for stable softmax
    logits = logits.to(torch.float32) * scale 

    # handle causal masking
    if mask is not None:
        if mask.dtype == torch.bool:
            m = mask
            # expect mask to be of shape [seq_len, seq_len], where True == keep
            while m.dim() < logits.dim():
                m = m.unsqueeze(0)
            logits = logits.masked_fill(~m, value=float("-inf"))
        else:
            # additive mask (e.g., 0 for keep, -inf for mask)
            logits = logits + mask

    weights: Float[Tensor, "... seq_len seq_len"] = \
        softmax(logits, dim=-1).to(q.dtype) # cast back
    out: Float[Tensor, "... seq_len head_dim"] = \
        weights @ v
    return out


class MultiHeadAttention(nn.Module):
    """MHA implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) \
        -> Float[Tensor, "batch_size seq_len d_model"]:
        """Forward pass.
        """


class MultiQueryAttention(nn.Module):
    """MQA implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) \
        -> Float[Tensor, "batch_size seq_len d_model"]:
        """Forward pass.
        """


class GroupedQueryAttention(nn.Module):
    """GQA implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) \
        -> Float[Tensor, "batch_size seq_len d_model"]:
        """Forward pass.
        """


class MultiHeadLatentAttention(nn.Module):
    """MLA implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) \
        -> Float[Tensor, "batch_size seq_len d_model"]:
        """Forward pass.
        """


class SlidingWindowAttention(nn.Module):
    """SWA implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) \
        -> Float[Tensor, "batch_size seq_len d_model"]:
        """Forward pass.
        """
