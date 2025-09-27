"""Normalization layers"""
from typing import Optional, Union
from jaxtyping import Float
import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    """Custom RMSNorm implementation"""
    def __init__(self,
                 d_model: int,
                 eps: Optional[float] = 1e-5,
                 bias: Optional[bool] = False,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initialization.

        Args:
            d_model: Model (embedding) dimension
            eps: Fudge factor for numerical stability when taking square root
            bias: Boolean flag to include a bias parameter
            device: Torch device
            dtype: Torch dtype

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones(self.d_model, device=self.device, dtype=self.dtype)
        )
        self.bias: Float[Tensor, "d_model"] = nn.Parameter(
                torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
        ) if bias else None

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Forward pass.

        Args:
            x: Input tensor
        
        Returns:
            RMS normalized tensor
        """
        x_dtype = x.dtype
        # Cast to float32 to prevent overflow when squaring
        x_f32 = x.to(torch.float32)
        inv_rms = x_f32.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x_f32 * inv_rms
        x_norm = x_norm * self.weight
        x_norm = x_norm + self.bias if self.bias is not None else x_norm
        # Cast back to input dtype
        x_norm = x_norm.to(x_dtype)
        return x_norm


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation"""
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 bias: Optional[bool] = False,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initialization.

        Args:
            d_model: Model (embedding dimension)
            eps: Fudge factor for numerical stability when dividing
            bias: Boolean flag to include a bias parameter
            device: Torch device
            dtype: Torch dtype

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones(d_model, device=self.device, dtype=self.dtype)
        )
        self.bias: Float[Tensor, "d_model"] = nn.Parameter(
                torch.zeros(d_model, device=self.device, dtype=self.dtype)
        ) if bias else None

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Layer normalized tensor
        """
        x_dtype = x.dtype
        # Cast to float32 to prevent overflow during reduction operations
        x_f32 = x.to(torch.float32)
        x_mean = x_f32.mean(dim=-1, keepdim=True)
        x_var = x_f32.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_f32 - x_mean) / torch.sqrt(x_var + self.eps)
        x_norm = x_norm * self.weight
        x_norm = x_norm + self.bias if self.bias is not None else x_norm
        # Cast back to input dtype
        x_norm = x_norm.to(x_dtype)
        return x_norm


if __name__ == "__main__":
    D_MODEL = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.randn(2, 3, D_MODEL, dtype=torch.float16, device=DEVICE)

    # RMSNorm
    custom = RMSNorm(D_MODEL, eps=1e-5).to(DEVICE)
    ref = nn.RMSNorm(D_MODEL, eps=1e-5).to(DEVICE)
    with torch.no_grad():
        ref.weight.copy_(custom.weight)
    torch.testing.assert_close(custom(X), ref(X), rtol=1e-5, atol=1e-5, msg="Failed!")

    # LayerNorm
    custom = LayerNorm(D_MODEL, eps=1e-5, bias=True).to(DEVICE)
    ref = nn.LayerNorm(D_MODEL, eps=1e-5, elementwise_affine=True, bias=True).to(DEVICE)
    with torch.no_grad():
        ref.weight.copy_(custom.weight)
        ref.bias.copy_(custom.bias)
    torch.testing.assert_close(custom(X), ref(X), rtol=1e-5, atol=1e-5, msg="Failed!")
