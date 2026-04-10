from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class FastLayerNorm(CustomOp):
    """
    LayerNorm implementation that inherits from CustomOp.
    Behaves identically to nn.LayerNorm.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
        Falls back to the reference implementation.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm
                return fast_layernorm(self, x)
            except ImportError as e:
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to nn.LayerNorm: %s",
                    e,
                )
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

class RMSNorm(CustomOp):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        output = torch_npu.npu_rms_norm(x, gamma=self.weight, epsilon=self.variance_epsilon)[0]

        return output

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x, scale, shift)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * out.to(input_dtype)