from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm implementation that inherits from nn.LayerNorm.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
        Falls back to nn.LayerNorm implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(normalized_shape=dim, eps=eps, elementwise_affine=elementwise_affine)
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self):
        if current_omni_platform.is_rocm():
            return self.forward_hip
        elif current_omni_platform.is_cuda():
            return self.forward_cuda
        elif current_omni_platform.is_npu():
            return self.forward_npu
        elif current_omni_platform.is_xpu():
            return self.forward_xpu
        else:
            return self.forward_native

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm

                return fast_layernorm(self, x)
            except ImportError as e:
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to FP32 layer_norm: %s",
                    e,
                )

        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

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
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * out.to(input_dtype)
