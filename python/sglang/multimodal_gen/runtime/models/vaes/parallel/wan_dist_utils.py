import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.parallel.decode_parallel import (
    ensure_local_height,
    gather_and_trim_height,
    halo_exchange,
    maybe_contiguous_for_decode_gather,
    split_for_parallel_decode,
    split_for_parallel_encode,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import (
    AvgDown3D,
    DupUp3D,
    WanCausalConv3d,
    WanRMS_norm,
    WanUpsample,
    attention_block_forward,
    match_conv3d_input_format,
    mid_block_forward,
    resample_forward,
    residual_block_forward,
    residual_down_block_forward,
    residual_up_block_forward,
    up_block_forward,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


class WanDistConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        height_padding: tuple[int, int] | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.height_halo_size = (self.kernel_size[-2] - 1) // 2
        if height_padding is None:
            height_padding = (self.padding[-2], self.padding[-2])
        self.height_pad_top, self.height_pad_bottom = height_padding

        self.padding: tuple[int, int]
        if self.height_halo_size > 0:
            self._padding = (0, 0, 0, 0)
        else:
            self._padding = (
                0,
                0,
                self.padding[0],
                self.padding[0],
            )

        self.padding = (0, self.padding[1])
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x):
        if any(self._padding):
            x = F.pad(x, self._padding)

        x_padded, self._halo_recv_top_buf, self._halo_recv_bottom_buf = halo_exchange(
            x,
            height_halo_size=self.height_halo_size,
            recv_top_buf=self._halo_recv_top_buf,
            recv_bottom_buf=self._halo_recv_bottom_buf,
        )

        pad_top = self.height_pad_top
        stride = self.stride[-2]
        global_start = self.rank * x.shape[-2]
        if self.height_halo_size > 0 and stride > 1:
            shift = (global_start - self.height_halo_size + pad_top) % stride
            if shift:
                x_padded = x_padded[..., shift:, :]
                global_start += shift

        out = super().forward(x_padded)

        if self.height_halo_size == 0:
            return out

        local_height = x.shape[-2]
        global_height = local_height * self.world_size
        halo = self.height_halo_size
        pad_bottom = self.height_pad_bottom
        kernel = self.kernel_size[-2]
        min_i = math.ceil(((-pad_top) - (global_start - halo)) / stride)
        max_i = math.floor(
            ((global_height - 1 + pad_bottom) - (kernel - 1) - (global_start - halo))
            / stride
        )
        start = max(min_i, 0)
        end = min(max_i + 1, out.shape[-2])
        if start != 0 or end != out.shape[-2]:
            out = out[..., start:end, :]

        return out


class WanDistCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.height_pad_top = self.padding[1]
        self.height_pad_bottom = self.padding[1]
        self.height_halo_size = (self.kernel_size[-2] - 1) // 2

        self.padding: tuple[int, int, int]
        # Set up causal padding, let the halo to control height padding
        if self.height_halo_size > 0:
            self._padding: tuple[int, ...] = (
                self.padding[2],
                self.padding[2],
                0,
                0,
                2 * self.padding[0],
                0,
            )
        else:
            self._padding: tuple[int, ...] = (
                self.padding[2],
                self.padding[2],
                self.padding[1],
                self.padding[1],
                2 * self.padding[0],
                0,
            )
        self.padding = (0, 0, 0)
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        x = F.pad(x, padding)

        x = (
            x if current_platform.is_amp_supported() else x.to(self.weight.dtype)
        )  # casting needed if amp isn't supported

        x_padded, self._halo_recv_top_buf, self._halo_recv_bottom_buf = halo_exchange(
            x,
            height_halo_size=self.height_halo_size,
            recv_top_buf=self._halo_recv_top_buf,
            recv_bottom_buf=self._halo_recv_bottom_buf,
        )

        pad_top = self.height_pad_top
        stride = self.stride[-2]
        global_start = self.rank * x.shape[-2]
        if self.height_halo_size > 0 and stride > 1:
            shift = (global_start - self.height_halo_size + pad_top) % stride
            if shift:
                x_padded = x_padded[..., shift:, :]
                global_start += shift

        x_padded = match_conv3d_input_format(x_padded, self.weight)
        out = super().forward(x_padded)

        if self.height_halo_size == 0:
            return out

        local_height = x.shape[-2]
        global_height = local_height * self.world_size
        halo = self.height_halo_size
        pad_bottom = self.height_pad_bottom
        kernel = self.kernel_size[-2]
        min_i = math.ceil(((-pad_top) - (global_start - halo)) / stride)
        max_i = math.floor(
            ((global_height - 1 + pad_bottom) - (kernel - 1) - (global_start - halo))
            / stride
        )
        start = max(min_i, 0)
        end = min(max_i + 1, out.shape[-2])
        if start != 0 or end != out.shape[-2]:
            out = out[..., start:end, :]

        return out


class WanDistZeroPad2d(nn.Module):
    """Apply 2D padding once globally across sequence-parallel height splits."""

    def __init__(self, padding: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.padding = padding  # (left, right, top, bottom)
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.padding
        if self.world_size <= 1:
            return F.pad(x, (left, right, top, bottom))
        # Only the first/last rank should contribute global top/bottom padding.
        top = top if self.rank == 0 else 0
        bottom = bottom if self.rank == self.world_size - 1 else 0
        return F.pad(x, (left, right, top, bottom))


class WanDistResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data used for parallel decoding.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        # We support parallel encode/decode; downsample uses halo exchange as well.
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanDistConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanDistConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                WanDistZeroPad2d((0, 1, 0, 0)),
                WanDistConv2d(dim, dim, 3, stride=(2, 2), height_padding=(0, 1)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                WanDistZeroPad2d((0, 1, 0, 0)),
                WanDistConv2d(dim, dim, 3, stride=(2, 2), height_padding=(0, 1)),
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return resample_forward(self, x)


class WanDistResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_act_fn(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanDistCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanDistCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanDistCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x):
        return residual_block_forward(self, x)


class WanDistAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()
        self.decode_group = get_decode_parallel_group_coordinator()

    def forward(self, x):
        if self.world_size > 1:
            x = self.decode_group.all_gather(
                maybe_contiguous_for_decode_gather(x), dim=-2
            )
            x = x.contiguous()
        x = attention_block_forward(self, x)
        if self.world_size > 1:
            x = torch.chunk(x, self.world_size, dim=-2)[self.rank]

        return x


class WanDistMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanDistResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanDistAttentionBlock(dim))
            resnets.append(WanDistResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x):
        return mid_block_forward(self, x)


class WanDistResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        num_res_blocks,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanDistResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanDistResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x):
        return residual_down_block_forward(self, x)


class WanDistResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanDistResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanDistResample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x):
        return residual_up_block_forward(self, x)


class WanDistUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanDistResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [WanDistResample(out_dim, mode=upsample_mode)]
            )

        self.gradient_checkpointing = False

    def forward(self, x):
        return up_block_forward(self, x)
