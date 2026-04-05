"""
BBNet-AAA: BBNet with Aperture-Aware Attention Bottleneck

Architecture: ResNet-D encoder (44x44) -> 3 Residual Groups x 3 AAB -> DPT decoder
- Replaces original 12x GlobalSparseAttn with 9x ApertureAttentionBlock
- Row-column decomposed attention: O(H^2*W + H*W^2) instead of O(N^2)
- Aperture-conditioned via DynRelPos2d decay masks from f_stop
- CoordConv at group boundaries for spatial awareness

Based on bbnet_upscale_bn.py (44x44 bottleneck variant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from timm.layers import DropPath

# Import AAA components from Bokehlicious
from method.nn_util import (
    ApertureAwareAttention,
    DynRelPos2d,
    DWConv2d,
)
from method.blocks import FeedForwardNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: ResNet-D (stops at 44x44, no 22x22 bottleneck layer)
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock without BatchNorm.

    Tensor trace (e.g. inplanes=128, planes=256, stride=2):
        Input:  (B, 128, 88, 88)
        conv1:  (B, 256, 44, 44)   -- stride=2 downsamples
        ReLU
        conv2:  (B, 256, 44, 44)
        + identity (via downsample: AvgPool2d + 1x1 conv)
        ReLU
        Output: (B, 256, 44, 44)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, split_stride=False):
        super(BasicBlock, self).__init__()
        if split_stride and stride > 3:
            self.conv1 = nn.Sequential(
                conv3x3(inplanes, planes, stride - stride // 2),
                nn.ReLU(inplace=True),
                conv3x3(planes, planes, stride=stride // 2)
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.activation(out)
        return out


class ResNet_D(nn.Module):
    """
    ResNet-D encoder producing features at 352^2, 176^2, 88^2, 44^2 (no 22x22 layer).

    Tensor trace (input_channels=6, 704x704 input):
        conv1 stride=2: (B, 32, 352, 352)
        conv2 stride=1: (B, 32, 352, 352)
        x1 = activation: (B, 32, 352, 352)
        x2 = layer1:     (B, 64, 176, 176)
        x3 = layer2:     (B, 128, 88, 88)
        x4 = layer3:     (B, 256, 44, 44)
    """

    def __init__(self, block, layers, input_channels=3, norm_layer=None, late_downsample=False,
                 width_mult=1.0, stride=[2, 2, 2, 2, 2],
                 encoder_channels=[32, 64, 128, 256, 512, 160]):
        super(ResNet_D, self).__init__()
        self.logger = logging.getLogger("Logger")
        self.channel = encoder_channels
        self.channel = [_make_divisible(c * width_mult, 8, 1) for c in self.channel]
        self.late_downsample = late_downsample
        self.midplanes = _make_divisible(32 * width_mult, 8, 1)
        self.inplanes = self.midplanes
        self.start_stride = [1, stride[0], 1, stride[1]] if late_downsample else [stride[0], 1, 1, stride[1]]

        self.conv1 = nn.Conv2d(input_channels, self.channel[0], kernel_size=3,
                               stride=self.start_stride[0], padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.channel[0], self.midplanes, kernel_size=3,
                               stride=self.start_stride[1], padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.midplanes, self.inplanes, kernel_size=3,
                               stride=self.start_stride[2], padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.channel[1], layers[0], stride=self.start_stride[3])
        self.layer2 = self._make_layer(block, self.channel[2], layers[1], stride=stride[2])
        self.layer3 = self._make_layer(block, self.channel[3], layers[2], stride=stride[3])
        # No layer_bottleneck — attention bottleneck operates at 44x44

        self.final_conv = nn.Sequential(
            conv1x1(self.inplanes, self.channel[5], 1),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.logger.debug("encoder conv1 weight shape: {}".format(str(self.conv1.weight.data.shape)))
        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                conv1x1(self.inplanes, planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)                  # (B, 32, 352, 352)
        x = self.activation(x)
        x = self.conv2(x)                  # (B, 32, 352, 352)
        x1 = self.activation(x)            # (B, 32, 352, 352)
        x2 = self.layer1(x1)               # (B, 64, 176, 176)
        x3 = self.layer2(x2)               # (B, 128, 88, 88)
        x4 = self.layer3(x3)               # (B, 256, 44, 44)
        return x1, x2, x3, x4


# ─────────────────────────────────────────────────────────────────────────────
# Scratch layers (feature reassembly)
# ─────────────────────────────────────────────────────────────────────────────

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape[0], kernel_size=3, stride=1,
                                  padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape[1], kernel_size=3, stride=1,
                                  padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape[2], kernel_size=3, stride=1,
                                  padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape[3], kernel_size=3, stride=1,
                                  padding=1, bias=False, groups=groups)
    return scratch


# ─────────────────────────────────────────────────────────────────────────────
# Decoder: DPT-style feature fusion
# ─────────────────────────────────────────────────────────────────────────────

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1,
                               bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1,
                               bias=not self.bn, groups=self.groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, out_features, activation, deconv=False, bn=False,
                 expand=False, align_corners=False):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1,
                                  padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, out_features, use_bn):
    return FeatureFusionBlock_custom(
        features, out_features, nn.ReLU(False),
        deconv=False, bn=use_bn, expand=False, align_corners=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Base model and interpolation utilities
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode,
                        align_corners=self.align_corners)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Aperture-Aware Attention Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class ApertureAttentionBlock(nn.Module):
    """
    Single Aperture-Aware Attention Block (AAB).

    Architecture (operates in BHWC format):
        x -> DWConv(pos) -> + x                         # local positional encoding
          -> LayerNorm -> AAA(x, rel_pos) -> + residual  # attention sub-block
          -> LayerNorm -> FFN(DWConv + GELU) -> + residual # feed-forward sub-block

    Tensor trace (dim=256, heads=4, ffn_dim=512, H=W=44):
        Input:  (B, 44, 44, 256)
        DWConv: (B, 44, 44, 256)   -- local positional encoding (LEPE-style)
        LN1:    (B, 44, 44, 256)
        AAA:    (B, 44, 44, 256)   -- row-col decomposed attention with aperture decay
        +res:   (B, 44, 44, 256)
        LN2:    (B, 44, 44, 256)
        FFN:    (B, 44, 44, 256)   -- fc1(256->512) -> GELU -> DWConv -> fc2(512->256)
        +res:   (B, 44, 44, 256)
        Output: (B, 44, 44, 256)

    Mathematical basis:
        Attention: softmax(Q_row * K_row^T / sqrt(d_k) + decay_mask) * V
        where decay_mask = -|i-j| * log(1 - 2^(-gamma)) encodes aperture-conditioned
        spatial locality. gamma is derived from f_stop via DynRelPos2d.
    """

    def __init__(self, dim, num_heads, ffn_dim, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.pos = DWConv2d(dim, 3, 1, 1)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ApertureAwareAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(dim, ffn_dim, subconv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rel_pos):
        """
        Args:
            x: (B, H, W, C) feature tensor in BHWC format
            rel_pos: tuple (mask_h, mask_w) from DynRelPos2d
        Returns:
            (B, H, W, C)
        """
        # Local positional encoding via depthwise conv
        x = x + self.pos(x)
        # Attention sub-block: LN -> AAA -> residual
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos))
        # FFN sub-block: LN -> FFN(DWConv+GELU) -> residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ResidualGroup(nn.Module):
    """
    Residual Group: N x AAB + auxiliary-conditioned conv + group-level skip.

    Handles BCHW <-> BHWC format conversions at group boundaries so the
    rest of the BBNet pipeline (encoder, decoder) stays in BCHW.

    Architecture:
        x_bchw -> permute(BHWC)
            -> AAB_1(x, rel_pos)
            -> AAB_2(x, rel_pos)
            -> AAB_3(x, rel_pos)
        -> permute(BCHW)
        -> cat(x, coord_map, kernel_map)  [if enabled]
        -> Conv2d(C+3 -> C, 3x3)
        -> + skip (group residual)

    Tensor trace (dim=256, 3 blocks, coord+kernel, 44x44):
        Input:  (B, 256, 44, 44)  BCHW
        permute: (B, 44, 44, 256) BHWC
        3x AAB: (B, 44, 44, 256) BHWC
        permute: (B, 256, 44, 44) BCHW
        cat:    (B, 259, 44, 44)  BCHW  -- +2 coord +1 kernel
        conv:   (B, 256, 44, 44)  BCHW
        +skip:  (B, 256, 44, 44)  BCHW
        Output: (B, 256, 44, 44)
    """

    def __init__(self, dim, num_heads, num_blocks=3, ffn_ratio=2.,
                 use_coord_conv=True, use_kernel_conv=True, drop_path=0.):
        super().__init__()
        ffn_dim = int(dim * ffn_ratio)

        # Handle per-block drop path rates
        if isinstance(drop_path, (list, tuple)):
            dpr = drop_path
        else:
            dpr = [drop_path] * num_blocks

        self.blocks = nn.ModuleList([
            ApertureAttentionBlock(dim, num_heads, ffn_dim, drop_path=dpr[i])
            for i in range(num_blocks)
        ])

        self.use_coord_conv = use_coord_conv
        self.use_kernel_conv = use_kernel_conv
        extra_ch = (2 if use_coord_conv else 0) + (1 if use_kernel_conv else 0)
        self.extra_ch = extra_ch
        self.conv = nn.Conv2d(dim + extra_ch, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rel_pos, coord_map=None, kernel_map=None):
        """
        Args:
            x: (B, C, H, W) feature tensor in BCHW format
            rel_pos: tuple (mask_h, mask_w) from DynRelPos2d
            coord_map: (B, 2, H, W) coordinate maps, or None
            kernel_map: (B, 1, H, W) CoC map at bottleneck resolution, or None
        Returns:
            (B, C, H, W) in BCHW format
        """
        skip = x

        # BCHW -> BHWC for attention blocks
        x = x.permute(0, 2, 3, 1)

        for block in self.blocks:
            x = block(x, rel_pos)

        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)

        # Concatenate auxiliary maps before group boundary conv
        aux = []
        if self.use_coord_conv:
            if coord_map is not None:
                aux.append(coord_map)                                       # (B, 2, H, W)
            else:
                B_, C_, H_, W_ = x.shape
                aux.append(torch.zeros(B_, 2, H_, W_, device=x.device, dtype=x.dtype))
        if self.use_kernel_conv:
            if kernel_map is not None:
                aux.append(kernel_map)                                      # (B, 1, H, W)
            else:
                B_, C_, H_, W_ = x.shape
                aux.append(torch.zeros(B_, 1, H_, W_, device=x.device, dtype=x.dtype))
        if aux:
            x = torch.cat([x] + aux, dim=1)                                 # (B, C+3, H, W)

        x = self.conv(x)

        return x + skip


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: BBNet with AAA Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class Inception_Encoder_Unet_Decoder(BaseModel):
    """
    BBNet-AAA: ResNet-D encoder + 3xRG AAA bottleneck + DPT decoder.

    Inputs (passed as *inputs tuple):
        inputs[0]: source       (B, 3, H, W)  - RGB image
        inputs[1]: kernel_map   (B, 1, H, W)  - Circle of Confusion map
        inputs[2]: bloom_input  (B, 1, H, W)  - Specular highlight map
        inputs[3]: coord_maps   (B, 2, H, W)  - Normalized spatial coordinates
        inputs[4]: f_stop       (B,)          - Aperture value

    Injection strategy:
        Encoder (6ch):  source(3) + kernel_map(1) + coord_maps(2)
        Bottleneck:     f_stop -> DynRelPos2d; coord_maps(2) + kernel_map(1) at RG conv
        Output head:    bloom_input(1) late injection

    Full pipeline tensor trace (B=1, 1408x1408):
        Encoder input: cat(source, kernel_map, coord_maps) -> (1, 6, 1408, 1408)
        |-- downsample 0.5x:  (1, 6, 704, 704)
        |-- ResNet-D encoder:
        |   x1: (1, 32, 352, 352)
        |   x2: (1, 64, 176, 176)
        |   x3: (1, 128, 88, 88)
        |   x4: (1, 256, 44, 44)
        |
        |-- Scratch reassembly:
        |   layer_1_rn: (1, 128, 352, 352)
        |   layer_2_rn: (1, 128, 176, 176)
        |   layer_3_rn: (1, 256, 88, 88)
        |   layer_4_rn: (1, 256, 44, 44)
        |
        |-- AAA Bottleneck @ 44x44:
        |   DynRelPos2d(f_stop) -> (mask_h, mask_w)
        |   RG1: 3x AAB(256, heads=4) + conv(C+3->C)  -> (1, 256, 44, 44)
        |   RG2: 3x AAB(256, heads=4) + conv(C+3->C)  -> (1, 256, 44, 44)
        |   RG3: 3x AAB(256, heads=4) + conv(C+3->C)  -> (1, 256, 44, 44)
        |
        |-- DPT decoder (refinenets):
        |   path_4: (1, 256, 88, 88)
        |   path_3: (1, 128, 176, 176)
        |   path_2: (1, 128, 352, 352)
        |   path_1: (1, 128, 704, 704)
        |
        |-- Head:
        |   head1: (1, 32, 1408, 1408) -- from decoder path
        |   head2: (1, 32, 1408, 1408) -- from HR source
        |   cat(head1, head2, bloom) + head: (1, 3, 1408, 1408)
        |
        Output: (1, 3, 1408, 1408)
    """

    def __init__(
            self,
            layers=[3, 4, 4, 2],
            encoder_start_filts=32,
            encoder_in_channels=6,
            out_channels=3,
            encoder_channels=[32, 64, 128, 256, 512, 160],
            width_mult=1,
            decoder_channels=[256, 512, 768, 768],
            features=[128, 128, 256, 256],
            non_negative=False,
            use_bn=False,
            enable_attention_hooks=False,
            scale_factor=2,
            # AAA bottleneck hyperparameters
            bottleneck_dim=256,
            bottleneck_heads=4,
            bottleneck_ffn_ratio=2.,
            num_residual_groups=3,
            blocks_per_group=3,
            use_coord_conv=True,
            use_kernel_conv=True,
            drop_path_rate=0.05,
            # DynRelPos2d parameters
            relpos_init_value=2,
            relpos_heads_range=6,
    ):
        super(Inception_Encoder_Unet_Decoder, self).__init__()

        # ── Encoder ──
        self.resnet = ResNet_D(BasicBlock, input_channels=encoder_in_channels,
                               layers=layers, encoder_channels=encoder_channels,
                               width_mult=width_mult)

        # ── Scratch reassembly (encoder_channels[0:4] for 44x44 variant) ──
        self.scratch = _make_scratch(encoder_channels[0:4], features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features[0], features[0], use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features[1], features[0], use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features[2], features[1], use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features[3], features[2], use_bn)

        # ── AAA Bottleneck ──
        # Shared aperture-aware relative position encoder
        self.relpos = DynRelPos2d(
            embed_dim=bottleneck_dim,
            num_heads=bottleneck_heads,
            initial_value=relpos_init_value,
            heads_range=relpos_heads_range,
        )

        # Stochastic depth: linearly increasing drop path across all blocks
        total_blocks = num_residual_groups * blocks_per_group
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # 3 Residual Groups
        self.residual_groups = nn.ModuleList()
        for rg_idx in range(num_residual_groups):
            start = rg_idx * blocks_per_group
            end = start + blocks_per_group
            self.residual_groups.append(
                ResidualGroup(
                    dim=bottleneck_dim,
                    num_heads=bottleneck_heads,
                    num_blocks=blocks_per_group,
                    ffn_ratio=bottleneck_ffn_ratio,
                    use_coord_conv=use_coord_conv,
                    use_kernel_conv=use_kernel_conv,
                    drop_path=dpr[start:end],
                )
            )

        # ── Decoder heads ──
        head1 = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features[0] // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity()
        )

        # head2: HR branch processes source RGB only (3ch)
        head2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
        )

        # head: +1 bloom channel for late specular highlight injection
        head = nn.Sequential(
            nn.Conv2d(64 + 1, 32, kernel_size=3, stride=1, padding=1),   # 32+32+1 bloom
            nn.ReLU(True),
            nn.Identity(),
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity()
        )

        self.scratch.output_conv = head
        self.scratch.output_conv1 = head1
        self.scratch.output_conv2 = head2

        self.downsample = Interpolate(scale_factor=0.5, mode="bilinear", align_corners=False)

    def forward(self, *inputs):
        """
        Forward pass.

        Args:
            inputs[0]: source       (B, 3, H, W)  - RGB image
            inputs[1]: kernel_map   (B, 1, H, W)  - Circle of Confusion (blur radius per pixel)
            inputs[2]: bloom_input  (B, 1, H, W)  - Specular highlight map
            inputs[3]: coord_maps   (B, 2, H, W)  - Normalized spatial coordinates
            inputs[4]: f_stop       (B,)          - Aperture value for attention decay

        Returns:
            (B, 3, H, W) output bokeh-rendered image
        """
        source, kernel_map, bloom_input, coord_maps, f_stop = inputs

        # ── Build encoder input: source(3) + kernel_map(1) + coord_maps(2) = 6ch ──
        encoder_input = torch.cat([source, kernel_map, coord_maps], dim=1)  # (B, 6, H, W)
        LR = self.downsample(encoder_input)                                 # (B, 6, H/2, W/2)

        # ── Encoder ──
        layer1, layer2, layer3, layer4 = self.resnet(LR)
        # layer1: (B, 32, H/4, W/4)    e.g. 352x352
        # layer2: (B, 64, H/8, W/8)    e.g. 176x176
        # layer3: (B, 128, H/16, W/16) e.g. 88x88
        # layer4: (B, 256, H/32, W/32) e.g. 44x44

        # ── Scratch reassembly ──
        layer_1_rn = self.scratch.layer1_rn(layer1)     # (B, 128, H/4, W/4)
        layer_2_rn = self.scratch.layer2_rn(layer2)     # (B, 128, H/8, W/8)
        layer_3_rn = self.scratch.layer3_rn(layer3)     # (B, 256, H/16, W/16)
        layer_4_rn = self.scratch.layer4_rn(layer4)     # (B, 256, H/32, W/32)

        # ── AAA Bottleneck ──
        B, C, H_bn, W_bn = layer_4_rn.shape             # (B, 256, 44, 44)

        # Generate aperture-aware positional decay masks from f_stop
        rel_pos = self.relpos((H_bn, W_bn), range_factor=f_stop)

        # Downsample auxiliary maps to bottleneck resolution
        coord_map_bn = F.interpolate(coord_maps, size=(H_bn, W_bn),
                                      mode='bilinear', align_corners=False)
        kernel_map_bn = F.interpolate(kernel_map, size=(H_bn, W_bn),
                                       mode='bilinear', align_corners=False)

        # Pass through 3 Residual Groups (coord + kernel conditioning at each)
        for rg in self.residual_groups:
            layer_4_rn = rg(layer_4_rn, rel_pos, coord_map_bn, kernel_map_bn)

        # ── DPT Decoder ──
        path_4 = self.scratch.refinenet4(layer_4_rn)            # (B, 256, H/16, W/16)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)    # (B, 128, H/8, W/8)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)    # (B, 128, H/4, W/4)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)    # (B, 128, H/2, W/2)

        # ── Output heads ──
        out1 = self.scratch.output_conv1(path_1)                # (B, 32, H, W)
        HR = self.scratch.output_conv2(source)                  # (B, 32, H, W) source-only

        # Late bloom injection: cat decoder + HR + bloom → head
        out = torch.cat((out1, HR, bloom_input), 1)             # (B, 65, H, W)
        out = self.scratch.output_conv(out)                     # (B, 3, H, W)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Inception_Encoder_Unet_Decoder()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    bottleneck_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'residual_groups' in name or 'relpos' in name
    )
    print(f"Total parameters:      {total_params:>12,}")
    print(f"Bottleneck parameters: {bottleneck_params:>12,}")
    print(f"Other parameters:      {total_params - bottleneck_params:>12,}")

    # Forward pass test with tuple inputs
    H, W = 352, 352
    source      = torch.rand((1, 3, H, W))
    kernel_map  = torch.rand((1, 1, H, W))
    bloom_input = torch.rand((1, 1, H, W))
    coord_maps  = torch.rand((1, 2, H, W))
    f_stop      = torch.tensor([2.0])

    print(f"\nsource:      {source.shape}")
    print(f"kernel_map:  {kernel_map.shape}")
    print(f"bloom_input: {bloom_input.shape}")
    print(f"coord_maps:  {coord_maps.shape}")
    print(f"f_stop:      {f_stop}")

    output = model(source, kernel_map, bloom_input, coord_maps, f_stop)
    print(f"Output:      {output.shape}")
