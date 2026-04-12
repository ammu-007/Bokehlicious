import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from functools import partial
import types
import math
import numpy as np
from PIL import Image
import PIL.Image as pil
from torchvision import transforms, datasets
import torch.utils.data as data
import glob
import gc
import os
from pdb import set_trace as stx
import numbers
import logging
from torchvision.utils import save_image
from safetensors.torch import load_file


# ============================================================================
# EfficientNet-B2 Encoder (Pure PyTorch, loaded from safetensors)
# ============================================================================
# Architecture exactly mirrors timm/efficientnet_b2.ra_in1k so that
# state-dict keys match 1-to-1 with the downloaded model.safetensors.
#
# Feature extraction points (out_indices 1-4):
#   After block 1 → channels=24,  reduction=/4
#   After block 2 → channels=48,  reduction=/8
#   After block 4 → channels=120, reduction=/16
#   After block 6 → channels=352, reduction=/32
# ============================================================================

class SqueezeExcite(nn.Module):
    """SE block: global-pool → reduce → expand → sigmoid gate."""
    def __init__(self, in_chs, reduce_chs):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_chs, reduce_chs, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduce_chs, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean(dim=(2, 3), keepdim=True)
        x_se = self.gate(self.conv_expand(self.act1(self.conv_reduce(x_se))))
        return x * x_se


class DepthwiseSeparableConv(nn.Module):
    """DW-separable conv used in EfficientNet block-0 (expand_ratio=1)."""
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1,
                 se_reduce=None, has_skip=False):
        super().__init__()
        self.has_skip = has_skip
        self.conv_dw = nn.Conv2d(in_chs, in_chs, kernel_size, stride=stride,
                                 padding=kernel_size // 2, groups=in_chs, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.se = SqueezeExcite(in_chs, se_reduce) if se_reduce else nn.Identity()
        self.conv_pw = nn.Conv2d(in_chs, out_chs, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        skip = x
        x = F.silu(self.bn1(self.conv_dw(x)))
        x = self.se(x)
        x = self.bn2(self.conv_pw(x))
        if self.has_skip:
            x = x + skip
        return x


class InvertedResidual(nn.Module):
    """MBConv block: PW-expand → DW → SE → PW-linear projection."""
    def __init__(self, in_chs, out_chs, exp_chs, kernel_size=3, stride=1,
                 se_reduce=None, has_skip=False):
        super().__init__()
        self.has_skip = has_skip
        self.conv_pw = nn.Conv2d(in_chs, exp_chs, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_chs)
        self.conv_dw = nn.Conv2d(exp_chs, exp_chs, kernel_size, stride=stride,
                                 padding=kernel_size // 2, groups=exp_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_chs)
        self.se = SqueezeExcite(exp_chs, se_reduce) if se_reduce else nn.Identity()
        self.conv_pwl = nn.Conv2d(exp_chs, out_chs, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        skip = x
        x = F.silu(self.bn1(self.conv_pw(x)))
        x = F.silu(self.bn2(self.conv_dw(x)))
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_skip:
            x = x + skip
        return x


class EfficientNetB2Features(nn.Module):
    """
    Pure-PyTorch EfficientNet-B2 feature extractor.
    Returns 4 feature maps with channels [24, 48, 120, 352].
    Loads pretrained ImageNet-1k weights from a safetensors file.
    """
    def __init__(self, in_chans=3, weights_path=None):
        super().__init__()
        # Stem
        self.conv_stem = nn.Conv2d(in_chans, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.blocks = nn.Sequential(
            # Block 0: DepthwiseSeparableConv, 32→16, stride=1
            nn.Sequential(
                DepthwiseSeparableConv(32, 16, 3, 1, se_reduce=8),
                DepthwiseSeparableConv(16, 16, 3, 1, se_reduce=4, has_skip=True),
            ),
            # Block 1: MBConv6 k3, 16→24, stride=2
            nn.Sequential(
                InvertedResidual(16,  24,   96, 3, stride=2, se_reduce=4),
                InvertedResidual(24,  24,  144, 3, stride=1, se_reduce=6,  has_skip=True),
                InvertedResidual(24,  24,  144, 3, stride=1, se_reduce=6,  has_skip=True),
            ),
            # Block 2: MBConv6 k5, 24→48, stride=2
            nn.Sequential(
                InvertedResidual(24,  48,  144, 5, stride=2, se_reduce=6),
                InvertedResidual(48,  48,  288, 5, stride=1, se_reduce=12, has_skip=True),
                InvertedResidual(48,  48,  288, 5, stride=1, se_reduce=12, has_skip=True),
            ),
            # Block 3: MBConv6 k3, 48→88, stride=2
            nn.Sequential(
                InvertedResidual(48,  88,  288, 3, stride=2, se_reduce=12),
                InvertedResidual(88,  88,  528, 3, stride=1, se_reduce=22, has_skip=True),
                InvertedResidual(88,  88,  528, 3, stride=1, se_reduce=22, has_skip=True),
                InvertedResidual(88,  88,  528, 3, stride=1, se_reduce=22, has_skip=True),
            ),
            # Block 4: MBConv6 k5, 88→120, stride=1
            nn.Sequential(
                InvertedResidual(88,  120, 528, 5, stride=1, se_reduce=22),
                InvertedResidual(120, 120, 720, 5, stride=1, se_reduce=30, has_skip=True),
                InvertedResidual(120, 120, 720, 5, stride=1, se_reduce=30, has_skip=True),
                InvertedResidual(120, 120, 720, 5, stride=1, se_reduce=30, has_skip=True),
            ),
            # Block 5: MBConv6 k5, 120→208, stride=2
            nn.Sequential(
                InvertedResidual(120, 208, 720,  5, stride=2, se_reduce=30),
                InvertedResidual(208, 208, 1248, 5, stride=1, se_reduce=52, has_skip=True),
                InvertedResidual(208, 208, 1248, 5, stride=1, se_reduce=52, has_skip=True),
                InvertedResidual(208, 208, 1248, 5, stride=1, se_reduce=52, has_skip=True),
                InvertedResidual(208, 208, 1248, 5, stride=1, se_reduce=52, has_skip=True),
            ),
            # Block 6: MBConv6 k3, 208→352, stride=1
            nn.Sequential(
                InvertedResidual(208, 352, 1248, 3, stride=1, se_reduce=52),
                InvertedResidual(352, 352, 2112, 3, stride=1, se_reduce=88, has_skip=True),
            ),
        )

        self.feature_channels = [24, 48, 120, 352]

        if weights_path and os.path.exists(weights_path):
            self._load_pretrained(weights_path, in_chans)

    def _load_pretrained(self, path, in_chans):
        state = load_file(path)

        # Strip classification head keys (conv_head, bn2, classifier)
        state = {k: v for k, v in state.items()
                 if not k.startswith(('conv_head', 'bn2.', 'classifier.'))}

        # Adapt stem for non-3-channel inputs
        if in_chans != 3:
            stem_w = state['conv_stem.weight']               # [32, 3, 3, 3]
            mean_w = stem_w.mean(dim=1, keepdim=True)        # [32, 1, 3, 3]
            new_w  = mean_w.repeat(1, in_chans, 1, 1)        # [32, C, 3, 3]
            new_w[:, :3] = stem_w                             # preserve RGB
            state['conv_stem.weight'] = new_w

        self.load_state_dict(state, strict=True)
        print(f"[EfficientNetB2] Loaded pretrained weights from {path}"
              f" (adapted stem: 3 -> {in_chans} channels)")

    def forward(self, x):
        x = F.silu(self.bn1(self.conv_stem(x)))       # /2  ch=32

        x = self.blocks[0](x)                          # /2  ch=16

        x = self.blocks[1](x)                          # /4  ch=24
        f1 = x

        x = self.blocks[2](x)                          # /8  ch=48
        f2 = x

        x = self.blocks[3](x)                          # /16 ch=88
        x = self.blocks[4](x)                          # /16 ch=120
        f3 = x

        x = self.blocks[5](x)                          # /32 ch=208
        x = self.blocks[6](x)                          # /32 ch=352
        f4 = x

        return [f1, f2, f3, f4]


# ============================================================================
# Bottleneck & Conditioning Modules
# ============================================================================

class MobileBottleneckBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Mobile NPU friendly: large kernel depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # LayerNorm expects channel last
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input + x


class FiLMLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(), 
            nn.Linear(32, in_channels * 2)
        )
    def forward(self, x, condition):
        affine = self.mlp(condition).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = affine.chunk(2, dim=1)
        return x * (1 + gamma) + beta


# ============================================================================
# Decoder Building Blocks
# ============================================================================

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape[0],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape[1],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape[2],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape[3],
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )
        self.bn1 = nn.BatchNorm2d(features) if self.bn else nn.Identity()

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )
        self.bn2 = nn.BatchNorm2d(features) if self.bn else nn.Identity()

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
            self,
            features,
            out_features,
            activation,
            deconv=False,
            bn=False,
            expand=False,
            align_corners=False,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self.pixel_shuffle_expand = nn.Conv2d(
            out_features, 
            out_features * 4, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = self.out_conv(output)
        output = self.pixel_shuffle_expand(output)
        output = self.pixel_shuffle(output)

        return output


def _make_fusion_block(features, out_features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        out_features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=False,
    )


# ============================================================================
# Base Model & Utilities
# ============================================================================

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


# ============================================================================
# Full Model
# ============================================================================

class Enet_Encoder_Unet_Decoder(BaseModel):
    def __init__(
            self,
            encoder_in_channels = 7,
            out_channels = 3,
            features=[128, 128, 256, 256],
            non_negative=False,
            use_bn=True,
            enable_attention_hooks=False,
            scale_factor=2,
            weights_path=None,
    ):
        super(Enet_Encoder_Unet_Decoder, self).__init__()

        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'weights', 'model.safetensors')

        self.encoder = EfficientNetB2Features(
            in_chans=encoder_in_channels,
            weights_path=weights_path
        )
        encoder_channels = self.encoder.feature_channels

        self.film = FiLMLayer(features[3])

        self.scratch = _make_scratch(encoder_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features[0], features[0], use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features[1], features[0], use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features[2], features[1], use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features[3], features[2], use_bn)
        
        self.transformer_layer = []
        for i in range(0,3):
            self.transformer_layer.append(MobileBottleneckBlock(dim=features[2]))

        self.transformer_layer = nn.Sequential(*self.transformer_layer)

        head1 = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features[0] // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReLU(True),
            nn.Identity()
        )

        head2 = nn.Sequential(
            nn.Conv2d(encoder_in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
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


    def forward(self, source, kernel_map, bloom_input, coord_maps, f_stop=None):
        enc_in = torch.cat([source, kernel_map, bloom_input, coord_maps], dim=1)  # [B, 7, 1408, 1408]
        LR = self.downsample(enc_in) # LR [B, 7, 704, 704]
        
        features = self.encoder(LR)
        layer1, layer2, layer3, layer4 = features[0], features[1], features[2], features[3]

        layer_1_rn = self.scratch.layer1_rn(layer1) 
        layer_2_rn = self.scratch.layer2_rn(layer2) 
        layer_3_rn = self.scratch.layer3_rn(layer3) 
        layer_4_rn = self.scratch.layer4_rn(layer4) 
        
        if f_stop is not None:
            layer_4_rn = self.film(layer_4_rn, f_stop)

        layer_4_rn = self.transformer_layer(layer_4_rn )

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out1 = self.scratch.output_conv1(path_1)
        HR = self.scratch.output_conv2(enc_in)
        
        out = torch.cat((out1, HR), 1)
        out = self.scratch.output_conv(out)
        
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Enet_Encoder_Unet_Decoder()
    source = torch.rand((1, 3, 1408, 1408))
    kernel_map = torch.rand((1, 1, 1408, 1408))
    bloom_input = torch.rand((1, 1, 1408, 1408))
    coord_maps = torch.rand((1, 2, 1408, 1408))
    f_stop = torch.rand((1, 1))
    
    print("Testing forward pass...")
    output = model(source, kernel_map, bloom_input, coord_maps, f_stop)
    
    print("Output shape:",output.shape)

    params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Params: {params / 1e6:.2f} M")
