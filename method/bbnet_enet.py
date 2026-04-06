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
import timm

class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=2):
        super().__init__()
        print("Improved GlobalSparseAttn")
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr= sr_ratio
        if self.sr > 1:
            self.sampler   = nn.AvgPool2d(kernel_size=1, stride=sr_ratio)
            self.LocalProp = nn.ConvTranspose2d(dim, dim,
                                                kernel_size=sr_ratio,
                                                stride=sr_ratio,
                                                groups=dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip = x                                                    # (B, C, H, W)
        B, C, H, W = x.shape

        if self.sr > 1:
            x = self.sampler(x)                                     # (B, C, H/sr, W/sr)

        x = x.flatten(2).transpose(1, 2)                           # (B, N', C)
        N = x.shape[1]

        qkv = self.qkv(x)                                          # (B, N', 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N', 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, B, heads, N', head_dim)
        q, k, v = qkv.unbind(0)                                    # each: (B, heads, N', head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B, heads, N', N')
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)                                             # (B, heads, N', head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)                    # (B, N', C)

        if self.sr > 1:
            Hs, Ws = H // self.sr, W // self.sr
            x = x.transpose(1, 2).reshape(B, C, Hs, Ws)           # (B, C, H/sr, W/sr)
            x = self.LocalProp(x)                                  # (B, C, H, W)
            x = x.flatten(2).transpose(1, 2)                      # (B, N, C),  N = H*W

        x = self.proj(x)                                           # (B, N, C)
        x = self.proj_drop(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)                 # (B, C, H, W)
        return x + skip


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


class Enet_Encoder_Unet_Decoder(BaseModel):
    def __init__(
            self,
            encoder_in_channels = 5,
            out_channels = 3,
            encoder_name = 'efficientnet_b2',
            features=[128, 128, 256, 256],
            non_negative=False,
            use_bn=True,
            enable_attention_hooks=False,
            scale_factor=2
    ):
        super(Enet_Encoder_Unet_Decoder, self).__init__()

        self.encoder = timm.create_model(encoder_name, pretrained=True, in_chans=encoder_in_channels, features_only=True, out_indices=(1, 2, 3, 4))
        encoder_channels = self.encoder.feature_info.channels()

        self.coord_adapter = nn.Conv2d(features[3] + 2, features[3], kernel_size=1)
        
        # Parallel HR Stream
        hr_dim = 16
        self.hr_entry = nn.Conv2d(encoder_in_channels, hr_dim, kernel_size=3, stride=1, padding=1)
        
        self.hr_block4 = ResidualConvUnit_custom(hr_dim, nn.ReLU(), bn=True)
        self.hr_fuse4 = nn.Sequential(
            nn.Conv2d(features[2], hr_dim, kernel_size=1),
            Interpolate(scale_factor=32, mode="bilinear", align_corners=False)
        )
        
        self.hr_block3 = ResidualConvUnit_custom(hr_dim, nn.ReLU(), bn=True)
        self.hr_fuse3 = nn.Sequential(
            nn.Conv2d(features[1], hr_dim, kernel_size=1),
            Interpolate(scale_factor=16, mode="bilinear", align_corners=False)
        )
        
        self.hr_block2 = ResidualConvUnit_custom(hr_dim, nn.ReLU(), bn=True)
        self.hr_fuse2 = nn.Sequential(
            nn.Conv2d(features[0], hr_dim, kernel_size=1),
            Interpolate(scale_factor=8, mode="bilinear", align_corners=False)
        )
        
        self.hr_block1 = ResidualConvUnit_custom(hr_dim, nn.ReLU(), bn=True)
        self.hr_fuse1 = nn.Sequential(
            nn.Conv2d(features[0], hr_dim, kernel_size=1),
            Interpolate(scale_factor=4, mode="bilinear", align_corners=False)
        )

        self.scratch = _make_scratch(encoder_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features[0], features[0], use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features[1], features[0], use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features[2], features[1], use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features[3], features[2], use_bn)
        
        self.transformer_layer = []
        for i in range(0,12):
            self.transformer_layer.append(GlobalSparseAttn(dim=features[2]))

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
            nn.Conv2d(encoder_in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
        )

        head = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Identity(),
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0),   # 4 channel op prediction RGB + segmask
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity()
        )

        self.scratch.output_conv = head
        self.scratch.output_conv1 = head1
        self.scratch.output_conv2 = head2

        self.downsample = Interpolate(scale_factor=0.5, mode="bilinear", align_corners=False)


    def forward(self, source, kernel_map, bloom_input, coord_maps, f_stop=None):
        enc_in = torch.cat([source, kernel_map, bloom_input], dim=1)  # [1, 5, 1408, 1408]
        LR = self.downsample(enc_in) # LR [1, 5, 704, 704] 
        
        # Parallel HR Stream Entry
        hr_stream = self.hr_entry(enc_in) # 1408x1408
        
        features = self.encoder(LR)
        layer1, layer2, layer3, layer4 = features[0], features[1], features[2], features[3]

        layer_1_rn = self.scratch.layer1_rn(layer1) 
        layer_2_rn = self.scratch.layer2_rn(layer2) 
        layer_3_rn = self.scratch.layer3_rn(layer3) 
        layer_4_rn = self.scratch.layer4_rn(layer4) 
        
        # Injection of Coordinate Maps into Bottleneck
        coords_down = F.interpolate(coord_maps, size=layer_4_rn.shape[-2:], mode='bilinear', align_corners=False)
        layer_4_rn = torch.cat([layer_4_rn, coords_down], dim=1)
        layer_4_rn = self.coord_adapter(layer_4_rn)

        layer_4_rn = self.transformer_layer(layer_4_rn )

        path_4 = self.scratch.refinenet4(layer_4_rn)
        hr_stream = self.hr_block4(hr_stream + self.hr_fuse4(path_4))

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        hr_stream = self.hr_block3(hr_stream + self.hr_fuse3(path_3))

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        hr_stream = self.hr_block2(hr_stream + self.hr_fuse2(path_2))

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        hr_stream = self.hr_block1(hr_stream + self.hr_fuse1(path_1))
        
        out1 = self.scratch.output_conv1(path_1)
        HR = self.scratch.output_conv2(enc_in)
        
        out = torch.cat((out1, HR, hr_stream), 1)
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
