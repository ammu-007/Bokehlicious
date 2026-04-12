"""
BBNet-AAA-NAF: BBNet with NAFBlock Encoder/Decoder + Aperture-Aware Attention Bottleneck

Architecture changes vs bbnet_aaa_v1.py:
  1. NAFBlock encoder (replaces ResNet-D) — SimpleGate + SCA + LayerNorm fix instability
  2. NAFBlock decoder (replaces DPT RefineNet) — learnable PixelShuffle upsampling
  3. 88×88 bottleneck with 6 AAB (replaces 44×44 with 9 AAB) — 2× spatial precision
  4. Separate RGB/guidance stems + SAFM modulation (replaces 6ch concat)

Resolution ladder:
  Input [B,3+4, 1408] → stems stride-2 → [B,32,704] → SAFM fuse
  Encoder: [B,32,704] → stem s2 → [B,32,352] → Stage0(2) → Stage1(4)@176 → Stage2(3)@88
  AAA Bottleneck: 2 RG × 3 AAB at 88×88 with DynRelPos2d(f_stop)
  Decoder: 88→176→352→704 (NAFBlock + PixelShuffle)
  Head: 704→1408 (PixelShuffle) + HR RGB → cat + bloom → RGB

Forward signature (unchanged from bbnet_aaa.py):
  model(source, kernel_map, bloom_input, coord_maps, f_stop)
  #      (B,3)    (B,1)       (B,1)       (B,2)     (B,)
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
# NAFBlock: Core building block for encoder and decoder
# ─────────────────────────────────────────────────────────────────────────────

class NAFBlock(nn.Module):
    """NAFNet block: LayerNorm → DWConv → SimpleGate → SCA → residual
                     LayerNorm → Conv1x1 (expand) → SimpleGate → Conv1x1 → residual

    Uses GroupNorm(1, C) as channel-wise LayerNorm for spatial feature maps.
    SimpleGate: chunk channels in half and multiply — content-adaptive gating
    with no learned activation parameters.

    Math:
        Spatial path:
            y = LN(x)
            y = Conv1×1(y)                    # expand C → 2C
            y = DWConv3×3(y)                  # spatial mixing
            y₁, y₂ = chunk(y, 2)             # SimpleGate split
            y = y₁ ⊙ y₂                      # gated activation
            y = y ⊙ Conv1×1(GAP(y))          # SCA: channel recalibration
            y = Conv1×1(y)                    # project back to C
            x = x + y

        FFN path:
            y = LN(x)
            y = Conv1×1(y) → DWConv3×3(y)    # expand + local mixing
            y₁, y₂ = chunk(y, 2)
            y = y₁ ⊙ y₂
            y = Conv1×1(y)
            x = x + y

    Tensor trace (c=128):
        Input:   (B, 128, H, W)
        Spatial: LN → Conv1×1(128→256) → DWConv → SG → (B,128,H,W) → SCA → Conv1×1 → +res
        FFN:     LN → Conv1×1(128→256) → DWConv → SG → (B,128,H,W) → Conv1×1 → +res
        Output:  (B, 128, H, W)
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = c * dw_expand

        # --- Spatial mixing path ---
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)                                    # Pointwise expand
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)          # DW spatial
        # After SimpleGate: dw_ch → dw_ch // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv2 = nn.Conv2d(dw_ch // 2, c, 1)                               # Pointwise project

        # --- Channel mixing path (FFN) ---
        self.norm2 = nn.GroupNorm(1, c)
        ffn_ch = c * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(c, ffn_ch, 1),
            nn.Conv2d(ffn_ch, ffn_ch, 3, 1, 1, groups=ffn_ch),                 # DW local mixing
        )
        # After SimpleGate: ffn_ch → ffn_ch // 2
        self.ffn_out = nn.Conv2d(ffn_ch // 2, c, 1)

    def forward(self, x):
        # --- Spatial mixing ---
        y = self.norm1(x)
        y = self.conv1(y)                   # (B, C, H, W) → (B, 2C, H, W)
        y = self.dw_conv(y)                 # Spatial mixing within each channel
        y1, y2 = y.chunk(2, dim=1)          # SimpleGate: split along channel dim
        y = y1 * y2                         # Content-adaptive gating: (B, C, H, W)
        y = y * self.sca(y)                 # SCA: channel recalibration via GAP
        y = self.conv2(y)                   # Project back: (B, C, H, W)
        x = x + y                           # Residual connection

        # --- Channel mixing (FFN) ---
        y = self.norm2(x)
        y = self.ffn(y)                     # (B, C, H, W) → (B, 2C, H, W) with DW
        y1, y2 = y.chunk(2, dim=1)          # SimpleGate
        y = y1 * y2                         # (B, C, H, W)
        y = self.ffn_out(y)                 # (B, C, H, W)
        x = x + y                           # Residual connection
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SAFM — Spatially-Adaptive Feature Modulation
# ─────────────────────────────────────────────────────────────────────────────

class SAFM(nn.Module):
    """Multi-scale spatial modulation (from SAFMN paper).

    Captures information at multiple spatial scales via pool → project → upsample,
    then modulates features multiplicatively. Naturally handles spatially-varying
    guidance signals like the CoC kernel map.

    Math:
        m_s = Upsample(Conv1×1(AvgPool_s(g)))  for s ∈ {2, 4}
        fused = Conv1×1(cat[g, m_2, m_4])
        output = x ⊙ fused

    Tensor trace (dim=32, scales=(1,2,4), H=W=704):
        g: (B, 32, 704, 704)     — guidance features
        pool_2: (B, 32, 352, 352) → conv → upsample → (B, 32, 704, 704)
        pool_4: (B, 32, 176, 176) → conv → upsample → (B, 32, 704, 704)
        cat: (B, 96, 704, 704)
        fuse: (B, 32, 704, 704)
        output = x ⊙ fuse: (B, 32, 704, 704)
    """

    def __init__(self, dim, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for _ in scales[1:]      # Skip scale=1
        ])
        self.fuse = nn.Conv2d(dim * len(scales), dim, 1)

    def forward(self, x, guidance):
        """
        Args:
            x: (B, C, H, W) features to be modulated (RGB stem output)
            guidance: (B, C, H, W) guidance features (guidance stem output)
        Returns:
            (B, C, H, W) modulated features
        """
        B, C, H, W = guidance.shape
        multi = [guidance]                              # Scale 1: identity
        for i, s in enumerate(self.scales[1:]):
            pooled = F.adaptive_avg_pool2d(guidance, (max(H // s, 1), max(W // s, 1)))
            projected = self.convs[i](pooled)
            upsampled = F.interpolate(projected, size=(H, W), mode='bilinear', align_corners=False)
            multi.append(upsampled)
        fused = self.fuse(torch.cat(multi, dim=1))      # (B, C*num_scales, H, W) → (B, C, H, W)
        return x * fused                                 # Spatial modulation


# ─────────────────────────────────────────────────────────────────────────────
# Input Stems — Separate RGB and Guidance pathways
# ─────────────────────────────────────────────────────────────────────────────

class InputStem(nn.Module):
    """Lightweight convolutional stem: stride-2 downsample + feature extraction.

    Tensor trace (in_ch=3, out_ch=32, input 1408×1408):
        Conv(3→32, k3, s2, p1): (B, 32, 704, 704)
        Conv(32→32, k3, s1, p1): (B, 32, 704, 704)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# NAFBlock Encoder (3 stages → 88×88)
# ─────────────────────────────────────────────────────────────────────────────

class NAFEncoder(nn.Module):
    """NAFBlock-based multi-scale encoder. Stops at 88×88 for the AAA bottleneck.

    Takes pre-fused features at 704×704 and produces:
        Stage 0: 352×352  (skip₀ for decoder)
        Stage 1: 176×176  (skip₁ for decoder)
        Stage 2: 88×88    (→ bottleneck input)

    All downsampling uses stride-2 convolutions (learned, not pooling).

    Tensor trace (in_ch=32, channels=(32, 128, 256), blocks=(2, 4, 3)):
        Input:   (B, 32, 704, 704)       — fused stem output
        Stem:    Conv(32→32, k3, s2, p1)  → (B, 32, 352, 352)
        Stage 0: 2×NAFBlock(32)           → (B, 32, 352, 352)    skip₀
        Down 0:  Conv(32→128, k2, s2)     → (B, 128, 176, 176)
        Stage 1: 4×NAFBlock(128)          → (B, 128, 176, 176)   skip₁
        Down 1:  Conv(128→256, k2, s2)    → (B, 256, 88, 88)
        Stage 2: 3×NAFBlock(256)          → (B, 256, 88, 88)     → bottleneck
    """

    def __init__(self, in_ch=32, channels=(32, 128, 256), blocks=(2, 4, 3)):
        super().__init__()
        # Internal stem: 704 → 352
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, stride=2, padding=1, bias=False),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1, bias=False),
        )

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels)):
            ch = channels[i]
            prev_ch = channels[i - 1] if i > 0 else channels[0]

            if i > 0:
                self.downs.append(nn.Conv2d(prev_ch, ch, 2, 2))    # Strided conv downsample

            self.stages.append(nn.Sequential(
                *[NAFBlock(ch) for _ in range(blocks[i])]
            ))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, in_ch, 704, 704) fused stem features
        Returns:
            list of 3 feature tensors at decreasing resolutions:
            [(B, 32, 352, 352), (B, 128, 176, 176), (B, 256, 88, 88)]
        """
        x = self.stem(x)                       # (B, 32, 704, 704) → (B, 32, 352, 352)
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downs[i - 1](x)       # Stride-2 downsample
            x = stage(x)                        # NAFBlock processing
            features.append(x)
        return features                         # [352, 176, 88]


# ─────────────────────────────────────────────────────────────────────────────
# NAFBlock Decoder (with PixelShuffle upsampling)
# ─────────────────────────────────────────────────────────────────────────────

class NAFDecoderBlock(nn.Module):
    """Single decoder stage: NAFBlock refinement + skip fusion + PixelShuffle ×2.

    Pipeline:
        1. If skip provided: project skip via 1×1 conv, add to input
        2. NAFBlock refinement (LayerNorm + SimpleGate + SCA)
        3. Conv1×1 expand channels 4× → PixelShuffle(2) for learned ×2 upsampling

    Tensor trace (in_ch=256, out_ch=128, skip_ch=128, H=W=88):
        Input:       (B, 256, 88, 88)
        skip:        (B, 128, 176, 176) — skip from encoder
            project: Conv1×1(128→256) → (B, 256, 176, 176)
            NOTE: skip is at 2× spatial resolution of input, so we upsample
                  input first via PixelShuffle, THEN add skip
        NAFBlock:    (B, out_ch, 176, 176) — refinement after fusion
        Output:      (B, out_ch, 176, 176)

    Actually, the cleaner design: upsample first, then fuse, then refine.
        Conv1×1(in_ch → out_ch×4) → PixelShuffle(2): (B, out_ch, 2H, 2W)
        + skip (projected to out_ch): (B, out_ch, 2H, 2W)
        NAFBlock(out_ch): (B, out_ch, 2H, 2W)
    """

    def __init__(self, in_ch, out_ch, skip_ch=None):
        super().__init__()
        # Learnable PixelShuffle ×2 upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1, bias=True),
            nn.PixelShuffle(2),
        )

        # Skip connection projection (if skip channels differ from out_ch)
        self.has_skip = skip_ch is not None
        if self.has_skip:
            self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False) if skip_ch != out_ch else nn.Identity()

        # Refinement after fusion
        self.refine = NAFBlock(out_ch)

    def forward(self, x, skip=None):
        """
        Args:
            x: (B, in_ch, H, W) features from deeper stage
            skip: (B, skip_ch, 2H, 2W) encoder skip connection, or None
        Returns:
            (B, out_ch, 2H, 2W) upsampled and refined features
        """
        # Upsample: (B, in_ch, H, W) → (B, out_ch, 2H, 2W)
        x = self.upsample(x)

        # Fuse with skip connection
        if self.has_skip and skip is not None:
            x = x + self.skip_proj(skip)

        # Refine
        x = self.refine(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Aperture-Aware Attention Bottleneck (from bbnet_aaa.py, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ApertureAttentionBlock(nn.Module):
    """
    Single Aperture-Aware Attention Block (AAB).

    Architecture (operates in BHWC format):
        x -> DWConv(pos) -> + x                         # local positional encoding
          -> LayerNorm -> AAA(x, rel_pos) -> + residual  # attention sub-block
          -> LayerNorm -> FFN(DWConv + GELU) -> + residual # feed-forward sub-block

    Tensor trace (dim=256, heads=4, ffn_dim=512, H=W=88):
        Input:  (B, 88, 88, 256)
        DWConv: (B, 88, 88, 256)   -- local positional encoding (LEPE-style)
        LN1:    (B, 88, 88, 256)
        AAA:    (B, 88, 88, 256)   -- row-col decomposed attention with aperture decay
        +res:   (B, 88, 88, 256)
        LN2:    (B, 88, 88, 256)
        FFN:    (B, 88, 88, 256)   -- fc1(256->512) -> GELU -> DWConv -> fc2(512->256)
        +res:   (B, 88, 88, 256)
        Output: (B, 88, 88, 256)

    Mathematical basis:
        Row attention:  softmax(Q_row · K_row^T / √d_k + mask_h) · V
        Col attention:  softmax(Q_col · K_col^T / √d_k + mask_w) · V
        where mask = -|i-j| · log(1 - 2^(-γ)) encodes aperture-conditioned
        spatial locality. γ is derived from f_stop via DynRelPos2d.
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
    Residual Group: N × AAB + auxiliary-conditioned conv + group-level skip.

    Handles BCHW <-> BHWC format conversions at group boundaries so the
    rest of the pipeline (encoder, decoder) stays in BCHW.

    Architecture:
        x_bchw -> permute(BHWC)
            -> AAB_1(x, rel_pos)
            -> AAB_2(x, rel_pos)
            -> AAB_3(x, rel_pos)
        -> permute(BCHW)
        -> cat(x, coord_map, kernel_map)  [if enabled]
        -> Conv2d(C+3 -> C, 3x3)
        -> + skip (group residual)

    Tensor trace (dim=256, 3 blocks, coord+kernel, 88×88):
        Input:  (B, 256, 88, 88)  BCHW
        permute: (B, 88, 88, 256) BHWC
        3× AAB: (B, 88, 88, 256) BHWC
        permute: (B, 256, 88, 88) BCHW
        cat:    (B, 259, 88, 88)  BCHW  -- +2 coord +1 kernel
        conv:   (B, 256, 88, 88)  BCHW
        +skip:  (B, 256, 88, 88)  BCHW
        Output: (B, 256, 88, 88)
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
            x = torch.cat([x] + aux, dim=1)                                # (B, C+3, H, W)

        x = self.conv(x)

        return x + skip


# ─────────────────────────────────────────────────────────────────────────────
# Utility modules
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)


class Interpolate(nn.Module):
    """Fixed-factor interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: BBNet-AAA-NAF
# ─────────────────────────────────────────────────────────────────────────────

class Inception_Encoder_Unet_Decoder(BaseModel):
    """
    BBNet-AAA-NAF: NAFBlock encoder/decoder + AAA bottleneck at 88×88.

    Inputs (passed as *inputs tuple):
        inputs[0]: source       (B, 3, H, W)  - RGB image
        inputs[1]: kernel_map   (B, 1, H, W)  - Circle of Confusion map
        inputs[2]: bloom_input  (B, 1, H, W)  - Specular highlight map
        inputs[3]: coord_maps   (B, 2, H, W)  - Normalized spatial coordinates
        inputs[4]: f_stop       (B,)          - Aperture value

    Injection strategy:
        RGB Stem:       source(3) → 32ch @ 704
        Guidance Stem:  kernel_map(1) + bloom(1) + coords(2) → 32ch @ 704
        SAFM Fusion:    rgb_feat ⊙ SAFM(guidance_feat) → 32ch @ 704
        Encoder:        NAFBlock stages → 88×88
        Bottleneck:     f_stop → DynRelPos2d; coord_maps(2) + kernel_map(1) at RG conv
        Output head:    bloom_input(1) late injection at full resolution

    Full pipeline tensor trace (B=1, H=W=1408):
        === Input Stems ===
        source:         (1, 3, 1408, 1408)
        guidance:       cat(kernel_map, bloom_input, coord_maps) = (1, 4, 1408, 1408)
        RGB stem:       (1, 3, 1408) → Conv(3→32, s2) → Conv(32→32) → (1, 32, 704, 704)
        Guidance stem:  (1, 4, 1408) → Conv(4→32, s2) → Conv(32→32) → (1, 32, 704, 704)

        === SAFM Fusion ===
        fused = rgb_feat ⊙ SAFM(guidance_feat): (1, 32, 704, 704)

        === NAFBlock Encoder ===
        Stem: Conv(32→32, s2) → (1, 32, 352, 352)
        Stage 0: 2×NAFBlock(32)  → (1, 32, 352, 352)    skip₀
        Down:    Conv(32→128, k2, s2) → (1, 128, 176, 176)
        Stage 1: 4×NAFBlock(128) → (1, 128, 176, 176)   skip₁
        Down:    Conv(128→256, k2, s2) → (1, 256, 88, 88)
        Stage 2: 3×NAFBlock(256) → (1, 256, 88, 88)     → bottleneck

        === AAA Bottleneck @ 88×88 ===
        DynRelPos2d(f_stop) → (mask_h, mask_w)
        coord_map_bn: interpolate(coord_maps, 88×88) → (1, 2, 88, 88)
        kernel_map_bn: interpolate(kernel_map, 88×88) → (1, 1, 88, 88)
        RG 1: 3×AAB(256, heads=4) + coord+kernel conv → (1, 256, 88, 88)
        RG 2: 3×AAB(256, heads=4) + coord+kernel conv → (1, 256, 88, 88)

        === NAFBlock Decoder ===
        Dec 2: upsample(256→128×4→PS) + skip₁@176 → NAFBlock → (1, 128, 176, 176)
        Dec 1: upsample(128→128×4→PS) + skip₀@352 → NAFBlock → (1, 128, 352, 352)
        Dec 0: upsample(128→128×4→PS) → NAFBlock           → (1, 128, 704, 704)

        === Output Head ===
        head1: Conv(128→64) → ReLU → Conv(64→128) → PixelShuffle(2) → (1, 32, 1408, 1408)
        head2: Conv(3→32, k3) on source                               → (1, 32, 1408, 1408)
        cat(head1, head2, bloom): (1, 65, 1408, 1408)
        head: Conv(65→32) → ReLU → Conv(32→3, k1) → (1, 3, 1408, 1408)
    """

    def __init__(
            self,
            out_channels=3,
            # Stem
            stem_channels=32,
            # Encoder
            encoder_channels=(32, 128, 256),
            encoder_blocks=(2, 4, 3),
            # AAA bottleneck
            bottleneck_dim=256,
            bottleneck_heads=4,
            bottleneck_ffn_ratio=2.,
            num_residual_groups=2,
            blocks_per_group=3,
            use_coord_conv=True,
            use_kernel_conv=True,
            drop_path_rate=0.05,
            # DynRelPos2d
            relpos_init_value=2,
            relpos_heads_range=6,
            # Decoder
            decoder_channels=(128, 128, 128),
            # Output
            non_negative=False,
            # Legacy params for API compatibility
            layers=None, encoder_start_filts=None, encoder_in_channels=None,
            width_mult=None, decoder_channels_legacy=None, features=None,
            use_bn=False, enable_attention_hooks=False, scale_factor=2,
    ):
        super(Inception_Encoder_Unet_Decoder, self).__init__()

        # ── Input Stems ──
        self.rgb_stem = InputStem(in_ch=3, out_ch=stem_channels)            # source RGB → 32ch
        self.guidance_stem = InputStem(in_ch=4, out_ch=stem_channels)       # kernel+bloom+coords → 32ch
        self.safm = SAFM(dim=stem_channels, scales=(1, 2, 4))

        # ── NAFBlock Encoder ──
        self.encoder = NAFEncoder(
            in_ch=stem_channels,
            channels=encoder_channels,
            blocks=encoder_blocks,
        )

        # ── AAA Bottleneck ──
        self.relpos = DynRelPos2d(
            embed_dim=bottleneck_dim,
            num_heads=bottleneck_heads,
            initial_value=relpos_init_value,
            heads_range=relpos_heads_range,
        )

        # Stochastic depth: linearly increasing drop path across all blocks
        total_blocks = num_residual_groups * blocks_per_group
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

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

        # ── NAFBlock Decoder ──
        # Dec 2: bottleneck 256@88 + skip₁ 128@176 → 128@176
        self.dec2 = NAFDecoderBlock(in_ch=encoder_channels[2], out_ch=decoder_channels[0],
                                    skip_ch=encoder_channels[1])
        # Dec 1: 128@176 + skip₀ 32@352 → 128@352
        self.dec1 = NAFDecoderBlock(in_ch=decoder_channels[0], out_ch=decoder_channels[1],
                                    skip_ch=encoder_channels[0])
        # Dec 0: 128@352 → 128@704 (no skip)
        self.dec0 = NAFDecoderBlock(in_ch=decoder_channels[1], out_ch=decoder_channels[2],
                                    skip_ch=None)

        # ── Output Head ──
        # head1: decoder output at 704 → PixelShuffle to 1408
        self.head1 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2] // 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(decoder_channels[2] // 2, 32 * 4, 3, 1, 1),
            nn.PixelShuffle(2),                                              # (B, 32, 1408, 1408)
            nn.ReLU(True),
        )

        # head2: HR branch processes source RGB only (3ch)
        self.head2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
        )

        # head: +1 bloom channel for late specular highlight injection
        self.head = nn.Sequential(
            nn.Conv2d(64 + 1, 32, 3, 1, 1),                                 # 32+32+1 bloom
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, 1, 1, 0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, *inputs):
        """
        Forward pass.

        Args:
            inputs[0]: source       (B, 3, H, W)  - RGB image
            inputs[1]: kernel_map   (B, 1, H, W)  - Circle of Confusion (CoC diameter)
            inputs[2]: bloom_input  (B, 1, H, W)  - Specular highlight map
            inputs[3]: coord_maps   (B, 2, H, W)  - Normalized spatial coordinates
            inputs[4]: f_stop       (B,)          - Aperture value for attention decay

        Returns:
            (B, 3, H, W) output bokeh-rendered image
        """
        source, kernel_map, bloom_input, coord_maps, f_stop = inputs

        # ── Input Stems ──
        # RGB pathway
        rgb_feat = self.rgb_stem(source)                                    # (B, 32, H/2, W/2)
        # Guidance pathway: kernel_map + bloom + coords = 4ch
        guidance_input = torch.cat([kernel_map, bloom_input, coord_maps], dim=1)  # (B, 4, H, W)
        guide_feat = self.guidance_stem(guidance_input)                     # (B, 32, H/2, W/2)

        # ── SAFM Fusion ──
        # Guidance modulates RGB via multi-scale spatial attention
        fused = self.safm(rgb_feat, guide_feat)                             # (B, 32, H/2, W/2)

        # ── NAFBlock Encoder ──
        # Returns features at [352, 176, 88] for 1408 input
        enc_features = self.encoder(fused)
        # enc_features[0]: (B, 32, H/4, W/4)   — skip₀ @ 352
        # enc_features[1]: (B, 128, H/8, W/8)  — skip₁ @ 176
        # enc_features[2]: (B, 256, H/16, W/16) — bottleneck @ 88

        # ── AAA Bottleneck @ 88×88 ──
        bottleneck_feat = enc_features[2]
        B, C, H_bn, W_bn = bottleneck_feat.shape                           # (B, 256, 88, 88)

        # Generate aperture-aware positional decay masks from f_stop
        rel_pos = self.relpos((H_bn, W_bn), range_factor=f_stop)

        # Downsample auxiliary maps to bottleneck resolution
        coord_map_bn = F.interpolate(coord_maps, size=(H_bn, W_bn),
                                      mode='bilinear', align_corners=False)
        kernel_map_bn = F.interpolate(kernel_map, size=(H_bn, W_bn),
                                       mode='bilinear', align_corners=False)

        # Pass through 2 Residual Groups (coord + kernel conditioning at each)
        for rg in self.residual_groups:
            bottleneck_feat = rg(bottleneck_feat, rel_pos, coord_map_bn, kernel_map_bn)

        # ── NAFBlock Decoder ──
        # Dec 2: 256@88 + skip₁@176 → 128@176
        dec2_out = self.dec2(bottleneck_feat, skip=enc_features[1])
        # Dec 1: 128@176 + skip₀@352 → 128@352
        dec1_out = self.dec1(dec2_out, skip=enc_features[0])
        # Dec 0: 128@352 → 128@704
        dec0_out = self.dec0(dec1_out)

        # ── Output Heads ──
        out1 = self.head1(dec0_out)                                         # (B, 32, H, W)
        hr = self.head2(source)                                             # (B, 32, H, W)

        # Late bloom injection: cat decoder + HR + bloom → head
        out = torch.cat((out1, hr, bloom_input), dim=1)                     # (B, 65, H, W)
        out = self.head(out)                                                # (B, 3, H, W)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Inception_Encoder_Unet_Decoder()

    # ── Parameter count ──
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    bottleneck_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'residual_groups' in name or 'relpos' in name
    )
    decoder_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith('dec')
    )
    stem_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'stem' in name or 'safm' in name
    )

    print(f"{'Component':<25} {'Parameters':>12}")
    print(f"{'-' * 25} {'-' * 12}")
    print(f"{'Input stems + SAFM':<25} {stem_params:>12,}")
    print(f"{'NAF Encoder':<25} {encoder_params:>12,}")
    print(f"{'AAA Bottleneck':<25} {bottleneck_params:>12,}")
    print(f"{'NAF Decoder':<25} {decoder_params:>12,}")
    print(f"{'Other (heads etc.)':<25} {total_params - stem_params - encoder_params - bottleneck_params - decoder_params:>12,}")
    print(f"{'-' * 25} {'-' * 12}")
    print(f"{'TOTAL':<25} {total_params:>12,}")

    # ── Forward pass test ──
    H, W = 1408, 1408
    source      = torch.rand((1, 3, H, W))
    kernel_map  = torch.rand((1, 1, H, W))
    bloom_input = torch.rand((1, 1, H, W))
    coord_maps  = torch.rand((1, 2, H, W))
    f_stop      = torch.tensor([2.0])

    print(f"\n{'Input':<15} {'Shape':>20}")
    print(f"{'-' * 15} {'-' * 20}")
    print(f"{'source':<15} {str(source.shape):>20}")
    print(f"{'kernel_map':<15} {str(kernel_map.shape):>20}")
    print(f"{'bloom_input':<15} {str(bloom_input.shape):>20}")
    print(f"{'coord_maps':<15} {str(coord_maps.shape):>20}")
    print(f"{'f_stop':<15} {str(f_stop.shape):>20}")

    with torch.no_grad():
        output = model(source, kernel_map, bloom_input, coord_maps, f_stop)

    print(f"\n{'output':<15} {str(output.shape):>20}")
    assert output.shape == (1, 3, H, W), \
        f"Shape mismatch! Expected (1, 3, {H}, {W}), got {output.shape}"
    print("\n[OK] Forward pass successful!")
