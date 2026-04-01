"""
bbnet_improved.py — BBNet with NAFBlock encoder, FFT bottleneck, PixelShuffle decoder, SAFM.

Architecture changes vs bbnet.py:
  1. NAFBlock encoder (replaces ResNet_D) — ~2.5x fewer params/block via DW conv + SimpleGate
  2. FourierBlock bottleneck at 44x44 (replaces 12x GSA at 22x22) — ~30x cheaper
  3. PixelShuffle upsampling (replaces bilinear) — learned upsampling for sharper edges
  4. SAFM multi-scale modulation in decoder — spatial awareness for varying blur disks
  5. GuidanceEncoder + FiLM DISCARDED — redundant with 6-channel input conditioning

Resolution ladder:
  Input [B,6,1408] → x0.5 → [B,6,704] → stem stride-2 → [B,64,352]
  Stage0: [B,64,352] → Stage1: [B,128,176] → Stage2: [B,256,88] → Stage3: [B,256,44]
  FourierBlock bottleneck at 44x44
  Decoder: 44→88→176→352 → head x4 → 1408, fuse with HR → RGB
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# NAFBlock Encoder
# ============================================================================

class NAFBlock(nn.Module):
    """NAFNet block: LayerNorm → DWConv → SimpleGate → SCA → residual
                     LayerNorm → Conv1x1 (expand) → SimpleGate → Conv1x1 → residual

    Uses GroupNorm(1, c) as LayerNorm equivalent for spatial feature maps.
    SimpleGate: chunk channels in half and multiply elementwise — no learned params,
    replaces ReLU/GELU while preserving gradient flow through the multiplicative path.

    Math:
        Spatial path:  y = x + Conv1x1(SCA(SimpleGate(DWConv(Conv1x1(LN(x))))))
        FFN path:      y = x + Conv1x1(SimpleGate(DWConv(Conv1x1(LN(x)))))

    The SCA (Simplified Channel Attention) is:
        SCA(z) = z ⊙ Conv1x1(GAP(z))
    where GAP is global average pooling. This is a lightweight SE-like mechanism.
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = c * dw_expand

        # --- Spatial mixing path ---
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)               # Pointwise expand
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)  # DW spatial
        # After SimpleGate: dw_ch → dw_ch // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv2 = nn.Conv2d(dw_ch // 2, c, 1)          # Pointwise project back

        # --- Channel mixing path (FFN) ---
        self.norm2 = nn.GroupNorm(1, c)
        ffn_ch = c * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(c, ffn_ch, 1),
            nn.Conv2d(ffn_ch, ffn_ch, 3, 1, 1, groups=ffn_ch),  # DW for local mixing
        )
        # After SimpleGate: ffn_ch → ffn_ch // 2
        self.ffn_out = nn.Conv2d(ffn_ch // 2, c, 1)

    def forward(self, x):
        # --- Spatial mixing ---
        # x: [B, C, H, W]
        y = self.norm1(x)
        y = self.conv1(y)                   # [B, C, H, W] → [B, 2C, H, W]
        y = self.dw_conv(y)                 # [B, 2C, H, W] → [B, 2C, H, W] (spatial mixing)
        y1, y2 = y.chunk(2, dim=1)          # SimpleGate: 2 x [B, C, H, W]
        y = y1 * y2                         # [B, C, H, W] — gated activation
        y = y * self.sca(y)                 # [B, C, H, W] — channel recalibration
        y = self.conv2(y)                   # [B, C, H, W] → [B, C, H, W]
        x = x + y                           # Residual

        # --- Channel mixing (FFN) ---
        y = self.norm2(x)
        y = self.ffn(y)                     # [B, C, H, W] → [B, 2C, H, W]
        y1, y2 = y.chunk(2, dim=1)          # SimpleGate
        y = y1 * y2                         # [B, C, H, W]
        y = self.ffn_out(y)                 # [B, C, H, W]
        x = x + y                           # Residual
        return x


class NAFEncoder(nn.Module):
    """NAFBlock-based multi-scale encoder.

    Produces features at 3 skip-connection scales + bottleneck:
        Stage 0: 352x352, 64ch  (no skip — too large for decoder)
        Stage 1: 176x176, 128ch → skip 0
        Stage 2: 88x88,   256ch → skip 1
        Stage 3: 44x44,   256ch → bottleneck input (goes to FourierBlock)

    Downsampling uses stride-2 convolutions (learned, not pooling).
    """

    def __init__(self, in_ch=6, channels=(64, 128, 256, 256), blocks=(2, 4, 4, 4)):
        super().__init__()
        # Stem: 704→352 with stride-2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, stride=2, padding=1),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1),
        )

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels)):
            ch = channels[i]
            prev_ch = channels[i - 1] if i > 0 else channels[0]

            if i > 0:
                self.downs.append(nn.Conv2d(prev_ch, ch, 2, 2))  # Strided conv downsample

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
        Input:  [B, 6, 704, 704]
        Output: list of 4 feature tensors at decreasing resolutions
                [B,64,352,352], [B,128,176,176], [B,256,88,88], [B,256,44,44]
        """
        x = self.stem(x)                   # [B, 6, 704, 704] → [B, 64, 352, 352]
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downs[i - 1](x)   # Stride-2 downsample
            x = stage(x)                    # NAFBlock processing
            features.append(x)
        return features


# ============================================================================
# FourierBlock Bottleneck
# ============================================================================

class FourierBlock(nn.Module):
    """FFT-based global modeling block (inspired by PW-FNet).

    Replaces self-attention with frequency-domain processing:
        1. Apply rfft2 to get spectral representation
        2. Apply learned 1x1 convolutions to real and imaginary parts independently
        3. Apply irfft2 to return to spatial domain
        4. SimpleGate + output projection

    Complexity: O(N log N) per channel vs O(N²) for attention.
    At 44x44: FFT ≈ 21K ops/channel vs attention ≈ 3.7M ops/channel.

    Math:
        X_freq = FFT2(Conv1x1(LN(x)))
        X_freq = GELU(Conv1x1(Re(X_freq))) + j·GELU(Conv1x1(Im(X_freq)))
        y = Conv1x1(SimpleGate(IFFT2(X_freq)))
        output = x + y
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv_in = nn.Conv2d(dim, dim * 2, 1)
        self.freq_conv = nn.Conv2d(dim * 2, dim * 2, 1)
        self.freq_act = nn.GELU()
        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Input:  [B, C, H, W]  (e.g. [B, 256, 44, 44])
        Output: [B, C, H, W]  (same shape, residual connection)
        """
        residual = x
        x = self.norm(x)
        x = self.conv_in(x)                # [B, C, H, W] → [B, 2C, H, W]

        # --- FFT global modeling ---
        # rfft2 output: [B, 2C, H, W//2+1] (complex)
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # Apply learned 1x1 conv to real and imaginary parts independently.
        # Same weights for both — treats the spectral representation symmetrically.
        x_freq = self.freq_conv(x_freq.real) + 1j * self.freq_conv(x_freq.imag)
        x_freq = self.freq_act(x_freq.real) + 1j * self.freq_act(x_freq.imag)

        # Back to spatial domain
        x = torch.fft.irfft2(x_freq, norm='ortho', s=residual.shape[-2:])
        # x: [B, 2C, H, W]

        # --- SimpleGate ---
        x1, x2 = x.chunk(2, dim=1)         # 2 x [B, C, H, W]
        x = x1 * x2                         # [B, C, H, W]

        x = self.conv_out(x)               # [B, C, H, W]
        return x + residual


# ============================================================================
# SAFM — Spatially-Adaptive Feature Modulation
# ============================================================================

class SAFM(nn.Module):
    """Multi-scale spatial modulation for decoder features (from SAFMN).

    Captures information at multiple spatial scales via pool→project→upsample,
    then modulates the input features multiplicatively. This gives the decoder
    spatial awareness of varying-size bokeh regions.

    Reduced to 3 scales [1, 2, 4] (vs original 4) to respect the "not KPI heavy" constraint.
    """

    def __init__(self, dim, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for _ in scales[1:]  # Skip scale=1
        ])
        self.fuse = nn.Conv2d(dim * len(scales), dim, 1)

    def forward(self, x):
        """
        Input:  [B, C, H, W]
        Output: [B, C, H, W] (multiplicatively modulated)
        """
        B, C, H, W = x.shape
        multi = [x]                         # Scale 1: identity
        for i, s in enumerate(self.scales[1:]):
            pooled = F.adaptive_avg_pool2d(x, (max(H // s, 1), max(W // s, 1)))
            projected = self.convs[i](pooled)
            upsampled = F.interpolate(projected, size=(H, W), mode='bilinear', align_corners=False)
            multi.append(upsampled)
        fused = self.fuse(torch.cat(multi, dim=1))  # [B, C*num_scales, H, W] → [B, C, H, W]
        return x * fused                    # Spatial modulation (element-wise)


# ============================================================================
# Decoder Components
# ============================================================================

class ResidualConvUnit(nn.Module):
    """Residual convolution module — two 3x3 convs with activation and skip."""

    def __init__(self, features, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block with PixelShuffle upsampling and SAFM modulation.

    Pipeline:
        1. SAFM spatial modulation on input features
        2. Fuse with skip connection (if provided) via ResidualConvUnit
        3. ResidualConvUnit refinement
        4. PixelShuffle x2 upsampling (Conv2d expand 4x then shuffle)
        5. 1x1 output projection
    """

    def __init__(self, features, out_features, activation, safm_scales=(1, 2, 4)):
        super().__init__()
        self.safm = SAFM(features, scales=safm_scales)
        self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True)

        # PixelShuffle upsampling: expand channels 4x, then rearrange to spatial x2
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 4, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, *xs):
        """
        Args:
            xs[0]: features from previous (deeper) decoder stage [B, C, H, W]
            xs[1]: skip connection from encoder (optional) [B, C, H, W]
        Returns:
            [B, out_features, 2H, 2W]
        """
        output = xs[0]

        # SAFM modulation — multi-scale spatial awareness before fusion
        output = self.safm(output)

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        # PixelShuffle x2 (learned upsampling)
        output = self.upsample(output)      # [B, C, H, W] → [B, C, 2H, 2W]

        output = self.out_conv(output)      # [B, C, 2H, 2W] → [B, out_C, 2H, 2W]
        return output


def _make_fusion_block(features, out_features):
    return FeatureFusionBlock(
        features, out_features,
        activation=nn.ReLU(False),
        safm_scales=(1, 2, 4),
    )


# ============================================================================
# Utility Modules
# ============================================================================

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


# ============================================================================
# Scratch — Channel Projection Layers
# ============================================================================

class Scratch(nn.Module):
    """3x3 conv channel projections for encoder features → decoder feature space."""

    def __init__(self, in_shape, out_shape):
        super().__init__()
        # Skip connections: encoder stages 1, 2 → decoder
        self.layer1_rn = nn.Conv2d(in_shape[0], out_shape[0], 3, 1, 1, bias=False)
        self.layer2_rn = nn.Conv2d(in_shape[1], out_shape[1], 3, 1, 1, bias=False)
        # Bottleneck projection (encoder stage 3 → FourierBlock input)
        self.bottleneck_rn = nn.Conv2d(in_shape[2], out_shape[2], 3, 1, 1, bias=False)


# ============================================================================
# Main Model
# ============================================================================

class Inception_Encoder_Unet_Decoder(BaseModel):
    """BBNet Improved — drop-in replacement for the original BBNet.

    Architecture:
        1. x0.5 downsample: 1408→704
        2. NAFEncoder: 704→{352, 176, 88, 44} multi-scale features
        3. FourierBlock bottleneck at 44x44 (global frequency modeling)
        4. Decoder with PixelShuffle + SAFM: 44→88→176→352
        5. Head: 352→1408 (x4 upsample), fuse with full-res input → RGB output

    Input:  [B, 6, 1408, 1408] — (RGB + depth + focus_distance + alpha)
    Output: [B, 3, 1408, 1408] — rendered bokeh RGB
    """

    def __init__(
            self,
            encoder_in_channels=6,
            out_channels=3,
            encoder_channels=(64, 128, 256, 256),
            encoder_blocks=(2, 4, 4, 4),
            num_fourier_blocks=3,
            # Decoder features: bottleneck, stage2, stage1
            decoder_features=(256, 256, 128),
            non_negative=False,
            # Legacy params kept for API compatibility (unused)
            layers=None, encoder_start_filts=None, width_mult=None,
            decoder_channels=None, features=None, use_bn=False,
            enable_attention_hooks=False, scale_factor=2,
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = NAFEncoder(
            in_ch=encoder_in_channels,
            channels=encoder_channels,
            blocks=encoder_blocks,
        )

        # --- Channel projections (encoder → decoder feature space) ---
        # Encoder stages 1,2,3 (skip at 176, 88; bottleneck at 44)
        # Stage 0 at 352 is not used as skip (spatial cost too high)
        self.scratch = Scratch(
            in_shape=[encoder_channels[1], encoder_channels[2], encoder_channels[3]],
            out_shape=[decoder_features[2], decoder_features[1], decoder_features[0]],
        )

        # --- FourierBlock bottleneck ---
        self.bottleneck = nn.Sequential(
            *[FourierBlock(dim=decoder_features[0]) for _ in range(num_fourier_blocks)]
        )

        # --- Decoder (3 stages: 44→88→176→352) ---
        # refinenet3: bottleneck 256@44 → 256@88, no skip (bottleneck is the deepest)
        self.scratch.refinenet3 = _make_fusion_block(decoder_features[0], decoder_features[1])
        # refinenet2: 256@88 → fuse with skip@88 → 128@176
        self.scratch.refinenet2 = _make_fusion_block(decoder_features[1], decoder_features[2])
        # refinenet1: 128@176 → fuse with skip@176 → 128@352
        self.scratch.refinenet1 = _make_fusion_block(decoder_features[2], decoder_features[2])

        # --- Output head ---
        # head1: path_1 at 352x352 → x4 upsample → 1408x1408
        head1 = nn.Sequential(
            nn.Conv2d(decoder_features[2], decoder_features[2] // 2, 3, 1, 1),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_features[2] // 2, 32, 3, 1, 1),
            nn.ReLU(True),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReLU(True),
        )

        # head2: full-res input → shallow feature extraction
        head2 = nn.Sequential(
            nn.Conv2d(encoder_in_channels, 32, 3, 1, 1),
            nn.ReLU(True),
        )

        # head: fuse LR decoded + HR input → final RGB
        head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, 1, 1, 0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        self.scratch.output_conv = head
        self.scratch.output_conv1 = head1
        self.scratch.output_conv2 = head2

        self.downsample = Interpolate(scale_factor=0.5, mode="bilinear", align_corners=False)

    def forward(self, x):
        """
        Input:  [B, 6, 1408, 1408]
        Output: [B, 3, 1408, 1408]
        """
        HR = x                              # [B, 6, 1408, 1408]
        LR = self.downsample(x)             # [B, 6, 704, 704]

        # --- Encoder ---
        # features: [B,64,352], [B,128,176], [B,256,88], [B,256,44]
        enc_features = self.encoder(LR)

        # --- Channel projections (skip connections) ---
        skip_176 = self.scratch.layer1_rn(enc_features[1])      # [B, 128, 176, 176]
        skip_88 = self.scratch.layer2_rn(enc_features[2])       # [B, 256, 88, 88]

        # --- Bottleneck ---
        bottleneck = self.scratch.bottleneck_rn(enc_features[3]) # [B, 256, 44, 44]
        bottleneck = self.bottleneck(bottleneck)                  # 3x FourierBlock

        # --- Decoder (with PixelShuffle + SAFM) ---
        path_3 = self.scratch.refinenet3(bottleneck)             # 256@44 → 256@88
        path_2 = self.scratch.refinenet2(path_3, skip_88)        # 256@88 + skip → 128@176
        path_1 = self.scratch.refinenet1(path_2, skip_176)       # 128@176 + skip → 128@352

        # --- Output head ---
        out1 = self.scratch.output_conv1(path_1)                 # [B, 32, 1408, 1408]
        HR = self.scratch.output_conv2(HR)                       # [B, 32, 1408, 1408]

        out = torch.cat((out1, HR), dim=1)                       # [B, 64, 1408, 1408]
        out = self.scratch.output_conv(out)                      # [B, 3, 1408, 1408]
        return out


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Inception_Encoder_Unet_Decoder()

    # --- Parameter count ---
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    bottleneck_params = sum(p.numel() for p in model.bottleneck.parameters())

    print(f"Total params:      {total_params:,}")
    print(f"Encoder params:    {encoder_params:,}")
    print(f"Bottleneck params: {bottleneck_params:,}")

    # --- Forward pass test ---
    model = model.to(device)
    # Use smaller input for quick test if no GPU
    test_size = 1408 if torch.cuda.is_available() else 352
    input_tensor = torch.rand((1, 6, test_size, test_size), device=device)
    print(f"\nInput shape:  {input_tensor.shape}")

    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(f"Output shape: {output_tensor.shape}")
    assert output_tensor.shape == (1, 3, test_size, test_size), \
        f"Shape mismatch! Expected [1, 3, {test_size}, {test_size}], got {output_tensor.shape}"
    print("\n✓ Forward pass successful!")
