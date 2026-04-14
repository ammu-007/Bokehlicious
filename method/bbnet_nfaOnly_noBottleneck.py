"""
BBNet-NFA-Only (No Bottleneck): Concat-first NAFBlock Encoder-Decoder for Bokeh Rendering

Architecture changes vs bbnet_bfa_no_bottleneck.py:
  - Input fusion:  REMOVED separate RGB/guidance stems + SAFM modulation.
  - ConcatStem:    ADDED single stride-2 conv on all inputs concatenated (7ch → 32ch @ 704×704).
                   source(3) ‖ kernel_map(1) ‖ bloom_input(1) ‖ coord_maps(2) = 7ch total.
  - Encoder:       Same NAFBlock encoder (channels=(32,128,256), blocks=(2,4,3)).
  - Bottleneck:    Same NFABottleneck — N×NAFBlock(256) at 88×88 (pure conv, no attention).
  - Decoder:       Same NAFBlock + PixelShuffle decoder.
  - Removed:       f_stop, DynRelPos2d, ApertureAwareAttention, ResidualGroup, DropPath, SAFM.
  - Forward sig:   model(source, kernel_map, bloom_input, coord_maps)
                   #      (B,3)    (B,1)       (B,1)       (B,2)
                   NOTE: f_stop is still accepted (5th arg) but silently ignored.

Resolution ladder (1408×1408 input):
  concat(source,kernel,bloom,coords)[7ch] → ConcatStem(stride-2) → 704×704 @ 32ch
  NAFEncoder:
    704 →(s2)→ 352 → Stage0(2×NAF, 32ch)  → skip₀
    352 →(s2)→ 176 → Stage1(4×NAF, 128ch) → skip₁
    176 →(s2)→  88 → Stage2(3×NAF, 256ch) → bottleneck_in
  NFABottleneck (no attention):
     88 → 2×NAFBlock(256) → bottleneck_out
  Decoder:
     88 →(PS×2)→ 176 + skip₁ → NAFBlock(128) → 176
    176 →(PS×2)→ 352 + skip₀ → NAFBlock(128) → 352
    352 →(PS×2)→ 704          → NAFBlock(128) → 704
  Head:
    704 →(PS×2)→ 1408 @ 32ch  + source(HR) @ 32ch + bloom(1ch)
    → Conv → 3ch output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# NAFBlock: SimpleGate + SCA + GroupNorm residual block
# ─────────────────────────────────────────────────────────────────────────────

class NAFBlock(nn.Module):
    """NAFNet block: two-path residual with SimpleGate activation and SCA.

    Math:
        Spatial: LN → Conv1×1(C→2C) → DWConv3×3 → SimpleGate → SCA → Conv1×1(C→C) → +res
        FFN:     LN → Conv1×1(C→2C) → DWConv3×3 → SimpleGate → Conv1×1(C→C) → +res

    SimpleGate: f(x) = x₁ ⊙ x₂  where cat(x₁,x₂) = x  (no learnable params)
    SCA: y = y ⊙ GAP(y) ← recalibrates channel importance via global context.

    Tensor trace (c=256, H=W=88):
        Input:   (B, 256, 88, 88)
        Spatial: GN → Conv1×1(256→512) → DW3×3 → chunk → ⊙ → GAP⊙ → Conv1×1(256→256) → +
        FFN:     GN → Conv1×1(256→512) → DW3×3 → chunk → ⊙ → Conv1×1(256→256) → +
        Output:  (B, 256, 88, 88)
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = c * dw_expand

        # Spatial mixing path
        self.norm1   = nn.GroupNorm(1, c)
        self.conv1   = nn.Conv2d(c, dw_ch, 1)
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)
        self.sca     = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(dw_ch // 2, dw_ch // 2, 1))
        self.conv2   = nn.Conv2d(dw_ch // 2, c, 1)

        # Channel mixing (FFN) path
        self.norm2   = nn.GroupNorm(1, c)
        ffn_ch = c * ffn_expand
        self.ffn     = nn.Sequential(nn.Conv2d(c, ffn_ch, 1),
                                     nn.Conv2d(ffn_ch, ffn_ch, 3, 1, 1, groups=ffn_ch))
        self.ffn_out = nn.Conv2d(ffn_ch // 2, c, 1)

    def forward(self, x):
        # Spatial mixing
        y = self.conv1(self.norm1(x))
        y = self.dw_conv(y)
        y1, y2 = y.chunk(2, dim=1)
        y = y1 * y2                    # SimpleGate
        y = y * self.sca(y)            # SCA recalibration
        x = x + self.conv2(y)

        # FFN
        y = self.ffn(self.norm2(x))
        y1, y2 = y.chunk(2, dim=1)
        x = x + self.ffn_out(y1 * y2)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ConcatStem — Single stride-2 stem (replaces dual RGB+Guidance stems + SAFM)
# ─────────────────────────────────────────────────────────────────────────────

class ConcatStem(nn.Module):
    """Concatenation-first input stem (bbnet.py style).

    All input signals are concatenated along the channel dimension and processed
    by a single stride-2 convolutional stem. Eliminates the dual RGB/guidance
    pathways and SAFM multi-scale fusion block.

    Input channels:
        source(3) ‖ kernel_map(1) ‖ bloom_input(1) ‖ coord_maps(2) = 7ch total

    Design rationale:
        The first Conv2d layer is given direct access to all input modalities
        simultaneously. The network learns cross-modal feature routing implicitly
        during training, without the prior of separate feature streams or
        multiplicative guidance modulation.

    Tensor trace (in_ch=7, out_ch=32, H=1408):
        Input:   (B, 7, 1408, 1408)
        Conv(7→32, k3, s2, p1)  → (B, 32, 704, 704)   — stride-2 downsample + project
        GN(1,32) → GELU         → (B, 32, 704, 704)   — normalise + activate
        Conv(32→32, k3, s1, p1) → (B, 32, 704, 704)   — feature refinement
        Output:  (B, 32, 704, 704)
    """

    def __init__(self, in_ch=7, out_ch=32):
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
# NAFEncoder
# ─────────────────────────────────────────────────────────────────────────────

class NAFEncoder(nn.Module):
    """3-stage NAFBlock encoder. Produces multi-scale skip features.

    Tensor trace (in_ch=32, channels=(32,128,256), blocks=(2,4,3)):
        Input:   (B, 32, 704, 704)   — concat-stem output
        Stem:    Conv(32→32, s2)     → (B,  32, 352, 352)
        Stage0:  2×NAFBlock(32)      → (B,  32, 352, 352)   skip₀
        Down0:   Conv(32→128, k2,s2) → (B, 128, 176, 176)
        Stage1:  4×NAFBlock(128)     → (B, 128, 176, 176)   skip₁
        Down1:   Conv(128→256,k2,s2) → (B, 256,  88,  88)
        Stage2:  3×NAFBlock(256)     → (B, 256,  88,  88)   → bottleneck
    """

    def __init__(self, in_ch=32, channels=(32, 128, 256), blocks=(2, 4, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, stride=2, padding=1, bias=False),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False),
        )
        self.stages = nn.ModuleList()
        self.downs  = nn.ModuleList()
        for i, ch in enumerate(channels):
            if i > 0:
                self.downs.append(nn.Conv2d(channels[i - 1], ch, 2, 2))
            self.stages.append(nn.Sequential(*[NAFBlock(ch) for _ in range(blocks[i])]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns list of skip features: [(B,32,H/4), (B,128,H/8), (B,256,H/16)]"""
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downs[i - 1](x)
            x = stage(x)
            features.append(x)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# NFABottleneck — NAFBlock stack, NO attention
# ─────────────────────────────────────────────────────────────────────────────

class NFABottleneck(nn.Module):
    """Pure convolutional bottleneck using N×NAFBlock at 88×88.

    Replaces the Aperture-Aware Attention (AAA) module entirely.
    Operates at the deepest encoder resolution (88×88 for 1408 input).

    No f_stop conditioning, no relative positional bias, no MHSA.
    SimpleGate + SCA in each NAFBlock provide content-adaptive mixing.

    Tensor trace (dim=256, num_blocks=2, H=W=88):
        Input:  (B, 256, 88, 88)
        Block1: NAFBlock(256) → (B, 256, 88, 88)
        Block2: NAFBlock(256) → (B, 256, 88, 88)
        Output: (B, 256, 88, 88)
    """

    def __init__(self, dim=256, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


# ─────────────────────────────────────────────────────────────────────────────
# NAFDecoderBlock — PixelShuffle ×2 + skip + NAFBlock refine
# ─────────────────────────────────────────────────────────────────────────────

class NAFDecoderBlock(nn.Module):
    """Single decoder stage: upsample → fuse skip → refine.

    Pipeline:
        x: (B, in_ch, H, W)
        Conv1×1(in_ch → out_ch * 4) + PixelShuffle(2): (B, out_ch, 2H, 2W)
        + skip_proj(skip):                              (B, out_ch, 2H, 2W)
        NAFBlock(out_ch):                               (B, out_ch, 2H, 2W)

    Tensor trace (in=256, out=128, skip=128, H=88):
        upsample: Conv1×1(256→512) → PS(2) → (B, 128, 176, 176)
        skip:     Conv1×1(128→128) → +      → (B, 128, 176, 176)
        refine:   NAFBlock(128)             → (B, 128, 176, 176)
    """

    def __init__(self, in_ch, out_ch, skip_ch=None):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1, bias=True),
            nn.PixelShuffle(2),
        )
        self.has_skip = skip_ch is not None
        if self.has_skip:
            self.skip_proj = (nn.Conv2d(skip_ch, out_ch, 1, bias=False)
                              if skip_ch != out_ch else nn.Identity())
        self.refine = NAFBlock(out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.has_skip and skip is not None:
            x = x + self.skip_proj(skip)
        return self.refine(x)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    def load(self, path):
        params = torch.load(path, map_location='cpu')
        if 'optimizer' in params:
            params = params['model']
        self.load_state_dict(params)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model: BBNet-NFA-Only (No Bottleneck, Concat-first)
# ─────────────────────────────────────────────────────────────────────────────

class Inception_Encoder_Unet_Decoder(BaseModel):
    """
    BBNet-NFA-Only: Concat-first fully convolutional NAFBlock encoder-decoder — zero attention.

    Key difference from bbnet_bfa_no_bottleneck.py:
        REMOVED: rgb_stem (3ch), guidance_stem (4ch), SAFM fusion block.
        ADDED:   ConcatStem — single stride-2 conv on all input modalities
                              concatenated (7ch → 32ch @ H/2×W/2).

    Inputs (passed as *inputs tuple):
        inputs[0]: source       (B, 3, H, W)  — RGB image
        inputs[1]: kernel_map   (B, 1, H, W)  — Circle of Confusion (CoC) map
        inputs[2]: bloom_input  (B, 1, H, W)  — Specular highlight map
        inputs[3]: coord_maps   (B, 2, H, W)  — Normalized spatial coordinates

    NOTE: f_stop is NOT used. Aperture-aware attention is removed entirely.
    The model still accepts it as an optional 5th argument to maintain
    pipeline compatibility, but it is silently ignored.

    Full pipeline tensor trace (B=1, H=W=1408):
        === ConcatStem ===
        concat_input: cat(source,kernel_map,bloom_input,coord_maps)
                      = (1, 7, 1408, 1408)
        stem:         Conv(7→32, s2) → GN → GELU → Conv(32→32) → (1, 32, 704, 704)

        === NAFEncoder ===
        Stem:    Conv(32→32, s2)                → (1,  32, 352, 352)
        Stage0:  2×NAFBlock(32)                 → (1,  32, 352, 352)   skip₀
        Down:    Conv(32→128)                   → (1, 128, 176, 176)
        Stage1:  4×NAFBlock(128)                → (1, 128, 176, 176)   skip₁
        Down:    Conv(128→256)                  → (1, 256,  88,  88)
        Stage2:  3×NAFBlock(256)                → (1, 256,  88,  88)

        === NFABottleneck (pure conv, NO attention) ===
        2×NAFBlock(256)                         → (1, 256,  88,  88)

        === NAFDecoder ===
        Dec2: PS+skip₁ → NAFBlock(128)          → (1, 128, 176, 176)
        Dec1: PS+skip₀ → NAFBlock(128)          → (1, 128, 352, 352)
        Dec0: PS       → NAFBlock(128)          → (1, 128, 704, 704)

        === Output Head ===
        head1: Conv(128→64)→ReLU→Conv(64→128)→PS(2) → (1, 32, 1408, 1408)
        head2: Conv(3→32) on source HR               → (1, 32, 1408, 1408)
        cat(head1, head2, bloom):                    → (1, 65, 1408, 1408)
        head:  Conv(65→32)→ReLU→Conv(32→3)           → (1,  3, 1408, 1408)
    """

    def __init__(
            self,
            out_channels=3,
            # Concat stem
            stem_in_ch=7,                       # source(3)+kernel(1)+bloom(1)+coords(2)
            stem_channels=32,
            # Encoder
            encoder_channels=(32, 128, 256),
            encoder_blocks=(2, 4, 3),
            # NFA Bottleneck (pure conv)
            bottleneck_blocks=2,
            # Decoder
            decoder_channels=(128, 128, 128),
            # Output
            non_negative=False,
            # Legacy API compatibility params (ignored)
            layers=None, encoder_start_filts=None, encoder_in_channels=None,
            width_mult=None, decoder_channels_legacy=None, features=None,
            use_bn=False, enable_attention_hooks=False, scale_factor=2,
            # AAA params (silently ignored — no attention bottleneck)
            bottleneck_dim=None, bottleneck_heads=None, bottleneck_ffn_ratio=None,
            num_residual_groups=None, blocks_per_group=None,
            use_coord_conv=None, use_kernel_conv=None, drop_path_rate=None,
            relpos_init_value=None, relpos_heads_range=None,
    ):
        super().__init__()

        # ── ConcatStem (replaces rgb_stem + guidance_stem + SAFM) ──
        # All 7 input channels concatenated and sent through a single stride-2 conv.
        self.stem = ConcatStem(in_ch=stem_in_ch, out_ch=stem_channels)

        # ── NAFEncoder ──
        self.encoder = NAFEncoder(
            in_ch=stem_channels,
            channels=encoder_channels,
            blocks=encoder_blocks,
        )

        # ── NFABottleneck: pure conv, no attention ──
        self.bottleneck = NFABottleneck(
            dim=encoder_channels[-1],
            num_blocks=bottleneck_blocks,
        )

        # ── NAFDecoder ──
        # Dec2: 256@88  + skip₁@176 → 128@176
        self.dec2 = NAFDecoderBlock(in_ch=encoder_channels[2], out_ch=decoder_channels[0],
                                    skip_ch=encoder_channels[1])
        # Dec1: 128@176 + skip₀@352 → 128@352
        self.dec1 = NAFDecoderBlock(in_ch=decoder_channels[0], out_ch=decoder_channels[1],
                                    skip_ch=encoder_channels[0])
        # Dec0: 128@352 → 128@704 (no skip)
        self.dec0 = NAFDecoderBlock(in_ch=decoder_channels[1], out_ch=decoder_channels[2],
                                    skip_ch=None)

        # ── Output Head ──
        self.head1 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2] // 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(decoder_channels[2] // 2, 32 * 4, 3, 1, 1),
            nn.PixelShuffle(2),                        # → (B, 32, 1408, 1408)
            nn.ReLU(True),
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
        )
        # 32 (head1) + 32 (head2) + 1 (bloom late inject) = 65ch
        self.head = nn.Sequential(
            nn.Conv2d(65, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, 1, 1, 0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, *inputs):
        """
        Forward pass.

        Args:
            inputs[0]: source       (B, 3, H, W)
            inputs[1]: kernel_map   (B, 1, H, W)
            inputs[2]: bloom_input  (B, 1, H, W)
            inputs[3]: coord_maps   (B, 2, H, W)
            inputs[4]: f_stop       (B,) — IGNORED (no aperture attention)

        Returns:
            (B, 3, H, W) bokeh-rendered output
        """
        source      = inputs[0]
        kernel_map  = inputs[1]
        bloom_input = inputs[2]
        coord_maps  = inputs[3]
        # inputs[4] = f_stop → silently unused

        # ── ConcatStem ──
        # Fuse all modalities by concatenation before the first conv.
        # cat: (B, 3+1+1+2, H, W) = (B, 7, H, W)
        concat_input = torch.cat([source, kernel_map, bloom_input, coord_maps], dim=1)
        stem_feat = self.stem(concat_input)                                 # (B, 32, H/2, W/2)

        # ── Encoder ──
        enc = self.encoder(stem_feat)
        # enc[0]: (B,  32, H/4,  W/4 ) skip₀
        # enc[1]: (B, 128, H/8,  W/8 ) skip₁
        # enc[2]: (B, 256, H/16, W/16) bottleneck input

        # ── NFABottleneck (pure conv, no attention) ──
        feat = self.bottleneck(enc[2])                                      # (B, 256, H/16, W/16)

        # ── Decoder ──
        feat = self.dec2(feat, skip=enc[1])                                # (B, 128, H/8,  W/8 )
        feat = self.dec1(feat, skip=enc[0])                                # (B, 128, H/4,  W/4 )
        feat = self.dec0(feat)                                              # (B, 128, H/2,  W/2 )

        # ── Output Heads ──
        out1 = self.head1(feat)                                             # (B, 32, H, W)
        hr   = self.head2(source)                                           # (B, 32, H, W)
        out  = self.head(torch.cat([out1, hr, bloom_input], dim=1))         # (B,  3, H, W)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = Inception_Encoder_Unet_Decoder()

    # Parameter breakdown
    total  = sum(p.numel() for p in model.parameters())
    stem_p = sum(p.numel() for p in model.stem.parameters())
    enc_p  = sum(p.numel() for p in model.encoder.parameters())
    bot_p  = sum(p.numel() for p in model.bottleneck.parameters())
    dec_p  = sum(p.numel() for name, p in model.named_parameters()
                 if name.startswith('dec'))
    head_p = total - stem_p - enc_p - bot_p - dec_p

    print(f"{'Component':<25} {'Parameters':>12}")
    print(f"{'-'*25} {'-'*12}")
    print(f"{'ConcatStem':<25} {stem_p:>12,}")
    print(f"{'NAF Encoder':<25} {enc_p:>12,}")
    print(f"{'NFA Bottleneck (NAF)':<25} {bot_p:>12,}")
    print(f"{'NAF Decoder':<25} {dec_p:>12,}")
    print(f"{'Output heads':<25} {head_p:>12,}")
    print(f"{'-'*25} {'-'*12}")
    print(f"{'TOTAL':<25} {total:>12,}")

    # Forward pass
    H, W = 1408, 1408
    s = torch.rand(1, 3, H, W)
    k = torch.rand(1, 1, H, W)
    b = torch.rand(1, 1, H, W)
    c = torch.rand(1, 2, H, W)
    f = torch.tensor([2.0])            # accepted but ignored

    print(f"\nForward pass at {H}×{W}...")
    with torch.no_grad():
        out = model(s, k, b, c, f)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 3, H, W), f"Shape mismatch: {out.shape}"
    print("[OK] Forward pass successful!")
