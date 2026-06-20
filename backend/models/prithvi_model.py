"""Faithful plain-PyTorch reconstruction of the Prithvi-EO-1.0-100M
multi-temporal crop/land classification model.

The published checkpoint is an mmsegmentation `.pth` (backbone + neck +
decode_head). Rather than fight mmcv/mmseg under torch 2.6, we rebuild the exact
modules so the legacy weights load directly. Architecture verified from the
checkpoint's state_dict shapes and the original geospatial_fm source:

  input (B, 6, 3, 224, 224)                      6 bands x 3 timesteps
  -> Conv3d patch embed (1x16x16)  -> tokens (B, 588, 768) + cls -> (B,589,768)
  -> 6 ViT blocks -> norm
  -> neck: drop cls, reshape to (B, 2304, 14, 14), two FPN x4 stages -> (B,2304,224,224)
  -> FCN head: Conv(2304->256)+BN+ReLU -> Conv(256->13)  -> logits (B,13,224,224)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 768
NUM_FRAMES = 3
PATCH = 16
DEPTH = 6
NUM_HEADS = 12
NUM_CLASSES = 13
IMG_SIZE = 224

# Per-band normalization (from the model's mmseg config), order:
# Blue, Green, Red, Narrow-NIR, SWIR1, SWIR2 (in HLS/S2 reflectance units).
BAND_MEANS = [494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917]
BAND_STDS = [284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808]

CLASSES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle Cropland", "Cotton", "Sorghum", "Other",
]


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(6, EMBED_DIM, kernel_size=(1, PATCH, PATCH), stride=(1, PATCH, PATCH))

    def forward(self, x):  # (B, 6, T, H, W)
        x = self.proj(x)  # (B, 768, T, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, T*14*14, 768)
        return x


class PrithviEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_patches = NUM_FRAMES * (IMG_SIZE // PATCH) ** 2  # 3 * 196 = 588
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Norm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Reshape tokens to a spatial grid (frames folded into channels) and
    upsample x16 back to image resolution with two FPN stages."""

    def __init__(self, embed_dim=EMBED_DIM * NUM_FRAMES, output_embed_dim=EMBED_DIM * NUM_FRAMES, hp=14, wp=14):
        super().__init__()
        self.hp, self.wp, self.output_embed_dim = hp, wp, output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, output_embed_dim, 2, 2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, 2, 2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, 2, 2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, 2, 2),
        )

    def forward(self, x):  # x: (B, 589, 768)
        x = x[:, 1:, :]  # drop cls
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.hp, self.wp)  # (B, 2304, 14, 14)
        x = self.fpn1(x)
        x = self.fpn2(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, ci, co, k, pad):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class FCNHead(nn.Module):
    def __init__(self, in_ch=EMBED_DIM * NUM_FRAMES, ch=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.convs = nn.Sequential(ConvModule(in_ch, ch, 3, 1))
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)
        return self.conv_seg(x)


class PrithviSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PrithviEncoder()
        self.neck = ConvTransformerTokensToEmbeddingNeck()
        self.decode_head = FCNHead()

    def forward(self, x):  # (B, 6, 3, 224, 224)
        tokens = self.backbone(x)
        feat = self.neck(tokens)
        logits = self.decode_head(feat)
        if logits.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return logits


def load_state_into(model: PrithviSeg, state_dict: dict) -> tuple[list, list]:
    """Load backbone/neck/decode_head weights (ignore mmseg auxiliary_head)."""
    wanted = {k: v for k, v in state_dict.items() if k.split(".")[0] in {"backbone", "neck", "decode_head"}}
    missing, unexpected = model.load_state_dict(wanted, strict=False)
    return missing, unexpected
