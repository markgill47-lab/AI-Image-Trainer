"""
VAE for Klein trainer.
Uses BFL (Black Forest Labs) format AutoEncoder for FLUX.2 models.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass, field
from typing import Optional
from safetensors.torch import load_file
from pathlib import Path


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """Attention block using Conv2d (BFL style)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """Residual block for VAE."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """VAE Encoder."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Quantization conv (applied after conv_out in BFL format)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    """VAE Decoder."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # Post-quantization conv (required for BFL VAE format)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # Post-quantization conv (must be applied before conv_in)
        z = self.post_quant_conv(z)

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


@dataclass
class AutoEncoderParams:
    """Parameters for the BFL AutoEncoder."""
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32  # Klein uses 32-channel VAE


class KleinVAE(nn.Module):
    """
    VAE for FLUX.2 Klein models (BFL format).
    Uses 32 latent channels with 2x2 packing (128 packed channels).
    """

    def __init__(self, params: Optional[AutoEncoderParams] = None):
        super().__init__()
        if params is None:
            params = AutoEncoderParams()
        self.params = params

        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        # BatchNorm for latent normalization (BFL-specific)
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]  # Packing size
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,  # 4 * 32 = 128 channels
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def latent_channels(self) -> int:
        """Number of packed latent channels (128 for Klein)."""
        return math.prod(self.ps) * self.params.z_channels

    def normalize(self, z: Tensor) -> Tensor:
        """Normalize latents using BatchNorm statistics."""
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        """Inverse normalize latents."""
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode images to latents.

        Args:
            x: Images tensor [B, 3, H, W] in range [-1, 1]

        Returns:
            Latents tensor [B, 128, H//16, W//16] (packed and normalized)
        """
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        # Pack latents: [B, 32, H/8, W/8] -> [B, 128, H/16, W/16]
        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor, debug: bool = False) -> Tensor:
        """
        Decode latents to images.

        Args:
            z: Latents tensor [B, 128, H//16, W//16]
            debug: If True, print debug statistics

        Returns:
            Images tensor [B, 3, H, W] in range approximately [-1, 1]
        """
        if debug:
            print(f"\n=== VAE DECODE DEBUG ===")
            print(f"Input z shape: {z.shape}")
            print(f"Input z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")

        z = self.inv_normalize(z)

        if debug:
            print(f"After inv_normalize: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")

        # Unpack latents: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )

        if debug:
            print(f"After unpack: shape={z.shape}")
            print(f"After unpack: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}")

        dec = self.decoder(z)

        if debug:
            print(f"Decoder output: min={dec.min().item():.4f}, max={dec.max().item():.4f}, mean={dec.mean().item():.4f}")
            print(f"=== VAE DECODE DEBUG END ===\n")

        return dec

    @classmethod
    def from_pretrained(cls, path: str, dtype: torch.dtype = torch.bfloat16) -> "KleinVAE":
        """
        Load VAE from a safetensors file.

        Args:
            path: Path to ae.safetensors (BFL format)
            dtype: Data type to load weights in

        Returns:
            Loaded KleinVAE model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VAE file not found: {path}")

        # Load state dict
        state_dict = load_file(str(path), device="cpu")

        # Check if it's BFL format (has decoder.conv_in.weight with shape [512, 32, 3, 3])
        if "decoder.conv_in.weight" in state_dict:
            z_channels = state_dict["decoder.conv_in.weight"].shape[1]
        else:
            raise ValueError("Cannot determine VAE format - missing decoder.conv_in.weight")

        # Verify this is a 32-channel VAE (required for Klein)
        if z_channels != 32:
            raise ValueError(f"Klein VAE must have 32 z_channels, found {z_channels}")

        # Detect format: diffusers uses "down_blocks/up_blocks", BFL uses "down/up"
        is_diffusers = any("down_blocks" in k or "up_blocks" in k for k in state_dict)
        if is_diffusers:
            print("  Detected diffusers-format VAE, converting to BFL format...")
            state_dict = cls._convert_diffusers_to_bfl(state_dict)

        # Remap quant_conv/post_quant_conv keys from root level to encoder/decoder
        # The BFL weights file has these at the root level, but our model structure
        # places them inside Encoder and Decoder classes
        key_remapping = {
            "quant_conv.weight": "encoder.quant_conv.weight",
            "quant_conv.bias": "encoder.quant_conv.bias",
            "post_quant_conv.weight": "decoder.post_quant_conv.weight",
            "post_quant_conv.bias": "decoder.post_quant_conv.bias",
        }
        for old_key, new_key in key_remapping.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        # Create model
        params = AutoEncoderParams(z_channels=z_channels)
        model = cls(params)

        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"Warning: Missing keys in VAE state dict: {missing}")

        if unexpected:
            print(f"Warning: Unexpected keys in VAE state dict: {unexpected}")

        # Verify BatchNorm stats are loaded
        if model.bn.running_mean is None or model.bn.running_var is None:
            raise ValueError("VAE BatchNorm statistics not loaded - check if bn.running_mean/var are in state dict")

        return model.to(dtype)

    @staticmethod
    def _convert_diffusers_to_bfl(sd: dict) -> dict:
        """Convert diffusers-format VAE state dict to BFL format."""
        import re
        new_sd = {}
        for key, val in sd.items():
            new_key = key

            # --- Encoder ---
            # down_blocks.{i}.resnets.{j} → down.{i}.block.{j}
            new_key = re.sub(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)',
                             r'encoder.down.\1.block.\2', new_key)
            # down_blocks.{i}.downsamplers.0.conv → down.{i}.downsample.conv
            new_key = re.sub(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv',
                             r'encoder.down.\1.downsample.conv', new_key)
            # mid_block.resnets.{j} → mid.block_{j+1}
            new_key = re.sub(r'encoder\.mid_block\.resnets\.(\d+)',
                             lambda m: f'encoder.mid.block_{int(m.group(1))+1}', new_key)

            # --- Decoder ---
            # up_blocks.{i}.resnets.{j} → up.{i}.block.{j}
            new_key = re.sub(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)',
                             r'decoder.up.\1.block.\2', new_key)
            # up_blocks.{i}.upsamplers.0.conv → up.{i}.upsample.conv
            new_key = re.sub(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv',
                             r'decoder.up.\1.upsample.conv', new_key)
            # mid_block.resnets.{j} → mid.block_{j+1}
            new_key = re.sub(r'decoder\.mid_block\.resnets\.(\d+)',
                             lambda m: f'decoder.mid.block_{int(m.group(1))+1}', new_key)

            # --- Both encoder and decoder ---
            # conv_shortcut → nin_shortcut
            new_key = new_key.replace('.conv_shortcut.', '.nin_shortcut.')
            # conv_norm_out → norm_out
            new_key = new_key.replace('conv_norm_out', 'norm_out')
            # Attention: mid_block.attentions.0 → mid.attn_1
            new_key = new_key.replace('mid_block.attentions.0.group_norm', 'mid.attn_1.norm')
            new_key = new_key.replace('mid_block.attentions.0.to_q', 'mid.attn_1.q')
            new_key = new_key.replace('mid_block.attentions.0.to_k', 'mid.attn_1.k')
            new_key = new_key.replace('mid_block.attentions.0.to_v', 'mid.attn_1.v')
            new_key = new_key.replace('mid_block.attentions.0.to_out.0', 'mid.attn_1.proj_out')

            new_sd[new_key] = val
        return new_sd


def encode_images(vae: KleinVAE, images: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Encode a batch of images to latents.

    Args:
        vae: Loaded KleinVAE
        images: Tensor of images [B, 3, H, W] normalized to [-1, 1]
        device: Device to run encoding on
        dtype: Data type for encoding

    Returns:
        Latents tensor [B, 128, H//16, W//16]
    """
    vae = vae.to(device)
    images = images.to(device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(images)

    return latents


def decode_latents(vae: KleinVAE, latents: Tensor, device: torch.device, dtype: torch.dtype, debug: bool = False) -> Tensor:
    """
    Decode latents to images.

    Args:
        vae: Loaded KleinVAE
        latents: Tensor [B, 128, H//16, W//16]
        device: Device to run decoding on
        dtype: Data type for decoding
        debug: If True, print debug statistics

    Returns:
        Images tensor [B, 3, H, W] approximately in [-1, 1]
    """
    vae = vae.to(device)
    latents = latents.to(device, dtype=dtype)

    with torch.no_grad():
        images = vae.decode(latents, debug=debug)

    return images
