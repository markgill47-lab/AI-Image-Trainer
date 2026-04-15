"""
FLUX.2 Klein Model Architecture
Klein models are guidance-free with 24 single blocks
"""
import torch
from einops import rearrange
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt
import math
from dataclasses import dataclass, field
from .model import (
    DoubleStreamBlock,
    SingleStreamBlock,
    LastLayer,
    Modulation,
    timestep_embedding,
    EmbedND,
    MLPEmbedder,
)


@dataclass
class Flux2KleinParams:
    """Architecture parameters for FLUX.2 Klein 9B model
    Both Klein 4B and 9B use 32-channel VAE → 128 packed channels (32 * 4)
    """
    in_channels: int = 128  # 32 VAE channels * 4 packing (2x2) = 128
    context_in_dim: int = 12288  # Klein 9B uses 12288
    hidden_size: int = 4096  # Klein 9B uses 4096
    num_heads: int = 32  # Klein 9B uses 32 heads
    depth: int = 8  # Klein 9B has 8 double blocks
    depth_single_blocks: int = 24  # Klein 9B has 24 single blocks
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0


@dataclass
class Flux2Klein4BParams:
    """Architecture parameters for FLUX.2 Klein 4B model
    Klein 4B uses 32-channel VAE → 128 packed channels (32 * 4)
    """
    in_channels: int = 128  # 32 VAE channels * 4 packing (2x2) = 128
    context_in_dim: int = 7680  # 4B uses 7680, not 12288
    hidden_size: int = 3072  # 4B uses 3072, not 4096
    num_heads: int = 24  # 3072 / 24 = 128 per head
    depth: int = 5  # 4B has 5 double blocks
    depth_single_blocks: int = 20  # 4B has 20 single blocks, not 24
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0


class FakeConfig:
    # for diffusers compatability
    def __init__(self):
        self.patch_size = 1


class Flux2Klein(nn.Module):
    """
    FLUX.2 Klein Model - Guidance-free variant with 24 single blocks
    """

    def __init__(self, params: Flux2KleinParams):
        super().__init__()
        self.config = FakeConfig()
        self._debug_generation = False  # Flag for generation debugging
        self._first_forward = False  # Track first training forward for debugging

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        # Klein models don't have guidance_in layer
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size, double=False, disable_bias=True
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            self.out_channels,
        )

        self.gradient_checkpointing = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None = None,  # Klein models ignore guidance
    ):
        num_txt_tokens = ctx.shape[1]
        num_img_tokens = x.shape[1]  # Track original image token count

        # DEBUG: Check for NaN/Inf in inputs during training
        if self.training and (torch.isnan(x).any() or torch.isinf(x).any()):
            print(f"WARNING: NaN/Inf in input x! NaN: {torch.isnan(x).sum()}, Inf: {torch.isinf(x).sum()}")
        if self.training and (torch.isnan(ctx).any() or torch.isinf(ctx).any()):
            print(f"WARNING: NaN/Inf in input ctx! NaN: {torch.isnan(ctx).sum()}, Inf: {torch.isinf(ctx).sum()}")

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        # Klein models don't use guidance - skip guidance_in

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        # DEBUG: Check modulation values
        if self._debug_generation and not self.training:
            mod_shift, mod_scale, mod_gate = single_block_mod
            print(f"    single_block_mod: shift mean={mod_shift.mean().item():.4f}, scale mean={mod_scale.mean().item():.4f}, gate mean={mod_gate.mean().item():.4f}")
            print(f"    single_block_mod gate: min={mod_gate.min().item():.4f}, max={mod_gate.max().item():.4f}")

        # prepare image
        img = self.img_in(x)
        img_ids = self.pe_embedder(x_ids)

        # prepare txt
        txt = self.txt_in(ctx)
        txt_ids = self.pe_embedder(ctx_ids)

        # DEBUG: Check shapes for training
        if self.training and hasattr(self, '_first_forward'):
            if not self._first_forward:
                print(f"\n=== FIRST TRAINING FORWARD ===")
                print(f"Input shapes: x={x.shape}, ctx={ctx.shape}")
                print(f"Position ID shapes: x_ids={x_ids.shape}, ctx_ids={ctx_ids.shape}")
                print(f"ctx (text embeds) stats: min={ctx.min().item():.4f}, max={ctx.max().item():.4f}, mean={ctx.mean().item():.4f}")
                print(f"After embedding: img={img.shape}, txt={txt.shape}")
                print(f"Position embeddings: img_ids={img_ids.shape}, txt_ids={txt_ids.shape}")
                print(f"num_img_tokens={num_img_tokens}, num_txt_tokens={num_txt_tokens}")
                self._first_forward = True

        # CRITICAL FIX: Don't concatenate position embeddings here!
        # DoubleStreamBlock expects separate pe and pe_ctx, and concatenates them internally
        # Keep img_ids and txt_ids separate for double blocks

        # encoder double stream blocks
        for i, block in enumerate(self.double_blocks):
            if self.gradient_checkpointing and self.training:
                img, txt = ckpt.checkpoint(
                    block,
                    img,
                    txt,
                    img_ids,  # pe parameter - image position embeddings
                    txt_ids,  # pe_ctx parameter - text position embeddings
                    double_block_mod_img,  # mod_img parameter
                    double_block_mod_txt,  # mod_txt parameter
                    use_reentrant=False,
                )
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    pe=img_ids,  # Pass image position embeddings
                    pe_ctx=txt_ids,  # Pass text position embeddings separately
                    mod_img=double_block_mod_img,
                    mod_txt=double_block_mod_txt,
                )

        # After double blocks, concatenate img and txt
        # Klein concatenates as (img, txt) - image first
        img = torch.cat((img, txt), 1)

        # For single blocks, concatenate position embeddings along sequence dimension
        # RoPE embeddings have shape: [batch, 1, seq, pe_dim//2, 2, 2]
        # After pe_embedder: [batch, 1, img_seq, 64, 2, 2] and [batch, 1, txt_seq, 64, 2, 2]
        # Concatenate on dim=2 (sequence dimension, not dim=1 which is the unsqueezed dim)
        ids = torch.cat((img_ids, txt_ids), dim=2)  # Concatenate on sequence dimension (dim=2, not dim=1)

        # encoder single stream blocks
        for i, block in enumerate(self.single_blocks):
            if self.gradient_checkpointing and self.training:
                img = ckpt.checkpoint(
                    block, img, ids, single_block_mod, use_reentrant=False
                )
            else:
                img = block(img, ids, single_block_mod)

        # Slice to extract only image tokens (first num_img_tokens)
        # After single blocks: img shape is [batch, img_seq+txt_seq, hidden]
        # We want: [batch, img_seq, hidden] where img_seq = num_img_tokens
        img = img[:, :num_img_tokens, ...]

        if self._debug_generation and not self.training:
            print(f"    Model forward: before final_layer min={img.min().item():.4f}, max={img.max().item():.4f}, mean={img.mean().item():.4f}")

        img = self.final_layer(img, vec=vec)

        if self._debug_generation and not self.training:
            print(f"    Model forward: after final_layer min={img.min().item():.4f}, max={img.max().item():.4f}, mean={img.mean().item():.4f}")

        # DEBUG: Check for NaN/Inf in output during training
        if self.training and (torch.isnan(img).any() or torch.isinf(img).any()):
            print(f"ERROR: NaN/Inf in model output! NaN: {torch.isnan(img).sum()}, Inf: {torch.isinf(img).sum()}")
            print(f"  Output stats: min={img.min().item() if not torch.isinf(img).any() else 'inf'}, max={img.max().item() if not torch.isinf(img).any() else 'inf'}")

        return img
