"""
LoRA (Low-Rank Adaptation) implementation for Klein models.
Clean, minimal implementation focused on training and inference correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import math
from safetensors.torch import save_file, load_file


class LoRALayer(nn.Module):
    """
    A single LoRA layer that wraps a linear layer.

    LoRA decomposes weight updates as: W' = W + BA
    where B is [out_features, rank] and A is [rank, in_features]
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 32,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.original_layer.requires_grad_(False)

        # Get device and dtype from original layer
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype

        # LoRA matrices - create on same device/dtype as original
        self.lora_A = nn.Linear(self.in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False, device=device, dtype=dtype)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Flag for merged state
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_layer(x)

        # Add LoRA contribution (only if not merged)
        if not self.merged:
            lora_out = self.lora_B(self.lora_A(self.dropout(x)))
            result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into the original layer for faster inference."""
        if self.merged:
            return

        with torch.no_grad():
            # Compute delta: B @ A * scaling
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.add_(delta)

        self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from the original layer."""
        if not self.merged:
            return

        with torch.no_grad():
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original_layer.weight.sub_(delta)

        self.merged = False

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA weights."""
        return {
            "lora_A.weight": self.lora_A.weight.data,
            "lora_B.weight": self.lora_B.weight.data,
        }


class LoRANetwork(nn.Module):
    """
    LoRA network that wraps a transformer model.

    Injects LoRA layers into attention Q, K, V, and output projections,
    as well as MLP layers in transformer blocks.
    """

    def __init__(
        self,
        transformer: nn.Module,
        rank: int = 32,
        alpha: int = 32,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        self.transformer = transformer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules for Flux2Klein architecture
        # These are the actual layer names in Flux2Klein:
        # - Double blocks: img_attn.qkv, img_attn.proj, txt_attn.qkv, txt_attn.proj, img_mlp.0, img_mlp.2, txt_mlp.0, txt_mlp.2
        # - Single blocks: linear1, linear2
        if target_modules is None:
            target_modules = [
                # Attention projections
                "img_attn.qkv", "img_attn.proj",
                "txt_attn.qkv", "txt_attn.proj",
                # MLP layers
                "img_mlp.0", "img_mlp.2",
                "txt_mlp.0", "txt_mlp.2",
                # Single block layers
                "linear1", "linear2",
            ]
        self.target_modules = target_modules

        # Store LoRA layers
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Inject LoRA into target modules
        self._inject_lora()

        print(f"LoRA network created:")
        print(f"  - Rank: {rank}")
        print(f"  - Alpha: {alpha}")
        print(f"  - LoRA layers: {len(self.lora_layers)}")
        print(f"  - Trainable params: {self.get_trainable_params():,}")

    def _inject_lora(self):
        """Inject LoRA layers into the transformer."""
        for name, module in self.transformer.named_modules():
            # Check if this module should have LoRA
            if not isinstance(module, nn.Linear):
                continue

            should_inject = False
            for target in self.target_modules:
                if target in name:
                    should_inject = True
                    break

            if not should_inject:
                continue

            # Create LoRA layer
            lora_layer = LoRALayer(
                original_layer=module,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )

            # Replace in parent module
            parent_name, child_name = self._get_parent_and_child_name(name)
            parent = self.transformer.get_submodule(parent_name) if parent_name else self.transformer

            setattr(parent, child_name, lora_layer)
            self.lora_layers[name] = lora_layer

    def _get_parent_and_child_name(self, name: str) -> Tuple[str, str]:
        """Split module name into parent path and child name."""
        parts = name.split(".")
        if len(parts) == 1:
            return "", parts[0]
        return ".".join(parts[:-1]), parts[-1]

    def forward(self, *args, **kwargs):
        """Forward pass through the transformer."""
        return self.transformer(*args, **kwargs)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(
            p.numel()
            for layer in self.lora_layers.values()
            for p in [layer.lora_A.weight, layer.lora_B.weight]
        )

    def parameters(self):
        """Return only LoRA parameters for training."""
        for layer in self.lora_layers.values():
            yield layer.lora_A.weight
            yield layer.lora_B.weight

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return named LoRA parameters for training."""
        for name, layer in self.lora_layers.items():
            yield f"{prefix}{name}.lora_A.weight", layer.lora_A.weight
            yield f"{prefix}{name}.lora_B.weight", layer.lora_B.weight

    def train(self, mode: bool = True):
        """Set training mode for LoRA layers and transformer."""
        for layer in self.lora_layers.values():
            layer.lora_A.train(mode)
            layer.lora_B.train(mode)
        # Set transformer to training mode too (required for gradient checkpointing!)
        # The base weights remain frozen via requires_grad=False
        self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for all components."""
        return self.train(False)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the underlying transformer."""
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            print("  Gradient checkpointing enabled on transformer")

    def merge_weights(self):
        """Merge all LoRA weights into original layers."""
        for layer in self.lora_layers.values():
            layer.merge_weights()

    def unmerge_weights(self):
        """Unmerge all LoRA weights from original layers."""
        for layer in self.lora_layers.values():
            layer.unmerge_weights()

    def get_state_dict(self, comfy_format: bool = True) -> Dict[str, torch.Tensor]:
        """Get state dict containing all LoRA weights.

        Args:
            comfy_format: If True, uses ComfyUI-compatible key format
        """
        state_dict = {}
        for name, layer in self.lora_layers.items():
            layer_state = layer.get_lora_state_dict()
            for key, value in layer_state.items():
                if comfy_format:
                    # ComfyUI expects: diffusion_model.<layer_path>.lora_A.weight
                    full_key = f"diffusion_model.{name}.{key}"
                else:
                    full_key = f"{name}.{key}"
                state_dict[full_key] = value
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load LoRA weights from state dict."""
        for name, layer in self.lora_layers.items():
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"

            if a_key in state_dict:
                layer.lora_A.weight.data.copy_(state_dict[a_key])
            elif strict:
                raise KeyError(f"Missing key: {a_key}")

            if b_key in state_dict:
                layer.lora_B.weight.data.copy_(state_dict[b_key])
            elif strict:
                raise KeyError(f"Missing key: {b_key}")

    def save_weights(self, path: str, dtype: torch.dtype = torch.bfloat16, model_variant: str = "4B"):
        """Save LoRA weights to safetensors file."""
        state_dict = self.get_state_dict(comfy_format=True)

        # Convert to specified dtype
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        # Add metadata for ComfyUI compatibility
        metadata = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_base_model_version": f"flux2_klein_{model_variant.lower()}",
            "modelspec.architecture": f"flux2-klein-{model_variant.lower()}-lora",
            "modelspec.implementation": "klein_trainer",
            # Legacy fields
            "rank": str(self.rank),
            "alpha": str(self.alpha),
            "target_modules": ",".join(self.target_modules),
        }

        save_file(state_dict, path, metadata=metadata)
        print(f"LoRA weights saved to {path}")

    @classmethod
    def load_weights(
        cls,
        transformer: nn.Module,
        path: str,
        device: torch.device = torch.device("cpu"),
    ) -> "LoRANetwork":
        """Load LoRA weights from safetensors file."""
        # Load safetensors
        state_dict = load_file(path, device=str(device))

        # Try to read metadata
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                metadata = f.metadata() or {}
        except:
            metadata = {}

        # Get config from metadata or use defaults
        rank = int(metadata.get("rank", 32))
        alpha = int(metadata.get("alpha", 32))
        target_modules = metadata.get("target_modules", "").split(",") if metadata.get("target_modules") else None

        # Create network
        network = cls(
            transformer=transformer,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )

        # Load weights
        network.load_state_dict(state_dict, strict=False)

        print(f"LoRA weights loaded from {path}")
        return network


def apply_lora_to_transformer(
    transformer: nn.Module,
    rank: int = 32,
    alpha: int = 32,
    dropout: float = 0.0,
) -> LoRANetwork:
    """
    Apply LoRA to a Klein transformer.

    This is the main entry point for LoRA application.

    Args:
        transformer: Klein transformer model
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout rate for LoRA layers

    Returns:
        LoRANetwork wrapping the transformer
    """
    return LoRANetwork(
        transformer=transformer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
