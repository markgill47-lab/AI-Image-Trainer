"""
FLUX.2 Klein Model Loader
Extends Flux2Model to support Klein-specific file structures
"""
import os
import torch
import huggingface_hub
from safetensors.torch import load_file
from .flux2_model import Flux2Model, MISTRAL_PATH, FLUX2_VAE_FILENAME, HF_TOKEN
from .src.model import Flux2, Flux2Params
from .src.model_klein import Flux2Klein, Flux2KleinParams, Flux2Klein4BParams
from .src.autoencoder import AutoEncoder, AutoEncoderParams
from toolkit.basic import flush

# Klein-specific filenames
FLUX2_KLEIN_9B_FILENAME = "flux-2-klein-base-9b.safetensors"
FLUX2_KLEIN_4B_FILENAME = "flux-2-klein-base-4b.safetensors"
FLUX2_KLEIN_VAE_FILENAME = "vae/diffusion_pytorch_model.safetensors"  # Klein uses different VAE path


class Flux2KleinModel(Flux2Model):
    """
    FLUX.2 Klein variant model loader
    Supports both 9B and 4B Klein models with correct file naming
    """
    arch = "flux2_klein"

    def __init__(self, device, model_config, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        # Override LoRA target modules to use Flux2Klein instead of Flux2
        self.target_lora_modules = ["Flux2Klein"]

    def load_model(self):
        """
        Load FLUX.2 Klein model with correct filenames
        """
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux2 Klein model")

        model_path = self.model_config.name_or_path
        transformer_path = model_path

        # Determine which Klein variant based on model path
        is_4b = "4b" in model_path.lower() or "4B" in model_path
        klein_filename = FLUX2_KLEIN_4B_FILENAME if is_4b else FLUX2_KLEIN_9B_FILENAME

        model_size = "4B" if is_4b else "9B"
        self.print_and_status_update(f"Loading FLUX.2 Klein {model_size} transformer")

        # Note: Klein model architecture will be created later after we know the VAE latent channels
        # We need to defer this because Klein's in_channels depends on the VAE z_channels
        # (in_channels = z_channels * 4, where z_channels is either 16 or 32)

        # Check for local file first
        if os.path.exists(os.path.join(transformer_path, klein_filename)):
            transformer_path = os.path.join(transformer_path, klein_filename)
            self.print_and_status_update(f"Using local Klein model: {transformer_path}")
        elif not os.path.exists(transformer_path):
            # Download from HuggingFace Hub
            self.print_and_status_update(f"Downloading {klein_filename} from HuggingFace Hub...")
            try:
                transformer_path = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename=klein_filename,
                    token=HF_TOKEN,
                )
                self.print_and_status_update(f"Downloaded Klein model to: {transformer_path}")
            except Exception as e:
                self.print_and_status_update(f"Error downloading Klein model: {e}")
                raise

        # Load transformer state dict (we'll create the model after determining VAE channels)
        self.print_and_status_update("Loading Klein transformer weights...")
        transformer_state_dict = load_file(transformer_path, device="cpu")

        # Klein models use Qwen3 text encoders (not Mistral!)
        # Klein 4B: Qwen3 with hidden_size=2560 (3 layers × 2560 = 7680)
        # Klein 9B: Qwen3 with hidden_size=4096 (3 layers × 4096 = 12288)
        klein_repo = f"black-forest-labs/FLUX.2-klein-base-{model_size}"
        self.print_and_status_update(f"Loading Qwen3 text encoder for Klein {model_size} on CPU")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from optimum.quanto import freeze
        from toolkit.util.quantize import quantize, get_qtype

        # Load Klein-specific Qwen3 text encoder directly to CPU first
        text_encoder = AutoModelForCausalLM.from_pretrained(
            klein_repo,
            subfolder="text_encoder",
            torch_dtype=dtype,
            device_map="cpu",  # Force CPU loading
        )
        flush()

        # Quantize BEFORE moving to GPU to save VRAM
        if hasattr(self.model_config, 'quantize_te') and self.model_config.quantize_te:
            self.print_and_status_update(f"Quantizing Qwen3 text encoder (8-bit) on CPU")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        # Move to GPU only if low_vram is disabled (for 16GB VRAM, keep on CPU)
        if hasattr(self.model_config, 'low_vram') and self.model_config.low_vram:
            self.print_and_status_update("Keeping text encoder on CPU (low_vram mode enabled)")
            # Text encoder stays on CPU to save VRAM during prompt caching
        else:
            self.print_and_status_update("Moving text encoder to GPU")
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        # Offload text encoder layers
        if (
            hasattr(self.model_config, 'low_vram') and
            self.model_config.layer_offloading and
            hasattr(self.model_config, 'layer_offloading_text_encoder_percent') and
            self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            self.print_and_status_update(f"Enabling text encoder offloading: {self.model_config.layer_offloading_text_encoder_percent*100:.0f}% of layers")
            from toolkit.memory_management.manager import MemoryManager
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = AutoTokenizer.from_pretrained(klein_repo, subfolder="tokenizer")

        # Load VAE on CPU to minimize VRAM usage
        # Both Klein 4B and 9B use 32-channel VAE (same as FLUX.1-dev) → 128 packed channels
        required_vae_channels = 32
        self.print_and_status_update(f"Loading VAE on CPU (Klein uses 32-channel VAE)")
        vae_path = None

        # Try multiple possible locations for the VAE file
        possible_vae_paths = []

        # 1. Check if user provided a custom VAE path
        if hasattr(self.model_config, 'vae_path') and self.model_config.vae_path:
            possible_vae_paths.append(self.model_config.vae_path)

        # 2. Check common local paths
        possible_vae_paths.extend([
            'models/flux2_vae/ae.safetensors',  # User's folder
            'models/ae.safetensors',
            'ae.safetensors',  # Root of project
            'vae/ae.safetensors',
        ])

        # Try each possible path and verify it has the correct number of channels
        for test_path in possible_vae_paths:
            if os.path.exists(test_path):
                self.print_and_status_update(f"Checking VAE at: {test_path}")
                try:
                    test_vae_sd = load_file(test_path, device="cpu")
                    decoder_shape = test_vae_sd.get('decoder.conv_in.weight', None)
                    if decoder_shape is not None:
                        channels = decoder_shape.shape[1]
                        if channels == required_vae_channels:
                            vae_path = os.path.abspath(test_path)
                            self.print_and_status_update(f"Found {required_vae_channels}-channel VAE: {vae_path}")
                            break
                        else:
                            self.print_and_status_update(f"  VAE has {channels} channels (need {required_vae_channels}), skipping")
                except Exception as e:
                    self.print_and_status_update(f"  Could not load VAE: {e}")

        if not vae_path:
            # Download Klein-specific 32-channel VAE from the Klein model repository
            klein_repo = f"black-forest-labs/FLUX.2-klein-base-{model_size}"
            self.print_and_status_update(f"No local 32-channel VAE found. Downloading Klein {model_size} VAE from {klein_repo}...")
            try:
                vae_path = huggingface_hub.hf_hub_download(
                    repo_id=klein_repo,
                    filename="vae/diffusion_pytorch_model.safetensors",
                    token=HF_TOKEN if HF_TOKEN else None,
                )
                self.print_and_status_update(f"Downloaded Klein {model_size} 32-channel VAE to: {vae_path}")
            except Exception as e:
                self.print_and_status_update(f"ERROR: Failed to download Klein {model_size} VAE: {e}")
                self.print_and_status_update("")
                self.print_and_status_update(f"Klein {model_size} requires its own 32-channel VAE from {klein_repo}.")
                self.print_and_status_update("Please download it manually and place it in one of these locations:")
                self.print_and_status_update("  - models/flux2_vae/ae.safetensors")
                self.print_and_status_update("  - models/ae.safetensors")
                self.print_and_status_update("")
                self.print_and_status_update(f"Download from: https://huggingface.co/{klein_repo}/blob/main/vae/diffusion_pytorch_model.safetensors")
                raise

        # Load state dict and verify it has the correct number of channels
        vae_state_dict = load_file(vae_path, device="cpu")

        # Verify VAE has the correct latent channels
        decoder_conv_in_shape = vae_state_dict.get('decoder.conv_in.weight', None)
        if decoder_conv_in_shape is not None:
            z_channels = decoder_conv_in_shape.shape[1]  # Second dimension is input channels to decoder
            self.print_and_status_update(f"VAE latent channels: {z_channels}")

            if z_channels != required_vae_channels:
                klein_repo = f"black-forest-labs/FLUX.2-klein-base-{model_size}"
                raise ValueError(
                    f"Klein {model_size} requires a {required_vae_channels}-channel VAE, but found {z_channels} channels. "
                    f"Each Klein model has its own 32-channel VAE in its repository. "
                    f"Please download from: https://huggingface.co/{klein_repo}/blob/main/vae/diffusion_pytorch_model.safetensors"
                )
        else:
            # Couldn't detect, use required channels
            z_channels = required_vae_channels
            self.print_and_status_update(f"Could not detect VAE channels from decoder.conv_in, assuming {required_vae_channels}")

        # Both Klein models use 32-channel VAE with 2x2 packing:
        # 32 latent channels * 4 packing (2x2) = 128 packed channels
        klein_in_channels = z_channels * 4
        self.print_and_status_update(f"Creating Klein transformer with in_channels={klein_in_channels} (z_channels={z_channels} * 4)")

        # Klein 4B and 9B are both guidance-free but have different architectures:
        # - Klein 9B: 0 double blocks, 24 single blocks, hidden_size=4096, context_in_dim=12288
        # - Klein 4B: 5 double blocks, 20 single blocks, hidden_size=3072, context_in_dim=7680
        # Both use the Flux2Klein class (guidance-free with double blocks support)
        if is_4b:
            # Klein 4B uses guidance-free architecture with double blocks
            klein_params = Flux2Klein4BParams()
            klein_params.in_channels = klein_in_channels

            self.print_and_status_update("Creating Klein 4B with Flux2Klein architecture (5 double + 20 single blocks)")
            with torch.device("meta"):
                transformer = Flux2Klein(klein_params)
        else:
            # Klein 9B uses guidance-free architecture with only single blocks
            klein_params = Flux2KleinParams()
            klein_params.in_channels = klein_in_channels

            self.print_and_status_update("Creating Klein 9B with Flux2Klein architecture (0 double + 24 single blocks)")
            with torch.device("meta"):
                transformer = Flux2Klein(klein_params)

        # Materialize and load transformer
        self.print_and_status_update("Initializing Klein transformer on CPU...")
        transformer = transformer.to_empty(device="cpu")

        # Cast transformer state dict to target dtype
        for key in transformer_state_dict:
            transformer_state_dict[key] = transformer_state_dict[key].to(dtype)

        # Load transformer state dict
        transformer.load_state_dict(transformer_state_dict, assign=True)
        self.print_and_status_update("Klein transformer loaded successfully")

        # Quantize transformer on CPU BEFORE moving to GPU to minimize VRAM spike
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Klein Transformer on CPU (this saves VRAM)")
            from toolkit.util.quantize import quantize_model

            # Quantize on CPU (much more memory-efficient than quantizing on GPU)
            # Pass self (base_model) first, then transformer
            quantize_model(self, transformer)
            flush()
            self.print_and_status_update("Klein transformer quantized on CPU")

        # Create VAE with detected architecture
        vae_params = AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=z_channels,
        )
        with torch.device("meta"):
            vae = AutoEncoder(vae_params)

        # Materialize on CPU
        vae = vae.to_empty(device="cpu")

        # Cast state dict to target dtype
        for key in vae_state_dict:
            vae_state_dict[key] = vae_state_dict[key].to(dtype)

        # Load with strict=False to handle missing quant_conv layers (FLUX.1 VAEs don't have these)
        missing_keys, unexpected_keys = vae.load_state_dict(vae_state_dict, strict=False, assign=True)

        # Log what was missing (expected for some VAE variants)
        if missing_keys:
            self.print_and_status_update(f"VAE missing keys (may be expected): {len(missing_keys)} keys")
        if unexpected_keys:
            self.print_and_status_update(f"VAE unexpected keys: {unexpected_keys}")

        # VAE will stay on CPU and be loaded only when needed for latent caching
        self.print_and_status_update("VAE loaded on CPU (will be used for latent caching)")
        flush()

        # Create scheduler and pipeline
        self.noise_scheduler = Flux2Model.get_train_scheduler()

        self.print_and_status_update("Making Klein pipeline")
        from .src.pipeline import Flux2Pipeline

        pipe = Flux2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        pipe.transformer = transformer

        # Override pipeline's embedding extraction for Qwen3
        # Klein uses Qwen3 (not Mistral) with simple tokenization (no chat template)
        def _get_qwen3_prompt_embeds(
            self,  # Need self parameter since this replaces a class method
            prompt,
            device=None,
            dtype=None,
            max_sequence_length=2048,  # Large default for Klein to handle all image sizes
        ):
            from einops import rearrange
            device = device or self._execution_device
            dtype = dtype or self.text_encoder.dtype

            if not isinstance(prompt, list):
                prompt = [prompt]

            # Use chat template for Qwen3 (required for correct encoding!)
            # BFL's Qwen3Embedder uses apply_chat_template with enable_thinking=False
            all_input_ids = []
            all_attention_masks = []

            for p in prompt:
                messages = [{"role": "user", "content": p}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                model_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max(max_sequence_length, 1),
                )

                all_input_ids.append(model_inputs["input_ids"])
                all_attention_masks.append(model_inputs["attention_mask"])

            # Move to device
            text_encoder_device = next(self.text_encoder.parameters()).device
            input_ids = torch.cat(all_input_ids, dim=0).to(text_encoder_device)
            attention_mask = torch.cat(all_attention_masks, dim=0).to(text_encoder_device)

            # Forward pass
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # BFL uses layers [9, 18, 27] for Qwen3, NOT [10, 20, 30]!
            OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
            out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
            prompt_embeds = rearrange(out, "b c l d -> b l (c d)")

            return prompt_embeds, None

        # Replace the Mistral method with Qwen3 method
        # Use types.MethodType to properly bind the function to the pipe instance
        import types
        pipe._get_mistral_prompt_embeds = types.MethodType(_get_qwen3_prompt_embeds, pipe)

        # Override encode_prompt to use larger default max_sequence_length for variable image sizes
        original_encode_prompt = pipe.encode_prompt
        def encode_prompt_with_large_default(
            self,  # Need self parameter since this is bound as a method
            prompt,
            device=None,
            num_images_per_prompt=1,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            max_sequence_length=2048,  # Large default for Klein to handle all image sizes
            **kwargs
        ):
            return original_encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                max_sequence_length=max_sequence_length,
                **kwargs
            )
        pipe.encode_prompt = types.MethodType(encode_prompt_with_large_default, pipe)

        self.print_and_status_update("Preparing Klein Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        flush()
        # Only move text encoder to GPU if low_vram is disabled
        if not (hasattr(self.model_config, 'low_vram') and self.model_config.low_vram):
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # Attach layer offloading to transformer for 16GB VRAM optimization
        if (
            hasattr(self.model_config, 'layer_offloading') and
            self.model_config.layer_offloading and
            hasattr(self.model_config, 'layer_offloading_transformer_percent') and
            self.model_config.layer_offloading_transformer_percent > 0
        ):
            self.print_and_status_update(f"Enabling layer offloading: {self.model_config.layer_offloading_transformer_percent*100:.0f}% of transformer layers")
            from toolkit.memory_management.manager import MemoryManager
            MemoryManager.attach(
                pipe.transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        # Save to model class
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update(f"FLUX.2 Klein {model_size} Model Loaded")

    def get_base_model_version(self):
        return "flux2_klein"

    @property
    def requires_dynamic_text_embeddings(self):
        """Klein models require text embeddings to be generated per-batch to match varying image sizes"""
        return True

    def get_prompt_embeds(self, prompt: str, max_sequence_length: int = 4096, **kwargs):
        """
        Override to use larger default max_sequence_length for Klein models.
        Klein requires text embeddings that can accommodate various image sizes.
        The embeddings will be truncated to match actual image sequence length during generation.
        """
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        # Use large max_sequence_length by default to support all image sizes
        # Will be truncated to match image sequence length in generate_single_image
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, device=self.device_torch, max_sequence_length=max_sequence_length
        )
        from toolkit.prompt_utils import PromptEmbeds
        pe = PromptEmbeds(prompt_embeds)
        return pe

    def generate_single_image(
        self,
        pipeline,
        gen_config,
        conditional_embeds,
        unconditional_embeds,
        generator,
        extra: dict,
    ):
        """Override to truncate text embeddings for Klein before generation"""
        from PIL import Image
        import torch

        # DEBUG: Check LoRA status
        print("\n=== GENERATION DEBUG START ===")
        print(f"Network active: {self.network.is_active if hasattr(self, 'network') and self.network else 'No network'}")
        if hasattr(self, 'network') and self.network:
            print(f"Network multiplier: {self.network.multiplier}")
            print(f"Network has weights: {len(list(self.network.parameters())) > 0 if hasattr(self.network, 'parameters') else 'N/A'}")

        # Enable debug output in Klein model
        if hasattr(self.model, '_debug_generation'):
            self.model._debug_generation = True

        gen_config.width = (
            gen_config.width // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()
        gen_config.height = (
            gen_config.height // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()

        # Calculate image sequence length
        # After VAE encoding (8x downscale) and 2x2 packing (2x downscale)
        # Total downscale = 16x
        img_seq_len = (gen_config.height // 16) * (gen_config.width // 16)

        # Truncate text embeddings to match image sequence length
        # conditional_embeds.text_embeds shape: [batch, seq, hidden]
        original_seq_len = conditional_embeds.text_embeds.shape[1]
        if original_seq_len > img_seq_len:
            conditional_embeds.text_embeds = conditional_embeds.text_embeds[:, :img_seq_len, :]

        print(f"Text embeds shape: {conditional_embeds.text_embeds.shape}")
        print(f"Text embeds stats: min={conditional_embeds.text_embeds.min().item():.4f}, max={conditional_embeds.text_embeds.max().item():.4f}, mean={conditional_embeds.text_embeds.mean().item():.4f}")

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        print(f"Generation config: {gen_config.width}x{gen_config.height}, steps={gen_config.num_inference_steps}, guidance={gen_config.guidance_scale}")

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            control_img_list=control_img_list,
            **extra,
        ).images[0]

        # DEBUG: Check output image stats
        import numpy as np
        img_array = np.array(img)
        print(f"Output image shape: {img_array.shape}")
        print(f"Output image stats: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
        print(f"Output image unique values: {len(np.unique(img_array))}")
        print("=== GENERATION DEBUG END ===\n")

        # Disable debug output in Klein model
        if hasattr(self.model, '_debug_generation'):
            self.model._debug_generation = False

        return img

    def save_model(self, output_path, meta, save_dtype):
        """
        Save Klein model (override to use Flux2Klein type)
        """
        from toolkit.accelerator import unwrap_model
        from toolkit.metadata import get_meta_for_safetensors
        from safetensors.torch import save_file
        from optimum.quanto import QTensor

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"

        # only save the unet
        transformer: Flux2Klein = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name="flux2_klein")
        save_file(save_dict, output_path, metadata=meta)
