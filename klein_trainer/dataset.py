"""
Dataset handling for Klein trainer.
Supports bucketing, caching, and caption loading.
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm


@dataclass
class ImageItem:
    """Single image item in the dataset."""
    path: Path
    caption: str
    original_size: Tuple[int, int]  # (width, height)
    bucket_size: Tuple[int, int]  # (width, height) after bucketing
    latent: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None


class BucketManager:
    """Manages resolution buckets for aspect ratio preserving training."""

    def __init__(
        self,
        base_resolution: int = 512,
        min_resolution: int = 256,
        max_resolution: int = 1024,
        step: int = 64,
    ):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.step = step

        # Generate valid bucket sizes
        self.buckets = self._generate_buckets()

    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """Generate all valid bucket sizes maintaining ~same pixel count."""
        buckets = []
        base_pixels = self.base_resolution * self.base_resolution

        for w in range(self.min_resolution, self.max_resolution + 1, self.step):
            # Calculate height to maintain approximately same pixel count
            h = base_pixels // w
            # Round to nearest step
            h = (h // self.step) * self.step
            h = max(self.min_resolution, min(h, self.max_resolution))

            if (w, h) not in buckets:
                buckets.append((w, h))
            if (h, w) not in buckets and w != h:
                buckets.append((h, w))

        return sorted(set(buckets))

    def get_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """Find the best bucket for an image of given dimensions."""
        aspect_ratio = width / height

        best_bucket = self.buckets[0]
        best_diff = float('inf')

        for bucket in self.buckets:
            bucket_aspect = bucket[0] / bucket[1]
            diff = abs(aspect_ratio - bucket_aspect)
            if diff < best_diff:
                best_diff = diff
                best_bucket = bucket

        return best_bucket


class KleinDataset(Dataset):
    """
    Dataset for Klein LoRA training.

    Features:
    - Automatic caption loading from .txt files
    - Resolution bucketing for aspect ratio preservation
    - Latent caching support
    - Text embedding caching support
    """

    def __init__(
        self,
        dataset_path: str,
        caption_extension: str = ".txt",
        resolution: int = 512,
        enable_bucket: bool = True,
        min_bucket_resolution: int = 256,
        max_bucket_resolution: int = 1024,
        bucket_resolution_step: int = 64,
        default_caption: str = "",
    ):
        self.dataset_path = Path(dataset_path)
        self.caption_extension = caption_extension
        self.resolution = resolution
        self.enable_bucket = enable_bucket
        self.default_caption = default_caption

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        # Initialize bucket manager
        if enable_bucket:
            self.bucket_manager = BucketManager(
                base_resolution=resolution,
                min_resolution=min_bucket_resolution,
                max_resolution=max_bucket_resolution,
                step=bucket_resolution_step,
            )
        else:
            self.bucket_manager = None

        # Find all images
        self.items: List[ImageItem] = []
        self._scan_dataset()

        # Group items by bucket for efficient batching
        self.bucket_groups: Dict[Tuple[int, int], List[int]] = {}
        self._group_by_bucket()

        print(f"Dataset loaded: {len(self.items)} images in {len(self.bucket_groups)} buckets")
        for bucket, indices in sorted(self.bucket_groups.items()):
            print(f"  {bucket[0]}x{bucket[1]}: {len(indices)} images")

    def _scan_dataset(self):
        """Scan dataset directory for images and captions."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        for file_path in sorted(self.dataset_path.rglob('*')):
            if file_path.suffix.lower() not in image_extensions:
                continue

            # Get image dimensions
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Could not open image {file_path}: {e}")
                continue

            # Find caption
            caption_path = file_path.with_suffix(self.caption_extension)
            if caption_path.exists():
                caption = caption_path.read_text(encoding='utf-8').strip()
            else:
                caption = self.default_caption

            # Determine bucket size
            if self.bucket_manager:
                bucket_size = self.bucket_manager.get_bucket(width, height)
            else:
                bucket_size = (self.resolution, self.resolution)

            self.items.append(ImageItem(
                path=file_path,
                caption=caption,
                original_size=(width, height),
                bucket_size=bucket_size,
            ))

    def _group_by_bucket(self):
        """Group dataset items by bucket size for efficient batching."""
        self.bucket_groups = {}
        for idx, item in enumerate(self.items):
            if item.bucket_size not in self.bucket_groups:
                self.bucket_groups[item.bucket_size] = []
            self.bucket_groups[item.bucket_size].append(idx)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        """Get a single training item."""
        item = self.items[idx]

        result = {
            'caption': item.caption,
            'bucket_size': item.bucket_size,
            'path': str(item.path),
        }

        # Use cached latent if available, otherwise load image
        if item.latent is not None:
            result['latent'] = item.latent
        else:
            # Load and preprocess image only if latent not cached
            image = Image.open(item.path).convert('RGB')
            target_w, target_h = item.bucket_size
            image = self._resize_and_crop(image, target_w, target_h)
            image_tensor = transforms.ToTensor()(image)
            image_tensor = image_tensor * 2.0 - 1.0
            result['image'] = image_tensor

        # Include cached text embeds if available
        if item.text_embeds is not None:
            result['text_embeds'] = item.text_embeds

        return result

    def _resize_and_crop(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize image preserving aspect ratio, then center crop to exact size."""
        orig_w, orig_h = image.size

        # Calculate scale to fill target
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))

        return image

    def cache_latents(self, vae, device: torch.device, dtype: torch.dtype):
        """
        Pre-compute and cache all latents.

        Args:
            vae: KleinVAE instance
            device: Device to run encoding on
            dtype: Data type for encoding
        """
        from .vae import encode_images

        print("Caching latents...")
        vae = vae.to(device)

        for idx, item in enumerate(tqdm(self.items, desc="Encoding latents")):
            # Load image
            image = Image.open(item.path).convert('RGB')
            target_w, target_h = item.bucket_size
            image = self._resize_and_crop(image, target_w, target_h)

            # Convert to tensor
            image_tensor = transforms.ToTensor()(image)
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim

            # Encode
            with torch.no_grad():
                latent = encode_images(vae, image_tensor, device, dtype)

            # Store on CPU
            self.items[idx].latent = latent.squeeze(0).cpu()

        # Move VAE back to CPU to save VRAM
        vae.cpu()
        torch.cuda.empty_cache()

    def cache_text_embeddings(self, text_encoder, tokenizer, device: torch.device, dtype: torch.dtype):
        """
        Pre-compute and cache all text embeddings.

        Args:
            text_encoder: Qwen3 text encoder
            tokenizer: Qwen3 tokenizer
            device: Device to run encoding on
            dtype: Data type for encoding
        """
        from einops import rearrange

        print("Caching text embeddings...")

        # Klein uses layers [9, 18, 27] for Qwen3
        OUTPUT_LAYERS_QWEN3 = [9, 18, 27]

        # Collect unique captions
        unique_captions = list(set(item.caption for item in self.items))
        caption_to_embeds = {}

        # Get encoder device
        encoder_device = next(text_encoder.parameters()).device

        for caption in tqdm(unique_captions, desc="Encoding captions"):
            with torch.no_grad():
                # Format using chat template (required for correct Qwen3 encoding)
                messages = [{"role": "user", "content": caption}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                ).to(encoder_device)

                # Get embeddings
                outputs = text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # Stack layers [9, 18, 27] and rearrange to match encode_text output
                out = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
                embeds = rearrange(out, "b c l d -> b l (c d)")

                caption_to_embeds[caption] = embeds.squeeze(0).to(dtype).cpu()

        # Assign to items
        for item in self.items:
            item.text_embeds = caption_to_embeds[item.caption]

        # Move text encoder back to CPU
        text_encoder.cpu()
        torch.cuda.empty_cache()
        print(f"  Cached {len(unique_captions)} unique captions")


class BucketBatchSampler:
    """
    Batch sampler that groups items by bucket.
    Ensures all items in a batch have the same resolution.
    """

    def __init__(self, dataset: KleinDataset, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Get all bucket groups
        bucket_indices = list(self.dataset.bucket_groups.values())

        # Shuffle buckets
        if self.shuffle:
            random.shuffle(bucket_indices)
            for indices in bucket_indices:
                random.shuffle(indices)

        # Yield batches
        for indices in bucket_indices:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        for indices in self.dataset.bucket_groups.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


def create_dataloader(
    dataset: KleinDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with bucket-aware batching."""
    sampler = BucketBatchSampler(dataset, batch_size, shuffle=shuffle)

    def collate_fn(batch):
        """Collate function for bucketed batches."""
        captions = [item['caption'] for item in batch]
        bucket_size = batch[0]['bucket_size']  # All same in a batch
        paths = [item['path'] for item in batch]  # Include paths for per-sample tracking

        result = {
            'captions': captions,
            'bucket_size': bucket_size,
            'paths': paths,
        }

        # Include images only if not using cached latents
        if 'image' in batch[0]:
            result['images'] = torch.stack([item['image'] for item in batch])

        # Include latents if cached
        if 'latent' in batch[0] and batch[0]['latent'] is not None:
            result['latents'] = torch.stack([item['latent'] for item in batch])

        # Include text embeds if cached
        if 'text_embeds' in batch[0] and batch[0]['text_embeds'] is not None:
            result['text_embeds'] = torch.stack([item['text_embeds'] for item in batch])

        return result

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
