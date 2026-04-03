"""Pipeline factories and guard registration.

``create_pipeline`` is the single entry-point for loading any supported model;
it dispatches on ``model_type`` so callers never need to import diffusers
pipeline classes directly.
"""

from __future__ import annotations

from typing import Optional

import torch
from diffusers import (
    CogVideoXPipeline,
    DDIMScheduler,
    FluxPipeline,
    HunyuanVideoPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)

from .artifact import GuardArtifact
from .steering import MultiLayerSparseSteeringController


# ─────────────────────────────────────────────────────────────────────────────
# Per-model loaders
# ─────────────────────────────────────────────────────────────────────────────

def create_sd_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float16,
):
    """Load a Stable Diffusion 1.x pipeline with a DDIM scheduler."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_sd3_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a Stable Diffusion 3 pipeline.

    Uses ``bfloat16`` by default (recommended for SD3 on Ampere/Ada GPUs).
    Keeps the model's native FlowMatchEulerDiscreteScheduler.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_flux_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a FLUX.1 pipeline (dev or schnell).

    Uses ``bfloat16`` by default.  The guidance embedding mode
    (dev vs schnell) is inferred automatically from the model config.
    """
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_cogvideox_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    devices: Optional[list] = None,
):
    """Load a CogVideoX pipeline.

    Uses ``bfloat16`` by default.  CogVideoX is a 3D DiT-based
    text-to-video model with T5-XXL text encoder.

    Parameters
    ----------
    devices :
        Optional list of torch.device for model parallelism.
        If provided with multiple devices, the model will be distributed
        across GPUs using device_map="auto".
    """
    if devices and len(devices) > 1:
        # For large video models, manually create device_map to split transformer
        print(f"Loading CogVideoX with multi-GPU support ({len(devices)} GPUs)...")
        
        gpu_ids = [d.index if hasattr(d, 'index') else int(str(d).split(':')[-1]) for d in devices]
        
        # CogVideoX-2B architecture:
        # - text_encoder (T5-XXL): ~10GB
        # - transformer (30 blocks): ~8GB  
        # - vae: ~1GB
        # We need to distribute transformer blocks across multiple GPUs
        
        # Manual device map
        device_map = {}
        
        # Text encoder on GPU 0
        device_map['text_encoder'] = gpu_ids[0]
        
        # VAE on last GPU
        device_map['vae'] = gpu_ids[-1]
        
        # Transformer: split blocks across ALL provided GPUs
        # CogVideoX-2B has 30 transformer blocks
        num_transformer_blocks = 30
        transformer_gpus = gpu_ids  # Use ALL provided GPUs
        blocks_per_gpu = num_transformer_blocks // len(transformer_gpus)
        
        # Map transformer components
        device_map['transformer.patch_embed'] = transformer_gpus[0]
        device_map['transformer.time_embed'] = transformer_gpus[0]
        device_map['transformer.embedding_norm'] = transformer_gpus[0]
        
        for i in range(num_transformer_blocks):
            gpu_idx = i // blocks_per_gpu
            if gpu_idx >= len(transformer_gpus):
                gpu_idx = len(transformer_gpus) - 1
            target_gpu = transformer_gpus[gpu_idx]
            device_map[f'transformer.transformer_blocks.{i}'] = target_gpu
        
        device_map['transformer.norm_final'] = transformer_gpus[-1]
        device_map['transformer.proj_out'] = transformer_gpus[-1]
        
        print(f"  Device map strategy:")
        print(f"    Text encoder -> GPU {gpu_ids[0]}")
        print(f"    VAE -> GPU {gpu_ids[-1]}")
        print(f"    Transformer blocks distributed across ALL GPUs {transformer_gpus}")
        
        # Print block distribution
        for gpu_id in transformer_gpus:
            blocks_on_gpu = blocks_per_gpu if gpu_id != transformer_gpus[-1] else (num_transformer_blocks - blocks_per_gpu * (len(transformer_gpus)-1))
            print(f"      GPU {gpu_id}: {blocks_on_gpu} transformer blocks")
        
        # Load on CPU first
        print("  Loading model on CPU (this may take a minute)...")
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
        
        # Dispatch transformer to multiple GPUs using accelerate
        import accelerate
        import torch.nn as nn
        
        # Analyze transformer structure
        print("  Analyzing transformer structure...")
        transformer_children = dict(pipe.transformer.named_children())
        print(f"    Top-level components: {list(transformer_children.keys())}")
        
        # Check if transformer_blocks is a ModuleList
        if isinstance(transformer_children.get('transformer_blocks'), nn.ModuleList):
            block_list = transformer_children['transformer_blocks']
            num_actual_blocks = len(block_list)
            block_names = [f'transformer_blocks.{i}' for i in range(num_actual_blocks)]
            print(f"    Found ModuleList with {num_actual_blocks} blocks")
        else:
            # Fallback: treat as single block
            num_actual_blocks = 1
            block_names = ['transformer_blocks']
            print(f"    WARNING: transformer_blocks is not a ModuleList, treating as single block")
        
        blocks_per_gpu = max(1, num_actual_blocks // len(transformer_gpus))
        print(f"    Distributing {blocks_per_gpu} blocks per GPU across {transformer_gpus}")
        
        # Create device_map - ALL on transformer_gpus to avoid cross-device errors
        transformer_device_map = {}
        
        # Map individual transformer blocks
        for i, block_name in enumerate(block_names):
            gpu_idx = min(i // blocks_per_gpu, len(transformer_gpus) - 1)
            target_gpu = transformer_gpus[gpu_idx]
            transformer_device_map[block_name] = target_gpu
            print(f"      {block_name} -> GPU {target_gpu}")
        
        # Map ALL other transformer components to first transformer GPU
        # This avoids cross-device errors
        first_gpu = transformer_gpus[0]
        last_gpu = transformer_gpus[-1]
        
        other_mappable = ['patch_embed', 'time_proj', 'time_embedding', 'embedding_dropout', 
                         'embedding_norm', 'norm_final', 'norm_out', 'proj_out']
        
        for name in other_mappable:
            if name in transformer_children:
                if name in ['norm_final', 'norm_out', 'proj_out']:
                    transformer_device_map[name] = last_gpu
                else:
                    transformer_device_map[name] = first_gpu
        
        print(f"  Dispatching transformer to GPUs {transformer_gpus}...")
        
        pipe.transformer = accelerate.dispatch_model(
            pipe.transformer,
            device_map=transformer_device_map,
            offload_buffers=False,
        )
        
        # CRITICAL: Move text_encoder to SAME GPU as first transformer component
        # to avoid cross-device errors during encoding
        text_encoder_gpu = transformer_gpus[0]
        print(f"  Moving text_encoder to GPU {text_encoder_gpu} (same as transformer)...")
        pipe.text_encoder = pipe.text_encoder.to(f'cuda:{text_encoder_gpu}')
        
        # VAE can stay on last GPU (only used at the end)
        print(f"  Moving vae to GPU {last_gpu}...")
        pipe.vae = pipe.vae.to(f'cuda:{last_gpu}')
        
        torch.cuda.empty_cache()
        print(f"✓ Loaded CogVideoX with multi-GPU distribution")
        print(f"  Layout: text_encoder=GPU{text_encoder_gpu}, transformer=GPUs{transformer_gpus}, vae=GPU{last_gpu}")
    else:
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device)
    
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_hunyuanvideo_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    devices: Optional[list] = None,
):
    """Load a HunyuanVideo pipeline.

    Uses ``bfloat16`` by default.  HunyuanVideo is a dual-stream
    DiT-based text-to-video model with LLAMA + CLIP encoders.

    Parameters
    ----------
    devices :
        Optional list of torch.device for model parallelism.
        If provided with multiple devices, the model will be distributed
        across GPUs using device_map="auto".
    """
    if devices and len(devices) > 1:
        print(f"Loading HunyuanVideo with multi-GPU support ({len(devices)} GPUs)...")
        print("  Note: HunyuanVideo multi-GPU not yet fully tested, using single GPU")
        # For now, use first GPU only
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(str(devices[0]))
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(device)
    
    pipe.set_progress_bar_config(disable=False)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Unified factory
# ─────────────────────────────────────────────────────────────────────────────

_DTYPE_DEFAULTS = {
    "sd1": torch.float16,
    "sd3": torch.bfloat16,
    "flux": torch.bfloat16,
    "cogvideox": torch.bfloat16,
    "hunyuanvideo": torch.bfloat16,
}


def create_pipeline(
    model_type: str,
    model_id: str,
    device: str,
    torch_dtype: Optional[torch.dtype] = None,
    devices: Optional[list] = None,
):
    """Unified pipeline factory dispatching on *model_type*.

    Parameters
    ----------
    model_type : ``"sd1"`` | ``"sd3"`` | ``"flux"`` | ``"cogvideox"`` | ``"hunyuanvideo"``
    model_id :
        HuggingFace Hub model ID or local path.
    device :
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    torch_dtype :
        Override the default dtype for the model type.
    devices :
        Optional list of devices for model parallelism (video models only).
    """
    key = model_type.lower()
    dtype = torch_dtype or _DTYPE_DEFAULTS.get(key, torch.float16)

    if key == "sd1":
        return create_sd_pipeline(model_id, device, dtype)
    elif key == "sd3":
        return create_sd3_pipeline(model_id, device, dtype)
    elif key == "flux":
        return create_flux_pipeline(model_id, device, dtype)
    elif key == "cogvideox":
        return create_cogvideox_pipeline(model_id, device, dtype, devices=devices)
    elif key == "hunyuanvideo":
        return create_hunyuanvideo_pipeline(model_id, device, dtype, devices=devices)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}.  "
            "Valid choices: sd1, sd3, flux, cogvideox, hunyuanvideo"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Guard registration
# ─────────────────────────────────────────────────────────────────────────────

def register_guard(
    pipe,
    artifact: GuardArtifact,
    device: Optional[torch.device] = None,
    alpha: Optional[float] = None,
) -> MultiLayerSparseSteeringController:
    """Attach forward hooks that apply the learned steering vectors during
    inference and return a controller object.

    The controller exposes:

    * ``controller.set_enabled(bool)`` — toggle steering per-step.
    * ``controller.remove()``          — detach all hooks when done.

    Works for SD1, SD3, and FLUX pipelines without any model-specific logic
    because the hook attachment goes through
    :func:`~cad_ssv_guard.steering._get_backbone`, which handles both
    ``pipe.unet`` and ``pipe.transformer`` transparently.
    """
    controller = MultiLayerSparseSteeringController.from_artifact(
        artifact=artifact,
        device=device,
        alpha=alpha,
    )
    return controller.register(pipe)
