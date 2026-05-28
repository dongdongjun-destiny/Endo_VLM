#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral RFT (Reinforcement Fine-Tuning) with Standard GRPO.

Training is image-only OR video-only (no mixed batches). Use either:
  - rft_grpo_image_train.py
  - rft_grpo_video_train.py
  - this module with --modality image|video
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import re
from difflib import SequenceMatcher
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template


def _patch_unsloth_grpo_loss_slow_call():
    """Unsloth cache mismatch: compute_loss must pass sampling_per_token_logps as 4th positional arg."""
    import importlib
    import sys

    for mod_name in tuple(sys.modules.keys()) + (
        "unsloth_compiled_cache.UnslothGRPOTrainer",
    ):
        try:
            mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        except Exception:
            continue
        if not mod_name.endswith("UnslothGRPOTrainer"):
            continue
        fn = getattr(mod, "grpo_compute_loss_slow", None)
        if fn is None or getattr(fn, "_grpo_sig_patched", False):
            continue
        _orig = getattr(fn, "__wrapped__", fn)

        def _grpo_compute_loss_slow_compat(ref, new, old, arg3, arg4, arg5, arg6, **kwargs):
            if "sampling_per_token_logps" in kwargs:
                sampling = kwargs.pop("sampling_per_token_logps")
                return _orig(ref, new, old, sampling, arg3, arg4, arg5, arg6, **kwargs)
            return _orig(ref, new, old, arg3, arg4, arg5, arg6, **kwargs)

        _grpo_compute_loss_slow_compat._grpo_sig_patched = True
        mod.grpo_compute_loss_slow = _grpo_compute_loss_slow_compat


_patch_unsloth_grpo_loss_slow_call()

# 引入 Qwen 官方视觉处理工具，用于提取视频帧和图片
from qwen_vl_utils import process_vision_info

# ==============================================================================
# UNSLOTH GRPO VISION TENSOR RESCUE PATCH
# ==============================================================================
pristine_vision_tensors = {}
_original_grpo_prepare_inputs = GRPOTrainer._prepare_inputs

def _patched_prepare_inputs(self, inputs):
    global pristine_vision_tensors
    is_mapping_like = hasattr(inputs, "get") and hasattr(inputs, "__setitem__")
    if isinstance(inputs, dict):
        inputs = dict(inputs)

    if is_mapping_like:
        # TRL splits flattened visual patches by grid_thw.prod(). GRPO prompt
        # expansion can duplicate pixel_values while leaving grid rows unexpanded.
        # Fix the grid before TRL's split_pixel_values_by_grid runs.
        if (
            isinstance(inputs.get("pixel_values"), torch.Tensor)
            and isinstance(inputs.get("image_grid_thw"), torch.Tensor)
        ):
            pixel_rows = inputs["pixel_values"].size(0)
            grid_rows = inputs["image_grid_thw"].prod(dim=-1).sum().item()
            if grid_rows > 0 and pixel_rows != grid_rows and pixel_rows % grid_rows == 0:
                repeat = pixel_rows // grid_rows
                inputs["image_grid_thw"] = inputs["image_grid_thw"].repeat_interleave(repeat, dim=0)

        if (
            isinstance(inputs.get("pixel_values_videos"), torch.Tensor)
            and isinstance(inputs.get("video_grid_thw"), torch.Tensor)
        ):
            video_pixel_rows = inputs["pixel_values_videos"].size(0)
            video_grid_rows = inputs["video_grid_thw"].prod(dim=-1).sum().item()
            if video_grid_rows > 0 and video_pixel_rows != video_grid_rows and video_pixel_rows % video_grid_rows == 0:
                repeat = video_pixel_rows // video_grid_rows
                inputs["video_grid_thw"] = inputs["video_grid_thw"].repeat_interleave(repeat, dim=0)

    prepared = _original_grpo_prepare_inputs(self, inputs)
    pristine_vision_tensors.clear()

    # 这里的 hook 完美支持了 image 和 video 两种 tensor
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if is_mapping_like and key in inputs and isinstance(inputs[key], torch.Tensor):
            tensor = inputs[key].to(self.args.device)
            pristine_vision_tensors[key] = tensor
            prepared[key] = tensor
    return prepared

GRPOTrainer._prepare_inputs = _patched_prepare_inputs

# TRL GRPO: strict single-modality flow (image-only or video-only).
_original_generate_single_turn = GRPOTrainer._generate_single_turn
_original_generate_and_score = GRPOTrainer._generate_and_score_completions
_original_compute_loss = GRPOTrainer._compute_loss
_original_get_per_token_logps = GRPOTrainer._get_per_token_logps_and_entropies

_GRPO_VISION_KEYS = (
    "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
)
_GRPO_LOGPS_EXTRA_KEYS = _GRPO_VISION_KEYS + ("num_videos",)


def _grpo_modality() -> str:
    """image | video — must be explicitly set by entrypoint/CLI."""
    value = os.environ.get("GRPO_MODALITY", "").strip().lower()
    if value in {"image", "video"}:
        return value
    raise RuntimeError(
        "GRPO_MODALITY must be 'image' or 'video'. "
        "Use rft_grpo_image_train.py / rft_grpo_video_train.py "
        "or run this file with --modality image|video."
    )


def _grpo_is_image_only_modality() -> bool:
    return _grpo_modality() == "image"


def _grpo_is_video_only_modality() -> bool:
    return _grpo_modality() == "video"


def _grpo_first_not_none(*values):
    """Pick first non-None value without bool()-testing tensors."""
    for value in values:
        if value is not None:
            return value
    return None


def _grpo_collect_logps_extra(self) -> dict:
    """Vision tensors for ref/policy logprob forwards (generation + loss paths)."""
    extra = {}
    if getattr(self, "_grpo_video_for_logps", None):
        extra.update(self._grpo_video_for_logps)
    fk = getattr(self, "_grpo_forward_kwargs", None)
    if isinstance(fk, dict):
        for key in _GRPO_LOGPS_EXTRA_KEYS:
            if key not in extra and fk.get(key) is not None:
                extra[key] = fk[key]
    return extra


def _grpo_has_images(images_entry) -> bool:
    return images_entry is not None and images_entry != [] and images_entry != [None]


def _grpo_has_videos(videos_entry) -> bool:
    return videos_entry is not None and videos_entry != [] and videos_entry != [None]


def _grpo_normalize_video_entry(videos_entry):
    if videos_entry is None:
        return None
    if isinstance(videos_entry, list) and len(videos_entry) == 1:
        return videos_entry[0]
    return videos_entry


def _grpo_count_videos(videos_entry) -> int:
    if videos_entry is None:
        return 0
    if isinstance(videos_entry, list):
        return len(videos_entry)
    return 1


def _grpo_is_video_by_media_type(media_type) -> bool:
    if media_type is None:
        return False
    t = str(media_type).strip().lower()
    return t in ("video", "sequence")


def _grpo_is_video_path(path) -> bool:
    if not path:
        return False
    ext = os.path.splitext(str(path).split("?")[0].lower())[1]
    return ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")


def _grpo_align_to_prompt_count(values, prompt_count):
    """Align per-sample modality metadata with GRPO's expanded prompt list."""
    if values is None:
        return None
    values = list(values)
    if len(values) == prompt_count:
        return values
    if values and prompt_count % len(values) == 0:
        repeat = prompt_count // len(values)
        return [v for v in values for _ in range(repeat)]
    if values and len(values) < prompt_count:
        return values + [values[-1]] * (prompt_count - len(values))
    return values[:prompt_count]


def _grpo_get_model_config(model):
    """Unwrap PEFT/Unsloth wrappers enough to reach Qwen vision config."""
    candidates = [model]
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    for attr in ("base_model", "model"):
        obj = getattr(model, attr, None)
        if obj is not None:
            candidates.append(obj)
            nested = getattr(obj, "model", None)
            if nested is not None:
                candidates.append(nested)
    for obj in candidates:
        cfg = getattr(obj, "config", None)
        if cfg is not None and (
            hasattr(cfg, "video_token_id") or hasattr(cfg, "image_token_id")
        ):
            return cfg
    return getattr(model, "config", None)


def _grpo_get_merge_length(config) -> int:
    vision_config = getattr(config, "vision_config", None)
    merge_size = getattr(config, "spatial_merge_size", None)
    if merge_size is None and vision_config is not None:
        merge_size = getattr(vision_config, "spatial_merge_size", None)
    try:
        merge_size = int(merge_size or 2)
    except Exception:
        merge_size = 2
    return max(1, merge_size * merge_size)


def _grpo_feature_rows_from_grid(grid_thw, merge_length: int) -> int:
    if not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() == 0:
        return 0
    return int((grid_thw.prod(dim=-1) // merge_length).sum().item())


def _grpo_count_tokens_in_batch(input_ids_batch: torch.Tensor, token_id: int) -> int:
    if token_id is None or input_ids_batch is None:
        return 0
    return int((input_ids_batch == int(token_id)).sum().item())


def _grpo_sanitize_vision_model_inputs(
    model_inputs: dict, model, trainer_self=None, processor=None
) -> None:
    """Drop or trim vision tensors that do not match placeholder tokens in input_ids."""
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return

    if processor is None and trainer_self is not None:
        processor = getattr(trainer_self, "processing_class", None)
    cfg = _grpo_get_model_config(model)
    merge_length = _grpo_get_merge_length(cfg)

    image_token_id = getattr(cfg, "image_token_id", None)
    video_token_id = getattr(cfg, "video_token_id", None)
    if processor is not None:
        if image_token_id is None:
            try:
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            except Exception:
                pass
        if video_token_id is None:
            try:
                video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            except Exception:
                pass

    modality = _grpo_modality()
    image_token_count = _grpo_count_tokens_in_batch(input_ids, image_token_id)
    video_token_count = _grpo_count_tokens_in_batch(input_ids, video_token_id)

    if modality == "video":
        model_inputs.pop("pixel_values", None)
        model_inputs.pop("image_grid_thw", None)
        model_inputs.pop("pixel_attention_mask", None)
        model_inputs.pop("image_sizes", None)
    if modality == "image":
        model_inputs.pop("pixel_values_videos", None)
        model_inputs.pop("video_grid_thw", None)
        model_inputs.pop("second_per_grid_ts", None)

    if image_token_count == 0:
        model_inputs.pop("pixel_values", None)
        model_inputs.pop("image_grid_thw", None)
        model_inputs.pop("pixel_attention_mask", None)
        model_inputs.pop("image_sizes", None)
    if video_token_count == 0:
        model_inputs.pop("pixel_values_videos", None)
        model_inputs.pop("video_grid_thw", None)
        model_inputs.pop("second_per_grid_ts", None)

    # Align feature rows with pad token counts when both are present.
    if (
        image_token_count > 0
        and "pixel_values" in model_inputs
        and "image_grid_thw" in model_inputs
    ):
        image_feature_rows = _grpo_feature_rows_from_grid(model_inputs["image_grid_thw"], merge_length)
        if image_feature_rows > 0 and image_token_count != image_feature_rows:
            if image_token_count > image_feature_rows and image_token_count % image_feature_rows == 0:
                repeat = image_token_count // image_feature_rows
                model_inputs["pixel_values"] = model_inputs["pixel_values"].repeat((repeat, 1))
                model_inputs["image_grid_thw"] = model_inputs["image_grid_thw"].repeat_interleave(repeat, dim=0)
            else:
                raw_rows = image_token_count * merge_length
                model_inputs["pixel_values"] = model_inputs["pixel_values"][:raw_rows]
                model_inputs["image_grid_thw"] = _grpo_make_truncated_grid(
                    image_token_count, merge_length, model_inputs["pixel_values"].device
                )

    if (
        video_token_count > 0
        and "pixel_values_videos" in model_inputs
        and "video_grid_thw" in model_inputs
    ):
        grid_rows = model_inputs["video_grid_thw"].size(0)
        if grid_rows > video_token_count:
            model_inputs["video_grid_thw"] = model_inputs["video_grid_thw"][:video_token_count]
        video_feature_rows = _grpo_feature_rows_from_grid(model_inputs["video_grid_thw"], merge_length)
        if video_feature_rows > 0 and video_token_count != video_feature_rows:
            if video_token_count > video_feature_rows and video_token_count % video_feature_rows == 0:
                repeat = video_token_count // video_feature_rows
                model_inputs["pixel_values_videos"] = model_inputs["pixel_values_videos"].repeat((repeat, 1))
                model_inputs["video_grid_thw"] = model_inputs["video_grid_thw"].repeat_interleave(repeat, dim=0)
            else:
                raw_rows = video_token_count * merge_length
                model_inputs["pixel_values_videos"] = model_inputs["pixel_values_videos"][:raw_rows]
                model_inputs["video_grid_thw"] = _grpo_make_truncated_grid(
                    video_token_count, merge_length, model_inputs["pixel_values_videos"].device
                )


def _grpo_make_truncated_grid(token_count: int, merge_length: int, device) -> torch.Tensor:
    """Create a small valid grid whose merged feature count equals token_count."""
    # For the common merge_length=4 case, [1, 2, token_count*2] gives
    # prod/4 == token_count and keeps h/w divisible by spatial merge size.
    side = int(round(merge_length ** 0.5))
    side = max(1, side)
    return torch.tensor([[1, side, max(side, token_count * side)]], device=device, dtype=torch.long)


def _grpo_prompt_contains_video(prompt_entry) -> bool:
    """Best-effort check: whether a chat prompt content includes a video node."""
    try:
        if isinstance(prompt_entry, str):
            text = prompt_entry.lower()
            # Qwen chat template commonly inserts video special tokens in text prompts.
            return (
                "<|video_pad|>" in text
                or "video_pad" in text
                or "<video>" in text
                or "</video>" in text
            )
        if isinstance(prompt_entry, dict):
            if prompt_entry.get("type") == "video":
                return True
            for v in prompt_entry.values():
                if _grpo_prompt_contains_video(v):
                    return True
            return False
        if isinstance(prompt_entry, list):
            return any(_grpo_prompt_contains_video(x) for x in prompt_entry)
        return False
    except Exception:
        return False


def _merge_forward_kwargs_list(fk_list: List[dict]) -> dict:
    merged = {}
    for key in _GRPO_VISION_KEYS:
        parts = [fk[key] for fk in fk_list if key in fk and fk[key] is not None]
        if not parts:
            continue
        if all(isinstance(p, torch.Tensor) for p in parts):
            merged[key] = torch.cat(parts, dim=0)
        else:
            merged[key] = parts[0]
    return merged


def _grpo_prompt_to_text(processor, prompt) -> str:
    """
    Match SFT/inference template path to keep multimodal tokenization consistent.
    """
    try:
        return processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return maybe_apply_chat_template({"prompt": prompt}, processor)["prompt"]


def _grpo_video_metadata_for_path(video_path: Optional[str]) -> dict:
    """Build a transformers VideoMetadata-compatible dict (file paths only)."""
    meta = {"total_num_frames": 1, "fps": 24.0}
    if not video_path:
        return meta
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frames > 0:
                meta["total_num_frames"] = frames
            if fps > 0:
                meta["fps"] = fps
                if frames > 0:
                    meta["duration"] = frames / fps
            if width > 0:
                meta["width"] = width
            if height > 0:
                meta["height"] = height
        cap.release()
    except Exception:
        pass
    return meta


def _grpo_prepare_prompt_inputs_for_generate(self, prompt_inputs):
    """
    Prepare processor outputs for generation without calling Trainer._prepare_inputs,
    because Unsloth's _prepare_inputs may recurse into _generate_and_score_completions.
    """
    global pristine_vision_tensors
    device = self.accelerator.device
    moved = {}
    for k, v in prompt_inputs.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v

    pristine_vision_tensors.clear()
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key in moved and isinstance(moved[key], torch.Tensor):
            pristine_vision_tensors[key] = moved[key]
    return moved


def _grpo_generate_video_batch(self, prompts, videos_list, fps_list):
    device = self.accelerator.device
    proc = self.processing_class
    video_token = getattr(proc, "video_token", "<|video_pad|>")
    prompts_text = [_grpo_prompt_to_text(proc, p) for p in prompts]

    # Qwen3-VL indexes video_metadata once per <|video_pad|> in all prompts combined.
    aligned_videos = []
    aligned_metadata = []
    for text, video_entry in zip(prompts_text, videos_list or []):
        video_count = text.count(video_token) or 1
        v = _grpo_normalize_video_entry(video_entry)
        video_path = None
        if isinstance(v, str):
            video_path = v
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            video_path = v[0]
        meta = _grpo_video_metadata_for_path(video_path)
        entry = video_path if video_path else v
        for _ in range(video_count):
            aligned_videos.append(entry)
            aligned_metadata.append(dict(meta))

    if not aligned_videos:
        return [], [], {}

    kwargs = {"videos": aligned_videos, "video_metadata": aligned_metadata}
    prompt_inputs = self.processing_class(
        text=prompts_text, padding=True, return_tensors="pt", **kwargs
    )
    prompt_inputs = _grpo_prepare_prompt_inputs_for_generate(self, prompt_inputs)
    forward_kwargs = {
        k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")
    }
    with torch.no_grad():
        prompt_completion_ids = self.model.generate(
            **prompt_inputs,
            generation_config=self.generation_config,
            disable_compile=True,
        )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    prompt_length = prompt_ids.size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]
    is_eos = completion_ids == self.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
    completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
    return prompt_ids, completion_ids, forward_kwargs


def _grpo_open_image(path: str) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
        if im.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        return im
    except Exception:
        return Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))


def _grpo_generate_image_batch(self, prompts, images_list):
    device = self.accelerator.device
    prompts_text = [_grpo_prompt_to_text(self.processing_class, p) for p in prompts]
    aligned_images = []
    total_image_tokens = sum(text.count("<|image_pad|>") for text in prompts_text)
    media_paths = getattr(self, "_grpo_per_media_path", None)
    for i, (text, image_entry) in enumerate(zip(prompts_text, images_list)):
        image_count = text.count("<|image_pad|>") or 1
        if isinstance(image_entry, list):
            imgs = [im for im in image_entry if im is not None]
        elif image_entry is not None:
            imgs = [image_entry]
        else:
            imgs = []
        if not imgs and media_paths and i < len(media_paths) and media_paths[i]:
            imgs = [_grpo_open_image(media_paths[i])]
        if imgs:
            if len(imgs) < image_count:
                imgs = imgs + [imgs[-1]] * (image_count - len(imgs))
            aligned_images.extend(imgs[:image_count])
    if total_image_tokens > 0 and not aligned_images:
        aligned_images = [Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))] * total_image_tokens
    images_list = aligned_images if aligned_images else images_list
    prompt_inputs = self.processing_class(
        text=prompts_text, images=images_list, padding=True, return_tensors="pt"
    )
    # Hard guard: image-only generation must not carry Qwen video tokens.
    # Some TRL/Unsloth paths can leave a video token in input_ids even after
    # modality splitting; Qwen then expects video_grid_thw and crashes.
    cfg = _grpo_get_model_config(self.model)
    video_token_id = getattr(cfg, "video_token_id", None)
    image_token_id = getattr(cfg, "image_token_id", None)
    if video_token_id is None:
        try:
            video_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        except Exception:
            video_token_id = None
    if image_token_id is None:
        try:
            image_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            image_token_id = None
    if (
        video_token_id is not None
        and image_token_id is not None
        and isinstance(prompt_inputs, dict)
        and "input_ids" in prompt_inputs
        and isinstance(prompt_inputs["input_ids"], torch.Tensor)
    ):
        video_mask = prompt_inputs["input_ids"] == int(video_token_id)
        if video_mask.any().item():
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"].masked_fill(
                video_mask, int(image_token_id)
            )

    prompt_inputs = _grpo_prepare_prompt_inputs_for_generate(self, prompt_inputs)
    forward_kwargs = {
        k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")
    }
    with torch.no_grad():
        prompt_completion_ids = self.model.generate(
            **prompt_inputs,
            generation_config=self.generation_config,
            disable_compile=True,
        )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    prompt_length = prompt_ids.size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]
    is_eos = completion_ids == self.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
    completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
    return prompt_ids, completion_ids, forward_kwargs


def _grpo_generate_one_from_messages(self, prompt, image_entry=None, video_entry=None):
    """Generate one sample using the same message -> vision path as SFT/inference."""
    device = self.accelerator.device
    try:
        prompt_text = self.processing_class.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"]
    try:
        image_inputs, video_inputs = process_vision_info(prompt)
        if image_inputs is not None:
            image_inputs = [im for im in image_inputs if im is not None]
        if video_inputs is not None:
            video_inputs = [vid for vid in video_inputs if vid is not None]
    except Exception:
        # Dataset/GRPO may serialize PIL objects in prompt content as None.
        # Fall back to the side-channel fields rebuilt in _patched_generate_and_score_completions.
        image_inputs = image_entry if _grpo_has_images(image_entry) else None
        video_inputs = video_entry if _grpo_has_videos(video_entry) else None

    if not image_inputs and _grpo_has_images(image_entry):
        image_inputs = image_entry
    if not video_inputs and _grpo_has_videos(video_entry):
        video_inputs = video_entry

    # Qwen processor expects the number of visual inputs to match the number
    # of visual tokens already rendered into prompt_text. GRPO expansion can
    # duplicate placeholders while side-channel media remains length 1.
    image_token_count = prompt_text.count("<|image_pad|>")
    video_token_count = prompt_text.count("<|video_pad|>")
    # Qwen3-VL may inject video/image pads during processor(), not in template text.
    if image_inputs and not image_token_count and not _grpo_is_image_only_modality():
        image_inputs = None
    if video_inputs and not video_token_count and not _grpo_is_video_only_modality():
        video_inputs = None
    if image_token_count and image_inputs:
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]
        if len(image_inputs) < image_token_count:
            image_inputs = image_inputs + [image_inputs[-1]] * (image_token_count - len(image_inputs))
        elif len(image_inputs) > image_token_count:
            image_inputs = image_inputs[:image_token_count]
    if video_token_count and video_inputs:
        if not isinstance(video_inputs, list):
            video_inputs = [video_inputs]
        if len(video_inputs) < video_token_count:
            video_inputs = video_inputs + [video_inputs[-1]] * (video_token_count - len(video_inputs))
        elif len(video_inputs) > video_token_count:
            video_inputs = video_inputs[:video_token_count]

    prompt_inputs = self.processing_class(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = _grpo_prepare_prompt_inputs_for_generate(self, prompt_inputs)
    forward_kwargs = {
        k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")
    }
    with torch.no_grad():
        prompt_completion_ids = self.model.generate(
            **prompt_inputs,
            generation_config=self.generation_config,
            disable_compile=True,
        )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    prompt_length = prompt_ids.size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]
    is_eos = completion_ids == self.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
    completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
    return prompt_ids[0], completion_ids[0], forward_kwargs

def _patched_generate_single_turn(self, prompts, images):
    per_images = getattr(self, "_grpo_per_images", images)
    per_videos = getattr(self, "_grpo_per_videos", None)
    per_fps = getattr(self, "_grpo_per_fps", None)
    per_media_type = getattr(self, "_grpo_per_media_type", None)

    n = len(prompts)
    per_images = _grpo_align_to_prompt_count(per_images, n)
    per_videos = _grpo_align_to_prompt_count(per_videos, n)
    per_fps = _grpo_align_to_prompt_count(per_fps, n)
    per_media_type = _grpo_align_to_prompt_count(per_media_type, n)

    if _grpo_is_image_only_modality():
        out_p, out_c, fk = _grpo_generate_image_batch(self, prompts, per_images)
        self._grpo_forward_kwargs = fk
        return out_p, out_c, None, fk

    if _grpo_is_video_only_modality():
        # Same path as SFT/inference: decode video from chat messages via process_vision_info.
        out_p, out_c, fk_list = [], [], []
        for i, prompt in enumerate(prompts):
            vid = per_videos[i] if per_videos and i < len(per_videos) else None
            p_ids, c_ids, fk = _grpo_generate_one_from_messages(self, prompt, None, vid)
            out_p.append(p_ids)
            out_c.append(c_ids)
            if fk:
                fk_list.append(fk)
        merged_fk = _merge_forward_kwargs_list(fk_list)
        self._grpo_forward_kwargs = merged_fk
        return out_p, out_c, None, merged_fk

    sample_has_video = []
    for i in range(n):
        vid = per_videos[i] if per_videos and i < len(per_videos) else None
        mtype = per_media_type[i] if per_media_type and i < len(per_media_type) else None
        sample_has_video.append(_grpo_is_video_by_media_type(mtype) or _grpo_has_videos(vid))

    if not any(sample_has_video):
        out_p, out_c, fk = _grpo_generate_image_batch(self, prompts, per_images)
        self._grpo_forward_kwargs = fk
        return out_p, out_c, None, fk

    out_p, out_c, fk_list = [], [], []
    for i, prompt in enumerate(prompts):
        im = per_images[i] if per_images and i < len(per_images) else None
        vid = per_videos[i] if per_videos and i < len(per_videos) else None
        p_ids, c_ids, fk = _grpo_generate_one_from_messages(self, prompt, im, vid)
        out_p.append(p_ids)
        out_c.append(c_ids)
        if fk:
            fk_list.append(fk)

    merged_fk = _merge_forward_kwargs_list(fk_list)
    self._grpo_forward_kwargs = merged_fk
    return out_p, out_c, None, merged_fk


def _patched_generate_and_score_completions(self, inputs):
    self._grpo_per_images = [ex.get("images") for ex in inputs]
    self._grpo_per_videos = [ex.get("videos") for ex in inputs]
    self._grpo_per_fps = [ex.get("fps") for ex in inputs]
    self._grpo_per_media_type = [ex.get("media_type", ex.get("type")) for ex in inputs]
    self._grpo_per_media_path = [ex.get("media_path") for ex in inputs]
    self._grpo_forward_kwargs = None

    fixed_inputs = []
    for ex, im, vid, mtype, mpath in zip(
        inputs,
        self._grpo_per_images,
        self._grpo_per_videos,
        self._grpo_per_media_type,
        self._grpo_per_media_path,
    ):
        row = dict(ex)
        if _grpo_is_image_only_modality():
            is_video = False
        elif _grpo_is_video_only_modality():
            is_video = True
        else:
            is_video = (
                _grpo_is_video_by_media_type(mtype)
                or _grpo_has_videos(vid)
                or _grpo_is_video_path(mpath)
            )
        if is_video:
            row["media_type"] = "video"
            row["images"] = []
            row["videos"] = vid if _grpo_has_videos(vid) else [mpath]
        else:
            row["media_type"] = "image"
            row["images"] = im if im is not None else []
            row["videos"] = []
        fixed_inputs.append(row)

    self._grpo_per_images = [ex.get("images") for ex in fixed_inputs]
    self._grpo_per_videos = [ex.get("videos") for ex in fixed_inputs]
    self._grpo_per_media_type = [ex.get("media_type") for ex in fixed_inputs]
    self._grpo_num_videos = [_grpo_count_videos(v) for v in self._grpo_per_videos]

    self._grpo_video_for_logps = None
    try:
        out = _original_generate_and_score(self, fixed_inputs)
    finally:
        self._grpo_video_for_logps = None

    if self._grpo_forward_kwargs:
        for key in _GRPO_VISION_KEYS:
            if key in self._grpo_forward_kwargs:
                out[key] = self._grpo_forward_kwargs[key]
    out["num_videos"] = self._grpo_num_videos
    return out


def _patched_get_per_token_logps_and_entropies(
    self,
    model,
    input_ids,
    attention_mask,
    logits_to_keep,
    batch_size=None,
    compute_entropy=False,
    pixel_values=None,
    image_grid_thw=None,
    num_images=None,
    num_videos=None,
    pixel_attention_mask=None,
    image_sizes=None,
    token_type_ids=None,
    **kwargs,
):
    extra = _grpo_collect_logps_extra(self)
    modality = _grpo_modality()

    # Ref logps: prefer vision tensors from our generation path, not stale TRL batch fields.
    if modality == "video":
        pixel_values = None
        image_grid_thw = None
        num_images = None
        pixel_values_videos = _grpo_first_not_none(
            extra.get("pixel_values_videos"), kwargs.pop("pixel_values_videos", None)
        )
        video_grid_thw = _grpo_first_not_none(
            extra.get("video_grid_thw"), kwargs.pop("video_grid_thw", None)
        )
        second_per_grid_ts = _grpo_first_not_none(
            extra.get("second_per_grid_ts"), kwargs.pop("second_per_grid_ts", None)
        )
        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
    elif modality == "image":
        pixel_values_videos = None
        video_grid_thw = None
        second_per_grid_ts = None
        num_videos = None
        pixel_values = kwargs.get("pixel_values") if pixel_values is None else pixel_values
        image_grid_thw = kwargs.get("image_grid_thw") if image_grid_thw is None else image_grid_thw
        kwargs.pop("pixel_values_videos", None)
        kwargs.pop("video_grid_thw", None)
    else:
        pixel_values_videos = extra.get("pixel_values_videos")
        video_grid_thw = extra.get("video_grid_thw")
        second_per_grid_ts = extra.get("second_per_grid_ts")

    if num_videos is None:
        num_videos = extra.get("num_videos")

    batch_size = batch_size or input_ids.size(0)
    all_logps, all_entropies = [], []

    if isinstance(num_images, torch.Tensor):
        num_images = num_images.tolist()
    if num_images is not None:
        num_images = [int(x) for x in num_images]
        if (
            image_grid_thw is not None
            and len(num_images) == input_ids.size(0)
            and sum(num_images) != image_grid_thw.size(0)
        ):
            # GRPO expands prompts by num_generations; visual inputs may be
            # duplicated accordingly while num_images stays unexpanded.
            if image_grid_thw.size(0) % input_ids.size(0) == 0:
                per_sample = image_grid_thw.size(0) // input_ids.size(0)
                num_images = [per_sample] * input_ids.size(0)
            else:
                image_sample_count = sum(1 for x in num_images if x > 0)
                if image_sample_count > 0 and image_grid_thw.size(0) % image_sample_count == 0:
                    per_image_sample = image_grid_thw.size(0) // image_sample_count
                    num_images = [per_image_sample if x > 0 else 0 for x in num_images]

    if (
        num_videos is None
        and video_grid_thw is not None
        and video_grid_thw.size(0) == input_ids.size(0)
    ):
        num_videos = [1] * input_ids.size(0)
    if isinstance(num_videos, torch.Tensor):
        num_videos = num_videos.tolist()
    if num_videos is not None:
        num_videos = [int(x) for x in num_videos]
        if (
            video_grid_thw is not None
            and len(num_videos) == input_ids.size(0)
            and sum(num_videos) != video_grid_thw.size(0)
        ):
            # Same expansion issue as images: num_videos is videos per sample,
            # not frames per video. It must sum to video_grid_thw rows.
            if video_grid_thw.size(0) % input_ids.size(0) == 0:
                per_sample = video_grid_thw.size(0) // input_ids.size(0)
                num_videos = [per_sample] * input_ids.size(0)
            else:
                video_sample_count = sum(1 for x in num_videos if x > 0)
                if video_sample_count > 0 and video_grid_thw.size(0) % video_sample_count == 0:
                    per_video_sample = video_grid_thw.size(0) // video_sample_count
                    num_videos = [per_video_sample if x > 0 else 0 for x in num_videos]

    for start in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[start : start + batch_size]
        attention_mask_batch = attention_mask[start : start + batch_size]
        model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

        if image_grid_thw is not None and pixel_values is not None and num_images is not None:
            rows_per_image = image_grid_thw.prod(dim=-1)
            rows_per_sample = torch.split(rows_per_image, num_images)
            rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
            cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
            row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
            if row_end > row_start and img_end > img_start:
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
        elif pixel_values is not None:
            pixel_slice = pixel_values[start : start + batch_size]
            if pixel_slice.numel() > 0:
                model_inputs["pixel_values"] = pixel_slice

        if (
            pixel_values_videos is not None
            and video_grid_thw is not None
            and num_videos is not None
            and len(num_videos) == input_ids.size(0)
        ):
            rows_per_video = video_grid_thw.prod(dim=-1)
            rows_per_sample = torch.split(rows_per_video, num_videos)
            rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
            cum_video_rows = torch.cat(
                [torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)]
            )
            video_row_start = cum_video_rows[start].item()
            video_row_end = cum_video_rows[start + batch_size].item()

            cum_videos = torch.tensor([0] + num_videos, device=video_grid_thw.device).cumsum(0)
            video_start = cum_videos[start]
            video_end = cum_videos[start + batch_size]
            if video_row_end > video_row_start and video_end > video_start:
                model_inputs["pixel_values_videos"] = pixel_values_videos[video_row_start:video_row_end]
                model_inputs["video_grid_thw"] = video_grid_thw[video_start:video_end]
        else:
            # Video-only: never attach an unsliced TRL video tensor (wrong grid/token count).
            if modality != "video":
                if pixel_values_videos is not None and pixel_values_videos.numel() > 0:
                    model_inputs["pixel_values_videos"] = pixel_values_videos
                if video_grid_thw is not None and video_grid_thw.numel() > 0:
                    model_inputs["video_grid_thw"] = video_grid_thw
        if second_per_grid_ts is not None:
            model_inputs["second_per_grid_ts"] = second_per_grid_ts
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        model_inputs["use_cache"] = False

        _grpo_sanitize_vision_model_inputs(model_inputs, model, self)

        logits = model(**model_inputs).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :] / self.temperature
        completion_ids = input_ids_batch[:, -logits_to_keep:]
        from trl.trainer.utils import selective_log_softmax, entropy_from_logits
        all_logps.append(selective_log_softmax(logits, completion_ids))
        if compute_entropy:
            with torch.no_grad():
                all_entropies.append(entropy_from_logits(logits))

    logps = torch.cat(all_logps, dim=0)
    entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
    return logps, entropies


def _patched_compute_loss(self, model, inputs):
    self._grpo_video_for_logps = {
        k: inputs[k]
        for k in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts", "num_videos")
        if k in inputs and inputs[k] is not None
    }
    try:
        return _original_compute_loss(self, model, inputs)
    finally:
        self._grpo_video_for_logps = None


GRPOTrainer._generate_and_score_completions = _patched_generate_and_score_completions
GRPOTrainer._generate_single_turn = _patched_generate_single_turn
GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_per_token_logps_and_entropies
GRPOTrainer._compute_loss = _patched_compute_loss
# ==============================================================================

from config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    SYSTEM_PROMPT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SEED,
    RFT_CONFIG, REWARD_CONFIG, get_grpo_train_config,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    AVAILABLE_STATIONS, POSITION_LABELS,
    build_user_prompt, build_target_text,
    parse_prediction, normalize_appearance, normalize_station,
    appearance_match,
    get_output_dirs, get_generation_config, is_thinking_model,
    setup_processor_image_size,
    build_train_video_content,
    normalize_training_sample, get_training_user_text,
)
from processor_utils import ensure_full_processor

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="EndoVLA-Oral RFT Training (Image & Video)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="", help="Optional if paths in jsonl are absolute")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--model", type=str, default="qwen3_8b_instruct", choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--runname", type=str, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--num_generations", type=int, default=None, help="Generations per prompt for GRPO")
    parser.add_argument("--temperature", type=float, default=None, help="GRPO rollout temperature (default from config, ~0.4 for clinical)")
    parser.add_argument("--top_p", type=float, default=None, help="GRPO rollout top_p")
    parser.add_argument("--top_k", type=int, default=None, help="GRPO rollout top_k")
    parser.add_argument("--max_prompt_length", type=int, default=None, help="GRPO max prompt tokens (lower saves VRAM)")
    parser.add_argument("--max_completion_length", type=int, default=None, help="GRPO max completion tokens")
    parser.add_argument("--doctor_pref", action="store_true", help="Enable doctor preference reward (pair or long chosen text only)")
    parser.add_argument(
        "--no_doctor_pref",
        action="store_true",
        help="Force-disable doctor preference even if enable_doctor_preference is True in config",
    )
    parser.add_argument("--doctor_pref_weight", type=float, default=None, help="Override doctor preference reward weight")

#自定义打分任务
    parser.add_argument("--task", type=str, default="gastrohun", choices=["legacy_endoscopy", "gastrohun"], help="Specify the task to use the corresponding reward mechanism")
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "video"],
        default=None,
        help="Required when running this file directly: image-only or video-only GRPO.",
    )

    parser.add_argument(
        "--reward_style",
        type=str,
        choices=["short", "cot"],
        default=None,
        help="GRPO format/accuracy rules: short (video/multimodal) or cot (image CoT1). "
        "Default: short for --modality video, cot for --modality image.",
    )
    parser.add_argument(
        "--no_gpu_guard",
        action="store_true",
        help="Disable GPU watchdog (do not block new GPU training jobs)",
    )

    return parser.parse_args()


def _apply_grpo_cli_overrides(config: dict, args) -> dict:
    """Merge CLI overrides into GRPO config (returns same dict, mutated)."""
    if args is None:
        return config
    if args.temperature is not None:
        config["grpo_temperature"] = args.temperature
    if args.top_p is not None:
        config["grpo_top_p"] = args.top_p
    if args.top_k is not None:
        config["grpo_top_k"] = args.top_k
    if args.max_prompt_length is not None:
        config["max_prompt_length"] = args.max_prompt_length
    if args.max_completion_length is not None:
        config["max_completion_length"] = args.max_completion_length
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_generations is not None:
        config["num_generations"] = args.num_generations
    if args.reward_style is not None:
        config["reward_style"] = args.reward_style
    return config


def _build_grpo_config(args, modality: str) -> dict:
    reward_style = args.reward_style
    cfg = get_grpo_train_config(modality=modality, reward_style=reward_style)
    return _apply_grpo_cli_overrides(cfg, args)


def filter_grpo_train_samples(
    samples: List[Dict[str, Any]], modality: str, max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Keep one modality only; apply max_samples after filtering."""
    if modality == "image":
        out = [s for s in samples if s.get("type") == "image"]
    elif modality == "video":
        out = [s for s in samples if s.get("type") in {"video", "sequence"}]
    else:
        raise ValueError(f"Unknown GRPO modality: {modality!r}")
    if max_samples:
        random.shuffle(out)
        out = out[:max_samples]
    return out


def run_grpo_training(modality: str, wandb_group: str, banner: str) -> None:
    """Shared entry: load data for one modality, eval, GRPO train, eval."""
    os.environ["GRPO_MODALITY"] = modality
    os.environ["WANDB_PROJECT"] = RFT_CONFIG["wandb_project"]
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    args = parse_args()
    print("\n" + "=" * 70 + f"\n{banner}\n" + "=" * 70)
    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME) if args.model else BASE_MODEL_NAME

    all_samples = load_rft_data(args.data_path, args.image_dir, max_samples=None)
    train_samples = filter_grpo_train_samples(all_samples, modality, args.max_samples)
    label = "Image" if modality == "image" else "Video"
    print(f"{label}-only samples: {len(train_samples)}")
    if not train_samples:
        print(f"ERROR: No {modality} samples found after filtering.")
        sys.exit(1)

    model, processor, training_mode = setup_model(model_name, args.checkpoint, args.mode)

    from gpu_guard import activate_gpu_guard_from_config
    activate_gpu_guard_from_config(enabled=not args.no_gpu_guard)

    print("\n--- Pre-RL Evaluation ---")
    pre_results = evaluate_model(
        model,
        processor,
        train_samples,
        model_name=model_name,
        num_samples=min(5, len(train_samples)),
        task_name=args.task,
    )
    print(f"Pre-RL full accuracy: {pre_results['full_accuracy']:.2%}")

    train_rft(model, processor, train_samples, args=args, config=_build_grpo_config(args, modality), training_mode=training_mode)

    print("\n--- Post-RL Evaluation ---")
    post_results = evaluate_model(
        model,
        processor,
        train_samples,
        model_name=model_name,
        num_samples=min(10, len(train_samples)),
        task_name=args.task,
    )
    print(f"Post-RL full accuracy: {post_results['full_accuracy']:.2%}")


# ==============================================================================

# ==============================================================================
# ==============================================================================
# DATA LOADING 
# ==============================================================================

import os
import json
from typing import Dict, List, Any, Optional
import cv2
import os

def check_media_validity(item, image_dir=""):
    """物理级验证图片或视频文件是否完整、未损坏"""
    try:
        # 1. 验证图片样本
        if item.get("type") == "image" or "image_path" in item:
            img_path = item.get("image_path")
            if not os.path.isabs(img_path) and image_dir:
                img_path = os.path.join(image_dir, img_path)
            
            if not os.path.exists(img_path):
                return False, "File not found"
                
            with Image.open(img_path) as img:
                img.verify() # 检查文件头是否损坏
            return True, ""

        # 2. 验证视频/序列样本
        elif item.get("type") in ("sequence", "video") or "video_path" in item:
            vid_path = item.get("video_path")
            if not os.path.isabs(vid_path) and image_dir:
                vid_path = os.path.join(image_dir, vid_path)
                
            if not os.path.exists(vid_path):
                return False, "File not found"
                
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                return False, "Corrupted video container"
            ret, _ = cap.read()
            cap.release()
            
            if not ret:
                return False, "Cannot decode video frames"
            return True, ""
            
        return True, "" 
        
    except Exception as e:
        return False, str(e)


def load_rft_data(data_path, image_dir="", max_samples=None):
    """加载数据并自动剔除坏死文件"""
    valid_samples = []
    bad_count = 0
    
    print(f"验证数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="验证媒体完整性"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                is_valid, err_msg = check_media_validity(item, image_dir)
                
                if is_valid:
                    sample = normalize_training_sample(item, image_dir)
                    if sample is not None:
                        valid_samples.append(sample)
                else:
                    bad_count += 1
                    
            except json.JSONDecodeError:
                bad_count += 1
            
    n_img = sum(1 for s in valid_samples if s["type"] == "image")
    n_vid = sum(1 for s in valid_samples if s["type"] == "video")
    print(f"有效样本: {len(valid_samples)} 条 (图片 {n_img}, 视频 {n_vid})，已跳过损坏/无效: {bad_count} 条。")
    
    if max_samples:
        random.shuffle(valid_samples)
        valid_samples = valid_samples[:max_samples]
        
    return valid_samples


# ==============================================================================
# MODEL SETUP
# ==============================================================================
def setup_model(model_name: str = BASE_MODEL_NAME, checkpoint_path: Optional[str] = None,
                training_mode: int = 2) -> tuple:
    print("\n" + "=" * 70 + "\nSetting up model\n" + "=" * 70)

    load_path = model_name if training_mode == 1 else checkpoint_path
    model, processor = FastVisionModel.from_pretrained(
        model_name=load_path, load_in_4bit=False, load_in_8bit=False, torch_dtype=torch.bfloat16, device_map="cuda:0", use_gradient_checkpointing="unsloth",
    )

    processor = ensure_full_processor(processor, load_path)
    processor = setup_processor_image_size(processor)

    if training_mode in [1, 3]:
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", random_state=SEED, use_dora=True
        )

    # 视觉张量逃生舱 (处理多模态 padding 逻辑)
    def vision_rescue_forward_pre_hook(module, args, kwargs):
        global pristine_vision_tensors
        device = next(module.parameters()).device if hasattr(module, "parameters") else torch.device("cuda:0")

        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0: input_ids = args[0]
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        # Image-only GRPO: rewrite stray video placeholder tokens before forward.
        # Video-only GRPO must keep <|video_pad|> or get_rope_index/image indexing breaks.
        video_token_id = None
        image_token_id = None
        modality = _grpo_modality()
        try:
            cfg = _grpo_get_model_config(module)
            video_token_id = getattr(cfg, "video_token_id", None)
            image_token_id = getattr(cfg, "image_token_id", None)
            if video_token_id is None:
                video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            if image_token_id is None:
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if (
                modality != "video"
                and input_ids is not None
                and video_token_id is not None
                and image_token_id is not None
                and kwargs.get("video_grid_thw") is None
            ):
                video_mask = input_ids == int(video_token_id)
                if video_mask.any().item():
                    new_input_ids = input_ids.masked_fill(video_mask, int(image_token_id))
                    if "input_ids" in kwargs:
                        kwargs["input_ids"] = new_input_ids
                    elif len(args) > 0:
                        args = (new_input_ids,) + tuple(args[1:])
                    input_ids = new_input_ids
        except Exception:
            pass

        has_video_tokens = False
        try:
            has_video_tokens = (
                input_ids is not None
                and video_token_id is not None
                and (input_ids == int(video_token_id)).any().item()
            )
        except Exception:
            has_video_tokens = False

        if modality == "image":
            kwargs["pixel_values_videos"] = None
            kwargs["video_grid_thw"] = None
            kwargs["second_per_grid_ts"] = None
        elif has_video_tokens:
            for k in ["pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]:
                if kwargs.get(k) is None and k in pristine_vision_tensors:
                    kwargs[k] = pristine_vision_tensors[k]
            if modality == "video":
                kwargs["pixel_values"] = None
                kwargs["image_grid_thw"] = None
        else:
            kwargs["pixel_values_videos"] = None
            kwargs["video_grid_thw"] = None
            kwargs["second_per_grid_ts"] = None

        has_image_tokens = False
        try:
            image_token_id = image_token_id or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            has_image_tokens = (
                input_ids is not None
                and image_token_id is not None
                and (input_ids == int(image_token_id)).any().item()
            )
        except Exception:
            has_image_tokens = False

        if modality == "video":
            kwargs["pixel_values"] = None
            kwargs["image_grid_thw"] = None
            pixel_values = None
            image_grid_thw = None
        elif has_image_tokens:
            for k in ["pixel_values", "image_grid_thw"]:
                if kwargs.get(k) is None and k in pristine_vision_tensors:
                    kwargs[k] = pristine_vision_tensors[k]
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            if modality == "image":
                kwargs["pixel_values_videos"] = None
                kwargs["video_grid_thw"] = None
                kwargs["second_per_grid_ts"] = None
        else:
            kwargs["pixel_values"] = None
            kwargs["image_grid_thw"] = None
            pixel_values = None
            image_grid_thw = None

        # Final dynamic alignment right before the compiled graph sees inputs.
        # Qwen compares visual placeholder token counts against merged visual
        # feature rows (grid_thw.prod / merge_length), not raw patch rows.
        try:
            cfg = _grpo_get_model_config(module)
            merge_length = _grpo_get_merge_length(cfg)
            if (
                input_ids is not None
                and image_token_id is not None
                and isinstance(kwargs.get("pixel_values"), torch.Tensor)
                and isinstance(kwargs.get("image_grid_thw"), torch.Tensor)
            ):
                img_tokens = int((input_ids == int(image_token_id)).sum().item())
                img_features = _grpo_feature_rows_from_grid(kwargs["image_grid_thw"], merge_length)
                if img_tokens == 0:
                    kwargs["pixel_values"] = None
                    kwargs["image_grid_thw"] = None
                elif img_features > 0 and img_tokens != img_features:
                    if img_tokens > img_features and img_tokens % img_features == 0:
                        repeat = img_tokens // img_features
                        kwargs["pixel_values"] = kwargs["pixel_values"].repeat((repeat, 1))
                        kwargs["image_grid_thw"] = kwargs["image_grid_thw"].repeat_interleave(repeat, dim=0)
                    else:
                        raw_rows = img_tokens * merge_length
                        kwargs["pixel_values"] = kwargs["pixel_values"][:raw_rows]
                        kwargs["image_grid_thw"] = _grpo_make_truncated_grid(img_tokens, merge_length, device)
        except Exception:
            pass

        try:
            cfg = _grpo_get_model_config(module)
            merge_length = _grpo_get_merge_length(cfg)
            if (
                input_ids is not None
                and video_token_id is not None
                and isinstance(kwargs.get("pixel_values_videos"), torch.Tensor)
                and isinstance(kwargs.get("video_grid_thw"), torch.Tensor)
            ):
                vid_tokens = int((input_ids == int(video_token_id)).sum().item())
                vid_features = _grpo_feature_rows_from_grid(kwargs["video_grid_thw"], merge_length)
                if vid_tokens == 0:
                    kwargs["pixel_values_videos"] = None
                    kwargs["video_grid_thw"] = None
                elif vid_features > 0 and vid_tokens != vid_features:
                    if vid_tokens > vid_features and vid_tokens % vid_features == 0:
                        repeat = vid_tokens // vid_features
                        kwargs["pixel_values_videos"] = kwargs["pixel_values_videos"].repeat((repeat, 1))
                        kwargs["video_grid_thw"] = kwargs["video_grid_thw"].repeat_interleave(repeat, dim=0)
                    else:
                        raw_rows = vid_tokens * merge_length
                        kwargs["pixel_values_videos"] = kwargs["pixel_values_videos"][:raw_rows]
                        kwargs["video_grid_thw"] = _grpo_make_truncated_grid(vid_tokens, merge_length, device)
        except Exception:
            pass

        _grpo_sanitize_vision_model_inputs(kwargs, module, processor=processor)
        return args, kwargs

    model.register_forward_pre_hook(vision_rescue_forward_pre_hook, with_kwargs=True)
    if hasattr(model, "base_model"): model.base_model.register_forward_pre_hook(vision_rescue_forward_pre_hook, with_kwargs=True)

    return model, processor, training_mode


# ==============================================================================
# INFERENCE & EVAL
# ==============================================================================
def run_inference(model, processor, sample: Dict[str, Any], model_name: str = BASE_MODEL_NAME) -> str:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    FastVisionModel.for_inference(model)

    sys_prompt = sample.get("system_instruction", SYSTEM_PROMPT)
    user_prompt = get_training_user_text(sample)

    if sample["type"] == "video":
        media_content = build_train_video_content(sample["media_path"])
    else:
        image = Image.open(sample["media_path"]).convert("RGB")
        media_content = {"type": "image", "image": image}

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [media_content, {"type": "text", "text": user_prompt}]}
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, 
        padding=True, return_tensors="pt"
    ).to("cuda")
    
    gen_config = get_generation_config(model_name, for_eval=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=gen_config.get("max_new_tokens", 256),
                                 temperature=gen_config.get("temperature", 0.1), use_cache=True)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if "assistant" in output_text.lower(): output_text = output_text.split("assistant")[-1].strip(": \n")
    return output_text.strip()


def evaluate_model(model, processor, samples: List[Dict[str, Any]], model_name: str = BASE_MODEL_NAME,
                   num_samples: int = 20, task_name: str = "gastrohun") -> Dict[str, Any]: # 👈 加了 task_name 参数
    FastVisionModel.for_inference(model)
    eval_samples = random.sample(samples, num_samples) if len(samples) > num_samples else samples
    results = {"total": 0, "label_correct": 0, "appearance_correct": 0, "station_correct": 0, "full_correct": 0}

    for sample in eval_samples:
        try:
            output = run_inference(model, processor, sample, model_name)
            pred = parse_prediction(output)
            results["total"] += 1
            
            target_label = str(sample.get("target_label", "")).strip().upper()
            
            # 👑 针对 GastroHUN 任务的评分标准 (只看 Label)
            if task_name == "gastrohun":
                if pred and pred["label"].upper() == target_label:
                    results["label_correct"] += 1
                    results["full_correct"] += 1
                elif target_label and target_label in output.upper():
                    results["label_correct"] += 1
                    results["full_correct"] += 1
                    
            # 📸 针对 legacy_endoscopy 旧版任务的评分标准 (必须三者全对)
            else:
                if pred is not None:
                    label_ok = pred["label"].upper() == target_label
                    appearance_ok = appearance_match(pred["appearance"], sample.get("gt_appearance", ""))
                    station_ok = normalize_station(pred["station"]) == sample.get("gt_station", "").lower()

                    if label_ok: results["label_correct"] += 1
                    if appearance_ok: results["appearance_correct"] += 1
                    if station_ok: results["station_correct"] += 1
                    if label_ok and appearance_ok and station_ok: results["full_correct"] += 1

        except Exception as e:
            print(f"Eval Error: {e}")
            continue


    t = results["total"]
    if t > 0:
        results.update({
            "label_accuracy": results["label_correct"] / t, 
            "appearance_accuracy": results["appearance_correct"] / t,
            "station_accuracy": results["station_correct"] / t, 
            "full_accuracy": results["full_correct"] / t
        })
    else:
        results.update({"label_accuracy": 0.0, "appearance_accuracy": 0.0, "station_accuracy": 0.0, "full_accuracy": 0.0})
    return results


# ==============================================================================
# GRPO DATASET CREATION
# ==============================================================================
# def create_grpo_dataset(samples: List[Dict[str, Any]], processor, task_name: str) -> Dataset:
#     data_list = []
#     for sample in samples:
#         sys_prompt = sample.get("system_instruction", SYSTEM_PROMPT)
#         user_prompt = build_user_prompt(sample["oral_instruction"])

#         if sample["type"] == "video":
#             media_content = {
#                 "type": "video", "video": sample["media_path"], 
#                 "fps": TRAIN_VIDEO_FPS, "max_pixels": TRAIN_VIDEO_MAX_PIXELS
#             }
#             messages = [
#                 {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
#                 {"role": "user", "content": [media_content, {"type": "text", "text": user_prompt}]},
#             ]
#             data_list.append({
#                 "prompt": messages,
#                 "target_label": sample["target_label"],
#                 "gt_appearance": sample["gt_appearance"],
#                 "gt_station": sample["gt_station"],
#                 "task": task_name,
#                 "media_path": sample["media_path"],#视频路径传给grpo图像打分
#             })
#         else:
#             try:
#                 image = Image.open(sample["media_path"]).convert("RGB")
#                 if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT): 
#                     image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
#             except Exception:
#                 image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

#             messages = [
#                 {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
#                 {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
#             ]
#             data_list.append({
#                 "prompt": messages,
#                 "images": [image], # TRL GRPO expects image obj here for image tasks
#                 "target_label": sample["target_label"],
#                 "gt_appearance": sample["gt_appearance"],
#                 "gt_station": sample["gt_station"],
#                 "task": task_name,
#                 "media_path": sample["media_path"],#图片路径传给grpo图像打分
#             })
#     return Dataset.from_list(data_list)



def create_grpo_dataset(samples: List[Dict[str, Any]], processor, task_name: str) -> Dataset:
    data_list = []
    
    # 用 tqdm 包裹 samples，并添加描述文本
    for sample in tqdm(samples, desc=f"Creating GRPO dataset ({task_name})"):
        sys_prompt = sample.get("system_instruction", SYSTEM_PROMPT)
        user_prompt = get_training_user_text(sample)

        if sample["type"] == "video":
            media_content = build_train_video_content(sample["media_path"])
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [media_content, {"type": "text", "text": user_prompt}]},
            ]
            # IMPORTANT: do not decode video frames during dataset creation.
            # Keep only lightweight video path refs and let processor decode lazily
            # at generation/training time per batch.
            row = {
                "prompt": messages,
                "media_type": "video",
                "images": [],
                "target_label": sample["target_label"],
                "gt_appearance": sample["gt_appearance"],
                "gt_station": sample["gt_station"],
                "chosen": sample.get("chosen", ""),
                "rejected": sample.get("rejected", ""),
                "doctor_score": sample.get("doctor_score", None),
                "task": task_name,
                "media_path": sample["media_path"],
                "videos": [sample["media_path"]],
            }
            data_list.append(row)
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            ]
            data_list.append({
                "prompt": messages,
                "media_type": "image",
                "images": [],
                "videos": [],
                "target_label": sample["target_label"],
                "gt_appearance": sample["gt_appearance"],
                "gt_station": sample["gt_station"],
                "chosen": sample.get("chosen", ""),
                "rejected": sample.get("rejected", ""),
                "doctor_score": sample.get("doctor_score", None),
                "task": task_name,
                "media_path": sample["media_path"],
            })
            
    return Dataset.from_list(data_list)

#grpo图像打分方式，耗资源

# def visual_reward_fn(completions, **kwargs) -> List[float]:
#     """
#     带图像的高级视觉裁判
#     """
#     media_paths = kwargs.get("media_path", [])
#     target_labels = kwargs.get("target_label", [])
#     tasks = kwargs.get("task", []) 
#     rewards = []

#     for i, completion in enumerate(completions):
#         text = completion[0]["content"] if isinstance(completion, list) else str(completion)
#         current_task = tasks[i] if i < len(tasks) else "gastrohun"
#         img_path = media_paths[i] if i < len(media_paths) else None
#         gt_label = str(target_labels[i]).upper() if i < len(target_labels) else ""
        
#         reward = 0.0

#         # 确保图片路径存在
#         if img_path and os.path.exists(img_path) and current_task == "gastrohun":
#             try:
#                 # 1. 裁判亲自打开图片！
#                 image = Image.open(img_path).convert("RGB")
                
#                 # 2. 提取模型生成的文本标签
#                 words = re.findall(r'\b[A-Z0-9]+\b', text.upper())
#                 found_labels = [w for w in words if w in VALID_SSS_LABELS]
                
#                 if len(found_labels) == 1:
#                     pred_label = found_labels[0]
                    
#                     # ==========================================================
#                     # 3. 核心视觉打分逻辑 (你需要在这里填入你的算法)
#                     # ==========================================================
#                     # 举例 A：使用 CLIP 计算图文匹配度 (需要额外加载 CLIP 模型)
#                     # text_feature = clip_model.encode_text(f"This is a stomach image of station {pred_label}")
#                     # image_feature = clip_model.encode_image(image)
#                     # visual_score = cosine_similarity(text_feature, image_feature)
#                     # reward += visual_score
                    
#                     # 举例 B：如果你有一个专门判断胃镜特征的 ResNet 小模型
#                     # is_correct_view = resnet_judge(image, pred_label) 
#                     # if is_correct_view: reward += 1.0
                    
#                     # 举例 C：如果你是做目标检测 (Bounding Box)
#                     # 提取文本里的坐标，去和原图上的特征比对...
#                     # ==========================================================
                    
#                     # 暂时用文本逻辑兜底占位，防止你直接运行报错
#                     if pred_label == gt_label:
#                         reward += 1.0 
                        
#             except Exception as e:
#                 print(f"视觉裁判读取图片出错: {e}")

#         rewards.append(max(0.0, min(1.0, reward)))
#     return rewards




# SSS 协议规定的 23 个合法解剖学标签
VALID_SSS_LABELS = {
    "A1", "A2", "A3", "A4", "A5", "A6", 
    "G1", "G2", "G3", "G4", 
    "L1", "L2", "L3", "L4", "L5", "L6", 
    "P1", "P2", "P3", "P4", "P5", "P6", 
    "NA"
}

# Dataset "chosen" is often only the gold label; treat longer text as real doctor feedback.
DOCTOR_CHOSEN_MIN_CHARS = 32

_active_grpo_reward_style = "short"


def _set_grpo_reward_style(style: str) -> None:
    global _active_grpo_reward_style
    _active_grpo_reward_style = style if style in ("short", "cot") else "short"


def _extract_gastrohun_pred_label(text: str) -> Optional[str]:
    """Parse predicted SSS label; cot style uses the last valid code in long reasoning."""
    upper = (text or "").upper()
    if _active_grpo_reward_style == "cot":
        labels = re.findall(r"\b([AGLP][1-6]|NA)\b", upper)
        return labels[-1] if labels else None
    words = re.findall(r"\b[A-Z0-9]+\b", upper)
    found = [w for w in words if w in VALID_SSS_LABELS]
    return found[0] if len(found) == 1 else None


def _gastrohun_tail_label(text: str) -> Optional[str]:
    clean = (text or "").strip()
    tail = re.search(
        r"(?:SSS|Final\s+SSS|Final)\s*[:：]\s*([AGLP][1-6]|NA)\s*$",
        clean,
        re.IGNORECASE,
    )
    return tail.group(1).upper() if tail else None


def _extract_gastrohun_final_label(text: str) -> Optional[str]:
    """Get the final predicted SSS label from a completion.

    Rule: must explicitly declare the final label on the last line (Final/SSS).
    We do NOT accept earlier mentions as the final answer.
    """
    clean = (text or "").strip()
    tail = _gastrohun_tail_label(clean)
    return tail


def _extract_gastrohun_declared_label(text: str) -> Optional[str]:
    """Extract an explicitly declared final label anywhere in the completion.

    Accepts lines like:
      - Final: P5
      - Final SSS: P5
      - SSS: P5
      - Answer: P5
      - 最终: P5 / 结论: P5
    We scan from bottom to top and take the first match (closest to the end).
    """
    clean = (text or "").strip()
    if not clean:
        return None
    pattern = re.compile(
        r"(?:(?:final\s+sss|final|sss|answer|最终|结论))\s*[:：]\s*([AGLP][1-6]|NA)\s*$",
        re.IGNORECASE,
    )
    for line in reversed([ln.strip() for ln in clean.splitlines() if ln.strip()]):
        m = pattern.search(line)
        if m:
            return m.group(1).upper()
    return None


def _gastrohun_last_line_is_uncertain(text: str) -> bool:
    """Heuristic: last line must not present multiple candidate labels (e.g. 'P4 or P5')."""
    clean = (text or "").strip()
    if not clean:
        return False
    last = clean.splitlines()[-1].strip().lower()
    if " or " in last:
        return True
    # If the last line contains 2+ valid labels, treat it as multiple-choice.
    labels = re.findall(r"\b([aglp][1-6]|na)\b", last)
    return len(labels) >= 2


def _extract_gastrohun_biased_label(text: str) -> Optional[str]:
    """Extract a preferred label from 'leaning toward / more likely / 倾向' style statements.

    Used when the completion mentions multiple candidates but explicitly prefers one.
    We scan from bottom to top and take the first match (closest to the end).
    """
    clean = (text or "").strip()
    if not clean:
        return None
    # Common bias cues in English + Chinese.
    cue = r"(?:lean(?:ing)?\s*toward|lean\s*to|prefer|preferred|more\s*likely|most\s*likely|i\s*think|i\s*believe|choose|selected|倾向|更可能|更像|更符合|更支持|选择|我认为)"
    pattern = re.compile(
        rf"(?:{cue})\s*(?:to|是|为|[:：]|\s)*\s*([AGLP][1-6]|NA)\b",
        re.IGNORECASE,
    )
    for line in reversed([ln.strip() for ln in clean.splitlines() if ln.strip()]):
        m = pattern.search(line)
        if m:
            return m.group(1).upper()
    return None


def _extract_gastrohun_any_label(text: str) -> Optional[str]:
    """Fallback parse: last valid label anywhere in the text (used for shaping)."""
    upper = (text or "").upper()
    labels = re.findall(r"\b([AGLP][1-6]|NA)\b", upper)
    return labels[-1] if labels else None


def _extract_gastrohun_decisive_label(text: str) -> Optional[str]:
    """Parse a single decisive final label from a completion.

    Accepts:
    - an explicit declared label anywhere (Final/SSS/Answer/最终/结论), preferred.

    Rejects ambiguous outputs like 'P4 or P5' or mentioning multiple different labels.
    """
    clean = (text or "").strip()
    if not clean:
        return None
    # If the model explicitly declares a final/answer label, trust that as the decisive result,
    # even if other labels are mentioned in the reasoning.
    declared = _extract_gastrohun_declared_label(clean)
    if declared:
        return declared

    biased = _extract_gastrohun_biased_label(clean)
    if biased:
        return biased

    # Otherwise, require exactly one unique label in the whole completion.
    if _gastrohun_last_line_is_uncertain(clean):
        return None
    labels = re.findall(r"\b([AGLP][1-6]|NA)\b", clean.upper())
    if not labels:
        return None
    return labels[-1] if len(set(labels)) == 1 else None




def oral_format_reward(completions, **kwargs) -> List[float]:
    tasks = kwargs.get("task", []) 
    chosen_list = kwargs.get("chosen", [])
    rejected_list = kwargs.get("rejected", [])
    doctor_scores = kwargs.get("doctor_score", [])
    rewards = []

    for i, completion in enumerate(completions):
        if _has_doctor_preference(i, chosen_list, rejected_list, doctor_scores):
            # Doctor preference has higher priority for this sample.
            rewards.append(0.0)
            continue
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        current_task = tasks[i] if i < len(tasks) else "gastrohun" 
        reward = 0.0

        # 1: legacy_endoscopy 
        if current_task == "legacy_endoscopy":
            parsed = parse_prediction(text)
            if parsed is not None:
                # 兼容旧版的 a,b,c 以及新版的 SSS 标签 (A1-P6, NA)
                valid_format_labels = [l.lower() for l in POSITION_LABELS] + [l.lower() for l in VALID_SSS_LABELS]
                if parsed.get("label", "").lower() in valid_format_labels: 
                    reward += 0.4
                
                pred_station = normalize_station(parsed.get("station", ""))
                if pred_station in [s.lower() for s in AVAILABLE_STATIONS]: 
                    reward += 0.3
                    
                appearance = parsed.get("appearance", "").strip()
                if appearance and len(appearance.split()) >= 1: 
                    reward += 0.2
                    
                if len(text.strip()) < 200: 
                    reward += 0.1

        # 2: gastrohun 格式打分
        elif current_task == "gastrohun":
            style = _active_grpo_reward_style
            clean_text = text.strip()
            text_len = len(clean_text)
            lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
            lower_text = clean_text.lower()

            # Shaping: if model outputs any valid label anywhere, give a small base reward
            # so training doesn't get stuck at all-zero rewards before it learns the Final/SSS line.
            any_label = _extract_gastrohun_any_label(clean_text)
            if any_label:
                reward += 0.05

            # High score requires a single decisive label (Final/SSS tail preferred but not required).
            pred_label = _extract_gastrohun_decisive_label(clean_text)
            tail_label = _gastrohun_tail_label(clean_text)
            if pred_label:
                # decisive answer present
                reward += 0.30
                if tail_label is not None and tail_label == pred_label:
                    reward += 0.10

                if _gastrohun_last_line_is_uncertain(clean_text):
                    reward -= 0.2
                if len(lines) >= 2 and len(" ".join(lines[:-1])) >= 12:
                    reward += 0.2

                short_cues = ("because", "due to", "based on", "reason", "view", "landmark")
                cot_cues = short_cues + (
                    "step 1", "step 2", "final region", "mucosal", "pylorus", "cardia", "fundus"
                )
                if any(cue in lower_text for cue in (cot_cues if style == "cot" else short_cues)):
                    reward += 0.2

                if style == "short":
                    if text_len <= 5:
                        reward += 0.0
                    elif 12 <= text_len <= 180:
                        reward += 0.2
                    else:
                        reward += 0.1
                else:
                    # CoT-style: encourage multi-step reasoning structure before the final label.
                    step_hits = len(re.findall(r"\bstep\s*\d+\b", lower_text))
                    if step_hits >= 3:
                        reward += 0.2
                    elif step_hits >= 2:
                        reward += 0.15
                    elif step_hits >= 1:
                        reward += 0.1

                    if 80 <= text_len <= 3500:
                        reward += 0.2
                    elif text_len >= 40:
                        reward += 0.1
                    if "step 1" in lower_text and ("final sss" in lower_text or tail_label):
                        reward += 0.1

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def oral_accuracy_reward(completions, **kwargs) -> List[float]:
    target_labels = kwargs.get("target_label", [])
    gt_appearances = kwargs.get("gt_appearance", [])
    gt_stations = kwargs.get("gt_station", [])
    tasks = kwargs.get("task", []) 
    chosen_list = kwargs.get("chosen", [])
    rejected_list = kwargs.get("rejected", [])
    doctor_scores = kwargs.get("doctor_score", [])
    rewards = []
    
    for i, completion in enumerate(completions):
        if _has_doctor_preference(i, chosen_list, rejected_list, doctor_scores):
            # Doctor preference has higher priority for this sample.
            rewards.append(0.0)
            continue
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        current_task = tasks[i] if i < len(tasks) else "gastrohun"
        reward = 0.0
        
        gt_label = str(target_labels[i]).upper() if i < len(target_labels) else ""
        
        if current_task == "legacy_endoscopy":
            gt_app = gt_appearances[i] if i < len(gt_appearances) else ""
            gt_sta = str(gt_stations[i]).lower() if i < len(gt_stations) else ""
            parsed = parse_prediction(text)
            if parsed is not None and gt_app and gt_sta:
                if parsed["label"].upper() == gt_label: reward += 0.40
                if gt_app and appearance_match(parsed["appearance"], gt_app): reward += 0.35
                if normalize_station(parsed["station"]) == gt_sta: reward += 0.25
                
        #  gastrohun 精度打分
        elif current_task == "gastrohun":
            # Only care about the (single) decisive label; Final/SSS tail is preferred but not required.
            pred_label = _extract_gastrohun_decisive_label(text)

            if pred_label and gt_label:
                if pred_label == gt_label:
                    reward += 1.0

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def _completion_to_text(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else str(completion)


def _safe_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _has_doctor_preference(i: int, chosen_list, rejected_list, doctor_scores) -> bool:
    chosen = (chosen_list[i] if i < len(chosen_list) else "") or ""
    rejected = (rejected_list[i] if i < len(rejected_list) else "") or ""
    chosen = str(chosen).strip()
    rejected = str(rejected).strip()
    if chosen and rejected:
        return True
    if len(chosen) >= DOCTOR_CHOSEN_MIN_CHARS:
        return True
    # Label-only "chosen" (e.g. "P5") is not doctor preference; keep format/accuracy rewards.
    return False


def doctor_preference_reward(completions, **kwargs) -> List[float]:
    """
    Preference reward from doctor feedback:
    - chosen/rejected pair: prefer outputs closer to chosen and farther from rejected.
    - optional doctor_score in [0,1]: blended as additional supervision.
    """
    chosen_list = kwargs.get("chosen", [])
    rejected_list = kwargs.get("rejected", [])
    doctor_scores = kwargs.get("doctor_score", [])
    rewards = []

    for i, completion in enumerate(completions):
        text = _completion_to_text(completion)
        chosen = chosen_list[i] if i < len(chosen_list) else ""
        rejected = rejected_list[i] if i < len(rejected_list) else ""
        has_pref = _has_doctor_preference(i, chosen_list, rejected_list, doctor_scores)

        if not has_pref:
            rewards.append(0.0)
            continue

        reward = 0.0
        has_pair = bool(chosen) and bool(rejected)
        if has_pair:
            sim_chosen = _safe_similarity(text, chosen)
            sim_rejected = _safe_similarity(text, rejected)
            reward = 0.5 + 0.5 * (sim_chosen - sim_rejected)
        elif chosen:
            reward = _safe_similarity(text, chosen)

        if i < len(doctor_scores) and doctor_scores[i] is not None:
            try:
                score = float(doctor_scores[i])
                score = max(0.0, min(1.0, score))
                reward = 0.7 * reward + 0.3 * score
            except (TypeError, ValueError):
                pass

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards

# ==============================================================================
# TRAINING
# ==============================================================================
def train_rft(model, processor, train_samples: List[Dict[str, Any]], args=None, config=None, training_mode: int = 2):
    print("\n" + "=" * 70 + "\nStarting EndoVLA-Oral RFT Training with Standard GRPO\n" + "=" * 70)

    config = dict(config or get_grpo_train_config())
    _set_grpo_reward_style(str(config.get("reward_style", "short")))

    num_epochs = args.epochs if args.epochs else config["num_epochs"]
    learning_rate = args.lr if args.lr else config["learning_rate"]
    batch_size = args.batch_size if args and args.batch_size else config["batch_size"]
    num_generations = args.num_generations if args.num_generations else config["num_generations"]
    if args and getattr(args, "no_doctor_pref", False):
        enable_doctor_pref = False
    else:
        enable_doctor_pref = bool(getattr(args, "doctor_pref", False)) or bool(
            config.get("enable_doctor_preference", False)
        )
    doctor_pref_weight = (
        args.doctor_pref_weight
        if args and args.doctor_pref_weight is not None
        else float(config.get("doctor_preference_weight", 0.3))
    )
    # Weights live in REWARD_CONFIG; .get fallback only if caller passes dict(RFT_CONFIG) without merge.
    format_weight = float(config.get("format_weight", REWARD_CONFIG["format_weight"]))
    accuracy_weight = float(config.get("accuracy_weight", REWARD_CONFIG["accuracy_weight"]))
    reward_weights = [format_weight, accuracy_weight]
    if enable_doctor_pref:
        reward_weights.append(1.0)  # doctor func already scaled by doctor_pref_weight
    print(
        f"GRPO rewards: oral_format + oral_accuracy (style={_active_grpo_reward_style})"
        + (f" + doctor_preference (weight={doctor_pref_weight:.3f})" if enable_doctor_pref else "")
        + f" | enable_doctor_preference={enable_doctor_pref}"
        + f" | reward_weights={reward_weights}"
    )

    if args and getattr(args, "output_dir", None):
        output_dir = args.output_dir
    elif args and getattr(args, "runname", None):
        output_dir = f"/home/ren9/yidong-code/exendovla/exvla/checkpoints/{args.runname}"
    else:
        output_dir = "/home/ren9/yidong-code/exendovla/exvla/checkpoints/default_rft_run"

    os.makedirs(output_dir, exist_ok=True)

    if args and getattr(args, "runname", None):
        model_save_dir = f"/home/ren9/yidong-code/exendovla/exvla/models/{args.runname}"
    else:
        model_save_dir = "/home/ren9/yidong-code/exendovla/exvla/models/default_rft_run"
        
    modality = _grpo_modality()
    if modality in ("image", "video"):
        print(f"GRPO modality lock: {modality} (no mixed image/video batches)")

    grpo_temperature = float(config.get("grpo_temperature", RFT_CONFIG["grpo_temperature"]))
    grpo_top_p = float(config.get("grpo_top_p", RFT_CONFIG["grpo_top_p"]))
    grpo_top_k = int(config.get("grpo_top_k", RFT_CONFIG["grpo_top_k"]))
    max_prompt_length = int(config.get("max_prompt_length", RFT_CONFIG["max_prompt_length"]))
    max_completion_length = int(config.get("max_completion_length", RFT_CONFIG["max_completion_length"]))
    print(
        f"GRPO sampling: temperature={grpo_temperature}, top_p={grpo_top_p}, top_k={grpo_top_k} "
        f"| max_prompt={max_prompt_length}, max_completion={max_completion_length} "
        f"| batch={batch_size}, num_generations={num_generations} "
        "(rollout only; eval uses INSTRUCT_GENERATION_CONFIG)"
    )
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if free_gb < 4.0:
            print(
                f"WARNING: only {free_gb:.1f} GiB GPU free. GRPO video/image is VRAM-heavy. "
                "Try --batch_size 1 --num_generations 2 --max_prompt_length 3072 "
                "and kill other GPU processes."
            )
        torch.cuda.empty_cache()

    print("\nCreating GRPO dataset...")
    train_dataset = create_grpo_dataset(train_samples, processor, task_name=args.task)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=num_epochs,
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim="adamw_torch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        report_to="wandb",
        run_name=f"endovla_oral_grpo_{args.runname}",
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        remove_unused_columns=False,
        temperature=grpo_temperature,
        top_p=grpo_top_p,
        top_k=grpo_top_k,
        reward_weights=reward_weights,
    )

    print("\nInitializing Standard GRPOTrainer...")
    reward_funcs = [oral_format_reward, oral_accuracy_reward]
    if enable_doctor_pref:
        print(
            f"Doctor preference reward enabled (weight={doctor_pref_weight:.3f}). "
            "Samples with doctor feedback use doctor reward first; others fallback to original rewards."
        )

        def weighted_doctor_preference_reward(completions, **kwargs):
            base_rewards = doctor_preference_reward(completions, **kwargs)
            return [r * doctor_pref_weight for r in base_rewards]

        reward_funcs.append(weighted_doctor_preference_reward)

    FastVisionModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
    )

    print("\nStarting GRPO reinforcement learning...")
    _patch_unsloth_grpo_loss_slow_call()

    if args.checkpoint and "checkpoint-" in args.checkpoint:
        print(f"断点续训，正在恢复优化器状态：{args.checkpoint}")
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        trainer.train()

    os.makedirs(model_save_dir, exist_ok=True)
    if args.merge:
        model.save_pretrained_merged(model_save_dir, processor, save_method="merged_16bit")
    else:
        model.save_pretrained(model_save_dir)
        processor.save_pretrained(model_save_dir)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    if args.modality not in ("image", "video"):
        print(
            "ERROR: Mixed image+video GRPO is disabled.\n"
            "  Use --modality image  OR  --modality video\n"
            "  Or run: rft_grpo_image_train.py / rft_grpo_video_train.py"
        )
        sys.exit(1)
    banner = (
        "EndoVLA-Oral RFT Training (Image Only)"
        if args.modality == "image"
        else "EndoVLA-Oral RFT Training (Video Only)"
    )
    run_grpo_training(args.modality, f"grpo_{args.modality}_only", banner)


if __name__ == "__main__":
    main()