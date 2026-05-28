#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processor utilities for Qwen-VL models loaded via Unsloth.

When loading from a fine-tuned checkpoint, Unsloth's FastVisionModel.from_pretrained()
may return only a PreTrainedTokenizerFast instead of the full multimodal processor.
This module provides helpers to:
  1. Detect and fix this by loading the full AutoProcessor
  2. Process image + text inputs using the correct Qwen-VL calling convention
"""

from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from config import BASE_MODEL_NAME


def ensure_full_processor(processor, load_path: Optional[str] = None):
    """
    Ensure processor has image processing capabilities (image_processor attribute).

    When loading from a checkpoint, Unsloth may return just a tokenizer.
    In that case, load the full AutoProcessor from the checkpoint or base model.

    Args:
        processor: The processor/tokenizer returned by FastVisionModel.from_pretrained
        load_path: The path that was used to load the model (checkpoint or base)

    Returns:
        A full processor with both tokenizer and image_processor
    """
    # Already a full processor
    if hasattr(processor, 'image_processor'):
        return processor

    print("  NOTE: Loaded processor is tokenizer-only, loading full AutoProcessor...")
    from transformers import AutoProcessor

    # Try loading from checkpoint path first
    if load_path:
        try:
            full_proc = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
            if hasattr(full_proc, 'image_processor'):
                print(f"  Loaded full processor from: {load_path}")
                return full_proc
        except Exception:
            pass

    # Try from the configured base model
    try:
        full_proc = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        print(f"  Loaded full processor from base: {BASE_MODEL_NAME}")
        return full_proc
    except Exception:
        pass

    # Try the original HuggingFace model name (strip Unsloth suffixes)
    for suffix in ["-unsloth-bnb-4bit", "-bnb-4bit"]:
        if suffix in BASE_MODEL_NAME:
            original = BASE_MODEL_NAME.replace(suffix, "").replace("unsloth/", "Qwen/")
            try:
                full_proc = AutoProcessor.from_pretrained(original, trust_remote_code=True)
                print(f"  Loaded full processor from: {original}")
                return full_proc
            except Exception:
                pass

    print("  WARNING: Could not load full processor, using tokenizer-only")
    return processor


def process_multimodal(processor, messages, image: Optional[Image.Image] = None):
    """
    Process chat messages + image into model inputs.

    Uses the standard Qwen-VL processor calling convention:
        processor(text=[str], images=[PIL.Image], return_tensors="pt")

    Args:
        processor: Full Qwen-VL processor (with image_processor)
        messages: Chat messages list (system, user, optionally assistant)
        image: PIL Image to include (if not already embedded in messages)

    Returns:
        Tuple of (inputs_dict, prompt_text_str)
    """
    # Get text from chat template
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # apply_chat_template may return list in newer transformers
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]

    # Collect images from messages if present
    image_list = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if img is not None and isinstance(img, Image.Image):
                        image_list.append(img)

    # Fall back to the explicitly provided image
    if not image_list and image is not None:
        image_list.append(image)

    # Call processor with explicit keyword arguments
    inputs = processor(
        text=[prompt_text],
        images=image_list if image_list else None,
        padding=True,
        return_tensors="pt",
    )

    return inputs, prompt_text