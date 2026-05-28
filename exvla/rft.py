#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral RFT (Reinforcement Fine-Tuning) with GRPO

Goal: Improve oral instruction → standardized command conversion through RL with:
  1. Format reward:   Output matches [label, appearance, station] format
  2. Accuracy reward:  Each field (label, appearance, station) is correct

Output format: [label, exact_appearance_adjectives, station]
  e.g. [b, orange-ish protruding, lesser curvature]

Training Modes:
    Mode 1: Train from base model (attach new LoRA adapter)
    Mode 2: Continue from checkpoint with existing adapters
    Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)

Usage:
    # Mode 2 – from SFT checkpoint (most common)
    python rft.py --checkpoint ./models/oral_sft_v1 \
        --data_path ./data/train.json --image_dir ./data/abc_images \
        --runname oral_rft_v1 --mode 2

    # Mode 3 – from merged SFT checkpoint
    python rft.py --checkpoint ./models/oral_sft_merged \
        --data_path ./data/train.json --image_dir ./data/abc_images \
        --runname oral_rft_v1 --mode 3

    # Mode 1 – from base model (less common)
    python rft.py --data_path ./data/train.json --image_dir ./data/abc_images \
        --runname oral_rft_base --mode 1

    # With merge after training
    python rft.py --checkpoint ./models/oral_sft_v1 \
        --data_path ./data/train.json --image_dir ./data/abc_images \
        --runname oral_rft_v1 --mode 2 --merge

    # Custom reward weights
    python rft.py --checkpoint ./models/oral_sft_v1 \
        --data_path ./data/train.json --image_dir ./data/abc_images \
        --runname oral_rft_v1 --mode 2 --format_weight 0.3 --accuracy_weight 0.7
"""

import os
import sys
import json
import random
import shutil
import argparse
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig

# ==============================================================================
# 🔥【新增修复1】TRL GRPOTrainer 拦截补丁：抢救被丢弃的图像张量 🔥
# ==============================================================================
pristine_vision_tensors = {}
_original_grpo_prepare_inputs = GRPOTrainer._prepare_inputs

def _patched_prepare_inputs(self, inputs):
    global pristine_vision_tensors
    prepared = _original_grpo_prepare_inputs(self, inputs)
    pristine_vision_tensors.clear()
    
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key in inputs:
            tensor = inputs[key].to(self.args.device)
            pristine_vision_tensors[key] = tensor
            prepared[key] = tensor
    return prepared

GRPOTrainer._prepare_inputs = _patched_prepare_inputs
# ==============================================================================

from config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    SYSTEM_PROMPT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SEED,
    RFT_CONFIG, REWARD_CONFIG,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS,
    AVAILABLE_STATIONS, POSITION_LABELS,
    build_user_prompt, build_target_text,
    parse_prediction, normalize_appearance, normalize_station,
    appearance_match,
    get_output_dirs, get_generation_config, is_thinking_model,
    setup_processor_image_size,
)
from processor_utils import ensure_full_processor


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="EndoVLA-Oral RFT Training (GRPO)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training JSON file (e.g., ./data/train.json)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing ABC composite images")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for mode 2 and 3)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=2,
                        help="Training mode: 1=base, 2=existing adapters, 3=merged+new LoRA")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Override model variant key (for mode 1)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--runname", type=str, required=True,
                        help="Run name for experiment tracking")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapters after training")
    parser.add_argument("--format_weight", type=float, default=None,
                        help="Override format reward weight")
    parser.add_argument("--accuracy_weight", type=float, default=None,
                        help="Override accuracy reward weight")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Override number of generations per prompt")
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_rft_data(
    json_path: str,
    image_dir: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load training data from JSON for RFT.

    Each sample needs: image_path, oral_instruction, gt_text,
    target_label, gt_appearance, gt_station, target_key.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for item in data:
        image_name = item.get("image_path", item.get("abc_image_name", ""))
        full_image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(full_image_path):
            skipped += 1
            continue

        oral = item.get("oral_instruction", "")
        gt_text = item.get("gt_text", "")
        if not oral or not gt_text:
            skipped += 1
            continue

        samples.append({
            "image_path": full_image_path,
            "image_name": image_name,
            "oral_instruction": oral,
            "gt_text": gt_text,
            "target_label": item.get("target_label", ""),
            "gt_appearance": item.get("gt_appearance", ""),
            "gt_station": item.get("gt_station", ""),
            "target_key": item.get("target_key", ""),
            "id": item.get("id", len(samples)),
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"Loaded {len(samples)} RFT samples, skipped {skipped}")

    # Target distribution
    target_counts = {}
    for s in samples:
        tk = s["target_key"]
        target_counts[tk] = target_counts.get(tk, 0) + 1
    print(f"Target distribution: {dict(sorted(target_counts.items()))}")

    return samples


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def setup_model(
    model_name: str = BASE_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    training_mode: int = 2,
) -> tuple:
    """
    Setup model with LoRA adapters based on specified training mode.

    Training Modes:
        Mode 1: Train from base model (attach new LoRA adapter)
        Mode 2: Continue from checkpoint with existing adapters
        Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)
    """
    print("\n" + "=" * 70)
    print("Setting up model")
    print("=" * 70)
    print(f"Requested training mode: {training_mode}")

    if training_mode in [2, 3] and not checkpoint_path:
        print("ERROR: Mode 2 and 3 require --checkpoint!")
        sys.exit(1)

    if training_mode == 1 and checkpoint_path:
        print("WARNING: Mode 1 ignores --checkpoint. Training from base model.")
        checkpoint_path = None

    load_path = model_name if training_mode == 1 else checkpoint_path
    print(f"Loading: {load_path}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=load_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Ensure we have full multimodal processor (not just tokenizer)
    processor = ensure_full_processor(processor, load_path)
    processor = setup_processor_image_size(processor)

    if training_mode in [1, 3]:
        action = "base model" if training_mode == 1 else "merged checkpoint"
        print(f"\n>>> MODE {training_mode}: Adding new LoRA adapters to {action}")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            random_state=SEED,
        )
        print("LoRA adapters added successfully.")
    elif training_mode == 2:
        print("\n>>> MODE 2: Continue with existing adapters from checkpoint")
        model, processor = FastVisionModel.from_pretrained(
        model_name=load_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        )
        if not _check_model_has_adapters(model):
            print("WARNING: No LoRA adapters found in checkpoint! Consider mode 3.")
        else:
            print("Existing LoRA adapters detected.")

    # ==============================================================================
    # 🔥【新增修复2】使用原生 PyTorch Hook 拦截参数，彻底解决 Python TypeError 🔥
    # ==============================================================================
    def vision_rescue_forward_pre_hook(module, args, kwargs):
        global pristine_vision_tensors
        
        # 1. 恢复被 TRL 切片丢弃的图像特征
        for k in ["pixel_values", "image_grid_thw"]:
            if kwargs.get(k) is None and k in pristine_vision_tensors:
                kwargs[k] = pristine_vision_tensors[k]

        # 获取 input_ids (处理位置参数和关键字参数两种可能)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        # 2. 如果存在 input_ids，检查是否被复制（GRPO多回答评估时，图像需倍增匹配）
        if input_ids is not None and pixel_values is not None and len(pixel_values) > 0:
            try:
                pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                if pad_id is not None:
                    total_pads = (input_ids == pad_id).sum().item()
                    current_patches = pixel_values.shape[0]
                    if total_pads > current_patches and current_patches > 0:
                        multiplier = total_pads // current_patches
                        if total_pads % current_patches == 0:
                            kwargs["pixel_values"] = pixel_values.repeat(multiplier, 1)
                            if image_grid_thw is not None:
                                kwargs["image_grid_thw"] = image_grid_thw.repeat(multiplier, 1)
            except Exception:
                pass
        return args, kwargs
    
    # 注册拦截钩子 (避免了之前覆写 model.forward 引发的 input_ids 冲突)
    model.register_forward_pre_hook(vision_rescue_forward_pre_hook, with_kwargs=True)
    # 为保万无一失，给底层的基础模型也挂上这个钩子
    if hasattr(model, "base_model"):
        model.base_model.register_forward_pre_hook(vision_rescue_forward_pre_hook, with_kwargs=True)
    # ==============================================================================

    print(f"\nLoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    return model, processor, training_mode


def _check_model_has_adapters(model) -> bool:
    """Check if model has LoRA/PEFT adapters attached."""
    if hasattr(model, 'peft_config'):
        return True
    if hasattr(model, 'active_adapter') or hasattr(model, 'active_adapters'):
        return True
    if 'Peft' in model.__class__.__name__:
        return True
    for name, _ in model.named_modules():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            return True
    return False


def _purge_unsloth_compiled_cache():
    """
    Fix Unsloth's compiled cache bugs for vision GRPO.

    Known bugs:
    1. `has_images` referenced before assignment in _generate_and_score_completions
    2. `self.processing_class.pad_token_id` fails for multimodal processors

    Strategy: try to patch the cached file in-place; if that fails, delete
    the cache so Unsloth regenerates it.
    """
    import glob

    search_dirs = [
        os.path.join(os.getcwd(), "unsloth_compiled_cache"),
        os.path.join(os.path.expanduser("~"), "unsloth_compiled_cache"),
    ]

    for cache_dir in search_dirs:
        grpo_file = os.path.join(cache_dir, "UnslothGRPOTrainer.py")
        if not os.path.exists(grpo_file):
            continue

        try:
            with open(grpo_file, "r") as f:
                content = f.read()

            patched = False

            # Patch 1: fix `has_images` undefined by defaulting to False
            # 🔥 坚决注释掉这里的 Patch 1，避免模型变瞎导致 768x0 🔥
            # if "if not has_images:" in content:
            #     old = "def _generate_and_score_completions(self, inputs):"
            #     if old in content and "has_images = False  # EndoVLA patch" not in content:
            #         new = old + "\n        has_images = False  # EndoVLA patch: default value"
            #         content = content.replace(old, new)
            #         patched = True
            #         print(f"  Patched has_images default in: {grpo_file}")

            # Patch 2: fix pad_token_id access on multimodal processor
            if "self.processing_class.pad_token_id" in content:
                old_expr = "self.processing_class.pad_token_id"
                new_expr = (
                    "getattr(self.processing_class, 'pad_token_id', "
                    "getattr(getattr(self.processing_class, 'tokenizer', None), "
                    "'pad_token_id', 0))"
                )
                if new_expr not in content:
                    content = content.replace(old_expr, new_expr)
                    patched = True
                    print(f"  Patched pad_token_id access in: {grpo_file}")

            if patched:
                with open(grpo_file, "w") as f:
                    f.write(content)
                print(f"  Unsloth cache patched successfully: {grpo_file}")
            else:
                print(f"  Unsloth cache inspected, no patches needed.")

        except Exception as e:
            print(f"  Warning: Could not patch Unsloth cache ({e})")
            print(f"  Trying to delete cache directory: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print(f"  Deleted: {cache_dir}")
            except Exception as e2:
                print(f"  Could not delete cache: {e2}")


# ==============================================================================
# MERGE AND SAVE
# ==============================================================================

def merge_and_save_model(model, processor, output_dir: str) -> str:
    """Merge LoRA adapters into base model and save as 16-bit."""
    print(f"\nMerging LoRA and saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained_merged(output_dir, processor, save_method="merged_16bit")
    print(f"Merged model saved to: {output_dir}")
    return output_dir


def save_adapters_only(model, processor, output_dir: str) -> str:
    """Save only the LoRA adapters (not merged)."""
    print(f"\nSaving adapters to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Adapters saved to: {output_dir}")
    return output_dir


# ==============================================================================
# INFERENCE (reused for pre/post evaluation)
# ==============================================================================

def run_inference(
    model,
    processor,
    image: Image.Image,
    oral_instruction: str,
    model_name: str = BASE_MODEL_NAME,
) -> str:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    FastVisionModel.for_inference(model)

    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    user_prompt = build_user_prompt(oral_instruction)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # apply_chat_template may return a list; ensure it's a plain string
    if isinstance(input_text, list):
        input_text = input_text[0] if len(input_text) == 1 else input_text[0]


    inputs = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    gen_config = get_generation_config(model_name, for_eval=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 256),
            temperature=gen_config.get("temperature", 0.1),
            top_p=gen_config.get("top_p", 0.95),
            top_k=gen_config.get("top_k", 20),
            do_sample=gen_config.get("do_sample", False),
            use_cache=True,
        )

    output_text = processor.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in output_text.lower():
        parts = output_text.split("assistant")
        output_text = parts[-1].strip(": \n")

    return output_text.strip()


def evaluate_model(
    model,
    processor,
    samples: List[Dict[str, Any]],
    model_name: str = BASE_MODEL_NAME,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate model on a subset of samples.

    Returns per-field accuracy: label, appearance, station, and full match.
    """
    FastVisionModel.for_inference(model)

    if len(samples) > num_samples:
        eval_samples = random.sample(samples, num_samples)
    else:
        eval_samples = samples

    results = {
        "total": 0,
        "label_correct": 0,
        "appearance_correct": 0,
        "station_correct": 0,
        "full_correct": 0,
    }

    print(f"\nEvaluating on {len(eval_samples)} samples...")

    for sample in eval_samples:
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            output = run_inference(model, processor, image,
                                   sample["oral_instruction"], model_name)
            pred = parse_prediction(output)
            results["total"] += 1

            if pred is None:
                continue

            # Label accuracy
            label_ok = pred["label"] == sample["target_label"].lower()
            if label_ok:
                results["label_correct"] += 1

            # Appearance accuracy (flexible matching)
            appearance_ok = appearance_match(pred["appearance"], sample["gt_appearance"])
            if appearance_ok:
                results["appearance_correct"] += 1

            # Station accuracy
            pred_station = normalize_station(pred["station"])
            station_ok = pred_station == sample["gt_station"].lower()
            if station_ok:
                results["station_correct"] += 1

            # Full match
            if label_ok and appearance_ok and station_ok:
                results["full_correct"] += 1

        except Exception as e:
            print(f"Eval error: {e}")
            continue

    total = results["total"]
    if total > 0:
        results["label_accuracy"] = results["label_correct"] / total
        results["appearance_accuracy"] = results["appearance_correct"] / total
        results["station_accuracy"] = results["station_correct"] / total
        results["full_accuracy"] = results["full_correct"] / total
    else:
        results["label_accuracy"] = 0.0
        results["appearance_accuracy"] = 0.0
        results["station_accuracy"] = 0.0
        results["full_accuracy"] = 0.0

    return results


# ==============================================================================
# GRPO DATASET PREPARATION
# ==============================================================================

def create_grpo_dataset(
    samples: List[Dict[str, Any]],
    processor,
) -> Dataset:
    """
    Create a HuggingFace Dataset for GRPO training.

    Each sample contains the prompt (as conversation messages with image)
    and metadata columns needed for reward computation.
    """
    data_list = []

    for idx, sample in enumerate(samples):
        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

        # Build user prompt
        user_prompt = build_user_prompt(sample["oral_instruction"])

        # Build conversation messages (prompt only, no assistant response)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        data_list.append({
            "prompt": messages,
            # ---- metadata for reward computation ----
            "target_text": sample["gt_text"],          # e.g. "[b, orange-ish protruding, lesser curvature]"
            "target_label": sample["target_label"],     # e.g. "b"
            "gt_appearance": sample["gt_appearance"],   # e.g. "orange-ish protruding"
            "gt_station": sample["gt_station"],         # e.g. "lesser curvature"
            "target_key": sample["target_key"],         # e.g. "lesser_curvature"
            "image_path": sample["image_path"],
            "idx": idx,
        })

    return Dataset.from_list(data_list)


# ==============================================================================
# REWARD FUNCTIONS
# ==============================================================================

def oral_format_reward(completions, **kwargs) -> List[float]:
    """
    Format reward: check if model output conforms to [label, appearance, station].

    Scoring:
        +0.4  Parseable format: [label, appearance, station] with valid label (a/b/c)
        +0.3  Station is one of the valid stations
        +0.2  Appearance field is non-empty and reasonable (2+ words or known pattern)
        +0.1  Output is concise (no long explanations beyond the command)

    Args:
        completions: list of model output dicts (each has "content" key)

    Returns:
        List of reward floats in [0, 1]
    """
    rewards = []

    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reward = 0.0

        parsed = parse_prediction(text)

        if parsed is not None:
            # Valid parseable format with a/b/c label
            if parsed["label"] in POSITION_LABELS:
                reward += 0.4

            # Station is one of the valid options
            pred_station = normalize_station(parsed["station"])
            if pred_station in [s.lower() for s in AVAILABLE_STATIONS]:
                reward += 0.3

            # Appearance is non-empty and reasonable
            appearance = parsed["appearance"].strip()
            if appearance and len(appearance.split()) >= 1:
                reward += 0.2

            # Concise: the core output should be short
            stripped = text.strip()
            if len(stripped) < 200:
                reward += 0.1

        else:
            # Partial credit: contains bracket format at all
            if "[" in text and "]" in text:
                reward += 0.1
            # Partial credit: mentions a label
            text_lower = text.lower()
            if any(f"({l})" in text_lower or f" {l}," in text_lower
                   for l in POSITION_LABELS):
                reward += 0.05

        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def oral_accuracy_reward(completions, **kwargs) -> List[float]:
    """
    Accuracy reward: check if each field matches ground truth.

    Scoring breakdown (total up to 1.0):
        +0.40  Label (a/b/c) is correct
        +0.35  Appearance matches (using flexible appearance_match)
        +0.25  Station is correct

    Args:
        completions: list of model output dicts
        kwargs: must contain target_label, gt_appearance, gt_station lists

    Returns:
        List of reward floats in [0, 1]
    """
    target_labels = kwargs.get("target_label", [])
    gt_appearances = kwargs.get("gt_appearance", [])
    gt_stations = kwargs.get("gt_station", [])

    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reward = 0.0

        # Get ground truth for this sample
        gt_label = target_labels[i].lower() if i < len(target_labels) else ""
        gt_app = gt_appearances[i] if i < len(gt_appearances) else ""
        gt_sta = gt_stations[i].lower() if i < len(gt_stations) else ""

        parsed = parse_prediction(text)

        if parsed is not None:
            # Label accuracy (0.40)
            if parsed["label"] == gt_label:
                reward += 0.40

            # Appearance accuracy (0.35) - flexible matching
            if gt_app and appearance_match(parsed["appearance"], gt_app):
                reward += 0.35

            # Station accuracy (0.25)
            pred_station = normalize_station(parsed["station"])
            if pred_station == gt_sta:
                reward += 0.25

        rewards.append(reward)

    return rewards


# ==============================================================================
# TRAINING
# ==============================================================================

def train_rft(
    model,
    processor,
    train_samples: List[Dict[str, Any]],
    args=None,
    config=None,
    training_mode: int = 2,
):
    """Run RFT training with GRPO."""
    print("\n" + "=" * 70)
    print("Starting EndoVLA-Oral RFT Training with GRPO")
    print("=" * 70)

    if config is None:
        config = RFT_CONFIG

    # Resolve hyperparameters (CLI > config defaults)
    num_epochs = args.epochs if args and args.epochs else config["num_epochs"]
    learning_rate = args.lr if args and args.lr else config["learning_rate"]
    batch_size = args.batch_size if args and args.batch_size else config["batch_size"]
    num_generations = (args.num_generations if args and args.num_generations
                       else config["num_generations"])
    format_weight = (args.format_weight if args and args.format_weight is not None
                     else REWARD_CONFIG["format_weight"])
    accuracy_weight = (args.accuracy_weight if args and args.accuracy_weight is not None
                       else REWARD_CONFIG["accuracy_weight"])

    if args and args.output_dir:
        output_dir = args.output_dir
    elif args and args.runname:
        output_dir = f"./checkpoints/{args.runname}"
    else:
        output_dir = get_output_dirs("rft")["checkpoint_dir"]

    os.makedirs(output_dir, exist_ok=True)

    model_save_dir = (f"./models/{args.runname}" if args and args.runname
                      else get_output_dirs("rft")["model_dir"])

    print(f"Training Mode:        {training_mode}")
    print(f"Epochs:               {num_epochs}")
    print(f"Learning rate:        {learning_rate}")
    print(f"Batch size:           {batch_size}")
    print(f"Num generations:      {num_generations}")
    print(f"Format weight:        {format_weight}")
    print(f"Accuracy weight:      {accuracy_weight}")
    print(f"Checkpoint dir:       {output_dir}")
    print(f"Final model dir:      {model_save_dir}")
    print(f"Merge after training: {args.merge if args else False}")
    print(f"Number of samples:    {len(train_samples)}")

    # ---- Build GRPO dataset ----
    print("\nCreating GRPO dataset...")
    train_dataset = create_grpo_dataset(train_samples, processor)
    print(f"Dataset created with {len(train_dataset)} samples")

    # ---- Reward functions (weighted) ----
    # GRPOTrainer accepts a list of reward functions and reward_weights.
    # Each function receives (prompts, completions, **dataset_columns).
    def format_reward_fn(prompts, completions, **kwargs):
        return oral_format_reward(completions, **kwargs)

    def accuracy_reward_fn(prompts, completions, **kwargs):
        return oral_accuracy_reward(completions, **kwargs)

    reward_funcs = [format_reward_fn, accuracy_reward_fn]
    reward_weights = [format_weight, accuracy_weight]

    running_name = args.runname if args else "default"

    # ==========================================================================
    # 🔥 计算安全步数，防止你传入 6 条测试数据导致 max_steps=0 报错 🔥
    # ==========================================================================
    estimated_steps = (len(train_dataset) // batch_size) * num_epochs if batch_size else 0
    safe_max_steps = 10 if len(train_dataset) < 20 and estimated_steps == 0 else -1

    # ---- GRPO Config ----
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        num_generations=num_generations,
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=SEED,
        report_to="wandb",
        run_name=f"endovla_oral_rft_{running_name}",
        # Unsloth GRPO-specific settings
        importance_sampling_level=config.get("importance_sampling_level", "token"),
        mask_truncated_completions=config.get("mask_truncated_completions", False),
        loss_type=config.get("loss_type", "dr_grpo"),
        remove_unused_columns=False, # 🔥 坚决不允许扔掉图像特征列
        max_steps=safe_max_steps,    # 防止 0 steps 崩溃
    )

    # ---- Workaround: Unsloth vision GRPO bugs ----
    # 1. Qwen VL Processor lacks top-level pad_token_id; Unsloth's compiled
    #    GRPOTrainer accesses self.processing_class.pad_token_id directly.
    if not hasattr(processor, "pad_token_id"):
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
            processor.pad_token_id = processor.tokenizer.pad_token_id
            print(f"  Patched processor.pad_token_id = {processor.pad_token_id}")
        else:
            processor.pad_token_id = 0
            print("  Patched processor.pad_token_id = 0 (fallback)")

    # 2. Unsloth compiled cache may have a stale `has_images` NameError.
    _purge_unsloth_compiled_cache()

    # ---- Initialize GRPOTrainer ----
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    # ---- Train ----
    print("\nStarting RFT training...")
    FastVisionModel.for_training(model) # 🔥 从评估模式切回训练模式
    trainer.train()

    # ---- Save ----
    os.makedirs(model_save_dir, exist_ok=True)
    if args and args.merge:
        merge_and_save_model(model, processor, model_save_dir)
    else:
        save_adapters_only(model, processor, model_save_dir)

    return trainer


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    # Set wandb env vars early
    os.environ["WANDB_PROJECT"] = RFT_CONFIG["wandb_project"]
    os.environ["WANDB_RUN_GROUP"] = RFT_CONFIG["wandb_run_group"]

    print("\n" + "=" * 70)
    print("EndoVLA-Oral RFT Training (GRPO)")
    print("=" * 70)

    # Determine model
    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME) if args.model else BASE_MODEL_NAME
    print(f"Model: {model_name}")
    print(f"Run name: {args.runname}")
    print(f"Training mode: {args.mode}")
    print(f"Merge after training: {args.merge}")

    # Use RFT config
    config = dict(RFT_CONFIG)

    # Load data
    print("\nLoading training data...")
    train_samples = load_rft_data(args.data_path, args.image_dir, args.max_samples)
    if not train_samples:
        print("ERROR: No training samples loaded!")
        sys.exit(1)

    # Setup model
    model, processor, training_mode = setup_model(
        model_name=model_name,
        checkpoint_path=args.checkpoint,
        training_mode=args.mode,
    )

    # Pre-RFT evaluation
    print("\n--- Pre-RFT Evaluation ---")
    pre_results = evaluate_model(model, processor, train_samples,
                                  model_name=model_name,
                                  num_samples=min(5, len(train_samples)))
    print(f"Pre-RFT full accuracy:       {pre_results['full_accuracy']:.2%}")
    print(f"  Label accuracy:            {pre_results['label_accuracy']:.2%}")
    print(f"  Appearance accuracy:       {pre_results['appearance_accuracy']:.2%}")
    print(f"  Station accuracy:          {pre_results['station_accuracy']:.2%}")

    # Show example output
    if train_samples:
        s = train_samples[0]
        image = Image.open(s["image_path"]).convert("RGB")
        out = run_inference(model, processor, image, s["oral_instruction"], model_name)
        print(f"\nExample:")
        print(f"  Ground truth: {s['gt_text']}")
        print(f"  Model output: {out[:300]}")

    # Train
    train_rft(model, processor, train_samples,
              args=args, config=config, training_mode=training_mode)

    # Post-RFT evaluation
    print("\n--- Post-RFT Evaluation ---")
    post_results = evaluate_model(model, processor, train_samples,
                                   model_name=model_name,
                                   num_samples=min(10, len(train_samples)))
    print(f"Post-RFT full accuracy:      {post_results['full_accuracy']:.2%}")
    print(f"  Label accuracy:            {post_results['label_accuracy']:.2%}")
    print(f"  Appearance accuracy:       {post_results['appearance_accuracy']:.2%}")
    print(f"  Station accuracy:          {post_results['station_accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("RFT Training Complete!")
    print("=" * 70)
    print(f"Pre-RFT full accuracy:  {pre_results['full_accuracy']:.2%}")
    print(f"Post-RFT full accuracy: {post_results['full_accuracy']:.2%}")
    print(f"Improvement:            {(post_results['full_accuracy'] - pre_results['full_accuracy']):+.2%}")
    print(f"")
    print(f"  Label:      {pre_results['label_accuracy']:.2%} → {post_results['label_accuracy']:.2%}")
    print(f"  Appearance: {pre_results['appearance_accuracy']:.2%} → {post_results['appearance_accuracy']:.2%}")
    print(f"  Station:    {pre_results['station_accuracy']:.2%} → {post_results['station_accuracy']:.2%}")

    if args.merge:
        print(f"\nModel saved as MERGED 16-bit: ./models/{args.runname}")
    else:
        print(f"Model saved as LoRA adapters: ./models/{args.runname}")


if __name__ == "__main__":
    main()