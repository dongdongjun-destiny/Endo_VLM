#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLCA-Planner RFT (Reinforcement Fine-Tuning) with GRPO

Goal: Improve model performance through reinforcement learning with:
  1. Format reward:    Encourages proper JSON output structure
  2. Precision reward:  Rewards correct planning predictions

Tasks:
  - GLOBAL_PLANNING:  JSON array of waypoints  e.g. ["B4","Target"]
  - NEXT_WAYPOINT:    JSON string of next goal  e.g. "B4"

Training Modes:
    Mode 1: Train from base model (attach new LoRA adapter)
    Mode 2: Continue from checkpoint with existing adapters
    Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)

Usage:
    # Mode 2 – from SFT checkpoint (most common)
    python vlca_rft.py --config data_config.yaml --checkpoint ./models/vlca_sft_v1 \\
        --runname vlca_rft_v1 --mode 2

    # Mode 3 – from merged SFT checkpoint
    python vlca_rft.py --config data_config.yaml --checkpoint ./models/vlca_sft_v1_merged \\
        --runname vlca_rft_v1 --mode 3

    # Mode 1 – from base model (less common)
    python vlca_rft.py --config data_config.yaml --runname vlca_rft_base --mode 1

    # With merge after training
    python vlca_rft.py --config data_config.yaml --checkpoint ./models/vlca_sft_v1 \\
        --runname vlca_rft_v1 --mode 2 --merge

    # Custom reward weights
    python vlca_rft.py --config data_config.yaml --checkpoint ./models/vlca_sft_v1 \\
        --runname vlca_rft_v1 --mode 2 --format_weight 0.2 --precision_weight 0.8
"""

import os
import sys
import json
import yaml
import random
import shutil
import argparse
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig

from vlca_config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    VLCA_SYSTEM_PROMPT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SEED,
    RFT_CONFIG, REWARD_CONFIG,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS,
    parse_global_planning_output, parse_next_waypoint_output,
    build_all_samples,
    get_output_dirs, get_generation_config, is_thinking_model,
    setup_processor_image_size, ensure_vision_processor,
)


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VLCA-Planner RFT Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to data_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for mode 2 and 3)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=2,
                        help="Training mode: 1=base, 2=existing adapters, 3=merged+new LoRA")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model variant key (e.g. 2b_thinking)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--task", type=str, default=None,
                        choices=["global_planning", "next_waypoint", "both"],
                        help="Override task type")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--runname", type=str, required=True,
                        help="Run name for experiment tracking")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapters after training")
    parser.add_argument("--format_weight", type=float, default=None,
                        help="Override format reward weight")
    parser.add_argument("--precision_weight", type=float, default=None,
                        help="Override precision reward weight")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Override number of generations per prompt")
    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ==============================================================================
# MODEL SETUP (same pattern as vlca_sft.py)
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
    processor = ensure_vision_processor(processor, load_path)
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
        if not _check_model_has_adapters(model):
            print("WARNING: No LoRA adapters found in checkpoint! Consider mode 3.")
        else:
            print("Existing LoRA adapters detected.")

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

    # Common locations for the compiled cache
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
            # Look for the function and insert `has_images = False` at the top
            if "if not has_images:" in content:
                # Add a safe default at the start of _generate_and_score_completions
                old = "def _generate_and_score_completions(self, inputs):"
                if old in content and "has_images = False  # VLCA patch" not in content:
                    new = old + "\n        has_images = False  # VLCA patch: default value"
                    content = content.replace(old, new)
                    patched = True
                    print(f"  Patched has_images default in: {grpo_file}")

            # Patch 2: fix pad_token_id access on multimodal processor
            if "self.processing_class.pad_token_id" in content:
                old_expr = "self.processing_class.pad_token_id"
                new_expr = "getattr(self.processing_class, 'pad_token_id', getattr(getattr(self.processing_class, 'tokenizer', None), 'pad_token_id', 0))"
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
    user_prompt: str,
    model_name: str = BASE_MODEL_NAME,
    for_eval: bool = True,
) -> str:
    """Run inference on a single image with the given prompt."""
    FastVisionModel.for_inference(model)

    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": VLCA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    input_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[input_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    gen_config = get_generation_config(model_name, for_eval=for_eval)

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
    """Evaluate model on a subset of samples."""
    FastVisionModel.for_inference(model)

    if len(samples) > num_samples:
        eval_samples = random.sample(samples, num_samples)
    else:
        eval_samples = samples

    correct = 0
    total = 0
    results_by_task = {"GLOBAL_PLANNING": {"correct": 0, "total": 0},
                       "NEXT_WAYPOINT": {"correct": 0, "total": 0}}

    print(f"\nEvaluating on {len(eval_samples)} samples...")

    for sample in eval_samples:
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            output = run_inference(model, processor, image,
                                   sample["user_prompt"], model_name)
            task_type = sample["task_type"]
            gt = sample["target_text"]

            if task_type == "GLOBAL_PLANNING":
                pred = parse_global_planning_output(output)
                gt_parsed = json.loads(gt)
                is_correct = pred == gt_parsed
            else:
                pred = parse_next_waypoint_output(output)
                gt_parsed = json.loads(gt)
                is_correct = pred == gt_parsed

            if is_correct:
                correct += 1
                results_by_task[task_type]["correct"] += 1
            total += 1
            results_by_task[task_type]["total"] += 1

        except Exception as e:
            print(f"Error evaluating: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_task": results_by_task,
    }


# ==============================================================================
# GRPO DATASET PREPARATION
# ==============================================================================

def create_grpo_dataset(
    samples: List[Dict[str, Any]],
    processor,
) -> Dataset:
    """
    Create a HuggingFace Dataset for GRPO training.

    Each sample contains the prompt text and metadata needed for reward
    computation. Images are referenced by path and loaded at training time.
    """
    data_list = []

    for idx, sample in enumerate(samples):
        # Load image for processing
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VLCA_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["user_prompt"]},
                ],
            },
        ]

        data_list.append({
            "prompt": messages,
            # ---- metadata for reward computation ----
            "task_type": sample["task_type"],
            "target_text": sample["target_text"],
            "image_path": sample["image_path"],
            "frame_planning_list": json.dumps(sample["frame_planning_list"]),
            "global_planning_list": json.dumps(sample["global_planning_list"]),
            "idx": idx,
        })

    return Dataset.from_list(data_list)


# ==============================================================================
# REWARD FUNCTIONS
# ==============================================================================

def vlca_format_reward(completions, task_type, **kwargs) -> List[float]:
    """
    Format reward: check if model output conforms to expected JSON schema.

    Scoring for GLOBAL_PLANNING:
        +0.5  Valid JSON array of strings
        +0.3  "Target" is the last element
        +0.2  All block IDs match B\\d+ pattern

    Scoring for NEXT_WAYPOINT:
        +0.5  Valid JSON string
        +0.3  Value is a valid blockId (B\\d+) or "Target"
        +0.2  Output is concise (no extra text / explanation)

    Args:
        completions: list of model output dicts (each has "content" key)
        task_type: list of task type strings per sample

    Returns:
        List of reward floats in [0, 1]
    """
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        t_type = task_type[i] if i < len(task_type) else "GLOBAL_PLANNING"
        reward = 0.0

        if t_type == "GLOBAL_PLANNING":
            parsed = parse_global_planning_output(text)
            if parsed is not None and isinstance(parsed, list):
                # Valid JSON array of strings
                reward += 0.5
                # "Target" is last element
                if len(parsed) > 0 and parsed[-1] == "Target":
                    reward += 0.3
                # All non-Target entries match B\d+ pattern
                import re
                block_entries = [x for x in parsed if x != "Target"]
                if all(re.match(r'^B\d+$', b) for b in block_entries):
                    reward += 0.2
            else:
                # Partial credit: contains a JSON-like array at all
                if "[" in text and "]" in text:
                    reward += 0.1

        elif t_type == "NEXT_WAYPOINT":
            parsed = parse_next_waypoint_output(text)
            if parsed is not None:
                # Valid parsed waypoint
                reward += 0.5
                # Is a valid blockId or Target
                import re
                if parsed == "Target" or re.match(r'^B\d+$', parsed):
                    reward += 0.3
                # Concise output (no long explanations)
                stripped = text.strip().strip('"')
                if len(stripped) < 20:
                    reward += 0.2
            else:
                # Partial credit for containing a block-like string
                if "Target" in text or "B" in text:
                    reward += 0.1

        rewards.append(max(0.0, min(1.0, reward)))

    return rewards


def vlca_precision_reward(completions, task_type, target_text, **kwargs) -> List[float]:
    """
    Precision reward: check if the prediction matches the ground truth.

    GLOBAL_PLANNING scoring:
        1.0   Exact match (same waypoints, same order)
        0.75  Same set of waypoints but different order
        0.5   Partial overlap + Target correct (last element)
        0.25  Some overlap but Target wrong
        0.0   No meaningful match

    NEXT_WAYPOINT scoring:
        1.0   Exact match
        0.0   Wrong

    Args:
        completions: list of model output dicts
        task_type: list of task type strings per sample
        target_text: list of ground truth JSON strings per sample

    Returns:
        List of reward floats in [0, 1]
    """
    rewards = []

    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        t_type = task_type[i] if i < len(task_type) else "GLOBAL_PLANNING"
        gt_str = target_text[i] if i < len(target_text) else "[]"
        reward = 0.0

        try:
            gt = json.loads(gt_str)
        except (json.JSONDecodeError, TypeError):
            gt = None

        if t_type == "GLOBAL_PLANNING" and gt is not None:
            pred = parse_global_planning_output(text)

            if pred is not None and isinstance(gt, list):
                if pred == gt:
                    # Exact match
                    reward = 1.0
                elif set(pred) == set(gt):
                    # Same waypoints, different order
                    reward = 0.75
                else:
                    # Partial overlap scoring
                    pred_set = set(pred)
                    gt_set = set(gt)
                    overlap = pred_set & gt_set

                    if len(gt_set) > 0:
                        overlap_ratio = len(overlap) / len(gt_set)
                    else:
                        overlap_ratio = 0.0

                    # Check if Target is correctly placed last
                    target_correct = (
                        len(pred) > 0 and pred[-1] == "Target"
                        and len(gt) > 0 and gt[-1] == "Target"
                    )

                    if target_correct and overlap_ratio > 0:
                        reward = 0.25 + 0.25 * overlap_ratio  # 0.25 – 0.50
                    elif overlap_ratio > 0:
                        reward = 0.25 * overlap_ratio  # 0.0 – 0.25

        elif t_type == "NEXT_WAYPOINT" and gt is not None:
            pred = parse_next_waypoint_output(text)

            if pred is not None and isinstance(gt, str):
                if pred == gt:
                    reward = 1.0

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
    print("Starting VLCA RFT Training with GRPO")
    print("=" * 70)

    if config is None:
        config = RFT_CONFIG

    # Resolve hyperparameters (CLI > yaml > defaults)
    num_epochs = args.epochs if args and args.epochs else config["num_epochs"]
    learning_rate = args.lr if args and args.lr else config["learning_rate"]
    batch_size = args.batch_size if args and args.batch_size else config["batch_size"]
    num_generations = (args.num_generations if args and args.num_generations
                       else config["num_generations"])
    format_weight = (args.format_weight if args and args.format_weight is not None
                     else REWARD_CONFIG["format_weight"])
    precision_weight = (args.precision_weight if args and args.precision_weight is not None
                        else REWARD_CONFIG["precision_weight"])

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
    print(f"Precision weight:     {precision_weight}")
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
        task_type = kwargs.pop("task_type", ["GLOBAL_PLANNING"] * len(completions))
        return vlca_format_reward(completions, task_type, **kwargs)

    def precision_reward_fn(prompts, completions, **kwargs):
        task_type = kwargs.pop("task_type", ["GLOBAL_PLANNING"] * len(completions))
        target_text = kwargs.pop("target_text", ["[]"] * len(completions))
        return vlca_precision_reward(completions, task_type, target_text, **kwargs)

    reward_funcs = [format_reward_fn, precision_reward_fn]
    reward_weights = [format_weight, precision_weight]

    running_name = args.runname if args else "default"

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
        run_name=f"vlca_rft_{num_epochs}ep_{running_name}",
        # Unsloth GRPO-specific settings
        importance_sampling_level=config.get("importance_sampling_level", "token"),
        mask_truncated_completions=config.get("mask_truncated_completions", False),
        loss_type=config.get("loss_type", "dr_grpo"),
    )

    # ---- Workaround: Unsloth vision GRPO bugs ----
    # 1. Qwen3VLProcessor lacks top-level pad_token_id; Unsloth's compiled
    #    GRPOTrainer accesses self.processing_class.pad_token_id directly.
    if not hasattr(processor, "pad_token_id"):
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
            processor.pad_token_id = processor.tokenizer.pad_token_id
            print(f"  Patched processor.pad_token_id = {processor.pad_token_id}")
        else:
            processor.pad_token_id = 0
            print("  Patched processor.pad_token_id = 0 (fallback)")

    # 2. Unsloth compiled cache may have a stale `has_images` NameError.
    #    Deleting the cache forces Unsloth to regenerate it on next run.
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
    yaml_cfg = load_yaml_config(args.config)
    wandb_cfg = yaml_cfg.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", RFT_CONFIG["wandb_project"])
    os.environ["WANDB_RUN_GROUP"] = wandb_cfg.get("rft_group", RFT_CONFIG["wandb_run_group"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]

    print("\n" + "=" * 70)
    print("VLCA-Planner RFT Training (GRPO)")
    print("=" * 70)

    # Determine model
    model_key = args.model or yaml_cfg.get("model_key", "2b_thinking")
    model_name = AVAILABLE_MODELS.get(model_key, BASE_MODEL_NAME)
    print(f"Model: {model_name}")
    print(f"Run name: {args.runname}")
    print(f"Training mode: {args.mode}")
    print(f"Merge after training: {args.merge}")

    # Override RFT config from YAML
    rft_overrides = yaml_cfg.get("rft", {})
    config = dict(RFT_CONFIG)
    config.update({k: v for k, v in rft_overrides.items() if v is not None})

    # Task selection
    task = args.task or yaml_cfg.get("task", "both")

    # Load data
    print("\nLoading training data...")
    data_pairs = yaml_cfg.get("data_pairs", [])
    if not data_pairs:
        print("ERROR: No data_pairs in config!")
        sys.exit(1)

    train_samples = build_all_samples(data_pairs, task=task, max_samples=args.max_samples)
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
    print(f"Pre-RFT accuracy: {pre_results['accuracy']:.2%}")
    for task_name, r in pre_results["by_task"].items():
        if r["total"] > 0:
            print(f"  {task_name}: {r['correct']}/{r['total']}")

    # Example output
    if train_samples:
        s = train_samples[0]
        image = Image.open(s["image_path"]).convert("RGB")
        out = run_inference(model, processor, image, s["user_prompt"], model_name)
        print(f"\nExample ({s['task_type']}):")
        print(f"  Ground truth: {s['target_text']}")
        print(f"  Model output: {out[:300]}")

    # Train
    train_rft(model, processor, train_samples,
              args=args, config=config, training_mode=training_mode)

    # Post-RFT evaluation
    print("\n--- Post-RFT Evaluation ---")
    post_results = evaluate_model(model, processor, train_samples,
                                   model_name=model_name,
                                   num_samples=min(10, len(train_samples)))
    print(f"Post-RFT accuracy: {post_results['accuracy']:.2%}")
    for task_name, r in post_results["by_task"].items():
        if r["total"] > 0:
            print(f"  {task_name}: {r['correct']}/{r['total']}")

    # Summary
    print("\n" + "=" * 70)
    print("RFT Training Complete!")
    print("=" * 70)
    print(f"Pre-RFT accuracy:  {pre_results['accuracy']:.2%}")
    print(f"Post-RFT accuracy: {post_results['accuracy']:.2%}")
    print(f"Improvement: {(post_results['accuracy'] - pre_results['accuracy']):.2%}")
    if args.merge:
        print(f"Model saved as MERGED 16-bit: ./models/{args.runname}")
    else:
        print(f"Model saved as LoRA adapters: ./models/{args.runname}")


if __name__ == "__main__":
    main()