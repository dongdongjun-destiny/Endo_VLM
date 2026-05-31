#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral SFT (Supervised Fine-Tuning)

Goal: Train VLM to convert noisy oral endoscopic descriptions into
standardized lesion selection commands: [label, appearance, station]

Training Modes:
    Mode 1: Train from base model (attach new LoRA adapter)
    Mode 2: Continue from checkpoint with existing adapters
    Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)

Usage:
    # Mode 1: From base model
    python sft.py --data_path ./data/train.json --image_dir ./data/abc_images \\
        --runname oral_sft_v1 --mode 1

    # Mode 2: Continue from checkpoint with existing adapters
    python sft.py --data_path ./data/train.json --image_dir ./data/abc_images \\
        --checkpoint ./models/oral_sft_v1 --runname oral_sft_v2 --mode 2

    # Mode 3: Continue from merged checkpoint (new LoRA)
    python sft.py --data_path ./data/train.json --image_dir ./data/abc_images \\
        --checkpoint ./models/oral_sft_merged --runname oral_sft_v3 --mode 3

    # With merge after training
    python sft.py --data_path ./data/train.json --image_dir ./data/abc_images \\
        --runname oral_sft_v1 --mode 1 --merge
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from datasets import Dataset
import torch.utils.data as torch_data

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    SYSTEM_PROMPT, SYSTEM_PROMPT_SIMPLE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SEED,
    SFT_CONFIG, IMAGE_WIDTH, IMAGE_HEIGHT,
    IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS,
    build_user_prompt, build_target_text, parse_prediction,
    get_output_dirs, get_generation_config, is_thinking_model,
    setup_processor_image_size,
)

# Set environment variables
os.environ["WANDB_PROJECT"] = SFT_CONFIG["wandb_project"]
os.environ["WANDB_RUN_GROUP"] = SFT_CONFIG["wandb_run_group"]


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="EndoVLA-Oral SFT Training")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training JSON file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing ABC composite images")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation JSON file (optional)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for mode 2 and 3)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=1,
                        help="Training mode: 1=base(new LoRA), 2=checkpoint(existing), 3=merged(new LoRA)")
    parser.add_argument("--model", type=str, default="qwen3_8b_instruct",
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Model variant (for mode 1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--runname", type=str, required=True,
                        help="Run name for experiment identification")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapters after training")
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sft_data(
    json_path: str,
    image_dir: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load training data from JSON."""
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

    print(f"Loaded {len(samples)} samples, skipped {skipped}")

    # Target distribution
    target_counts = {}
    for s in samples:
        tk = s["target_key"]
        target_counts[tk] = target_counts.get(tk, 0) + 1
    print(f"Target distribution: {dict(sorted(target_counts.items()))}")

    return samples


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class OralSFTDataset(torch_data.Dataset):
    """Dataset for SFT training on oral instruction task."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.system_prompt = SYSTEM_PROMPT

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load ABC composite image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

        # Build user prompt from oral instruction
        user_prompt = build_user_prompt(sample["oral_instruction"])

        # Target text: [label, appearance, station]
        target_text = sample["gt_text"]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ]

        return {"messages": messages}


# ==============================================================================
# MODEL SETUP
# ==============================================================================

def setup_model(
    model_name: str = BASE_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    training_mode: int = 1,
) -> tuple:
    """
    Setup model with LoRA adapters based on training mode.

    Mode 1: Base model → new LoRA
    Mode 2: Checkpoint → continue with existing adapters
    Mode 3: Merged checkpoint → new LoRA
    """
    print(f"\nTraining Mode: {training_mode}")

    if training_mode in [2, 3] and not checkpoint_path:
        print("ERROR: Mode 2 and 3 require --checkpoint")
        sys.exit(1)

    if training_mode == 1 and checkpoint_path:
        print("WARNING: Mode 1 ignores checkpoint, training from base model.")
        checkpoint_path = None

    load_path = model_name if training_mode == 1 else checkpoint_path
    print(f"Loading: {load_path}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=load_path,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        use_gradient_checkpointing="unsloth",
    )

    processor = setup_processor_image_size(processor)

    if training_mode == 1:
        print(">>> MODE 1: Attaching new LoRA adapters to base model")
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
            use_dora=True
        )
    elif training_mode == 2:
        print(">>> MODE 2: Continuing with existing adapters from checkpoint")
        # Adapters already loaded from checkpoint
    elif training_mode == 3:
        print(">>> MODE 3: Loaded merged model, attaching new LoRA adapters")
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
            use_dora=True
        )

    return model, processor, training_mode


def merge_and_save_model(model, processor, output_dir: str) -> str:
    """Merge LoRA adapters and save as 16-bit model."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained_merged(output_dir, processor, save_method="merged_16bit")
    print(f"Merged model saved → {output_dir}")
    return output_dir


def save_adapters_only(model, processor, output_dir: str) -> str:
    """Save LoRA adapters only (not merged)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Adapters saved → {output_dir}")
    return output_dir


# ==============================================================================
# INFERENCE / EVALUATION
# ==============================================================================

def run_inference(model, processor, image, oral_instruction, model_name=BASE_MODEL_NAME):
    """Run inference on a single sample."""
    FastVisionModel.for_inference(model)

    user_prompt = build_user_prompt(oral_instruction)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    gen_config = get_generation_config(model_name, for_eval=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 256),
            temperature=gen_config.get("temperature", 0.1),
            do_sample=gen_config.get("do_sample", False),
            use_cache=True,
        )

    output_text = processor.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in output_text.lower():
        parts = output_text.split("assistant")
        output_text = parts[-1].strip(": \n")

    return output_text.strip()


def evaluate_model(model, processor, samples, model_name=BASE_MODEL_NAME, num_samples=50):
    """Quick evaluation on a subset of samples."""
    FastVisionModel.for_inference(model)

    eval_samples = random.sample(samples, min(num_samples, len(samples)))
    correct = 0
    total = 0

    for sample in eval_samples:
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            output = run_inference(model, processor, image, sample["oral_instruction"], model_name)
            pred = parse_prediction(output)

            if pred and pred["label"] == sample["target_label"]:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Eval error: {e}")
            continue

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


# ==============================================================================
# TRAINING
# ==============================================================================

def train(model, processor, train_dataset, val_dataset=None, args=None, training_mode=1):
    """Run SFT training."""
    config = SFT_CONFIG

    num_epochs = args.epochs if args and args.epochs else config["num_epochs"]
    learning_rate = args.lr if args and args.lr else config["learning_rate"]
    batch_size = args.batch_size if args and args.batch_size else config["batch_size"]


    if args and args.output_dir:
        output_dir = args.output_dir
    elif args and args.runname:
        output_dir = f"/home/rennc1/Documents/Yidong_code/exvla/checkpoints/{args.runname}"
    else:
 
        output_dir = "/home/rennc1/Documents/Yidong_code/exvla/checkpoints/default_sft_run"

    os.makedirs(output_dir, exist_ok=True)


    if args and args.runname:
        model_save_dir = f"/home/rennc1/Documents/Yidong_code/exvla/models/{args.runname}"
    else:
        model_save_dir = "/home/rennc1/Documents/Yidong_code/exvla/models/default_sft_run"
    print(f"\nSFT Training Config:")
    print(f"  Mode: {training_mode}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"  Effective batch: {batch_size * config['gradient_accumulation_steps']}")
    print(f"  Checkpoint dir: {output_dir}")
    print(f"  Model save dir: {model_save_dir}")

    running_name = args.runname if args else "default"

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_seq_length=config["max_length"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=SEED,
        report_to="wandb",
        run_name=f"endovla_oral_sft_{running_name}",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=UnslothVisionDataCollator(model, processor),
    )

    print("\nStarting SFT training...")
    trainer.train()

    # Save
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
    print("=" * 70)
    print("EndoVLA-Oral SFT Training")
    print("=" * 70)

    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME)
    print(f"Model: {model_name}")
    print(f"Run: {args.runname}, Mode: {args.mode}, Merge: {args.merge}")

    # Load data
    train_samples = load_sft_data(args.data_path, args.image_dir, args.max_samples)
    if not train_samples:
        print("ERROR: No training samples!")
        sys.exit(1)

    val_samples = None
    if args.val_data_path:
        val_samples = load_sft_data(args.val_data_path, args.image_dir, max_samples=100)

    # Setup model
    model, processor, training_mode = setup_model(model_name, args.checkpoint, args.mode)

    # Create datasets
    train_dataset = OralSFTDataset(train_samples)
    val_dataset = OralSFTDataset(val_samples) if val_samples else None

    # Pre-training eval
    pre_results = evaluate_model(model, processor, train_samples, model_name, num_samples=min(5, len(train_samples)))
    print(f"Pre-training accuracy: {pre_results['accuracy']:.2%}")

    # Train
    train(model, processor, train_dataset, val_dataset, args, training_mode)

    # Post-training eval
    post_results = evaluate_model(model, processor, train_samples, model_name, num_samples=min(10, len(train_samples)))
    print(f"\nPost-training accuracy: {post_results['accuracy']:.2%}")
    print(f"Improvement: {post_results['accuracy'] - pre_results['accuracy']:+.2%}")


if __name__ == "__main__":
    main()