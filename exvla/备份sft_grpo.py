#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral SFT (Supervised Fine-Tuning) 
Supports both Images & Videos (Sequences)

Usage:
    # Mode 1: From base model
    python sft.py --data_path ./gastrohun_llm_en_images_train.jsonl \
        --runname oral_sft_v1 --mode 1
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import torch
from PIL import Image
from datasets import Dataset
import torch.utils.data as torch_data

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 引入 Qwen 官方视觉处理工具，用于提取视频帧和图片
from qwen_vl_utils import process_vision_info

from config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    SYSTEM_PROMPT, SYSTEM_PROMPT_SIMPLE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SEED,
    SFT_CONFIG, IMAGE_WIDTH, IMAGE_HEIGHT,
    IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS,
    build_train_video_content,
    normalize_training_sample, get_training_user_text,
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
    parser = argparse.ArgumentParser(description="EndoVLA-Oral SFT Training (Image & Video)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training JSON/JSONL file")
    # image_dir 设为选填，因为新 jsonl 已经是绝对路径了
    parser.add_argument("--image_dir", type=str, default="",
                        help="Directory containing images (Optional if paths in jsonl are absolute)")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation JSON/JSONL file (optional)")
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


def load_sft_data(data_path, image_dir="", max_samples=None):
  
    valid_samples = []
    bad_count = 0
    
    print(f"🔍 正在扫描并验证数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="验证媒体完整性"):
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
class OralSFTDataset(torch_data.Dataset):
    """Dataset for SFT training handling both Images and Videos."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sys_prompt = sample.get("system_instruction", SYSTEM_PROMPT)
        
        # 针对图片和视频构建不同的 media_content 节点
        if sample["type"] == "video":
            media_content = build_train_video_content(sample["media_path"])
        else:
            try:
                image = Image.open(sample["media_path"]).convert("RGB")
            except Exception as e:
                print(f"Error loading {sample['media_path']}: {e}")
                image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))
            
            media_content = {"type": "image", "image": image}

        user_prompt = get_training_user_text(sample)
        target_text = sample["gt_text"]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    media_content,
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
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained_merged(output_dir, processor, save_method="merged_16bit")
    print(f"Merged model saved → {output_dir}")
    return output_dir

def save_adapters_only(model, processor, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Adapters saved → {output_dir}")
    return output_dir


# ==============================================================================
# INFERENCE / EVALUATION
# ==============================================================================

def run_inference(model, processor, sample, model_name=BASE_MODEL_NAME):
    """Run inference dynamically on either an image or a video sample."""
    FastVisionModel.for_inference(model)

    sys_prompt = sample.get("system_instruction", SYSTEM_PROMPT)
    user_prompt = get_training_user_text(sample)

    # 构建带媒介的 prompt 结构
    if sample["type"] == "video":
        media_content = build_train_video_content(sample["media_path"])
    else:
        image = Image.open(sample["media_path"]).convert("RGB")
        media_content = {"type": "image", "image": image}

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [media_content, {"type": "text", "text": user_prompt}]},
    ]

    # 使用官方的 process_vision_info 自动处理多模态信息 (支持图片/视频混杂)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    gen_config = get_generation_config(model_name, for_eval=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 128),
            temperature=gen_config.get("temperature", 0.1),
            do_sample=gen_config.get("do_sample", False),
        )

    # 剔除 prompt 部分，只提取新生成的文字
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def evaluate_model(model, processor, samples, model_name=BASE_MODEL_NAME, num_samples=50):
    FastVisionModel.for_inference(model)

    eval_samples = random.sample(samples, min(num_samples, len(samples)))
    correct = 0
    total = 0

    for sample in eval_samples:
        try:
            output = run_inference(model, processor, sample, model_name)
            pred = parse_prediction(output)
            
            # 由于标签可能是 "G1" 或 "A3"，如果你的 parse_prediction 未覆盖，可直接做字符串匹配:
            target = sample["target_label"].strip().upper()
            if pred and pred["label"].upper() == target:
                correct += 1
            # 容错降级：如果提取不到，就看输出里有没有这个正确的站位代码
            elif target in output.upper():
                correct += 1
                
            total += 1
        except Exception as e:
            print(f"Eval error on {sample['media_path']}: {e}")
            continue

    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


# ==============================================================================
# TRAINING
# ==============================================================================

def train(model, processor, train_dataset, val_dataset=None, args=None, training_mode=1):
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
    print(f"  Checkpoint dir: {output_dir}")

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
        run_name=f"endovla_oral_sft_{args.runname if args else 'default'}",
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
    print("EndoVLA SFT Training (Supports Images & Videos)")
    print("=" * 70)

    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME)
    
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