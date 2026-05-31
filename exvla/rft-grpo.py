#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral RFT (Reinforcement Fine-Tuning) with Standard GRPO

Goal: Improve oral instruction → standardized command conversion through RL.
Instead of using DPO/SPO contrastive pairs which cause collator conflicts,
this standard version uses GRPO to naturally generate and score outputs using:
  1. Format reward
  2. Accuracy reward
"""

import os
import sys

import json
import random
import argparse
from typing import Dict, List, Any, Optional


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel

from trl import GRPOTrainer, GRPOConfig


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
    parser = argparse.ArgumentParser(description="EndoVLA-Oral RFT Training (Standard GRPO)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
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

    parser.add_argument("--num_generations", type=int, default=4, help="Generations per prompt for GRPO")
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_rft_data(json_path: str, image_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    skipped = 0

    for item in data:
        image_name = item.get("image_path", item.get("abc_image_name", ""))
        full_image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(full_image_path):
            skipped += 1;
            continue

        oral = item.get("oral_instruction", "")
        gt_text = item.get("gt_text", "")
        if not oral or not gt_text:
            skipped += 1;
            continue

        samples.append({
            "image_path": full_image_path,
            "oral_instruction": oral,
            "gt_text": gt_text,
            "target_label": item.get("target_label", ""),
            "gt_appearance": item.get("gt_appearance", ""),
            "gt_station": item.get("gt_station", ""),
        })

        if max_samples and len(samples) >= max_samples: break

    print(f"Loaded {len(samples)} Standard RL samples, skipped {skipped}")
    return samples


# ==============================================================================
# MODEL SETUP
# ==============================================================================
def setup_model(model_name: str = BASE_MODEL_NAME, checkpoint_path: Optional[str] = None,
                training_mode: int = 2) -> tuple:
    print("\n" + "=" * 70 + "\nSetting up model\n" + "=" * 70)

    load_path = model_name if training_mode == 1 else checkpoint_path
    model, processor = FastVisionModel.from_pretrained(
        model_name=load_path, load_in_4bit=False,load_in_8bit=False, torch_dtype=torch.bfloat16, device_map="cuda:0", use_gradient_checkpointing="unsloth",
    )

    processor = ensure_full_processor(processor, load_path)
    processor = setup_processor_image_size(processor)

    if training_mode in [1, 3]:
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", random_state=SEED, use_dora = True
        )


    def vision_rescue_forward_pre_hook(module, args, kwargs):
        global pristine_vision_tensors
        for k in ["pixel_values", "image_grid_thw"]:
            if kwargs.get(k) is None and k in pristine_vision_tensors:
                kwargs[k] = pristine_vision_tensors[k]

        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0: input_ids = args[0]
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

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
                            if image_grid_thw is not None: kwargs["image_grid_thw"] = image_grid_thw.repeat(multiplier,
                                                                                                            1)
            except Exception:
                pass
        return args, kwargs

    model.register_forward_pre_hook(vision_rescue_forward_pre_hook, with_kwargs=True)
    if hasattr(model, "base_model"): model.base_model.register_forward_pre_hook(vision_rescue_forward_pre_hook,
                                                                                with_kwargs=True)

    return model, processor, training_mode


# ==============================================================================
# INFERENCE & EVAL
# ==============================================================================
def run_inference(model, processor, image: Image.Image, oral_instruction: str,
                  model_name: str = BASE_MODEL_NAME) -> str:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    FastVisionModel.for_inference(model)

    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT): image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    user_prompt = build_user_prompt(oral_instruction)
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}]
    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(input_text, list): input_text = input_text[0]

    inputs = processor(text=[input_text], images=[image], return_tensors="pt", padding=True).to("cuda")
    gen_config = get_generation_config(model_name, for_eval=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=gen_config.get("max_new_tokens", 256),
                                 temperature=gen_config.get("temperature", 0.1), use_cache=True)
    output_text = processor.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in output_text.lower(): output_text = output_text.split("assistant")[-1].strip(": \n")
    return output_text.strip()


def evaluate_model(model, processor, samples: List[Dict[str, Any]], model_name: str = BASE_MODEL_NAME,
                   num_samples: int = 20) -> Dict[str, Any]:
    FastVisionModel.for_inference(model)
    eval_samples = random.sample(samples, num_samples) if len(samples) > num_samples else samples
    results = {"total": 0, "label_correct": 0, "appearance_correct": 0, "station_correct": 0, "full_correct": 0}

    for sample in eval_samples:
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            output = run_inference(model, processor, image, sample["oral_instruction"], model_name)
            pred = parse_prediction(output)
            results["total"] += 1
            if pred is None: continue

            label_ok = pred["label"] == sample["target_label"].lower()
            appearance_ok = appearance_match(pred["appearance"], sample["gt_appearance"])
            station_ok = normalize_station(pred["station"]) == sample["gt_station"].lower()

            if label_ok: results["label_correct"] += 1
            if appearance_ok: results["appearance_correct"] += 1
            if station_ok: results["station_correct"] += 1
            if label_ok and appearance_ok and station_ok: results["full_correct"] += 1
        except Exception:
            continue

    t = results["total"]
    if t > 0:
        results.update(
            {"label_accuracy": results["label_correct"] / t, "appearance_accuracy": results["appearance_correct"] / t,
             "station_accuracy": results["station_correct"] / t, "full_accuracy": results["full_correct"] / t})
    else:
        results.update(
            {"label_accuracy": 0.0, "appearance_accuracy": 0.0, "station_accuracy": 0.0, "full_accuracy": 0.0})
    return results


# ==============================================================================

# ==============================================================================
def create_grpo_dataset(samples: List[Dict[str, Any]], processor) -> Dataset:
    data_list = []
    for sample in samples:
        try:
            with Image.open(sample["image_path"]) as img:
                image = img.convert("RGB")
                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT): image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT),
                                                                                   Image.LANCZOS)
                image.load()
        except Exception:
            image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

        user_prompt = build_user_prompt(sample["oral_instruction"])

        # 对于 GRPO，prompt 是对话格式，包含图像
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]

        data_list.append({
            "prompt": messages,
            "images": [image],
            # 携带 metadata 传递给下方的裁判函数
            "target_label": sample["target_label"],
            "gt_appearance": sample["gt_appearance"],
            "gt_station": sample["gt_station"],
        })
    return Dataset.from_list(data_list)


# ==============================================================================

# ==============================================================================
def oral_format_reward(completions, **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reward = 0.0
        parsed = parse_prediction(text)
        if parsed is not None:
            if parsed["label"] in POSITION_LABELS: reward += 0.4
            pred_station = normalize_station(parsed["station"])
            if pred_station in [s.lower() for s in AVAILABLE_STATIONS]: reward += 0.3
            appearance = parsed["appearance"].strip()
            if appearance and len(appearance.split()) >= 1: reward += 0.2
            if len(text.strip()) < 200: reward += 0.1
        else:
            if "[" in text and "]" in text: reward += 0.1
            text_lower = text.lower()
            if any(f"({l})" in text_lower or f" {l}," in text_lower for l in POSITION_LABELS): reward += 0.05
        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def oral_accuracy_reward(completions, **kwargs) -> List[float]:
    target_labels = kwargs.get("target_label", [])
    gt_appearances = kwargs.get("gt_appearance", [])
    gt_stations = kwargs.get("gt_station", [])
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reward = 0.0
        gt_label = target_labels[i].lower() if i < len(target_labels) else ""
        gt_app = gt_appearances[i] if i < len(gt_appearances) else ""
        gt_sta = gt_stations[i].lower() if i < len(gt_stations) else ""
        parsed = parse_prediction(text)
        if parsed is not None:
            if parsed["label"] == gt_label: reward += 0.40
            if gt_app and appearance_match(parsed["appearance"], gt_app): reward += 0.35
            pred_station = normalize_station(parsed["station"])
            if pred_station == gt_sta: reward += 0.25
        rewards.append(reward)
    return rewards


# ==============================================================================
# TRAINING
# ==============================================================================
def train_rft(model, processor, train_samples: List[Dict[str, Any]], args=None, config=None, training_mode: int = 2):
    print("\n" + "=" * 70 + "\nStarting EndoVLA-Oral RFT Training with Standard GRPO\n" + "=" * 70)

    num_epochs = args.epochs if args.epochs else config["num_epochs"]
    learning_rate = args.lr if args.lr else config["learning_rate"]
    batch_size = args.batch_size if args.batch_size else config["batch_size"]
    num_generations = args.num_generations


    if args and getattr(args, "output_dir", None):
        output_dir = args.output_dir
    elif args and getattr(args, "runname", None):
        output_dir = f"/home/rennc1/Documents/Yidong_code/exvla/checkpoints/{args.runname}"
    else:
        output_dir = "/home/rennc1/Documents/Yidong_code/exvla/checkpoints/default_rft_run"

    os.makedirs(output_dir, exist_ok=True)

   
    if args and getattr(args, "runname", None):
        model_save_dir = f"/home/rennc1/Documents/Yidong_code/exvla/models/{args.runname}"
    else:
        model_save_dir = "/home/rennc1/Documents/Yidong_code/exvla/models/default_rft_run"
    print("\nCreating GRPO dataset...")
    train_dataset = create_grpo_dataset(train_samples, processor)

    # ---- GRPO Config
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
        # GRPO 专属配置
        num_generations=num_generations,
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=256,
        remove_unused_columns=False,  # 防止删掉 metadata
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
    )

    print("\nInitializing Standard GRPOTrainer...")
    FastVisionModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[oral_format_reward, oral_accuracy_reward],  # 注入你的打分系统
    )

    print("\nStarting GRPO reinforcement learning...")
   
    #  checkpoint 路径里带着 "checkpoint-" 文件夹，触发续训
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
    os.environ["WANDB_PROJECT"] = RFT_CONFIG["wandb_project"]
    os.environ["WANDB_RUN_GROUP"] = "grpo_standard"

    print("\n" + "=" * 70 + "\nEndoVLA-Oral RFT Training (Standard RL)\n" + "=" * 70)
    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME) if args.model else BASE_MODEL_NAME

    train_samples = load_rft_data(args.data_path, args.image_dir, args.max_samples)
    model, processor, training_mode = setup_model(model_name, args.checkpoint, args.mode)

    print("\n--- Pre-RL Evaluation ---")
    pre_results = evaluate_model(model, processor, train_samples, model_name=model_name,
                                 num_samples=min(5, len(train_samples)))
    print(f"Pre-RL full accuracy: {pre_results['full_accuracy']:.2%}")

    train_rft(model, processor, train_samples, args=args, config=dict(RFT_CONFIG), training_mode=training_mode)

    print("\n--- Post-RL Evaluation ---")
    post_results = evaluate_model(model, processor, train_samples, model_name=model_name,
                                  num_samples=min(10, len(train_samples)))
    print(f"Post-RL full accuracy: {post_results['full_accuracy']:.2%}")


if __name__ == "__main__":
    main()