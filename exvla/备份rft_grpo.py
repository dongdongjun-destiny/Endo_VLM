#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral RFT (Reinforcement Fine-Tuning) with Standard GRPO
Supports both Images & Videos (Sequences) + JSON & JSONL Formats.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import re
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template

# 引入 Qwen 官方视觉处理工具，用于提取视频帧和图片
from qwen_vl_utils import process_vision_info

# ==============================================================================
# UNSLOTH GRPO VISION TENSOR RESCUE PATCH
# ==============================================================================
pristine_vision_tensors = {}
_original_grpo_prepare_inputs = GRPOTrainer._prepare_inputs

def _patched_prepare_inputs(self, inputs):
    global pristine_vision_tensors
    prepared = _original_grpo_prepare_inputs(self, inputs)
    pristine_vision_tensors.clear()

    # 这里的 hook 完美支持了 image 和 video 两种 tensor
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key in inputs:
            tensor = inputs[key].to(self.args.device)
            pristine_vision_tensors[key] = tensor
            prepared[key] = tensor
    return prepared

GRPOTrainer._prepare_inputs = _patched_prepare_inputs

# TRL GRPO: mixed image/video batch — sub-batch by modality, merge vision tensors for loss.
_original_generate_single_turn = GRPOTrainer._generate_single_turn
_original_generate_and_score = GRPOTrainer._generate_and_score_completions
_original_compute_loss = GRPOTrainer._compute_loss
_original_get_per_token_logps = GRPOTrainer._get_per_token_logps_and_entropies

_GRPO_VISION_KEYS = (
    "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
)


def _grpo_has_images(images_entry) -> bool:
    return images_entry is not None and images_entry != [] and images_entry != [None]


def _grpo_has_videos(videos_entry) -> bool:
    return videos_entry is not None


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


def _grpo_generate_video_batch(self, prompts, videos_list, fps_list):
    device = self.accelerator.device
    videos_arg = [_grpo_normalize_video_entry(v) for v in videos_list]
    kwargs = {"videos": videos_arg}
    if fps_list and any(f is not None for f in fps_list):
        flat_fps = []
        for f in fps_list:
            if isinstance(f, list):
                flat_fps.extend(f)
            elif f is not None:
                flat_fps.append(f)
        if flat_fps:
            kwargs["fps"] = flat_fps
    prompts_text = [
        maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts
    ]
    prompt_inputs = self.processing_class(
        text=prompts_text, padding=True, return_tensors="pt", **kwargs
    )
    prompt_inputs = GRPOTrainer._prepare_inputs(self, prompt_inputs)
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


def _patched_generate_single_turn(self, prompts, images):
    per_images = getattr(self, "_grpo_per_images", images)
    per_videos = getattr(self, "_grpo_per_videos", None)
    per_fps = getattr(self, "_grpo_per_fps", None)

    has_any_video = per_videos and any(_grpo_has_videos(v) for v in per_videos)
    if not has_any_video:
        return _original_generate_single_turn(self, prompts, per_images)

    n = len(prompts)
    img_idx = [i for i in range(n) if _grpo_has_images(per_images[i] if per_images else None)]
    vid_idx = [i for i in range(n) if _grpo_has_videos(per_videos[i])]

    if not img_idx and not vid_idx:
        return _original_generate_single_turn(self, prompts, None)

    if img_idx and not vid_idx:
        sub_p = [prompts[i] for i in img_idx]
        sub_im = [per_images[i] for i in img_idx]
        p_ids, c_ids, _, fk = _original_generate_single_turn(self, sub_p, sub_im)
        if len(img_idx) == n:
            return p_ids, c_ids, None, fk
        out_p, out_c, fk_list = [None] * n, [None] * n, []
        for j, i in enumerate(img_idx):
            out_p[i], out_c[i] = p_ids[j], c_ids[j]
        fk_list.append(fk)
        self._grpo_forward_kwargs = _merge_forward_kwargs_list(fk_list)
        return out_p, out_c, None, self._grpo_forward_kwargs

    if vid_idx and not img_idx:
        sub_p = [prompts[i] for i in vid_idx]
        sub_v = [per_videos[i] for i in vid_idx]
        sub_f = [per_fps[i] if per_fps else None for i in vid_idx]
        p_ids, c_ids, fk = _grpo_generate_video_batch(self, sub_p, sub_v, sub_f)
        if len(vid_idx) == n:
            self._grpo_forward_kwargs = fk
            return p_ids, c_ids, None, fk
        out_p, out_c, fk_list = [None] * n, [None] * n, []
        for j, i in enumerate(vid_idx):
            out_p[i], out_c[i] = p_ids[j], c_ids[j]
        fk_list.append(fk)
        self._grpo_forward_kwargs = _merge_forward_kwargs_list(fk_list)
        return out_p, out_c, None, self._grpo_forward_kwargs

    out_p, out_c, fk_list = [None] * n, [None] * n, []
    if img_idx:
        sub_p = [prompts[i] for i in img_idx]
        sub_im = [per_images[i] for i in img_idx]
        p_ids, c_ids, _, fk = _original_generate_single_turn(self, sub_p, sub_im)
        for j, i in enumerate(img_idx):
            out_p[i], out_c[i] = p_ids[j], c_ids[j]
        fk_list.append(fk)
    if vid_idx:
        sub_p = [prompts[i] for i in vid_idx]
        sub_v = [per_videos[i] for i in vid_idx]
        sub_f = [per_fps[i] if per_fps else None for i in vid_idx]
        p_ids, c_ids, fk = _grpo_generate_video_batch(self, sub_p, sub_v, sub_f)
        for j, i in enumerate(vid_idx):
            out_p[i], out_c[i] = p_ids[j], c_ids[j]
        fk_list.append(fk)

    merged_fk = _merge_forward_kwargs_list(fk_list)
    self._grpo_forward_kwargs = merged_fk
    return out_p, out_c, None, merged_fk


def _patched_generate_and_score_completions(self, inputs):
    self._grpo_per_images = [ex.get("images") for ex in inputs]
    self._grpo_per_videos = [ex.get("videos") for ex in inputs]
    self._grpo_per_fps = [ex.get("fps") for ex in inputs]
    self._grpo_num_videos = [_grpo_count_videos(v) for v in self._grpo_per_videos]
    self._grpo_forward_kwargs = None

    fixed_inputs = []
    for ex, im, vid in zip(inputs, self._grpo_per_images, self._grpo_per_videos):
        row = dict(ex)
        row.setdefault("images", im if im is not None else [])
        row.setdefault("videos", vid)
        fixed_inputs.append(row)

    out = _original_generate_and_score(self, fixed_inputs)

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
    extra = getattr(self, "_grpo_video_for_logps", None) or {}
    pixel_values_videos = extra.get("pixel_values_videos")
    video_grid_thw = extra.get("video_grid_thw")
    second_per_grid_ts = extra.get("second_per_grid_ts")
    if num_videos is None:
        num_videos = extra.get("num_videos")

    batch_size = batch_size or input_ids.size(0)
    all_logps, all_entropies = [], []

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
            model_inputs["pixel_values"] = pixel_values[row_start:row_end]
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
            model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
        elif pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values[start : start + batch_size]

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
            model_inputs["pixel_values_videos"] = pixel_values_videos[video_row_start:video_row_end]

            cum_videos = torch.tensor([0] + num_videos, device=video_grid_thw.device).cumsum(0)
            video_start = cum_videos[start]
            video_end = cum_videos[start + batch_size]
            model_inputs["video_grid_thw"] = video_grid_thw[video_start:video_end]
        else:
            if pixel_values_videos is not None:
                model_inputs["pixel_values_videos"] = pixel_values_videos
            if video_grid_thw is not None:
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
    RFT_CONFIG, REWARD_CONFIG,
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

#自定义打分任务
    parser.add_argument("--task", type=str, default="gastrohun", choices=["legacy_endoscopy", "gastrohun"], help="Specify the task to use the corresponding reward mechanism")

    return parser.parse_args()


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
        # 兼容 Image 和 Video 的 Tensor 恢复
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if kwargs.get(k) is None and k in pristine_vision_tensors:
                kwargs[k] = pristine_vision_tensors[k]

        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0: input_ids = args[0]
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        # 原有的 image padding multiplier 逻辑
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
                            if image_grid_thw is not None: kwargs["image_grid_thw"] = image_grid_thw.repeat(multiplier, 1)
            except Exception:
                pass
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
            _, video_inputs, video_kw = process_vision_info(
                messages, return_video_kwargs=True
            )
            row = {
                "prompt": messages,
                "target_label": sample["target_label"],
                "gt_appearance": sample["gt_appearance"],
                "gt_station": sample["gt_station"],
                "task": task_name,
                "media_path": sample["media_path"],
                "videos": video_inputs,
            }
            if video_kw and video_kw.get("fps"):
                row["fps"] = video_kw["fps"]
            data_list.append(row)
        else:
            try:
                image = Image.open(sample["media_path"]).convert("RGB")
                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT): 
                    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            except Exception:
                image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(0, 0, 0))

            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            ]
            data_list.append({
                "prompt": messages,
                "images": [image],
                "videos": None,
                "target_label": sample["target_label"],
                "gt_appearance": sample["gt_appearance"],
                "gt_station": sample["gt_station"],
                "task": task_name,
                "media_path": sample["media_path"], # 图片路径传给grpo图像打分
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
#                 # 👑 1. 裁判亲自打开图片！
#                 image = Image.open(img_path).convert("RGB")
                
#                 # 👑 2. 提取模型生成的文本标签
#                 words = re.findall(r'\b[A-Z0-9]+\b', text.upper())
#                 found_labels = [w for w in words if w in VALID_SSS_LABELS]
                
#                 if len(found_labels) == 1:
#                     pred_label = found_labels[0]
                    
#                     # ==========================================================
#                     # 👑 3. 核心视觉打分逻辑 (你需要在这里填入你的算法)
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




def oral_format_reward(completions, **kwargs) -> List[float]:
    tasks = kwargs.get("task", []) 
    rewards = []

    for i, completion in enumerate(completions):
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
            # 提取文本中所有独立的字母数字组合
            words = re.findall(r'\b[A-Z0-9]+\b', text.upper())
            # 过滤出存在于 SSS 协议中的合法标签
            found_labels = [w for w in words if w in VALID_SSS_LABELS]

            if len(found_labels) == 1:
                reward += 0.4  
                
                clean_text = text.strip()
                text_len = len(clean_text)
                
                if text_len <= 5:
         
                    reward += 0.0
                    
                elif 8 < text_len < 100:
               
                    reward += 0.6  
                    
                else:
             
                    reward += 0.1  

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def oral_accuracy_reward(completions, **kwargs) -> List[float]:
    target_labels = kwargs.get("target_label", [])
    gt_appearances = kwargs.get("gt_appearance", [])
    gt_stations = kwargs.get("gt_station", [])
    tasks = kwargs.get("task", []) 
    rewards = []
    
    for i, completion in enumerate(completions):
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
            words = re.findall(r'\b[A-Z0-9]+\b', text.upper())
            found_labels = [w for w in words if w in VALID_SSS_LABELS]
            
           
            if len(found_labels) == 1 and gt_label:
                pred_label = found_labels[0]
                
                if pred_label == gt_label:
                    reward += 1.0  # 完全命中！
                elif pred_label != "NA" and gt_label != "NA" and len(pred_label) == 2 and len(gt_label) == 2:
                    # 拆解字母
                    pred_wall, pred_depth = pred_label[0], pred_label[1]
                    gt_wall, gt_depth = gt_label[0], gt_label[1]
                    
                    if pred_wall == gt_wall:
                        reward += 0.4  # 真实是 A1，猜了 A2
                    elif pred_depth == gt_depth:
                        reward += 0.4  # 真实是 A1，猜了 G1

        rewards.append(max(0.0, min(1.0, reward)))
    return rewards

# ==============================================================================
# TRAINING
# ==============================================================================
def train_rft(model, processor, train_samples: List[Dict[str, Any]], args=None, config=None, training_mode: int = 2):
    print("\n" + "=" * 70 + "\nStarting EndoVLA-Oral RFT Training with Standard GRPO\n" + "=" * 70)

    num_epochs = args.epochs if args.epochs else config["num_epochs"]
    learning_rate = args.lr if args.lr else config["learning_rate"]
    batch_size = args.batch_size if args and args.batch_size else config["batch_size"]
    num_generations = args.num_generations if args.num_generations else config["num_generations"]

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
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        remove_unused_columns=False,  
        temperature=1.1,

    )

    print("\nInitializing Standard GRPOTrainer...")

    FastVisionModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[oral_format_reward, oral_accuracy_reward], 
    )

    print("\nStarting GRPO reinforcement learning...")
    
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

    print("\n" + "=" * 70 + "\nEndoVLA-Oral RFT Training (Image & Video Standard RL)\n" + "=" * 70)
    model_name = AVAILABLE_MODELS.get(args.model, BASE_MODEL_NAME) if args.model else BASE_MODEL_NAME

    train_samples = load_rft_data(args.data_path, args.image_dir, args.max_samples)
    model, processor, training_mode = setup_model(model_name, args.checkpoint, args.mode)

    print("\n--- Pre-RL Evaluation ---")
    pre_results = evaluate_model(model, processor, train_samples, model_name=model_name,
                                 num_samples=min(5, len(train_samples)),task_name=args.task)
    print(f"Pre-RL full accuracy: {pre_results['full_accuracy']:.2%}")

    train_rft(model, processor, train_samples, args=args, config=dict(RFT_CONFIG), training_mode=training_mode)

    print("\n--- Post-RL Evaluation ---")
    post_results = evaluate_model(model, processor, train_samples, model_name=model_name,
                                  num_samples=min(10, len(train_samples)),task_name=args.task)
    print(f"Post-RL full accuracy: {post_results['full_accuracy']:.2%}")

if __name__ == "__main__":
    main()