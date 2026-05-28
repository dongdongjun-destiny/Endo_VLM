#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral Training Configuration

Oral Instruction → Standardized Lesion Command for Endoscopic Target Selection

Task: Given a noisy oral description and a composite ABC image of 3 endoscopic keyframes,
identify which panel (a/b/c) contains the target lesion and output:
    [label, exact_appearance_adjectives, station]

The appearance field should contain the EXACT descriptive adjectives from the oral
instruction, not a canonical form. Each sample may have different appearance text.

Example outputs:
    [b, orange-ish protruding, lesser curvature]
    [a, whitish oval-shaped, greater curvature]
    [c, tiny circular, pyloric antrum]

Training Modes:
    Mode 1: Train from base model (attach new LoRA adapter)
    Mode 2: Continue from checkpoint with existing adapters
    Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)
lo
Usage:
    from config import *
"""

import os
import re
import math
from typing import Optional, Dict, Any, List

# ==============================================================================
# PROJECT METADATA
# ==============================================================================

PROJECT_NAME = "endovla_oral"
PROJECT_VERSION = "1.1.0"
PROJECT_DESCRIPTION = (
    "VLM fine-tuning for converting noisy oral endoscopic instructions "
    "into standardized lesion selection commands with exact appearance extraction"
)

# ==============================================================================
# PATHS CONFIGURATION
# ==============================================================================

# --- Data Paths (UPDATE THESE) ---
# Each entry is a dict with "image_dir" and "json_path"
# Supports multiple dataset files for flexible training/evaluation
DATA_SOURCES = [
    {
        "name": "train",
        "image_dir": "./data/abc_images",
        "json_path": "./data/train.json",
        "split": "train",
    },
    {
        "name": "eval",
        "image_dir": "./data/abc_images",
        "json_path": "./data/eval.json",
        "split": "eval",
    },
]

# Raw image folders (source endoscopic images, 3 per folder)
RAW_IMAGE_FOLDERS = {
    "greater_curvature": "./data/raw/greater_curvature",
    "lesser_curvature": "./data/raw/lesser_curvature",
    "pyloric_antrum": "./data/raw/pyloric_antrum",
}

# Generated data output
GENERATED_DATA_DIR = "./data"
ABC_IMAGE_DIR = os.path.join(GENERATED_DATA_DIR, "abc_images")

# --- Model Paths ---

BASE_MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct"

AVAILABLE_MODELS = {
    "qwen2.5_3b": "unsloth/Qwen2.5-VL-3B-Instruct",
    "qwen2.5_7b": "unsloth/Qwen2.5-VL-7B-Instruct",
    "qwen3_2b_thinking": "unsloth/Qwen3-VL-2B-Thinking",
    "qwen3_2b_instruct": "unsloth/Qwen3-VL-2B-Instruct",
    "qwen3_4b_thinking": "unsloth/Qwen3-VL-4B-Thinking",
    "qwen3_4b_instruct": "unsloth/Qwen3-VL-4B-Instruct",
    "qwen3_8b_thinking": "unsloth/Qwen3-VL-8B-Thinking",  
    "qwen3_8b_instruct": "unsloth/Qwen3-VL-8B-Instruct",  
}
# Checkpoint paths
SFT_CHECKPOINT_PATH = "./models/endovla_oral_sft"
RFT_CHECKPOINT_PATH = "./models/endovla_oral_rft"


# ==============================================================================
# TASK DEFINITIONS
# ==============================================================================
# 1. 定义 SSS 协议标签 (A1-P6, NA)
SSS_LABELS = (
    [f"A{i}" for i in range(1, 7)] + 
    [f"G{i}" for i in range(1, 5)] + 
    [f"L{i}" for i in range(1, 7)] + 
    [f"P{i}" for i in range(1, 7)] + 
    ["NA"]
)

# 旧版描述性标签
LEGACY_STATIONS = ["greater curvature", "lesser curvature", "pyloric antrum"]


AVAILABLE_STATIONS = SSS_LABELS + LEGACY_STATIONS
POSITION_LABELS = ["a", "b", "c"]

# Mapping from folder key to canonical output (kept for backward compatibility)
FOLDER_TO_APPEARANCE = {
    "greater_curvature": "white oval",
    "lesser_curvature": "orange protruding",
    "pyloric_antrum": "small round nodule",
}

FOLDER_TO_STATION = {
    "greater_curvature": "greater curvature",
    "lesser_curvature": "lesser curvature",
    "pyloric_antrum": "pyloric antrum",
}

# All known exact appearance adjectives (for reference / fuzzy matching in eval)
# These are the exact adjectives the model should extract from oral instructions
ALL_EXACT_APPEARANCES = {
    "greater_curvature": [
        "whitish oval-shaped", "white elliptical", "pale oval",
        "bright white oval", "white-ish elongated", "milky white oval",
        "light colored oval", "whitish egg-shaped", "white flattened oval",
    ],
    "lesser_curvature": [
        "orange-ish protruding", "orangey raised", "reddish-orange protruding",
        "orange elevated", "orange colored protrusion", "orange raised",
        "tangerine colored protruding", "orange-ish bulging", "bright orange",
    ],
    "pyloric_antrum": [
        "small round", "tiny circular", "little round raised",
        "small spherical", "small rounded", "petite round",
        "small circular nodular", "tiny round elevated", "small round-shaped",
        "compact round little",
    ],
}

# Flat set of all known exact appearances for quick lookup
ALL_EXACT_APPEARANCES_FLAT = set()
for _apps in ALL_EXACT_APPEARANCES.values():
    ALL_EXACT_APPEARANCES_FLAT.update(a.lower() for a in _apps)


# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================

# SYSTEM_PROMPT = """You are an expert endoscopic assistant. Your task is to interpret a user's oral description of a suspicious lesion and match it to one of three endoscopic keyframe images labeled (a), (b), and (c).

# Given the user's oral description (which may contain noise, filler words, or imprecise language) and the composite image showing three keyframes:
# 1. Identify which keyframe (a, b, or c) matches the described lesion
# 2. Extract the EXACT appearance adjectives from the oral instruction that describe the lesion (e.g., "whitish oval-shaped", "orange-ish protruding", "tiny circular")
# 3. Determine the correct anatomical station from the available options

# Available stations: greater curvature; lesser curvature; pyloric antrum

# Output format: [label, exact_appearance_adjectives, station]
# Examples:
#   [b, orange-ish protruding, lesser curvature]
#   [a, whitish oval-shaped, greater curvature]
#   [c, tiny circular, pyloric antrum]

# Important: The appearance field should use the exact descriptive adjectives from the user's oral instruction, not a generic category."""

# SYSTEM_PROMPT_SIMPLE = """You are an endoscopic assistant. Match the user's oral description to one of three keyframe images (a, b, c) and output the standardized command.

# Extract the exact appearance adjectives from the oral instruction.
# Available stations: greater curvature; lesser curvature; pyloric antrum
# Output format: [label, exact_appearance_adjectives, station]"""


# # ==============================================================================
# # PROMPT BUILDING
# # ==============================================================================

# def build_user_prompt(oral_instruction: str) -> str:
#     """Build the user prompt from oral instruction."""
#     return f"""According to the user's oral description and the 3 endoscopic keyframe images attached (labeled a, b, c), help me point out which suspicious lesion the user is referring to.

# User's oral description: "{oral_instruction}"

# Available stations: greater curvature; lesser curvature; pyloric antrum

# Please identify:
# 1. Which keyframe (a, b, or c) shows the described lesion
# 2. The exact appearance adjectives from the oral description (e.g., "whitish oval-shaped", "orangey raised", "small spherical")
# 3. The correct anatomical station

# Output your answer in the format: [label, exact_appearance_adjectives, station]"""


# def build_target_text(label: str, appearance: str, station: str) -> str:
#     """Build the ground truth target text."""
#     return f"[{label}, {appearance}, {station}]"




#new project
# ==============================================================================
# SYSTEM PROMPTS new
# ==# ==============================================================================
# SYSTEM PROMPTS (SSS 协议三任务版)
# ==============================================================================

SYSTEM_PROMPT = """You are an expert endoscopic assistant. Your task is to interpret a user's oral description of a suspicious lesion and analyze the provided gastroscopy image.

Given the user's oral description and the endoscopic image:
1. Identify the SSS Label Code (e.g., A1, G4, L2, or NA) according to the anatomical context.
2. Extract the EXACT appearance adjectives from the oral instruction (e.g., "whitish oval-shaped", "orange-ish protruding").
3. Determine the correct anatomical station name.

Output format: [label, exact_appearance_adjectives, station]
Examples:
  [L2, orange-ish protruding, lower gastric body - lesser curvature]
  [G4, whitish oval-shaped, fundus and cardia region]
  [NA, tiny circular, unknown station]

Important: Please provide your answer as a brief, natural diagnostic sentence containing the SSS code (e.g., 'The lesion is located at A1.'). Do not just output the code alone. The appearance field must use the exact descriptive adjectives from the user's oral instruction. The label must be an SSS code."""

SYSTEM_PROMPT_SIMPLE = """You are an endoscopic assistant. Analyze the user's oral description and the image to output:
[label, exact_appearance_adjectives, station]

Label must be an SSS code (e.g., A1, G4, NA). Extract exact appearance adjectives from the instruction."""


# ==============================================================================
# PROMPT BUILDING (SSS 协议三任务版)
# ==============================================================================

def build_user_prompt(oral_instruction: str) -> str:
    """Build the user prompt from oral instruction for 3-task output."""
    return f"""According to the user's oral description and the endoscopic image attached, help me analyze the lesion.

User's oral description: "{oral_instruction}"

Please identify:
1. The SSS label code (e.g., A1, G4, L5, P6, or NA)
2. The exact appearance adjectives from the oral description
3. The correct anatomical station name

Output your answer in the format: [label, exact_appearance_adjectives, station]"""


def build_target_text(label: str, appearance: str, station: str) -> str:
    """Build the ground truth target text with 3 elements."""
    # 重新加回 label 参数，确保输出与 parse_prediction 结构匹配
    return f"[{label}, {appearance}, {station}]"


# ==============================================================================
# PARSING FUNCTIONS
# ==============================================================================

# def parse_prediction(text: str) -> Optional[Dict[str, str]]:
#     """
#     Parse model output to extract [label, appearance, station].

#     The appearance field should be the exact adjectives from the oral instruction.

#     Returns:
#         Dict with keys: label, appearance, station (or None if parse fails)
#     """
#     text = text.strip()

#     # Pattern: [label, appearance, station]
#     pattern = r'\[\s*([abc])\s*,\s*(.+?)\s*,\s*(.+?)\s*\]'
#     match = re.search(pattern, text, re.IGNORECASE)
#     if match:
#         return {
#             "label": match.group(1).lower().strip(),
#             "appearance": match.group(2).strip().lower(),
#             "station": match.group(3).strip().lower(),
#         }

#     # Fallback: try to find components separately
#     # Label
#     label_match = re.search(r'\b([abc])\b', text.lower())
#     label = label_match.group(1) if label_match else None

#     # Station
#     station = None
#     for s in AVAILABLE_STATIONS:
#         if s.lower() in text.lower():
#             station = s
#             break

#     # Appearance: try to find any known exact appearance in the text
#     appearance = None
#     text_lower = text.lower()

#     # First try exact match against all known appearances
#     for exact_app in sorted(ALL_EXACT_APPEARANCES_FLAT, key=len, reverse=True):
#         if exact_app in text_lower:
#             appearance = exact_app
#             break

#     # If no exact match, try keyword-based detection
#     if appearance is None:
#         appearance_keywords = {
#             "greater_curvature": ["white", "oval", "whitish", "pale", "elliptical", "elongated", "egg-shaped"],
#             "lesser_curvature": ["orange", "protruding", "protrusion", "orangey", "bulging", "elevated", "tangerine", "reddish"],
#             "pyloric_antrum": ["round", "nodule", "circular", "small", "tiny", "spherical", "nodular", "petite", "compact"],
#         }
#         best_match_count = 0
#         best_match_key = None
#         for target_key, keywords in appearance_keywords.items():
#             matches = sum(1 for kw in keywords if kw in text_lower)
#             if matches > best_match_count:
#                 best_match_count = matches
#                 best_match_key = target_key

#         # Extract appearance words from the text that match keywords
#         if best_match_key and best_match_count >= 2:
#             keywords = appearance_keywords[best_match_key]
#             found_words = [kw for kw in keywords if kw in text_lower]
#             appearance = " ".join(found_words[:3])  # Take up to 3 matching keywords

#     if label and appearance and station:
#         return {"label": label, "appearance": appearance, "station": station}

#     return None


#多情况时候



def parse_prediction(text: str) -> Optional[Dict[str, str]]:
    """
    解析模型输出以提取 [label, appearance, station]。
    
    兼容性说明：
    1. 支持 [A1, whitish, greater curvature] (新格式)
    2. 支持 [a, whitish, greater curvature] (旧格式)
    3. 支持纯标签输出 "L2" (自动填充结构)
    """
    text = text.strip()

    # 1.核心模式：匹配三元素 [label, appearance, station]
    # label 匹配范围：a/b/c 或 SSS标签 (A1-A6, G1-G4, L1-L6, P1-P6, NA)
    label_regex = r'([abc]|[AGLP][1-6]|NA)'
    pattern_3 = rf'\[\s*{label_regex}\s*,\s*(.+?)\s*,\s*(.+?)\s*\]'
    
    match_3 = re.search(pattern_3, text, re.IGNORECASE)
    if match_3:
        return {
            "label": match_3.group(1).upper().strip(),
            "appearance": match_3.group(2).strip().lower(),
            "station": match_3.group(3).strip().lower(),
        }

    # 2.兜底模式 A：识别纯 SSS 标签 (针对 GastroHUN 只有 "L2" 这种回答的情况)
    sss_standalone_pattern = r'\b([AGLP][1-6]|NA)\b'
    sss_match = re.search(sss_standalone_pattern, text.upper())
    if sss_match:
        code = sss_match.group(1).upper()
        return {
            "label": code,
            "appearance": "",  # 纯标签模式下外观为空
            "station": code    # 站位信息暂由代码填充
        }

    # 3. 🧩 兜底模式 B：独立组件提取 (Fallback)
    # 提取 Label (支持 a/b/c 或 A1-P6)
    label_match = re.search(rf'\b{label_regex}\b', text, re.IGNORECASE)
    label = label_match.group(1).upper() if label_match else None

    # 提取 Station (解剖位置描述)
    station = None
    # AVAILABLE_STATIONS 需要在外部定义，例如 ["greater curvature", ...]
    try:
        for s in AVAILABLE_STATIONS:
            if s.lower() in text.lower():
                station = s
                break
    except NameError:
        station = label # 如果没定义 AVAILABLE_STATIONS，则用 label 兜底

    # 提取 Appearance (外观形容词)
    appearance = None
    text_lower = text.lower()
    
    # 优先尝试从全局已知列表匹配 (ALL_EXACT_APPEARANCES_FLAT 需要在外部定义)
    try:
        for exact_app in sorted(ALL_EXACT_APPEARANCES_FLAT, key=len, reverse=True):
            if exact_app in text_lower:
                appearance = exact_app
                break
    except NameError:
        pass

    # 如果还没找到，使用关键字逻辑匹配 (由你提供的代码逻辑保留)
    if appearance is None:
        appearance_keywords = {
            "greater_curvature": ["white", "oval", "whitish", "pale", "elliptical", "elongated", "egg-shaped"],
            "lesser_curvature": ["orange", "protruding", "protrusion", "orangey", "bulging", "elevated", "tangerine", "reddish"],
            "pyloric_antrum": ["round", "nodule", "circular", "small", "tiny", "spherical", "nodular", "petite", "compact"],
        }
        best_match_count = 0
        best_match_key = None
        for target_key, keywords in appearance_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_match_count:
                best_match_count = matches
                best_match_key = target_key

        if best_match_key and best_match_count >= 2:
            keywords = appearance_keywords[best_match_key]
            found_words = [kw for kw in keywords if kw in text_lower]
            appearance = " ".join(found_words[:3])

    # 最终组装
    if label:
        return {
            "label": label,
            "appearance": appearance if appearance else "",
            "station": station if station else label
        }

    return None


def normalize_appearance(appearance_text: str) -> str:
    """
    Normalize a predicted appearance for comparison.

    Since we now use exact appearance adjectives (not canonical forms),
    normalization is minimal: lowercase, strip whitespace, normalize hyphens.
    """
    text = appearance_text.lower().strip()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_station(station_text: str) -> Optional[str]:
    """支持旧的三大部位，也支持 SSS 协议的解剖描述"""
    text = station_text.lower().strip()
    
    # 1. 优先匹配 SSS 编码前缀 (如 "a1" 或 "a1 - anterior wall")
    sss_prefix = re.search(r'([aglp][1-6]|na)', text)
    if sss_prefix:
        return sss_prefix.group(1).upper()

    # 2. 旧版兼容逻辑
    station_map = {
        "greater curvature": ["greater curvature", "greater", "outer"],
        "lesser_curvature": ["lesser curvature", "lesser", "inner"],
        "pyloric antrum": ["pyloric antrum", "pyloric", "antrum"],
    }
    for canon, aliases in station_map.items():
        if any(alias in text for alias in aliases):
            return canon
    return text

def appearance_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted appearance matches ground truth.

    Supports:
    1. Exact match (after normalization)
    2. Containment match (pred contains GT or GT contains pred)
    3. Word overlap match (>= 60% of GT words found in pred)
    """
    pred = normalize_appearance(predicted)
    gt = normalize_appearance(ground_truth)

    # Exact match
    if pred == gt:
        return True

    # Containment match
    if gt in pred or pred in gt:
        return True

    # Word overlap match
    gt_words = set(gt.split())
    pred_words = set(pred.split())
    if len(gt_words) > 0:
        overlap = len(gt_words & pred_words)
        if overlap / len(gt_words) >= 0.6:
            return True

    return False


# ==============================================================================
# IMAGE CONFIGURATION
# ==============================================================================

# Individual keyframe size
# KEYFRAME_WIDTH = 320
# KEYFRAME_HEIGHT = 240

# # Composite ABC image: 3 keyframes side by side with labels
# ABC_IMAGE_WIDTH = KEYFRAME_WIDTH * 3 + 40   # 3 panels + margins
# ABC_IMAGE_HEIGHT = KEYFRAME_HEIGHT + 60       # panel + label space

# IMAGE_WIDTH = ABC_IMAGE_WIDTH    # 1000 pixels wide
# IMAGE_HEIGHT = ABC_IMAGE_HEIGHT  # 300 pixels tall
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
# Min/max pixels for Qwen VL models
IMAGE_MIN_PIXELS = 256 * 28 * 28
IMAGE_MAX_PIXELS = 1280 * 28 * 28

# Per-frame video resolution (4:3 like image; ~1/10 pixels vs 1024×768, 10–16 frames/step)
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240

# Training video limits for Qwen VL models (~10–15s clips @ 1 fps → 10–16 frames)
TRAIN_VIDEO_FPS = 1.0
TRAIN_VIDEO_MIN_FRAMES = 10
TRAIN_VIDEO_MAX_FRAMES = 16
# Per-frame pixel bounds (keep below IMAGE_* ; ~7× fewer pixels/frame than 1024×768)
TRAIN_VIDEO_MIN_PIXELS = 224 * 28 * 28
TRAIN_VIDEO_MAX_PIXELS = 256 * 28 * 28
# Qwen3-VL video_processor: total-clip pixels ≈ frames × per-frame
TRAIN_VIDEO_TOTAL_MIN_PIXELS = TRAIN_VIDEO_MIN_PIXELS * TRAIN_VIDEO_MIN_FRAMES
TRAIN_VIDEO_TOTAL_MAX_PIXELS = TRAIN_VIDEO_MAX_PIXELS * TRAIN_VIDEO_MAX_FRAMES

# ==============================================================================
# MODEL GENERATION CONFIGS
# ==============================================================================

THINKING_GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

INSTRUCT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

EVAL_GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 20,
    "max_new_tokens": 256,
    "do_sample": False,
}


def get_generation_config(model_name: str, for_eval: bool = False) -> dict:
    """Get generation config based on model type."""
    if for_eval:
        return EVAL_GENERATION_CONFIG.copy()
    is_thinking = "thinking" in model_name.lower()
    return THINKING_GENERATION_CONFIG.copy() if is_thinking else INSTRUCT_GENERATION_CONFIG.copy()


def is_thinking_model(model_name: str) -> bool:
    return "thinking" in model_name.lower()


# ==============================================================================
# LORA CONFIGURATION
# ==============================================================================

LORA_R = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
SEED = 3407


# ==============================================================================
# SFT TRAINING CONFIGURATION
# ==============================================================================

SFT_CONFIG = {
    "training_mode": "auto_detect",
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "max_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,
    "lr_scheduler_type": "cosine",
    "save_steps": 5000,
    "logging_steps": 10,
    "save_total_limit": 3,
    "eval_steps": 500,
    "wandb_project": "endovla_oral",
    "wandb_run_group": "sft",
}


# ==============================================================================
# RFT TRAINING CONFIGURATION (GRPO)
# ==============================================================================

RFT_CONFIG = {
    "training_mode": "auto_detect",#无用
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "max_prompt_length": 1024,
    "max_completion_length": 256,
    "max_seq_length": 1024,
    "num_epochs": 5,
    "learning_rate": 1e-6,
    "warmup_ratio": 0.1,
    "weight_decay": 0.1,
    "max_grad_norm": 0.1,
    "lr_scheduler_type": "cosine",
    "save_steps": 5000,
    "logging_steps": 10,
    "importance_sampling_level": "token",
    "mask_truncated_completions": False,
    "loss_type": "dr_grpo",
    "wandb_project": "endovla_oral",
    "wandb_run_group": "rft",
}


# ==============================================================================
# RFT REWARD CONFIGURATION
# ==============================================================================

REWARD_CONFIG = {
    "format_weight": 0.30,
    "accuracy_weight": 0.70,
}


# ==============================================================================
# WANDB CONFIGURATION
# ==============================================================================

WANDB_CONFIG = {
    "project": "endovla_oral",
    "entity": None,  # Set to your wandb team/username or None for default
    "tags": ["endoscopy", "oral-instruction", "lesion-selection", "vlm", "exact-appearance"],
    "notes": "Oral instruction to standardized lesion selection command with exact appearance extraction",
    "log_model": False,
    "log_code": True,
}


# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================

EVAL_CONFIG = {
    # Evaluation metrics: 3 rows × 4 columns
    # Rows: greater_curvature, lesser_curvature, pyloric_antrum
    # Columns: conversion_success, selection_accuracy, appearance_accuracy, station_accuracy
    "metric_columns": [
        "conversion_success",      # All 3 fields correct
        "selection_accuracy",      # a/b/c label correct
        "appearance_accuracy",     # Exact appearance adjectives correct
        "station_accuracy",        # Station selection correct
    ],
    "metric_rows": [
        "greater_curvature",       # White oval lesion (various exact adjectives)
        "lesser_curvature",        # Orange protruding lesion (various exact adjectives)
        "pyloric_antrum",          # Small round nodule (various exact adjectives)
    ],
    "eval_samples_per_target": 300,  # 300 per lesion type = 900 total
    "output_dir": "./eval_results",
    "save_predictions": True,
    "save_latex_table": True,
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_output_dirs(train_type: str, runname: str = None) -> dict:
    """Get output directories based on training type and runname."""
    run_tag = runname if runname else f"endovla_oral_{train_type}"
    return {
        "run_tag": run_tag,
        "checkpoint_dir": f"./checkpoints/{run_tag}",
        "model_dir": f"./models/{run_tag}",
    }


def normalize_training_sample(item: dict, image_dir: str = "") -> Optional[dict]:
    """Unify jsonl fields so image/video samples share one training schema."""
    raw_type = item.get("type", "image")
    if raw_type in ("sequence", "video") or item.get("video_path"):
        media_type = "video"
        media_path = item.get("video_path") or item.get("media_path", "")
    else:
        media_type = "image"
        media_path = (
            item.get("image_path")
            or item.get("media_path")
            or item.get("abc_image_name", "")
        )

    if not media_path:
        return None

    if not os.path.isabs(media_path) and image_dir:
        media_path = os.path.join(image_dir, media_path)

    label = (
        item.get("target_label")
        or item.get("label_code")
        or item.get("answer", "")
    )
    label = str(label).strip()

    oral = item.get("oral_instruction", "")
    user_instruction = item.get("user_instruction", "")
    if not oral and user_instruction:
        oral = user_instruction

    gt_text = item.get("gt_text")
    if not gt_text:
        appearance = item.get("gt_appearance", item.get("appearance_adjectives", ""))
        station = item.get("gt_station", item.get("station_name", ""))
        if appearance or station:
            gt_text = build_target_text(label, appearance, station)
        else:
            gt_text = label

    return {
        "type": media_type,
        "media_path": media_path,
        "oral_instruction": oral,
        "user_instruction": user_instruction,
        "gt_text": gt_text,
        "target_label": label,
        "gt_appearance": item.get("gt_appearance", item.get("appearance_adjectives", "")),
        "gt_station": item.get("gt_station", item.get("station_name", "")),
        "system_instruction": item.get("system_instruction", SYSTEM_PROMPT),
        "id": item.get("id", ""),
    }


def get_training_user_text(sample: dict) -> str:
    """User text for training: prefer gastrohun user_instruction, else oral 3-task prompt."""
    if sample.get("user_instruction"):
        return sample["user_instruction"]
    return build_user_prompt(sample.get("oral_instruction", ""))


def build_train_video_content(video_path: str) -> dict:
    """Build a Qwen-VL video content dict with training pixel/frame limits."""
    return {
        "type": "video",
        "video": video_path,
        "fps": TRAIN_VIDEO_FPS,
        "min_frames": TRAIN_VIDEO_MIN_FRAMES,
        "max_frames": TRAIN_VIDEO_MAX_FRAMES,
        "min_pixels": TRAIN_VIDEO_MIN_PIXELS,
        "max_pixels": TRAIN_VIDEO_MAX_PIXELS,
    }


def setup_processor_image_size(processor):
    """Configure processor image and video size settings for Qwen VL models."""
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {
            "shortest_edge": min(IMAGE_WIDTH, IMAGE_HEIGHT),
            "longest_edge": max(IMAGE_WIDTH, IMAGE_HEIGHT),
        }
        if hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = IMAGE_MIN_PIXELS
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = IMAGE_MAX_PIXELS

    if hasattr(processor, "video_processor"):
        vp = processor.video_processor
        if hasattr(vp, "min_pixels"):
            vp.min_pixels = TRAIN_VIDEO_MIN_PIXELS
        if hasattr(vp, "max_pixels"):
            vp.max_pixels = TRAIN_VIDEO_MAX_PIXELS
        if hasattr(vp, "size"):
            if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
                vp.size = {
                    "shortest_edge": min(VIDEO_WIDTH, VIDEO_HEIGHT),
                    "longest_edge": max(VIDEO_WIDTH, VIDEO_HEIGHT),
                }
            else:
                vp.size = {
                    "shortest_edge": TRAIN_VIDEO_TOTAL_MIN_PIXELS,
                    "longest_edge": TRAIN_VIDEO_TOTAL_MAX_PIXELS,
                }
        if hasattr(vp, "fps"):
            vp.fps = TRAIN_VIDEO_FPS
        if hasattr(vp, "min_frames"):
            vp.min_frames = TRAIN_VIDEO_MIN_FRAMES
        if hasattr(vp, "max_frames"):
            vp.max_frames = TRAIN_VIDEO_MAX_FRAMES

    return processor


def validate_config():
    """Validate configuration and print summary."""
    print("\n" + "=" * 70)
    print(f"{PROJECT_NAME} v{PROJECT_VERSION} — Configuration Summary")
    print("=" * 70)

    print(f"\nModel: {BASE_MODEL_NAME}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"Image: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Video: {VIDEO_WIDTH}x{VIDEO_HEIGHT}/frame, "
          f"fps={TRAIN_VIDEO_FPS}, frames={TRAIN_VIDEO_MIN_FRAMES}-{TRAIN_VIDEO_MAX_FRAMES}")
    print(f"Stations: {AVAILABLE_STATIONS}")

    print(f"\nExact appearance mode: ENABLED")
    print(f"  Total unique appearances: {len(ALL_EXACT_APPEARANCES_FLAT)}")
    for tk, apps in ALL_EXACT_APPEARANCES.items():
        print(f"  {tk}: {len(apps)} variants")

    print(f"\nSFT: batch={SFT_CONFIG['batch_size']}, epochs={SFT_CONFIG['num_epochs']}, "
          f"lr={SFT_CONFIG['learning_rate']}")
    print(f"RFT: batch={RFT_CONFIG['batch_size']}, epochs={RFT_CONFIG['num_epochs']}, "
          f"lr={RFT_CONFIG['learning_rate']}, gens={RFT_CONFIG['num_generations']}")
    print(f"Reward: format={REWARD_CONFIG['format_weight']}, "
          f"accuracy={REWARD_CONFIG['accuracy_weight']}")

    print(f"\nWandB: project={WANDB_CONFIG['project']}, entity={WANDB_CONFIG['entity']}")
    print(f"Eval: {EVAL_CONFIG['eval_samples_per_target']} per target × 3 = "
          f"{EVAL_CONFIG['eval_samples_per_target'] * 3} total")

    print(f"\nData sources:")
    for ds in DATA_SOURCES:
        exists_json = os.path.exists(ds["json_path"]) if ds["json_path"] else False
        exists_img = os.path.exists(ds["image_dir"]) if ds["image_dir"] else False
        print(f"  [{ds['split']}] {ds['name']}: json={'✓' if exists_json else '✗'} "
              f"images={'✓' if exists_img else '✗'}")

    print("\nTraining Modes:")
    print("  Mode 1: Train from base model (attach new LoRA adapter)")
    print("  Mode 2: Continue from checkpoint with existing adapters")
    print("  Mode 3: Continue from checkpoint with merged adapters (attach new LoRA adapter)")


if __name__ == "__main__":
    validate_config()
