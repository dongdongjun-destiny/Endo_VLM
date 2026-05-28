#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EndoVLA-Oral Evaluation Pipeline

Evaluates oral instruction → standardized command conversion with a 3×4 metrics table:

Rows (3 lesion targets):
    - Greater curvature (white oval lesion)
    - Lesser curvature (orange protruding lesion)
    - Pyloric antrum (small round nodule)

Columns (4 metrics):
    1. Conversion Success Rate: All 3 fields correct (full match)
    2. Selection Accuracy:      a/b/c label correct
    3. Appearance Accuracy:     Appearance keyword correct
    4. Station Accuracy:        Station selection correct

Total: 12 metric cells + aggregate row (Overall)

Usage:
    # Evaluate a fine-tuned model
    python evaluate.py --model_path ./models/oral_rft_v1 \\
        --eval_data ./data/eval.json --image_dir ./data/abc_images \\
        --output_dir ./eval_results --run_tag rft_v1

    # Evaluate base model (baseline)
    python evaluate.py --model_name qwen2.5_3b \\
        --eval_data ./data/eval.json --image_dir ./data/abc_images \\
        --output_dir ./eval_results --run_tag baseline

    # Evaluate external API model (e.g., Grok)
    python evaluate.py --api_model grok \\
        --eval_data ./data/eval.json --image_dir ./data/abc_images \\
        --output_dir ./eval_results --run_tag grok_baseline
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
from PIL import Image

from config import (
    BASE_MODEL_NAME, AVAILABLE_MODELS,
    SYSTEM_PROMPT, AVAILABLE_STATIONS,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    EVAL_CONFIG, FOLDER_TO_APPEARANCE, FOLDER_TO_STATION,
    build_user_prompt, parse_prediction,
    normalize_appearance, normalize_station,
    get_generation_config, is_thinking_model,
    setup_processor_image_size,
)


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="EndoVLA-Oral Evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--model_name", type=str, default=None,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Base model name (for baseline evaluation)")
    parser.add_argument("--api_model", type=str, default=None,
                        choices=["grok", "gpt4o", "gemini"],
                        help="External API model for baseline comparison")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation JSON file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing ABC images")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory for results")
    parser.add_argument("--run_tag", type=str, required=True,
                        help="Tag for this evaluation run")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max evaluation samples (default: all)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size")
    return parser.parse_args()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_eval_data(json_path, image_dir, max_samples=None):
    """Load evaluation data."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        image_name = item.get("image_path", item.get("abc_image_name", ""))
        full_path = os.path.join(image_dir, image_name)
        if not os.path.exists(full_path):
            continue

        samples.append({
            "image_path": full_path,
            "image_name": image_name,
            "oral_instruction": item.get("oral_instruction", ""),
            "target_label": item.get("target_label", ""),
            "gt_appearance": item.get("gt_appearance", ""),
            "gt_station": item.get("gt_station", ""),
            "gt_text": item.get("gt_text", ""),
            "target_key": item.get("target_key", ""),
            "id": item.get("id", len(samples)),
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"Loaded {len(samples)} evaluation samples")
    target_counts = {}
    for s in samples:
        tk = s["target_key"]
        target_counts[tk] = target_counts.get(tk, 0) + 1
    print(f"  Per target: {dict(sorted(target_counts.items()))}")
    return samples


# ==============================================================================
# MODEL INFERENCE
# ==============================================================================

def load_model(model_path=None, model_name=None):
    """Load model for evaluation."""
    from unsloth import FastVisionModel

    if model_path:
        load_path = model_path
    elif model_name:
        load_path = AVAILABLE_MODELS.get(model_name, BASE_MODEL_NAME)
    else:
        load_path = BASE_MODEL_NAME

    print(f"Loading model: {load_path}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=load_path, load_in_4bit=True,
    )
    processor = setup_processor_image_size(processor)
    """Evaluate model on a subset of samples."""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    FastVisionModel.for_inference(model)
    return model, processor


def run_local_inference(model, processor, image, oral_instruction, model_name=BASE_MODEL_NAME):
    """Run inference with local model."""
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
            do_sample=False,
            use_cache=True,
        )

    output_text = processor.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in output_text.lower():
        parts = output_text.split("assistant")
        output_text = parts[-1].strip(": \n")

    return output_text.strip()


def run_api_inference(api_model, image_path, oral_instruction):
    """
    Run inference with external API model.

    Currently supports: grok (xAI), gpt4o (OpenAI), gemini (Google)
    Requires corresponding API keys in environment variables.

    Returns raw response text.
    """
    import base64

    user_prompt = build_user_prompt(oral_instruction)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # Encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    if api_model == "grok":
        return _call_grok_api(full_prompt, img_b64)
    elif api_model == "gpt4o":
        return _call_openai_api(full_prompt, img_b64)
    elif api_model == "gemini":
        return _call_gemini_api(full_prompt, img_b64)
    else:
        raise ValueError(f"Unknown API model: {api_model}")


def _call_grok_api(prompt, img_b64):
    """Call xAI Grok API."""
    import requests
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise ValueError("XAI_API_KEY not set")

    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "grok-2-vision-latest",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_openai_api(prompt, img_b64):
    """Call OpenAI GPT-4o API."""
    import requests
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini_api(prompt, img_b64):
    """Call Google Gemini API."""
    import requests
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the 3×4 evaluation metrics table + overall aggregate.

    Returns:
        Dict with:
            - per_target: {target_key: {metric: value}} (3×4 table)
            - overall: {metric: value} (aggregate row)
            - raw_counts: detailed counts for each cell
    """
    target_keys = ["greater_curvature", "lesser_curvature", "pyloric_antrum"]
    metric_names = ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]

    # Initialize counts
    counts = {tk: {m: {"correct": 0, "total": 0} for m in metric_names} for tk in target_keys}
    overall_counts = {m: {"correct": 0, "total": 0} for m in metric_names}

    for pred in predictions:
        tk = pred["target_key"]
        if tk not in counts:
            continue

        gt_label = pred["gt_label"].lower()
        gt_app = pred["gt_appearance"].lower()
        gt_station = pred["gt_station"].lower()

        parsed = pred.get("parsed_prediction")

        # Check each metric
        label_correct = False
        app_correct = False
        station_correct = False

        if parsed:
            label_correct = parsed["label"] == gt_label
            app_correct = normalize_appearance(parsed["appearance"]) == gt_app
            station_correct = normalize_station(parsed["station"]) == gt_station

        all_correct = label_correct and app_correct and station_correct

        # Update counts
        for metric, is_correct in [
            ("conversion_success", all_correct),
            ("selection_accuracy", label_correct),
            ("appearance_accuracy", app_correct),
            ("station_accuracy", station_correct),
        ]:
            counts[tk][metric]["total"] += 1
            counts[tk][metric]["correct"] += int(is_correct)
            overall_counts[metric]["total"] += 1
            overall_counts[metric]["correct"] += int(is_correct)

    # Compute rates
    per_target = {}
    for tk in target_keys:
        per_target[tk] = {}
        for m in metric_names:
            total = counts[tk][m]["total"]
            correct = counts[tk][m]["correct"]
            per_target[tk][m] = correct / total if total > 0 else 0.0

    overall = {}
    for m in metric_names:
        total = overall_counts[m]["total"]
        correct = overall_counts[m]["correct"]
        overall[m] = correct / total if total > 0 else 0.0

    return {
        "per_target": per_target,
        "overall": overall,
        "raw_counts": counts,
        "overall_counts": overall_counts,
    }


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

def print_metrics_table(metrics: Dict[str, Any], run_tag: str = ""):
    """Print the 3×4 metrics table in a formatted way."""
    per_target = metrics["per_target"]
    overall = metrics["overall"]

    target_display = {
        "greater_curvature": "Greater Curvature",
        "lesser_curvature": "Lesser Curvature",
        "pyloric_antrum": "Pyloric Antrum",
    }

    metric_display = {
        "conversion_success": "Conv. Success",
        "selection_accuracy": "Selection Acc.",
        "appearance_accuracy": "Appearance Acc.",
        "station_accuracy": "Station Acc.",
    }

    print(f"\n{'=' * 80}")
    print(f"Evaluation Results: {run_tag}")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Target':<22} | {'Conv. Success':>14} | {'Selection':>10} | {'Appearance':>11} | {'Station':>10}"
    print(header)
    print("-" * 80)

    # Per-target rows
    for tk in ["greater_curvature", "lesser_curvature", "pyloric_antrum"]:
        row = f"{target_display[tk]:<22}"
        for m in ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]:
            val = per_target[tk][m]
            width = 14 if m == "conversion_success" else (10 if m in ["selection_accuracy", "station_accuracy"] else 11)
            row += f" | {val:>{width}.2%}"
        print(row)

    print("-" * 80)

    # Overall row
    row = f"{'Overall':<22}"
    for m in ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]:
        val = overall[m]
        width = 14 if m == "conversion_success" else (10 if m in ["selection_accuracy", "station_accuracy"] else 11)
        row += f" | {val:>{width}.2%}"
    print(row)
    print("=" * 80)


def generate_latex_table(metrics: Dict[str, Any], run_tag: str = "") -> str:
    """Generate LaTeX table for paper."""
    per_target = metrics["per_target"]
    overall = metrics["overall"]

    target_display = {
        "greater_curvature": "Greater Curvature",
        "lesser_curvature": "Lesser Curvature",
        "pyloric_antrum": "Pyloric Antrum",
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Oral Instruction Conversion Results ({run_tag})}}")
    lines.append(f"\\label{{tab:eval_{run_tag.replace(' ', '_')}}}")
    lines.append(r"\begin{tabular}{l|cccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Target Station} & \textbf{Conv. Success} & \textbf{Selection Acc.} & \textbf{Appearance Acc.} & \textbf{Station Acc.} \\")
    lines.append(r"\midrule")

    for tk in ["greater_curvature", "lesser_curvature", "pyloric_antrum"]:
        vals = [per_target[tk][m] for m in
                ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]]
        vals_str = " & ".join([f"{v:.1%}" for v in vals])
        lines.append(f"{target_display[tk]} & {vals_str} \\\\")

    lines.append(r"\midrule")

    overall_vals = [overall[m] for m in
                    ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]]
    overall_str = " & ".join([f"\\textbf{{{v:.1%}}}" for v in overall_vals])
    lines.append(f"\\textbf{{Overall}} & {overall_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_combined_latex_table(all_results: Dict[str, Dict], model_names: List[str]) -> str:
    """Generate combined LaTeX table comparing multiple models."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of Oral Instruction Conversion Across Models}")
    lines.append(r"\label{tab:model_comparison}")

    # columns: Model | then 4 metrics per target + overall
    ncols = 1 + 4  # model + 4 metrics
    lines.append(f"\\begin{{tabular}}{{l|{'c' * 4}}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Conv. Succ.} & \textbf{Select.} & \textbf{Appear.} & \textbf{Station} \\")
    lines.append(r"\midrule")

    for name in model_names:
        metrics = all_results[name]
        overall = metrics["overall"]
        vals = [overall[m] for m in
                ["conversion_success", "selection_accuracy", "appearance_accuracy", "station_accuracy"]]
        vals_str = " & ".join([f"{v:.1%}" for v in vals])
        lines.append(f"{name} & {vals_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ==============================================================================
# PREDICTION LOGGING
# ==============================================================================

def save_predictions(predictions, output_path):
    """Save all predictions to CSV for analysis."""
    if not predictions:
        return

    fieldnames = [
        "id", "target_key", "gt_label", "gt_appearance", "gt_station", "gt_text",
        "raw_output", "parsed_label", "parsed_appearance", "parsed_station",
        "label_correct", "appearance_correct", "station_correct", "all_correct",
        "oral_instruction", "image_name",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for p in predictions:
            parsed = p.get("parsed_prediction") or {}
            gt_label = p["gt_label"].lower()
            gt_app = p["gt_appearance"].lower()
            gt_station = p["gt_station"].lower()

            p_label = parsed.get("label", "")
            p_app = normalize_appearance(parsed.get("appearance", "")) if parsed else ""
            p_station = normalize_station(parsed.get("station", "")) if parsed else ""

            label_ok = p_label == gt_label
            app_ok = p_app == gt_app
            station_ok = p_station == gt_station

            writer.writerow({
                "id": p.get("id", ""),
                "target_key": p["target_key"],
                "gt_label": gt_label,
                "gt_appearance": gt_app,
                "gt_station": gt_station,
                "gt_text": p.get("gt_text", ""),
                "raw_output": p.get("raw_output", ""),
                "parsed_label": p_label,
                "parsed_appearance": p_app,
                "parsed_station": p_station,
                "label_correct": int(label_ok),
                "appearance_correct": int(app_ok),
                "station_correct": int(station_ok),
                "all_correct": int(label_ok and app_ok and station_ok),
                "oral_instruction": p.get("oral_instruction", ""),
                "image_name": p.get("image_name", ""),
            })

    print(f"Predictions saved → {output_path}")


# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

def evaluate(args):
    """Run full evaluation."""
    print("=" * 70)
    print("EndoVLA-Oral Evaluation Pipeline")
    print(f"Run tag: {args.run_tag}")
    print("=" * 70)

    # Load data
    samples = load_eval_data(args.eval_data, args.image_dir, args.max_samples)
    if not samples:
        print("ERROR: No evaluation samples!")
        sys.exit(1)

    # Setup model
    model, processor = None, None
    use_api = args.api_model is not None

    if not use_api:
        model, processor = load_model(args.model_path, args.model_name)

    # Run inference
    predictions = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s, ETA: {eta:.0f}s")

        try:
            if use_api:
                raw_output = run_api_inference(
                    args.api_model, sample["image_path"], sample["oral_instruction"]
                )
            else:
                image = Image.open(sample["image_path"]).convert("RGB")
                raw_output = run_local_inference(
                    model, processor, image, sample["oral_instruction"],
                    model_name=args.model_path or BASE_MODEL_NAME,
                )

            parsed = parse_prediction(raw_output)

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            raw_output = f"ERROR: {e}"
            parsed = None

        predictions.append({
            "id": sample["id"],
            "target_key": sample["target_key"],
            "gt_label": sample["target_label"],
            "gt_appearance": sample["gt_appearance"],
            "gt_station": sample["gt_station"],
            "gt_text": sample["gt_text"],
            "raw_output": raw_output,
            "parsed_prediction": parsed,
            "oral_instruction": sample["oral_instruction"],
            "image_name": sample["image_name"],
        })

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(predictions)} samples in {elapsed:.1f}s "
          f"({len(predictions)/elapsed:.1f} samples/s)")

    # Compute metrics
    metrics = compute_metrics(predictions)

    # Print table
    print_metrics_table(metrics, args.run_tag)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save predictions CSV
    pred_path = os.path.join(args.output_dir, f"predictions_{args.run_tag}.csv")
    save_predictions(predictions, pred_path)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, f"metrics_{args.run_tag}.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "run_tag": args.run_tag,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "elapsed_seconds": elapsed,
            "per_target": metrics["per_target"],
            "overall": metrics["overall"],
            "raw_counts": {
                tk: {m: dict(v) for m, v in ms.items()}
                for tk, ms in metrics["raw_counts"].items()
            },
        }, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # Save LaTeX table
    latex = generate_latex_table(metrics, args.run_tag)
    latex_path = os.path.join(args.output_dir, f"table_{args.run_tag}.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved → {latex_path}")

    return metrics


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
