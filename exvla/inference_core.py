"""
EndoVLA inference engine: single/batch image & video, thinking parse, clinical metrics.
Used by inference_ui.py and inference_test.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    from config import (
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        SYSTEM_PROMPT,
        build_train_video_content,
        get_generation_config,
        get_grpo_system_prompt,
        get_grpo_user_text,
        get_training_user_text,
        is_thinking_model,
        normalize_training_sample,
        parse_final_sss_first_line,
        parse_gastrohun_eval_label,
        setup_processor_image_size,
    )
except ImportError:
    IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768
    SYSTEM_PROMPT = "You are an expert gastrointestinal endoscopist."
    build_train_video_content = None
    get_generation_config = None
    get_grpo_system_prompt = lambda _=None: SYSTEM_PROMPT
    get_grpo_user_text = None
    get_training_user_text = None
    is_thinking_model = lambda _: False
    normalize_training_sample = None
    parse_final_sss_first_line = None
    parse_gastrohun_eval_label = None
    setup_processor_image_size = lambda p: p

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferenceResult:
    raw_text: str
    thinking: str
    answer: str
    pred_label: str
    latency_sec: float
    media_path: str = ""
    media_type: str = "image"
    error: Optional[str] = None


@dataclass
class TestRowMetrics:
    sample_id: str
    media_path: str
    gt_answer: str
    pred_label: str
    raw_response: str
    thinking: str
    is_depth: bool
    is_depth_tolerant: bool
    is_wall: bool
    is_feature_hit: bool
    exact_match: bool
    unique_label_ok: bool
    latency_sec: float
    error: Optional[str] = None


def parse_thinking_output(text: str) -> Tuple[str, str]:
    """Split model output into (thinking, answer) like ChatGPT reasoning UI."""
    raw = (text or "").strip()
    if not raw:
        return "", ""

    _f = re.DOTALL | re.IGNORECASE
    _tags = ("think", "thinking", "reasoning")
    patterns = [
        (rf"<{tag}>(.*?)</{tag}>", _f) for tag in _tags
    ]
    thinking_parts: List[str] = []
    remainder = raw
    for pat, flags in patterns:
        for m in re.finditer(pat, remainder, flags):
            thinking_parts.append(m.group(1).strip())
        remainder = re.sub(pat, "", remainder, flags=flags).strip()

    if thinking_parts:
        thinking = "\n\n".join(thinking_parts).strip()
        answer = remainder.strip()
        if not answer:
            answer = raw
        return thinking, answer

    # Leading Final SSS format: first line is the answer
    if parse_final_sss_first_line:
        head = parse_final_sss_first_line(raw)
        if head:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            answer = lines[0] if lines else raw
            thinking = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            return thinking, answer

    tail = re.search(
        r"((?:Final\s+SSS|SSS|Final)\s*[:：]\s*[AGLP][1-6]|NA)\s*$",
        raw,
        re.IGNORECASE | re.MULTILINE,
    )
    if tail:
        answer = tail.group(0).strip()
        thinking = raw[: tail.start()].strip()
        return thinking, answer

    if len(raw) > 120 and "\n" in raw:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return "\n".join(lines[1:]), lines[0]

    return "", raw


def extract_sss_label(text: str) -> str:
    """Extract SSS label; align with GRPO eval (Final SSS first line)."""
    if parse_gastrohun_eval_label:
        label = parse_gastrohun_eval_label(text)
        if label:
            return label
    if parse_final_sss_first_line:
        label = parse_final_sss_first_line(text)
        if label:
            return label
    upper = (text or "").upper()
    labels = re.findall(r"\b([AGLP][1-6]|NA|OTHERCLASS)\b", upper)
    if not labels:
        match = re.search(r"\b([AGLP][1-6]|NA|OTHERCLASS)\b", upper)
        return match.group(1) if match else upper.strip()[:32]
    if len(labels) == 1:
        return labels[0]
    return labels[0]


def count_sss_labels(text: str) -> int:
    upper = (text or "").upper()
    return len(re.findall(r"\b([AGLP][1-6]|NA|OTHERCLASS)\b", upper))


def compute_clinical_metrics(gt_answer: str, pred_label: str) -> Dict[str, bool]:
    gt = str(gt_answer).strip().upper()
    pred = str(pred_label).strip().upper()
    is_depth = is_depth_tolerant = is_wall = is_feature_hit = False

    if gt == "OTHERCLASS" or pred == gt:
        is_depth = is_depth_tolerant = is_wall = is_feature_hit = True
    elif (
        len(gt) == 2
        and len(pred) == 2
        and gt != "NA"
        and pred != "NA"
        and pred[0] in "AGLP"
        and gt[0] in "AGLP"
    ):
        gt_wall, gt_num = gt[0], int(gt[1])
        pred_wall, pred_num = pred[0], int(pred[1])
        if gt_num == pred_num:
            is_depth = True
        if abs(gt_num - pred_num) <= 1:
            is_depth_tolerant = True
        if gt_wall == pred_wall:
            is_wall = True
        is_feature_hit = is_depth or is_wall

    exact = pred == gt
    return {
        "is_depth": is_depth,
        "is_depth_tolerant": is_depth_tolerant,
        "is_wall": is_wall,
        "is_feature_hit": is_feature_hit,
        "exact_match": exact,
    }


def load_jsonl(path: str, image_dir: str = "") -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if normalize_training_sample:
                norm = normalize_training_sample(item, image_dir=image_dir)
                if norm:
                    samples.append(norm)
                    continue
            samples.append(item)
    return samples


def detect_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    return "image"


class EndoVLAInferenceEngine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor = None
        self.model = None
        self._model_name_hint = os.path.basename(model_path.rstrip("/"))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self, progress: Optional[Callable[[str], None]] = None) -> str:
        if self.is_loaded:
            return f"Model already loaded: {self.model_path}"

        def log(msg: str):
            if progress:
                progress(msg)

        log("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if setup_processor_image_size:
            self.processor = setup_processor_image_size(self.processor)

        log("Loading model (may take 1–3 min)...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        self.model.eval()
        return f"Loaded: {self.model_path}"

    def unload(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _gen_kwargs(self, for_eval: bool, max_new_tokens: int) -> Dict[str, Any]:
        if get_generation_config:
            cfg = get_generation_config(self._model_name_hint, for_eval=for_eval)
        else:
            cfg = {"temperature": 0.1 if for_eval else 0.7, "max_new_tokens": max_new_tokens}
        kwargs = {"max_new_tokens": cfg.get("max_new_tokens", max_new_tokens)}
        if cfg.get("do_sample", not for_eval):
            kwargs["do_sample"] = True
            for k in ("temperature", "top_p", "top_k"):
                if k in cfg:
                    kwargs[k] = cfg[k]
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _build_messages(
        self,
        media_path: str,
        media_type: str,
        system_instruction: str,
        user_instruction: str,
    ) -> List[Dict[str, Any]]:
        if media_type == "video":
            if not build_train_video_content:
                raise RuntimeError("Video requires config.build_train_video_content and qwen_vl_utils")
            media_content = build_train_video_content(media_path)
        else:
            image = Image.open(media_path).convert("RGB")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
            media_content = {"type": "image", "image": image}

        return [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [media_content, {"type": "text", "text": user_instruction}],
            },
        ]

    def predict(
        self,
        media_path: str,
        system_instruction: str = "",
        user_instruction: str = "",
        media_type: Optional[str] = None,
        for_eval: bool = False,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        if not self.is_loaded:
            raise RuntimeError("Call load() first")

        media_type = media_type or detect_media_type(media_path)
        sys_p = system_instruction or SYSTEM_PROMPT
        user_p = user_instruction or "Analyze this gastroscopy media and provide your reasoning, then the SSS label."

        if not os.path.exists(media_path):
            return InferenceResult(
                raw_text="",
                thinking="",
                answer="",
                pred_label="",
                latency_sec=0.0,
                media_path=media_path,
                media_type=media_type,
                error=f"File not found: {media_path}",
            )

        try:
            messages = self._build_messages(media_path, media_type, sys_p, user_p)
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if process_vision_info:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                image = Image.open(media_path).convert("RGB")
                inputs = self.processor(
                    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
                )

            inputs = inputs.to(self.model.device)

            t0 = time.time()
            gen_kw = self._gen_kwargs(for_eval=for_eval, max_new_tokens=max_new_tokens)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kw)

            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            if "assistant" in raw.lower():
                raw = raw.split("assistant")[-1].strip(": \n")

            latency = time.time() - t0
            thinking, answer = parse_thinking_output(raw)
            pred = extract_sss_label(answer or raw)

            return InferenceResult(
                raw_text=raw,
                thinking=thinking,
                answer=answer or raw,
                pred_label=pred,
                latency_sec=latency,
                media_path=media_path,
                media_type=media_type,
            )
        except Exception as e:
            return InferenceResult(
                raw_text="",
                thinking="",
                answer="",
                pred_label="",
                latency_sec=0.0,
                media_path=media_path,
                media_type=media_type,
                error=str(e),
            )

    def predict_sample(self, sample: Dict[str, Any], for_eval: bool = True) -> InferenceResult:
        media_path = sample.get("media_path") or sample.get("image_path") or sample.get("video_path", "")
        media_type = sample.get("type", detect_media_type(media_path))
        if media_type in ("sequence",):
            media_type = "video"
        if get_grpo_user_text:
            sys_p = get_grpo_system_prompt("gastrohun")
            user_p = get_grpo_user_text(sample, "gastrohun")
        elif get_training_user_text:
            sys_p = sample.get("system_instruction", SYSTEM_PROMPT)
            user_p = get_training_user_text(sample)
        else:
            sys_p = sample.get("system_instruction", SYSTEM_PROMPT)
            user_p = sample.get("user_instruction", sample.get("oral_instruction", ""))
        gt = sample.get("target_label") or sample.get("label_code") or sample.get("answer", "")
        res = self.predict(media_path, sys_p, user_p, media_type=media_type, for_eval=for_eval)
        if not res.error and gt:
            res.answer = res.answer  # keep
        return res

    def run_test_evaluation(
        self,
        jsonl_path: str,
        image_dir: str = "",
        output_excel: str = "",
        max_samples: Optional[int] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float], str]:
        samples = load_jsonl(jsonl_path, image_dir=image_dir)
        if max_samples:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError(f"No samples loaded from {jsonl_path}")

        rows: List[Dict[str, Any]] = []
        n = len(samples)
        depth_ok = depth_tol_ok = wall_ok = feature_ok = exact_ok = unique_ok = 0
        valid = 0

        for i, sample in enumerate(samples):
            if progress:
                progress((i + 1) / n, f"{i + 1}/{n}")

            sid = sample.get("id", f"sample_{i}")
            media_path = sample.get("media_path") or sample.get("image_path", "")
            gt = str(
                sample.get("target_label")
                or sample.get("label_code")
                or sample.get("answer", "")
            ).strip().upper()

            if not os.path.exists(media_path):
                rows.append(
                    {
                        "样本 ID": sid,
                        "媒体路径": media_path,
                        "媒体类型": sample.get("type", ""),
                        "金标准": gt,
                        "预测标签": "FILE_NOT_FOUND",
                        "完整回复": "",
                        "思考过程": "",
                        "唯一标签合规": False,
                        "完全匹配": False,
                        "纵向深度命中": False,
                        "±1级深度容错": False,
                        "圆周方位命中": False,
                        "特征有效捕获": False,
                        "推理延迟 (秒)": 0.0,
                        "错误": "file not found",
                    }
                )
                continue

            res = self.predict_sample(sample, for_eval=True)
            if res.error:
                rows.append(
                    {
                        "样本 ID": sid,
                        "媒体路径": media_path,
                        "媒体类型": res.media_type,
                        "金标准": gt,
                        "预测标签": "ERROR",
                        "完整回复": "",
                        "思考过程": "",
                        "唯一标签合规": False,
                        "完全匹配": False,
                        "纵向深度命中": False,
                        "±1级深度容错": False,
                        "圆周方位命中": False,
                        "特征有效捕获": False,
                        "推理延迟 (秒)": 0.0,
                        "错误": res.error,
                    }
                )
                continue

            n_labels = count_sss_labels(res.raw_text)
            unique_label_ok = n_labels == 1
            m = compute_clinical_metrics(gt, res.pred_label)
            valid += 1
            if m["is_depth"]:
                depth_ok += 1
            if m["is_depth_tolerant"]:
                depth_tol_ok += 1
            if m["is_wall"]:
                wall_ok += 1
            if m["is_feature_hit"]:
                feature_ok += 1
            if m["exact_match"]:
                exact_ok += 1
            if unique_label_ok:
                unique_ok += 1

            rows.append(
                {
                    "样本 ID": sid,
                    "媒体路径": media_path,
                    "媒体类型": res.media_type,
                    "金标准": gt,
                    "预测标签": res.pred_label,
                    "完整回复": res.raw_text,
                    "思考过程": res.thinking,
                    "唯一标签合规": unique_label_ok,
                    "完全匹配": m["exact_match"],
                    "纵向深度命中": m["is_depth"],
                    "±1级深度容错": m["is_depth_tolerant"],
                    "圆周方位命中": m["is_wall"],
                    "特征有效捕获": m["is_feature_hit"],
                    "推理延迟 (秒)": round(res.latency_sec, 4),
                    "错误": "",
                }
            )

        summary = {}
        if valid > 0:
            summary = {
                "valid_count": valid,
                "exact_match_rate": exact_ok / valid,
                "unique_label_rate": unique_ok / valid,
                "depth_acc": depth_ok / valid,
                "depth_tol_acc": depth_tol_ok / valid,
                "wall_acc": wall_ok / valid,
                "feature_acc": feature_ok / valid,
                "avg_latency": sum(r["推理延迟 (秒)"] for r in rows if r.get("推理延迟 (秒)")) / max(len(rows), 1),
            }

        df = pd.DataFrame(rows)
        if summary:
            summary_row = {
                "样本 ID": "【临床能力评估】",
                "媒体路径": f"总计: {valid}",
                "媒体类型": "",
                "金标准": "",
                "预测标签": "汇总 ->",
                "完整回复": f"完全匹配: {summary['exact_match_rate']*100:.2f}%",
                "思考过程": f"唯一标签: {summary['unique_label_rate']*100:.2f}%",
                "唯一标签合规": "",
                "完全匹配": f"{summary['exact_match_rate']*100:.2f}%",
                "纵向深度命中": f"{summary['depth_acc']*100:.2f}%",
                "±1级深度容错": f"{summary['depth_tol_acc']*100:.2f}%",
                "圆周方位命中": f"{summary['wall_acc']*100:.2f}%",
                "特征有效捕获": f"{summary['feature_acc']*100:.2f}%",
                "推理延迟 (秒)": f"平均: {summary['avg_latency']:.4f}s",
                "错误": "",
            }
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        out_path = output_excel or os.path.join(
            os.path.dirname(jsonl_path) or ".",
            "evaluation_report_ui.xlsx",
        )
        df.to_excel(out_path, index=False)
        return df, summary, out_path


def format_chat_html(thinking: str, answer: str, pred_label: str, latency: float, error: Optional[str]) -> str:
    if error:
        return f'<div style="color:#b91c1c;padding:12px;">Error: {error}</div>'

    think_block = ""
    if thinking:
        think_block = f"""
        <details open style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:8px;padding:8px;background:#f9fafb;">
          <summary style="cursor:pointer;font-weight:600;color:#374151;">推理过程 (Thinking)</summary>
          <pre style="white-space:pre-wrap;font-size:13px;color:#4b5563;margin:8px 0 0 0;">{_esc(thinking)}</pre>
        </details>
        """

    return f"""
    {think_block}
    <div style="border:1px solid #d1d5db;border-radius:8px;padding:12px;background:#fff;">
      <div style="font-weight:600;color:#111827;margin-bottom:8px;">最终回答</div>
      <pre style="white-space:pre-wrap;font-size:14px;margin:0;">{_esc(answer)}</pre>
      <div style="margin-top:10px;font-size:13px;color:#6b7280;">
        解析标签: <strong style="color:#059669;">{_esc(pred_label)}</strong>
        &nbsp;|&nbsp; 耗时: {latency:.2f}s
      </div>
    </div>
    """


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
