#!/usr/bin/env python3
"""
EndoVLA Gradio UI: chat (thinking + answer), batch inference, clinical test Excel.

  python inference_ui.py
  python inference_ui.py --model_path /path/to/merged --port 7860
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from inference_core import (
    EndoVLAInferenceEngine,
    detect_media_type,
    format_chat_html,
    load_jsonl,
)

DEFAULT_MODEL = "/home/rennc1/Documents/Yidong_code/exvla/models/grpo_5090_10percent_image_newDora_Ultimate_Merged"
DEFAULT_JSONL = "/home/rennc1/Documents/Yidong_code/exvla/output_dir/gastrohun_llm_en_images_train.jsonl"
DEFAULT_EXCEL = "/home/rennc1/Documents/Yidong_code/exvla/evaluation_report_clinical_metrics.xlsx"

_engine: Optional[EndoVLAInferenceEngine] = None
_engine_path: Optional[str] = None


def get_engine(model_path: str) -> EndoVLAInferenceEngine:
    global _engine, _engine_path
    path = model_path.strip()
    if _engine is None or _engine_path != path:
        if _engine is not None:
            _engine.unload()
        _engine = EndoVLAInferenceEngine(path)
        _engine_path = path
    return _engine


def ui_load_model(model_path: str) -> str:
    try:
        eng = get_engine(model_path)
        return eng.load()
    except Exception as e:
        return f"Load failed: {e}"


def ui_chat(
    model_path: str,
    media_file,
    media_kind: str,
    system_text: str,
    user_text: str,
    use_eval_mode: bool,
) -> Tuple[str, str, str]:
    if media_file is None:
        return "请上传图片或视频", "", ""
    eng = get_engine(model_path)
    if not eng.is_loaded:
        return "请先点击「加载模型」", "", ""

    path = media_file if isinstance(media_file, str) else getattr(media_file, "name", str(media_file))
    mtype = "video" if media_kind == "video" else detect_media_type(path)

    res = eng.predict(
        path,
        system_instruction=system_text,
        user_instruction=user_text,
        media_type=mtype,
        for_eval=use_eval_mode,
        max_new_tokens=512,
    )
    html = format_chat_html(res.thinking, res.answer, res.pred_label, res.latency_sec, res.error)
    meta = f"类型: {res.media_type} | 标签: {res.pred_label} | {res.latency_sec:.2f}s"
    if res.error:
        meta = res.error
    return html, res.raw_text, meta


def _collect_media_paths(
    jsonl_path: str,
    image_dir: str,
    upload_files: Optional[List],
    folder_path: str,
) -> List[dict]:
    items: List[dict] = []

    if jsonl_path and os.path.isfile(jsonl_path):
        for s in load_jsonl(jsonl_path, image_dir=image_dir):
            mp = s.get("media_path") or s.get("image_path", "")
            if mp and os.path.exists(mp):
                items.append(
                    {
                        "path": mp,
                        "type": s.get("type", detect_media_type(mp)),
                        "system": s.get("system_instruction", ""),
                        "user": s.get("user_instruction", ""),
                        "id": s.get("id", ""),
                    }
                )
        return items

    paths: List[str] = []
    if upload_files:
        for f in upload_files:
            p = f if isinstance(f, str) else getattr(f, "name", "")
            if p:
                paths.append(p)
    if folder_path and os.path.isdir(folder_path):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov"):
            paths.extend(glob.glob(os.path.join(folder_path, ext)))

    for p in sorted(set(paths)):
        items.append(
            {
                "path": p,
                "type": detect_media_type(p),
                "system": "",
                "user": "Look at the gastroscopy media, reason step by step, then give the SSS label.",
                "id": os.path.basename(p),
            }
        )
    return items


def ui_batch(
    model_path: str,
    jsonl_path: str,
    image_dir: str,
    upload_files,
    folder_path: str,
    system_text: str,
    user_text: str,
    max_items: int,
    progress=gr.Progress(),
) -> Tuple[pd.DataFrame, str]:
    eng = get_engine(model_path)
    if not eng.is_loaded:
        return pd.DataFrame(), "请先加载模型"

    items = _collect_media_paths(jsonl_path, image_dir, upload_files, folder_path)
    if not items:
        return pd.DataFrame(), "未找到可推理文件（检查 jsonl / 上传 / 文件夹路径）"

    if max_items > 0:
        items = items[:max_items]

    rows = []
    n = len(items)
    for i, it in enumerate(items):
        progress((i + 1) / n, desc=f"{i + 1}/{n}")
        sys_p = it["system"] or system_text
        usr_p = it["user"] or user_text
        res = eng.predict(it["path"], sys_p, usr_p, media_type=it["type"], for_eval=True)
        rows.append(
            {
                "ID": it.get("id", ""),
                "路径": it["path"],
                "类型": res.media_type,
                "预测标签": res.pred_label,
                "思考过程": (res.thinking or "")[:500],
                "最终回答": (res.answer or "")[:300],
                "延迟(s)": round(res.latency_sec, 3),
                "错误": res.error or "",
            }
        )

    df = pd.DataFrame(rows)
    return df, f"完成 {len(rows)} 条推理"


def ui_test(
    model_path: str,
    jsonl_path: str,
    image_dir: str,
    excel_path: str,
    max_samples: int,
    progress=gr.Progress(),
) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    eng = get_engine(model_path)
    if not eng.is_loaded:
        return pd.DataFrame(), "请先加载模型", "", None

    if not jsonl_path or not os.path.isfile(jsonl_path):
        return pd.DataFrame(), "请填写有效的 jsonl 路径", "", None

    out = excel_path.strip() or DEFAULT_EXCEL
    lim = int(max_samples) if max_samples and max_samples > 0 else None

    def prog(p: float, msg: str):
        progress(p, desc=msg)

    try:
        df, summary, saved = eng.run_test_evaluation(
            jsonl_path,
            image_dir=image_dir,
            output_excel=out,
            max_samples=lim,
            progress=prog,
        )
    except Exception as e:
        return pd.DataFrame(), str(e), "", None

    if not summary:
        return df, "无有效样本", saved, saved

    md = f"""### 测试汇总
| 指标 | 数值 |
|------|------|
| 有效样本 | {summary['valid_count']} |
| 完全匹配 | **{summary['exact_match_rate']*100:.2f}%** |
| 唯一标签合规 | {summary['unique_label_rate']*100:.2f}% |
| 纵向深度 | {summary['depth_acc']*100:.2f}% |
| ±1级深度容错 | **{summary['depth_tol_acc']*100:.2f}%** |
| 圆周方位 | {summary['wall_acc']*100:.2f}% |
| 特征捕获 | **{summary['feature_acc']*100:.2f}%** |
| 平均延迟 | {summary['avg_latency']:.3f}s |

报表: `{saved}`
"""
    return df, md, saved, saved


def build_ui(default_model: str) -> gr.Blocks:
    css = """
    .thinking-box { background:#f3f4f6; border-radius:8px; padding:12px; }
    """
    with gr.Blocks(title="EndoVLA 临床推理", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# EndoVLA 推理工作台\n"
            "对话（展示思考过程）· 批量图片/视频 · 测试模式（Excel 临床指标）"
        )

        with gr.Row():
            model_path = gr.Textbox(label="模型路径", value=default_model, scale=3)
            load_btn = gr.Button("加载模型", variant="primary")
            load_status = gr.Textbox(label="状态", interactive=False, scale=2)

        load_btn.click(
            ui_load_model,
            [model_path],
            [load_status],
        )

        with gr.Tabs():
            # ---- Chat ----
            with gr.Tab("对话推理"):
                gr.Markdown(
                    "类似 ChatGPT / Gemini：先展示 **推理过程**，再展示 **最终回答**。"
                    " Thinking 模型请关闭「评测采样模式」。"
                )
                with gr.Row():
                    chat_image = gr.Image(label="图片（二选一）", type="filepath")
                    chat_video = gr.Video(label="视频（二选一）", sources=["upload"])
                with gr.Row():
                    media_kind = gr.Radio(["image", "video"], value="image", label="媒体类型")
                    eval_mode = gr.Checkbox(
                        label="评测采样模式（低温、短输出，Thinking 建议关闭）",
                        value=False,
                    )
                system_tb = gr.Textbox(
                    label="System",
                    lines=4,
                    value="You are an expert gastrointestinal endoscopist...",
                )
                user_tb = gr.Textbox(
                    label="User",
                    lines=3,
                    value="Look at the gastroscopy image, reason about which SSS station it corresponds to, then answer with ONLY the label code.",
                )
                chat_btn = gr.Button("开始推理", variant="primary")
                chat_html = gr.HTML(label="回复")
                with gr.Accordion("原始文本", open=False):
                    raw_tb = gr.Textbox(label="模型完整输出", lines=8)
                chat_meta = gr.Textbox(label="元信息", interactive=False)

                def _run_chat(mp, mv, mk, sys, usr, ev, mpath):
                    media = mv if mk == "video" and mv else mp
                    return ui_chat(mpath, media, mk, sys, usr, ev)

                chat_btn.click(
                    _run_chat,
                    [chat_image, chat_video, media_kind, system_tb, user_tb, eval_mode, model_path],
                    [chat_html, raw_tb, chat_meta],
                )

            # ---- Batch ----
            with gr.Tab("批量推理"):
                gr.Markdown("支持：**jsonl 数据集** / **多文件上传** / **文件夹路径**（可混合图片与视频）")
                with gr.Row():
                    batch_jsonl = gr.Textbox(label="JSONL 路径（优先）", value="")
                    batch_img_dir = gr.Textbox(label="image_dir（相对路径前缀）", value="")
                with gr.Row():
                    batch_upload = gr.File(
                        label="上传多个文件",
                        file_count="multiple",
                        file_types=["image", "video"],
                    )
                    batch_folder = gr.Textbox(label="或：文件夹绝对路径", value="")
                batch_sys = gr.Textbox(label="默认 System（jsonl 为空时用）", lines=2)
                batch_user = gr.Textbox(label="默认 User", lines=2)
                batch_max = gr.Number(label="最多处理条数（0=全部）", value=0, precision=0)
                batch_btn = gr.Button("批量推理", variant="primary")
                batch_status = gr.Textbox(label="状态")
                batch_table = gr.Dataframe(label="结果预览", interactive=False)

                batch_btn.click(
                    ui_batch,
                    [
                        model_path,
                        batch_jsonl,
                        batch_img_dir,
                        batch_upload,
                        batch_folder,
                        batch_sys,
                        batch_user,
                        batch_max,
                    ],
                    [batch_table, batch_status],
                )

            # ---- Test ----
            with gr.Tab("测试评估"):
                gr.Markdown(
                    "与 `inference_test.py` 相同临床指标，导出 **Excel**（含思考过程、唯一标签合规等列）。"
                )
                with gr.Row():
                    test_jsonl = gr.Textbox(label="测试 JSONL", value=DEFAULT_JSONL)
                    test_img_dir = gr.Textbox(label="image_dir", value="/media/rennc1/Elements/exvla_clinical")
                test_excel = gr.Textbox(label="输出 Excel", value=DEFAULT_EXCEL)
                test_max = gr.Number(label="最多样本（0=全部）", value=0, precision=0)
                test_btn = gr.Button("运行测试并导出 Excel", variant="primary")
                test_summary = gr.Markdown()
                test_table = gr.Dataframe(label="结果（含汇总行）", interactive=False)
                test_file = gr.File(label="下载 Excel")

                test_btn.click(
                    ui_test,
                    [model_path, test_jsonl, test_img_dir, test_excel, test_max],
                    [test_table, test_summary, test_file, test_file],
                )

        gr.Markdown(
            "---\n"
            "**启动**: `python inference_ui.py`  \n"
            "**依赖**: `pip install gradio openpyxl`"
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.model_path)
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
