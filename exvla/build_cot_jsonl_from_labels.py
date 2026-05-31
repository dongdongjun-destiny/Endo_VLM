#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge StoBrain-style CoT1 labels into GastroHUN JSONL for image CoT-SFT.

CoT1 = image → Step1/2/3 reasoning → Final SSS label (matches your jsonl images).
CoT2 = navigation (current→target region); NOT used for SSS image classification.

Usage:
  python build_cot_jsonl_from_labels.py \\
    --input_jsonl output_dir/gastrohun_llm_en_multimodal_train.jsonl \\
    --cot1_json ../stobrain/ds/cot1_labels.json \\
    --output_jsonl output_dir/gastrohun_llm_en_images_train_cot1.jsonl \\
    --images_only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_image_key(image_path: str) -> Optional[Tuple[int, str]]:
    """Extract (patient_id, filename) from .../Labeled_Images/{pid}/{file}.jpg"""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.name:
        return None
    try:
        patient_id = int(p.parent.name)
    except ValueError:
        return None
    return patient_id, p.name


def load_cot1_index(
    cot1_path: str,
    require_reliable: bool = True,
) -> Dict[Tuple[int, str], dict]:
    with open(cot1_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[Tuple[int, str], dict] = {}
    for item in data:
        if require_reliable:
            if item.get("region_conf") != "reliable":
                continue
            if item.get("station_conf") not in (None, "reliable"):
                continue
        key = (int(item["patient_id"]), item["image_name"])
        index[key] = item
    return index


def build_cot1_target(
    cot_text: str,
    station_label: str,
    region_label: str,
    confidence: str = "reliable",
) -> str:
    """StoBrain CoT1 format + explicit SSS final line for GastroHUN."""
    cot = (cot_text or "").strip()
    final_region = f"Final region: {region_label} (confidence: {confidence})"
    final_sss = f"Final SSS: {station_label}"
    return f"{cot}\n\n{final_region}\n{final_sss}"


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_cot1_into_jsonl(
    input_jsonl: str,
    cot1_json: str,
    output_jsonl: str,
    images_only: bool = True,
    require_reliable: bool = True,
) -> dict:
    cot_index = load_cot1_index(cot1_json, require_reliable=require_reliable)
    rows = read_jsonl(input_jsonl)

    stats = {
        "total_in": len(rows),
        "images_in": 0,
        "merged": 0,
        "missing_cot": 0,
        "label_mismatch": 0,
        "skipped_non_image": 0,
    }
    out_rows: List[dict] = []

    for row in rows:
        row_type = row.get("type", "image")
        if images_only and row_type not in ("image",):
            stats["skipped_non_image"] += 1
            continue

        stats["images_in"] += 1
        image_path = row.get("image_path") or row.get("media_path", "")
        key = parse_image_key(image_path)
        if key is None or key not in cot_index:
            stats["missing_cot"] += 1
            continue

        cot_item = cot_index[key]
        station = str(cot_item.get("station_label", "")).strip().upper()
        region = str(cot_item.get("region_label", "")).strip().upper()
        jsonl_label = str(row.get("label_code") or row.get("answer", "")).strip().upper()

        if station and jsonl_label and station != jsonl_label:
            stats["label_mismatch"] += 1
            # Keep jsonl gold label; still attach cot but final SSS uses jsonl label.
            station = jsonl_label

        if not station:
            station = jsonl_label

        row = dict(row)
        row["gt_text_cot"] = build_cot1_target(
            cot_item["cot"],
            station_label=station,
            region_label=region or "OTHERCLASS",
            confidence=str(cot_item.get("region_conf", "reliable")),
        )
        row["cot_source"] = "cot1_labels.json"
        out_rows.append(row)
        stats["merged"] += 1

    write_jsonl(output_jsonl, out_rows)
    stats["output"] = output_jsonl
    stats["cot1_keys"] = len(cot_index)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge CoT1 labels into GastroHUN image JSONL")
    parser.add_argument(
        "--input_jsonl",
        default="output_dir/gastrohun_llm_en_multimodal_train.jsonl",
    )
    parser.add_argument(
        "--cot1_json",
        default="../stobrain/ds/cot1_labels.json",
        help="Path to cot1_labels.json (from WeChat copy to stobrain/ds/)",
    )
    parser.add_argument(
        "--output_jsonl",
        default="output_dir/gastrohun_llm_en_images_train_cot1.jsonl",
    )
    parser.add_argument("--images_only", action="store_true", default=True)
    parser.add_argument("--include_sequences", action="store_true", help="Also keep video rows (no cot merge)")
    parser.add_argument("--no_require_reliable", action="store_true")
    args = parser.parse_args()

    images_only = not args.include_sequences
    stats = merge_cot1_into_jsonl(
        args.input_jsonl,
        args.cot1_json,
        args.output_jsonl,
        images_only=images_only,
        require_reliable=not args.no_require_reliable,
    )

    print("=" * 60)
    print("CoT1 merge summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
    print("Train with:")
    print(
        f"  python sft_train.py --data_path {stats['output']} "
        f"--image_dir /media/rennc1/Elements/exvla_clinical --runname sft_cot1_image --mode 1"
    )


if __name__ == "__main__":
    main()
