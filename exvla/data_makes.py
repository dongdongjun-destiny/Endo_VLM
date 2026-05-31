#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build GastroHUN multimodal JSONL datasets with a unified framework for:
1) Standard training
2) CoT-SFT (gt_text_cot)
3) Doctor preference learning (chosen/rejected/doctor_score)

Inputs:
    official_splits/image_classification.csv
    official_splits/sequence_classification.csv

Outputs (ROOT_DIR):
    gastrohun_llm_en_images_{train|val|test}.jsonl
    gastrohun_llm_en_seqs_{train|val|test}.jsonl
    gastrohun_llm_en_multimodal_{train|val|test}.jsonl
    gastrohun_llm_en_images_pref_{train|val|test}.jsonl
    gastrohun_llm_en_seqs_pref_{train|val|test}.jsonl
    gastrohun_llm_en_multimodal_pref_{train|val|test}.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

ROOT_DIR = Path("/media/rennc1/Elements/exvla_clinical")
IMAGE_SPLIT_CSV = ROOT_DIR / "official_splits" / "image_classification.csv"
SEQ_SPLIT_CSV = ROOT_DIR / "official_splits" / "sequence_classification.csv"

IMAGE_LABEL_COL = "FG2-G1 agreement"
SEQ_LABEL_COL = "FG2-G1 agreement"

BUILD_IMAGES = True
BUILD_SEQUENCES = True
BUILD_MERGED_MULTIMODAL = True

SPLIT_MAP = {"Train": "train", "Validation": "val", "Test": "test"}

VALID_LABELS = {
    "A1", "A2", "A3", "A4", "A5", "A6",
    "G1", "G2", "G3", "G4",
    "L1", "L2", "L3", "L4", "L5", "L6",
    "P1", "P2", "P3", "P4", "P5", "P6",
    "NA",
}
VOTE_COLUMNS = ["FG1 (Team A)", "FG2 (Team A)", "G1 (Team B)", "G2 (Team B)"]

# =============================================================================
# LABEL DESCRIPTIONS & INSTRUCTIONS
# =============================================================================

LABEL_DESC_EN = {
    "G1": "Antrum - greater curvature (antegrade view, just above the pylorus).",
    "A1": "Antrum - anterior wall (antegrade view, just above the pylorus).",
    "L1": "Antrum - lesser curvature (antegrade view, just above the pylorus).",
    "P1": "Antrum - posterior wall (antegrade view, just above the pylorus).",
    "G2": "Lower gastric body - greater curvature (antegrade view).",
    "A2": "Lower gastric body - anterior wall (antegrade view).",
    "L2": "Lower gastric body - lesser curvature (antegrade view).",
    "P2": "Lower gastric body - posterior wall (antegrade view).",
    "G3": "Middle/upper gastric body - greater curvature (antegrade view).",
    "A3": "Middle/upper gastric body - anterior wall (antegrade view).",
    "L3": "Middle/upper gastric body - lesser curvature (antegrade view).",
    "P3": "Middle/upper gastric body - posterior wall (antegrade view).",
    "G4": "Fundus/cardia region - greater curvature (retroflex view).",
    "A4": "Fundus/cardia region - anterior wall around the cardia (retroflex view).",
    "L4": "Fundus/cardia region - lesser curvature near the cardia (retroflex view).",
    "P4": "Fundus/cardia region - posterior wall (retroflex view).",
    "A5": "Middle body near the incisura - anterior wall (retroflex view).",
    "L5": "Middle body near the incisura angularis - lesser curvature/incisura (retroflex view).",
    "P5": "Middle body near the incisura - posterior wall opposite the incisura (retroflex view).",
    "A6": "Proximal body/cardia - anterior wall panoramic view (retroflex).",
    "L6": "Proximal body/cardia - lesser curvature panoramic view including the incisura (retroflex).",
    "P6": "Proximal body/cardia - posterior wall panoramic view including the incisura region (retroflex).",
    "NA": "Unqualified view: target anatomical station not clearly visualized or dominated by lesion, fluid, bubbles or instruments.",
}

SYSTEM_INSTRUCTION_EN = (
    "You are an expert gastrointestinal endoscopist. Your task is to classify a white-light "
    "gastroscopy image or sequence according to the Kenshi Yao SSS protocol. "
    "Provide a concise and clinically grounded reasoning process first, then output exactly one label code."
)

IMG_USER_INSTRUCTION_EN = (
    "Analyze this gastroscopy image. Think step by step about anatomical cues (view mode, pylorus/cardia/fundus, "
    "curvature orientation, wall position), then output the final SSS code."
)

SEQ_USER_INSTRUCTION_EN = (
    "Analyze this gastroscopy video sequence. Think step by step about anatomical cues from representative frames, "
    "then output the final SSS code."
)

SEQ_GROUPS = [
    ("Labeled_Sequences_Group1_Patients_7-113", 7, 113),
    ("Labeled_Sequences_Group2_Patients_115-191", 115, 191),
    ("Labeled_Sequences_Group3_Patients_192-229", 192, 229),
    ("Labeled_Sequences_Group4_Patients_231-273", 231, 273),
    ("Labeled_Sequences_Group5_Patients_274-318_group5", 274, 318),
    ("Labeled_Sequences_Group6_Patients_319-375", 319, 375),
    ("Labeled_Sequences_Group7_Patients_376-387", 376, 387),
]


def normalize_label(label: str) -> str:
    code = str(label).strip().upper()
    if code in {"", "NAN", "NONE"}:
        return ""
    if code == "OTHERCLASS":
        return "NA"
    return code if code in VALID_LABELS else "NA"


def get_set_split(value: str) -> Optional[str]:
    return SPLIT_MAP.get(str(value).strip(), None)


def get_sequence_path(num_patient: int, filename: str) -> Optional[Path]:
    for dirname, lo, hi in SEQ_GROUPS:
        if lo <= num_patient <= hi:
            return ROOT_DIR / dirname / str(num_patient) / filename
    return None


def get_image_path(num_patient: int, filename: str) -> Optional[Path]:
    # Keep compatibility with both folder names that appear in your environment.
    candidates = [
        ROOT_DIR / "Labeled Images" / str(num_patient) / filename,
        ROOT_DIR / "Labeled_Images" / str(num_patient) / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_votes(row: pd.Series) -> List[str]:
    votes: List[str] = []
    for col in VOTE_COLUMNS:
        if col in row:
            code = normalize_label(row.get(col, ""))
            if code:
                votes.append(code)
    return votes


def get_preference_fields(chosen: str, votes: List[str]) -> Dict[str, object]:
    rejected = ""
    for v in votes:
        if v != chosen:
            rejected = v
            break
    if not votes:
        doctor_score = None
    else:
        doctor_score = sum(1 for v in votes if v == chosen) / len(votes)
    return {
        "chosen": chosen,
        "rejected": rejected,
        "doctor_score": doctor_score,
    }


def build_cot_text(code: str, desc_en: str) -> str:
    reasoning = (
        "The visual anatomy is matched against SSS landmarks and wall orientation, "
        "then mapped to the most consistent station code."
    )
    return f"<think>{reasoning}</think>\n{code}"


def build_common_fields(code: str, desc_en: str, user_instruction: str, votes: List[str]) -> Dict[str, object]:
    pref = get_preference_fields(code, votes)
    return {
        "label_code": code,
        "label_name_en": desc_en,
        "system_instruction": SYSTEM_INSTRUCTION_EN,
        "user_instruction": user_instruction,
        "answer": code,
        # CoT-SFT fields
        "gt_text": code,
        "gt_text_cot": build_cot_text(code, desc_en),
        # Preference-learning fields
        "chosen": pref["chosen"],
        "rejected": pref["rejected"],
        "doctor_score": pref["doctor_score"],
    }


def build_image_samples(df: pd.DataFrame, split_name: str) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    patient_col = "num patient"
    for idx, row in df.iterrows():
        code = normalize_label(row.get(IMAGE_LABEL_COL, ""))
        if not code:
            continue

        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        try:
            patient_id = int(row.get(patient_col))
        except Exception:
            continue

        img_path = get_image_path(patient_id, filename)
        if img_path is None:
            print(f"[WARN][image] not found, skip: patient={patient_id}, file={filename}")
            continue

        desc_en = LABEL_DESC_EN.get(code, LABEL_DESC_EN["NA"])
        votes = collect_votes(row)
        item = {
            "id": f"img_{split_name}_{len(samples):06d}",
            "split": split_name.capitalize(),
            "type": "image",
            "image_path": str(img_path),
        }
        item.update(build_common_fields(code, desc_en, IMG_USER_INSTRUCTION_EN, votes))
        samples.append(item)
    return samples


def build_seq_samples(df: pd.DataFrame, split_name: str) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    patient_col = "num_patient"
    for _, row in df.iterrows():
        code = normalize_label(row.get(SEQ_LABEL_COL, ""))
        if not code:
            continue

        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        try:
            patient_id = int(row.get(patient_col))
        except Exception:
            continue

        seq_path = get_sequence_path(patient_id, filename)
        if seq_path is None or not seq_path.exists():
            print(f"[WARN][seq] not found, skip: patient={patient_id}, file={filename}")
            continue

        desc_en = LABEL_DESC_EN.get(code, LABEL_DESC_EN["NA"])
        votes = collect_votes(row)
        item = {
            "id": f"seq_{split_name}_{len(samples):06d}",
            "split": split_name.capitalize(),
            "type": "sequence",
            "video_path": str(seq_path),
        }
        item.update(build_common_fields(code, desc_en, SEQ_USER_INSTRUCTION_EN, votes))
        samples.append(item)
    return samples


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pref_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    pref_rows = [r for r in records if r.get("chosen") and (r.get("rejected") or r.get("doctor_score") is not None)]
    write_jsonl(path, pref_rows)


def split_from_csv(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "set_type" not in df.columns:
        raise ValueError("CSV missing required column: set_type")
    df = df.copy()
    df["set_type"] = df["set_type"].astype(str).str.strip()
    grouped: Dict[str, pd.DataFrame] = {}
    for raw_name, final_name in SPLIT_MAP.items():
        grouped[final_name] = df[df["set_type"] == raw_name]
    return grouped


def build_images_jsonl() -> Dict[str, List[Dict[str, object]]]:
    df = pd.read_csv(IMAGE_SPLIT_CSV)
    split_dfs = split_from_csv(df)
    out: Dict[str, List[Dict[str, object]]] = {}
    for split_name, split_df in split_dfs.items():
        records = build_image_samples(split_df, split_name)
        out[split_name] = records
        base_path = ROOT_DIR / f"gastrohun_llm_en_images_{split_name}.jsonl"
        pref_path = ROOT_DIR / f"gastrohun_llm_en_images_pref_{split_name}.jsonl"
        write_jsonl(base_path, records)
        write_pref_jsonl(pref_path, records)
        print(f"[OK][images] {split_name}: {len(records)} -> {base_path}")
    return out


def build_sequences_jsonl() -> Dict[str, List[Dict[str, object]]]:
    df = pd.read_csv(SEQ_SPLIT_CSV)
    split_dfs = split_from_csv(df)
    out: Dict[str, List[Dict[str, object]]] = {}
    for split_name, split_df in split_dfs.items():
        records = build_seq_samples(split_df, split_name)
        out[split_name] = records
        base_path = ROOT_DIR / f"gastrohun_llm_en_seqs_{split_name}.jsonl"
        pref_path = ROOT_DIR / f"gastrohun_llm_en_seqs_pref_{split_name}.jsonl"
        write_jsonl(base_path, records)
        write_pref_jsonl(pref_path, records)
        print(f"[OK][seqs] {split_name}: {len(records)} -> {base_path}")
    return out


def build_multimodal_jsonl(
    image_records: Dict[str, List[Dict[str, object]]],
    seq_records: Dict[str, List[Dict[str, object]]],
) -> None:
    for split_name in ["train", "val", "test"]:
        merged = []
        merged.extend(image_records.get(split_name, []))
        merged.extend(seq_records.get(split_name, []))
        base_path = ROOT_DIR / f"gastrohun_llm_en_multimodal_{split_name}.jsonl"
        pref_path = ROOT_DIR / f"gastrohun_llm_en_multimodal_pref_{split_name}.jsonl"
        write_jsonl(base_path, merged)
        write_pref_jsonl(pref_path, merged)
        print(f"[OK][multi] {split_name}: {len(merged)} -> {base_path}")


def main() -> None:
    image_records: Dict[str, List[Dict[str, object]]] = {}
    seq_records: Dict[str, List[Dict[str, object]]] = {}
    if BUILD_IMAGES:
        image_records = build_images_jsonl()
    if BUILD_SEQUENCES:
        seq_records = build_sequences_jsonl()
    if BUILD_MERGED_MULTIMODAL and (BUILD_IMAGES or BUILD_SEQUENCES):
        build_multimodal_jsonl(image_records, seq_records)


if __name__ == "__main__":
    main()

