#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Generation Pipeline for EndoVLA-Oral

Generates:
1. Composite ABC images from 3 source folders (3 keyframes side-by-side with a/b/c labels)
2. Oral instruction prompts (noisy speech-style)
3. Ground truth labels in format: [label, exact_appearance, station]
   where exact_appearance is the specific adjectives from the oral instruction
4. Train/eval JSON splits

Data Math:
    - 3 folders × 3 images each = 9 source images
    - 3! = 6 position permutations (which folder → which a/b/c slot)
    - 3^3 = 27 image selections (which image from each folder)
    - 6 × 27 = 162 unique ABC composite images
    - 3 targets × 10 appearances × 10 spatials = 300 oral instructions
    - Training: 300 instructions × 27 image variants = 8,100 samples
    - Evaluation: 300 instructions × 3 image variants = 900 samples (300 per target)

Usage:
    python generate_data.py \\
        --gc_dir ./data/raw/greater_curvature \\
        --lc_dir ./data/raw/lesser_curvature \\
        --pa_dir ./data/raw/pyloric_antrum \\
        --output_dir ./data \\
        --train_images_per_instruction 27 \\
        --eval_images_per_instruction 3
"""

import os
import sys
import json
import random
import argparse
import itertools
from typing import List, Dict, Tuple, Any
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate training/eval data for EndoVLA-Oral")
    parser.add_argument("--gc_dir", type=str, required=True,
                        help="Path to greater_curvature image folder")
    parser.add_argument("--lc_dir", type=str, required=True,
                        help="Path to lesser_curvature image folder")
    parser.add_argument("--pa_dir", type=str, required=True,
                        help="Path to pyloric_antrum image folder")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for generated data")
    parser.add_argument("--train_images_per_instruction", type=int, default=27,
                        help="Number of ABC image variants per instruction for training")
    parser.add_argument("--eval_images_per_instruction", type=int, default=3,
                        help="Number of ABC image variants per instruction for evaluation")
    parser.add_argument("--keyframe_width", type=int, default=320,
                        help="Width of each keyframe panel")
    parser.add_argument("--keyframe_height", type=int, default=240,
                        help="Height of each keyframe panel")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# ==============================================================================
# IMAGE LOADING
# ==============================================================================

def load_images_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    """
    Load all images from a folder.

    Returns:
        List of (filename, full_path) tuples, sorted by name.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = []
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            images.append((fname, os.path.join(folder_path, fname)))
    return images


# ==============================================================================
# ABC IMAGE GENERATION
# ==============================================================================

def create_abc_image(
    images_abc: List[Tuple[str, str]],   # [(folder_key, image_path), ...] for a, b, c
    keyframe_width: int = 320,
    keyframe_height: int = 240,
) -> Image.Image:
    """
    Create a composite ABC image with 3 keyframes labeled (a), (b), (c).

    Args:
        images_abc: List of 3 tuples (folder_key, image_path) for positions a, b, c
        keyframe_width: Width of each panel
        keyframe_height: Height of each panel

    Returns:
        PIL Image of the composite
    """
    margin = 10
    label_height = 40
    total_width = keyframe_width * 3 + margin * 4
    total_height = keyframe_height + label_height + margin * 2

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a decent font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

    labels = ["(a)", "(b)", "(c)"]

    for i, (folder_key, img_path) in enumerate(images_abc):
        # Load and resize keyframe
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((keyframe_width, keyframe_height), Image.LANCZOS)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            img = Image.new("RGB", (keyframe_width, keyframe_height), color=(128, 128, 128))

        # Paste keyframe
        x_offset = margin + i * (keyframe_width + margin)
        y_offset = margin
        canvas.paste(img, (x_offset, y_offset))

        # Draw border
        draw.rectangle(
            [x_offset - 1, y_offset - 1,
             x_offset + keyframe_width, y_offset + keyframe_height],
            outline=(0, 0, 0), width=2,
        )

        # Draw label centered below
        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        lw = bbox[2] - bbox[0]
        lx = x_offset + (keyframe_width - lw) // 2
        ly = y_offset + keyframe_height + 5
        draw.text((lx, ly), label, fill=(0, 0, 0), font=font)

    return canvas


def generate_all_abc_images(
    folder_images: Dict[str, List[Tuple[str, str]]],
    output_dir: str,
    keyframe_width: int = 320,
    keyframe_height: int = 240,
) -> List[Dict[str, Any]]:
    """
    Generate all possible ABC composite images.

    Args:
        folder_images: Dict mapping folder_key → list of (filename, path)
        output_dir: Directory to save ABC images
        keyframe_width: Width of each panel
        keyframe_height: Height of each panel

    Returns:
        List of dicts describing each ABC image with position mapping
    """
    os.makedirs(output_dir, exist_ok=True)

    folder_keys = list(folder_images.keys())  # 3 folder keys
    # All permutations of folder-to-position assignment
    position_perms = list(itertools.permutations(folder_keys))

    # All image index combinations (one image per folder)
    image_counts = {k: len(v) for k, v in folder_images.items()}
    image_index_combos = list(itertools.product(
        *[range(image_counts[k]) for k in folder_keys]
    ))

    abc_images = []
    img_count = 0

    for perm in position_perms:
        # perm = (folder_for_a, folder_for_b, folder_for_c)
        for idx_combo in image_index_combos:
            # idx_combo = (img_idx_for_folder0, img_idx_for_folder1, img_idx_for_folder2)
            # Map to actual images based on permutation
            images_abc = []
            position_map = {}
            for pos_idx, folder_key in enumerate(perm):
                # Which original folder index is this key?
                orig_folder_idx = folder_keys.index(folder_key)
                img_idx = idx_combo[orig_folder_idx]
                fname, fpath = folder_images[folder_key][img_idx]
                images_abc.append((folder_key, fpath))
                position_map[["a", "b", "c"][pos_idx]] = {
                    "folder_key": folder_key,
                    "image_file": fname,
                    "image_path": fpath,
                }

            # Create composite image
            abc_img = create_abc_image(images_abc, keyframe_width, keyframe_height)

            # Save
            img_name = f"abc_{img_count:05d}.png"
            img_path = os.path.join(output_dir, img_name)
            abc_img.save(img_path, quality=95)

            abc_images.append({
                "abc_image_name": img_name,
                "abc_image_path": img_path,
                "position_map": position_map,
                "permutation": list(perm),
                "image_indices": list(idx_combo),
            })

            img_count += 1

    return abc_images


# ==============================================================================
# DATA SAMPLE GENERATION
# ==============================================================================

def generate_samples(
    abc_images: List[Dict[str, Any]],
    instructions: List[Dict[str, Any]],
    images_per_instruction: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate training/eval samples by pairing instructions with ABC images.

    For each instruction (targeting a specific lesion type), we pair it with
    multiple ABC images. The ground truth label depends on which position
    contains the target folder.

    The gt_appearance is the EXACT adjectives from the oral instruction
    (not the canonical form), so each sample may have different appearance text.

    Args:
        abc_images: List of ABC image descriptors
        instructions: List of oral instruction descriptors
        images_per_instruction: How many ABC images to pair per instruction
        seed: Random seed

    Returns:
        List of sample dicts ready for JSON output
    """
    rng = random.Random(seed)

    samples = []
    sample_id = 0

    for instr in instructions:
        target_key = instr["target_key"]

        # Filter ABC images that contain the target folder
        # (they all should, since each ABC has all 3 folders)
        valid_images = abc_images  # all contain all 3 folders

        # Sample a subset
        if len(valid_images) > images_per_instruction:
            selected = rng.sample(valid_images, images_per_instruction)
        else:
            selected = valid_images

        for abc_info in selected:
            # Find which position (a/b/c) contains the target
            target_label = None
            for pos, pos_info in abc_info["position_map"].items():
                if pos_info["folder_key"] == target_key:
                    target_label = pos
                    break

            if target_label is None:
                continue  # shouldn't happen

            # Generate the oral prompt
            oral_prompt = _build_oral_prompt(
                instr["appearance_desc"],
                instr["spatial_desc"],
                rng,
            )

            # Ground truth: use EXACT appearance from this instruction
            gt_appearance = instr["exact_appearance"]
            gt_station = instr["canonical_station"]
            gt_text = f"[{target_label}, {gt_appearance}, {gt_station}]"

            samples.append({
                "id": sample_id,
                "image_path": abc_info["abc_image_name"],
                "oral_instruction": oral_prompt,
                "target_key": target_key,
                "target_label": target_label,
                "gt_appearance": gt_appearance,
                "gt_station": gt_station,
                "gt_text": gt_text,
                "canonical_appearance": instr["canonical_appearance"],
                "appearance_idx": instr["appearance_idx"],
                "spatial_idx": instr["spatial_idx"],
                "abc_image_name": abc_info["abc_image_name"],
                "position_map": abc_info["position_map"],
            })

            sample_id += 1

    return samples


# Oral prompt templates (noisy, conversational)
_ORAL_TEMPLATES = [
    "Hey um can you look at these three images and tell me which one shows {appearance} {spatial}? I think it might be suspicious.",
    "So I'm looking at these endoscopic views and I need help identifying which image has {appearance} {spatial}.",
    "Um could you check these three keyframes and point out which one has {appearance} {spatial} please?",
    "I need to find the lesion that looks like {appearance} {spatial}, which of these three is it?",
    "Can you help me figure out which of these three endoscopy images shows {appearance} {spatial}?",
    "Looking at these three views, I think one of them has {appearance} {spatial}, can you tell me which?",
    "Hey so um there should be one with {appearance} {spatial} among these three, which one is it?",
    "I'm trying to identify which endoscopic image shows {appearance} {spatial}, can you help?",
    "One of these three keyframes should show {appearance} {spatial}, please point it out.",
    "Which of the three images shows the lesion that is {appearance} {spatial}? I need to know.",
]

def _build_oral_prompt(appearance_desc: str, spatial_desc: str, rng: random.Random) -> str:
    """Build a noisy oral prompt from appearance and spatial descriptions."""
    template = rng.choice(_ORAL_TEMPLATES)
    return template.format(appearance=appearance_desc, spatial=spatial_desc)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("EndoVLA-Oral Data Generation Pipeline")
    print("=" * 70)

    # --- Load source images ---
    folder_images = {}
    folder_dirs = {
        "greater_curvature": args.gc_dir,
        "lesser_curvature": args.lc_dir,
        "pyloric_antrum": args.pa_dir,
    }

    for key, dir_path in folder_dirs.items():
        if not os.path.exists(dir_path):
            print(f"ERROR: Folder not found: {dir_path}")
            sys.exit(1)
        images = load_images_from_folder(dir_path)
        if len(images) == 0:
            print(f"ERROR: No images found in {dir_path}")
            sys.exit(1)
        folder_images[key] = images
        print(f"  {key}: {len(images)} images from {dir_path}")

    # --- Generate ABC composite images ---
    abc_dir = os.path.join(args.output_dir, "abc_images")
    print(f"\nGenerating ABC composite images → {abc_dir}")

    abc_images = generate_all_abc_images(
        folder_images,
        abc_dir,
        keyframe_width=args.keyframe_width,
        keyframe_height=args.keyframe_height,
    )
    print(f"  Generated {len(abc_images)} unique ABC images")
    print(f"  (= {len(list(itertools.permutations(folder_images.keys())))} permutations "
          f"× {len(list(itertools.product(*[range(len(v)) for v in folder_images.values()])))} "
          f"image combos)")

    # --- Generate oral instructions ---
    # Import descriptions module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from descriptions import get_all_oral_instructions

    instructions = get_all_oral_instructions()
    print(f"\nGenerated {len(instructions)} oral instructions "
          f"({len(instructions) // 3} per target)")

    # Show exact appearance diversity
    exact_apps = {}
    for instr in instructions:
        tk = instr["target_key"]
        ea = instr["exact_appearance"]
        exact_apps.setdefault(tk, set()).add(ea)
    print("\nExact appearance variants per target:")
    for tk, apps in exact_apps.items():
        print(f"  {tk}: {len(apps)} unique → {sorted(apps)}")

    # --- Generate training samples ---
    print(f"\nGenerating training samples "
          f"({args.train_images_per_instruction} images/instruction)...")
    train_samples = generate_samples(
        abc_images, instructions,
        images_per_instruction=args.train_images_per_instruction,
        seed=args.seed,
    )
    print(f"  Training samples: {len(train_samples)}")

    # --- Generate eval samples (separate image selection) ---
    print(f"Generating eval samples "
          f"({args.eval_images_per_instruction} images/instruction)...")
    eval_samples = generate_samples(
        abc_images, instructions,
        images_per_instruction=args.eval_images_per_instruction,
        seed=args.seed + 9999,  # Different seed for eval
    )
    print(f"  Evaluation samples: {len(eval_samples)}")

    # --- Statistics ---
    print("\n--- Data Statistics ---")
    for split_name, split_samples in [("train", train_samples), ("eval", eval_samples)]:
        target_counts = {}
        label_counts = {}
        appearance_counts = {}
        for s in split_samples:
            tk = s["target_key"]
            target_counts[tk] = target_counts.get(tk, 0) + 1
            label_counts[s["target_label"]] = label_counts.get(s["target_label"], 0) + 1
            appearance_counts[s["gt_appearance"]] = appearance_counts.get(s["gt_appearance"], 0) + 1
        print(f"  {split_name}: {len(split_samples)} total")
        print(f"    Per target: {dict(sorted(target_counts.items()))}")
        print(f"    Per label (a/b/c): {dict(sorted(label_counts.items()))}")
        print(f"    Unique gt_appearances: {len(appearance_counts)}")

    # --- Save to JSON ---
    def clean_sample(s):
        """Remove non-serializable fields and convert position_map."""
        clean = {k: v for k, v in s.items() if k != "position_map"}
        # Simplify position_map to just folder_key per position
        pos_map_simple = {}
        for pos, info in s["position_map"].items():
            pos_map_simple[pos] = info["folder_key"]
        clean["position_map"] = pos_map_simple
        return clean

    train_path = os.path.join(args.output_dir, "train.json")
    eval_path = os.path.join(args.output_dir, "eval.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump([clean_sample(s) for s in train_samples], f, indent=2, ensure_ascii=False)
    print(f"\nSaved training data → {train_path}")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump([clean_sample(s) for s in eval_samples], f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation data → {eval_path}")

    # --- Save ABC image index ---
    abc_index_path = os.path.join(args.output_dir, "abc_image_index.json")
    abc_index = []
    for abc in abc_images:
        entry = {
            "abc_image_name": abc["abc_image_name"],
            "permutation": abc["permutation"],
            "image_indices": abc["image_indices"],
            "position_map": {
                pos: {"folder_key": info["folder_key"], "image_file": info["image_file"]}
                for pos, info in abc["position_map"].items()
            },
        }
        abc_index.append(entry)

    with open(abc_index_path, "w", encoding="utf-8") as f:
        json.dump(abc_index, f, indent=2, ensure_ascii=False)
    print(f"Saved ABC image index → {abc_index_path}")

    print(f"\n{'=' * 70}")
    print("Data generation complete!")
    print(f"  ABC images: {len(abc_images)}")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Evaluation samples: {len(eval_samples)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()