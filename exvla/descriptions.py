#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral Descriptions for Endoscopic Lesion Identification

Contains:
- 10 appearance descriptions per lesion type (simulating noisy oral speech)
- Each with an `exact_appearance` field: the exact adjectives to extract
- 10 spatial descriptions per anatomical station
- Utility functions for generating oral prompts

Lesion Types:
    1. White oval lesion       → Greater curvature
    2. Orange protruding lesion → Lesser curvature
    3. Small round nodule      → Pyloric antrum

Available Stations: greater curvature; lesser curvature; pyloric antrum
"""

# ==============================================================================
# LESION / STATION DEFINITIONS
# ==============================================================================

LESION_TYPES = {
    "greater_curvature": {
        "canonical_appearance": "white oval",
        "canonical_station": "greater curvature",
        "folder_name": "greater_curvature",
    },
    "lesser_curvature": {
        "canonical_appearance": "orange protruding",
        "canonical_station": "lesser curvature",
        "folder_name": "lesser_curvature",
    },
    "pyloric_antrum": {
        "canonical_appearance": "small round nodule",
        "canonical_station": "pyloric antrum",
        "folder_name": "pyloric_antrum",
    },
}

AVAILABLE_STATIONS = ["greater curvature", "lesser curvature", "pyloric antrum"]

# Mapping from folder key to canonical info
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


# ==============================================================================
# APPEARANCE DESCRIPTIONS (10 per lesion, simulating oral/noisy speech)
# Each entry is a dict with:
#   - "text": the full noisy oral appearance description
#   - "exact_appearance": the exact adjectives the model should extract
# ==============================================================================

APPEARANCE_DESCRIPTIONS = {
    # White oval lesion (greater curvature)
    "greater_curvature": [
        {
            "text": "I remembered a whitish oval-shaped lesion on the mucosa",
            "exact_appearance": "whitish oval-shaped",
        },
        {
            "text": "there was like a white elliptical spot on the stomach wall",
            "exact_appearance": "white elliptical",
        },
        {
            "text": "um there's this pale oval thing that looks abnormal",
            "exact_appearance": "pale oval",
        },
        {
            "text": "I remembered a bright white oval mark right there",
            "exact_appearance": "bright white oval",
        },
        {
            "text": "there seems to be a white-ish elongated patch on the lining",
            "exact_appearance": "white-ish elongated",
        },
        {
            "text": "so there was this milky white oval area that stands out",
            "exact_appearance": "milky white oval",
        },
        {
            "text": "I noticed a light colored oval abnormality on the surface",
            "exact_appearance": "light colored oval",
        },
        {
            "text": "there was a whitish egg-shaped lesion over there",
            "exact_appearance": "whitish egg-shaped",
        },
        {
            "text": "um I thought there's a white flattened oval spot on the wall",
            "exact_appearance": "white flattened oval",
        },
        {
            "text": "I saw something that looks like a pale oval bump on mucosa",
            "exact_appearance": "pale oval",
        },
    ],

    # Orange protruding lesion (lesser curvature)
    "lesser_curvature": [
        {
            "text": "there was an orange-ish protruding mass on the wall",
            "exact_appearance": "orange-ish protruding",
        },
        {
            "text": "I saw this orangey raised lesion sticking out",
            "exact_appearance": "orangey raised",
        },
        {
            "text": "um there was a reddish-orange bump protruding from the surface",
            "exact_appearance": "reddish-orange protruding",
        },
        {
            "text": "there was like an orange elevated growth on the lining",
            "exact_appearance": "orange elevated",
        },
        {
            "text": "I noticed an orange colored protrusion right there",
            "exact_appearance": "orange colored protrusion",
        },
        {
            "text": "so there was this orange raised abnormality I can see",
            "exact_appearance": "orange raised",
        },
        {
            "text": "there was a tangerine colored protruding thing on the mucosa",
            "exact_appearance": "tangerine colored protruding",
        },
        {
            "text": "I remembered an orange-ish bulging lesion on the wall",
            "exact_appearance": "orange-ish bulging",
        },
        {
            "text": "um there was this orange elevated mass sticking up there",
            "exact_appearance": "orange elevated",
        },
        {
            "text": "I remembered a bright orange protuberance on the stomach lining",
            "exact_appearance": "bright orange",
        },
    ],

    "pyloric_antrum": [
        {
            "text": "there was a small round nodule on the pyloric antrum",
            "exact_appearance": "small round",
        },
        {
            "text": "I remembered this tiny circular bump on the wall",
            "exact_appearance": "tiny circular",
        },
        {
            "text": "um there was a little round raised spot right there",
            "exact_appearance": "little round raised",
        },
        {
            "text": "there was like a small spherical nodule on the mucosa",
            "exact_appearance": "small spherical",
        },
        {
            "text": "I noticed a small rounded growth on the lining",
            "exact_appearance": "small rounded",
        },
        {
            "text": "so there was this petite round lump on the surface",
            "exact_appearance": "petite round",
        },
        {
            "text": "there was a small circular nodular thing I remembered",
            "exact_appearance": "small circular nodular",
        },
        {
            "text": "I remembered a tiny round elevated lesion on the wall",
            "exact_appearance": "tiny round elevated",
        },
        {
            "text": "um there was this small round-shaped nodule over there",
            "exact_appearance": "small round-shaped",
        },
        {
            "text": "I remembered a compact round little bump on the mucosa",
            "exact_appearance": "compact round little",
        },
    ],
}


# ==============================================================================
# SPATIAL DESCRIPTIONS (10 per station, based on anatomical features)
# ==============================================================================

SPATIAL_DESCRIPTIONS = {
    # Greater curvature: convex, lateral, outer, longer border
    "greater_curvature": [
        "along the outer convex wall of the stomach",
        "on the lateral border of the stomach body",
        "at the convex side of the gastric wall",
        "um along the longer curvature on the left side",
        "on the left-side outer curve of the stomach wall",
        "at the outer edge where the greater omentum attaches",
        "along the bigger curve on the outside of the stomach",
        "on the convex lateral margin of the stomach",
        "at the lower lateral wall area of the gastric body",
        "along the long outer curved surface of the stomach",
    ],

    # Lesser curvature: concave, medial, inner, shorter border
    "lesser_curvature": [
        "on the inner concave wall of the stomach",
        "along the medial border of the stomach",
        "at the concave side of the gastric wall",
        "um on the shorter curvature of the upper right",
        "along the right-side inner curve of the stomach",
        "on the upper medial surface near the incisura",
        "at the inner edge close to the hepatogastric area",
        "along the shorter concave border of the stomach",
        "on the concave medial margin of the gastric wall",
        "at the inner curved surface of the stomach body",
    ],

    # Pyloric antrum: distal, near outlet, funnel-shaped, pre-pyloric
    "pyloric_antrum": [
        "near the outlet of the stomach toward the duodenum",
        "in the distal part of the stomach before the pylorus",
        "close to the pyloric region of the stomach",
        "at the gastric antrum",
        "in the funnel-shaped lower part of the stomach",
        "at the antral region just proximal to the pyloric canal",
        "in the distal portion before the pyloric sphincter",
        "at the pyloric antrum",
        "at the lower end of the stomach near the pylorus",
        "at the antrum of the stomach",
    ],
}


# ==============================================================================
# ORAL PROMPT TEMPLATES
# ==============================================================================

ORAL_PROMPT_TEMPLATES = [
    "Hey um can you look at these three images and tell me which one shows {appearance} {spatial}? I think it might be suspicious.",
    "So I'm looking at these endoscopic views and I need help identifying which image has {appearance} {spatial}.",
    "Um could you check these three keyframes and point out which one has {appearance} {spatial} please?",
    "I need to find the lesion that looks like {appearance} {spatial}, which of these three is it?",
    "Can you help me figure out which of these three endoscopy images shows {appearance} {spatial}?",
]


def generate_oral_prompt(appearance_desc: str, spatial_desc: str, template_idx: int = None) -> str:
    """
    Generate a noisy oral-style prompt combining appearance and spatial descriptions.

    Args:
        appearance_desc: Appearance description string
        spatial_desc: Spatial description string
        template_idx: Optional template index (random if None)

    Returns:
        Generated oral prompt string
    """
    import random
    if template_idx is None:
        template_idx = random.randint(0, len(ORAL_PROMPT_TEMPLATES) - 1)
    template = ORAL_PROMPT_TEMPLATES[template_idx % len(ORAL_PROMPT_TEMPLATES)]
    return template.format(appearance=appearance_desc, spatial=spatial_desc)


def get_all_oral_instructions():
    """
    Generate all 300 oral instruction combinations.

    Returns:
        List of dicts with keys:
            - target_key: folder key (e.g., "greater_curvature")
            - appearance_idx: index of appearance description
            - spatial_idx: index of spatial description
            - appearance_desc: the full appearance text (noisy oral)
            - spatial_desc: the spatial text
            - exact_appearance: the exact adjectives to extract
            - canonical_appearance: standardized appearance keyword
            - canonical_station: standardized station name
    """
    instructions = []
    for target_key in LESION_TYPES:
        info = LESION_TYPES[target_key]
        for a_idx, a_entry in enumerate(APPEARANCE_DESCRIPTIONS[target_key]):
            for s_idx, s_desc in enumerate(SPATIAL_DESCRIPTIONS[target_key]):
                instructions.append({
                    "target_key": target_key,
                    "appearance_idx": a_idx,
                    "spatial_idx": s_idx,
                    "appearance_desc": a_entry["text"],
                    "spatial_desc": s_desc,
                    "exact_appearance": a_entry["exact_appearance"],
                    "canonical_appearance": info["canonical_appearance"],
                    "canonical_station": info["canonical_station"],
                })
    return instructions


# ==============================================================================
# HELPER: Get all unique exact appearances (for evaluation reference)
# ==============================================================================

def get_all_exact_appearances():
    """
    Return a dict mapping target_key → list of unique exact_appearance strings.
    Useful for building evaluation lookup tables.
    """
    result = {}
    for target_key, entries in APPEARANCE_DESCRIPTIONS.items():
        result[target_key] = list(set(e["exact_appearance"] for e in entries))
    return result


if __name__ == "__main__":
    instructions = get_all_oral_instructions()
    print(f"Total oral instruction combinations: {len(instructions)}")
    print(f"  Per lesion type: {len(instructions) // 3}")

    # Show samples
    for key in LESION_TYPES:
        sample = [i for i in instructions if i["target_key"] == key][0]
        prompt = generate_oral_prompt(sample["appearance_desc"], sample["spatial_desc"], 0)
        print(f"\n--- {key} ---")
        print(f"  Prompt: {prompt}")
        print(f"  Exact Appearance: {sample['exact_appearance']}")
        print(f"  Canonical Appearance: {sample['canonical_appearance']}")
        print(f"  GT Station: {sample['canonical_station']}")

    # Show all unique exact appearances
    print("\n--- All Unique Exact Appearances ---")
    all_exact = get_all_exact_appearances()
    for key, appearances in all_exact.items():
        print(f"  {key}: {appearances}")