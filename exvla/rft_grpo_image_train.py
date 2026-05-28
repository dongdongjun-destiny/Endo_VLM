#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-only GRPO entrypoint (sets GRPO_MODALITY=image before shared patches load)."""

import os

os.environ["GRPO_MODALITY"] = "image"

from rft_grpo_core import run_grpo_training

if __name__ == "__main__":
    run_grpo_training(
        modality="image",
        wandb_group="grpo_image_train",
        banner="EndoVLA-Oral RFT Training (Image Train)",
    )
