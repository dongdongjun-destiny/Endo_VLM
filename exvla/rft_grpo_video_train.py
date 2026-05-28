#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video-only GRPO entrypoint (sets GRPO_MODALITY=video before shared patches load)."""

import os

os.environ["GRPO_MODALITY"] = "video"

from rft_grpo_core import run_grpo_training

if __name__ == "__main__":
    run_grpo_training(
        modality="video",
        wandb_group="grpo_video_train",
        banner="EndoVLA-Oral RFT Training (Video Train)",
    )
