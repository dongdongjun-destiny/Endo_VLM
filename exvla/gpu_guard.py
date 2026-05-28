#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protect an ongoing exvla training job on a shared GPU.

Default conservative mode:
  - kill only when cmdline matches training_cmd_hints
  - then apply memory thresholds for that matched process

Own process tree (current sft_train / rft_grpo_* run) is always kept.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Iterable, List, Optional, Set, Tuple

_ACTIVE_GUARD: Optional["GpuGuard"] = None

DEFAULT_MIN_KILL_MEM_MIB = 8192  # 8 GiB
DEFAULT_MIN_TRAINING_CMD_MEM_MIB = 512

# External training jobs only (not sft_train.py / rft_grpo_* entry scripts).
DEFAULT_TRAINING_CMD_HINTS = (
    "finetune",
    "openpi",
    "compute_norm_stats",
    "torchrun",
    "deepspeed",
    "accelerate launch",
)

# Keyword path never targets our own entry scripts (protected by process tree + this list).
OWN_ENTRY_SCRIPTS = (
    "sft_train.py",
    "rft_grpo_video_train.py",
    "rft_grpo_image_train.py",
)


def _list_gpu_compute_pids() -> List[Tuple[int, str, int]]:
    """Return (pid, process_name, used_mib) for GPU compute processes."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    rows: List[Tuple[int, str, int]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1] if len(parts) > 1 else ""
        try:
            mem_mib = int(float(parts[2])) if len(parts) > 2 else 0
        except ValueError:
            mem_mib = 0
        rows.append((pid, name, mem_mib))
    return rows


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", errors="replace").lower()
    except OSError:
        return ""


def _iter_child_pids(root_pid: int) -> Iterable[int]:
    children_path = f"/proc/{root_pid}/task/{root_pid}/children"
    try:
        with open(children_path, encoding="utf-8") as handle:
            child_text = handle.read().strip()
    except OSError:
        return
    if not child_text:
        return
    for token in child_text.split():
        try:
            child_pid = int(token)
        except ValueError:
            continue
        yield child_pid
        yield from _iter_child_pids(child_pid)


def _collect_allowed_pids(root_pid: Optional[int] = None) -> Set[int]:
    """Current training job: this process and all child processes."""
    root = root_pid or os.getpid()
    allowed = {root, os.getppid()}
    for child in _iter_child_pids(root):
        allowed.add(child)
    return allowed


def _should_kill(
    pid: int,
    mem_mib: int,
    min_kill_mem_mib: int,
    training_cmd_hints: Tuple[str, ...],
    min_training_cmd_mem_mib: int,
    require_cmd_match_for_kill: bool,
) -> bool:
    cmdline = _read_cmdline(pid)
    if not cmdline:
        return False
    if any(script in cmdline for script in OWN_ENTRY_SCRIPTS):
        return False
    matched = any(hint in cmdline for hint in training_cmd_hints)
    if require_cmd_match_for_kill and not matched:
        return False
    if mem_mib > min_kill_mem_mib:
        return True
    if mem_mib < min_training_cmd_mem_mib:
        return False
    return matched


def _kill_pid(pid: int, force_after_sec: float = 1.5) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        print(f"[gpu_guard] Permission denied when killing PID {pid}", file=sys.stderr)
        return False

    deadline = time.time() + force_after_sec
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False

    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        print(f"[gpu_guard] Permission denied when force-killing PID {pid}", file=sys.stderr)
        return False


class GpuGuard:
    def __init__(
        self,
        watchdog: bool = True,
        watchdog_poll_sec: float = 10.0,
        min_kill_mem_mib: int = DEFAULT_MIN_KILL_MEM_MIB,
        min_training_cmd_mem_mib: int = DEFAULT_MIN_TRAINING_CMD_MEM_MIB,
        training_cmd_hints: Tuple[str, ...] = DEFAULT_TRAINING_CMD_HINTS,
        require_cmd_match_for_kill: bool = True,
    ) -> None:
        self.watchdog = watchdog
        self.watchdog_poll_sec = watchdog_poll_sec
        self.min_kill_mem_mib = min_kill_mem_mib
        self.min_training_cmd_mem_mib = min_training_cmd_mem_mib
        self.training_cmd_hints = training_cmd_hints
        self.require_cmd_match_for_kill = require_cmd_match_for_kill

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def activate(self) -> "GpuGuard":
        if self.watchdog:
            self._thread = threading.Thread(
                target=self._watchdog_loop,
                name="gpu-guard-watchdog",
                daemon=True,
            )
            self._thread.start()
            print(
                f"[gpu_guard] Watchdog active (poll={self.watchdog_poll_sec:.0f}s); "
                f"{'require cmd match; ' if self.require_cmd_match_for_kill else ''}"
                f"kill threshold={self.min_kill_mem_mib} MiB; keep PID {os.getpid()} + children."
            )
        else:
            print("[gpu_guard] Disabled (no watchdog).")
        return self

    def deactivate(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.watchdog_poll_sec + 2.0)
            self._thread = None

    def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            allowed = _collect_allowed_pids()
            for pid, name, mem_mib in _list_gpu_compute_pids():
                if pid in allowed:
                    continue
                if not _should_kill(
                    pid,
                    mem_mib,
                    self.min_kill_mem_mib,
                    self.training_cmd_hints,
                    self.min_training_cmd_mem_mib,
                    self.require_cmd_match_for_kill,
                ):
                    continue
                print(
                    f"[gpu_guard] PID {pid} ({name}, {mem_mib} MiB) "
                    "- terminating to protect current training."
                )
                _kill_pid(pid)
            self._stop.wait(self.watchdog_poll_sec)


def activate_gpu_guard(
    enabled: bool = True,
    watchdog: bool = True,
    watchdog_poll_sec: float = 10.0,
    min_kill_mem_mib: int = DEFAULT_MIN_KILL_MEM_MIB,
    min_training_cmd_mem_mib: int = DEFAULT_MIN_TRAINING_CMD_MEM_MIB,
    training_cmd_hints: Tuple[str, ...] = DEFAULT_TRAINING_CMD_HINTS,
    require_cmd_match_for_kill: bool = True,
) -> Optional[GpuGuard]:
    """Activate GPU protection once per process."""
    global _ACTIVE_GUARD
    if not enabled:
        return None
    if _ACTIVE_GUARD is not None:
        return _ACTIVE_GUARD

    guard = GpuGuard(
        watchdog=watchdog,
        watchdog_poll_sec=watchdog_poll_sec,
        min_kill_mem_mib=min_kill_mem_mib,
        min_training_cmd_mem_mib=min_training_cmd_mem_mib,
        training_cmd_hints=training_cmd_hints,
        require_cmd_match_for_kill=require_cmd_match_for_kill,
    ).activate()
    _ACTIVE_GUARD = guard
    atexit.register(deactivate_gpu_guard)
    return guard


def deactivate_gpu_guard() -> None:
    global _ACTIVE_GUARD
    if _ACTIVE_GUARD is None:
        return
    _ACTIVE_GUARD.deactivate()
    _ACTIVE_GUARD = None


def activate_gpu_guard_from_config(enabled: bool = True) -> Optional[GpuGuard]:
    from config import GPU_GUARD_CONFIG

    cfg = dict(GPU_GUARD_CONFIG)
    if not cfg.get("enabled", True) or not enabled:
        return None

    hints = tuple(cfg.get("training_cmd_hints", DEFAULT_TRAINING_CMD_HINTS))

    return activate_gpu_guard(
        enabled=True,
        watchdog=bool(cfg.get("watchdog", True)),
        watchdog_poll_sec=float(cfg.get("watchdog_poll_sec", 10.0)),
        min_kill_mem_mib=int(cfg.get("min_kill_mem_mib", DEFAULT_MIN_KILL_MEM_MIB)),
        min_training_cmd_mem_mib=int(
            cfg.get("min_training_cmd_mem_mib", DEFAULT_MIN_TRAINING_CMD_MEM_MIB)
        ),
        training_cmd_hints=hints,
        require_cmd_match_for_kill=bool(cfg.get("require_cmd_match_for_kill", True)),
    )


def preview_watchdog_targets(
    min_kill_mem_mib: int = DEFAULT_MIN_KILL_MEM_MIB,
    min_training_cmd_mem_mib: int = DEFAULT_MIN_TRAINING_CMD_MEM_MIB,
    training_cmd_hints: Tuple[str, ...] = DEFAULT_TRAINING_CMD_HINTS,
    require_cmd_match_for_kill: bool = True,
) -> None:
    """Print which GPU compute jobs would be kept vs killed."""
    allowed = _collect_allowed_pids()
    for pid, name, mem_mib in _list_gpu_compute_pids():
        if pid in allowed:
            status = "self (keep)"
        elif _should_kill(
            pid,
            mem_mib,
            min_kill_mem_mib,
            training_cmd_hints,
            min_training_cmd_mem_mib,
            require_cmd_match_for_kill,
        ):
            status = "would-kill"
        else:
            status = "ignore"
        cmd = _read_cmdline(pid)[:120]
        print(f"[gpu_guard] PID {pid} ({mem_mib} MiB) [{status}] {name} :: {cmd}")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protect GPU for exvla training.")
    parser.add_argument("--preview", action="store_true", help="Show keep vs kill decisions.")
    parser.add_argument("--no-watchdog", action="store_true", help="Skip background watchdog.")
    parser.add_argument(
        "--min-mem-mib",
        type=int,
        default=DEFAULT_MIN_KILL_MEM_MIB,
        help="Kill other GPU jobs above this MiB (default 8192 = 8 GiB).",
    )
    parser.add_argument("--poll-sec", type=float, default=10.0, help="Watchdog poll interval.")
    parser.add_argument(
        "--allow-unmatched-kill",
        action="store_true",
        help="Aggressive mode: allow killing high-VRAM jobs even without cmdline hint match.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    require_cmd_match_for_kill = not args.allow_unmatched_kill
    if args.preview:
        preview_watchdog_targets(
            min_kill_mem_mib=args.min_mem_mib,
            require_cmd_match_for_kill=require_cmd_match_for_kill,
        )
        return

    guard = activate_gpu_guard(
        watchdog=not args.no_watchdog,
        watchdog_poll_sec=args.poll_sec,
        min_kill_mem_mib=args.min_mem_mib,
        require_cmd_match_for_kill=require_cmd_match_for_kill,
    )
    if guard is None:
        return

    print("[gpu_guard] Running until Ctrl+C.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[gpu_guard] Stopping.")
    finally:
        deactivate_gpu_guard()


if __name__ == "__main__":
    main()
