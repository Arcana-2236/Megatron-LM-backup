# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Environment-gated CUDA memory liveness snapshots for benchmark studies."""

import json
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch


_ENABLED_VALUES = {"1", "true", "yes", "on"}
_HISTORY_CONFIGURED = False
_CAPTURED_KEYS: Set[str] = set()


def memory_liveness_enabled() -> bool:
    """Return whether memory liveness snapshot capture is enabled."""
    return os.environ.get("MEMORY_SNAPSHOT_PHASES", "").strip().lower() in _ENABLED_VALUES


def _distributed_rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _rank_selected(rank: int) -> bool:
    ranks = os.environ.get("MEMORY_SNAPSHOT_RANKS", "all").strip().lower()
    if ranks in {"", "all", "*"}:
        return True
    selected: Set[int] = set()
    for item in ranks.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            selected.add(int(item))
        except ValueError:
            continue
    return rank in selected


def _snapshot_dir() -> Path:
    path = os.environ.get("MEMORY_SNAPSHOT_DIR")
    if not path:
        path = os.path.join(os.getcwd(), "memory_snapshots")
    snapshot_dir = Path(path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _configure_memory_history() -> None:
    global _HISTORY_CONFIGURED
    if _HISTORY_CONFIGURED:
        return
    _HISTORY_CONFIGURED = True
    if os.environ.get("MEMORY_SNAPSHOT_RECORD_HISTORY", "1").strip().lower() not in _ENABLED_VALUES:
        return

    record_memory_history = getattr(torch.cuda.memory, "_record_memory_history", None)
    if record_memory_history is None:
        return

    max_entries = int(os.environ.get("MEMORY_SNAPSHOT_MAX_HISTORY", "200000"))
    attempts = (
        {"enabled": True, "stacks": "all", "max_entries": max_entries},
        {"enabled": "all", "stacks": "all", "max_entries": max_entries},
        {"enabled": True},
    )
    for kwargs in attempts:
        try:
            record_memory_history(**kwargs)
            return
        except TypeError:
            continue
        except Exception:
            return
    try:
        record_memory_history(True)
    except Exception:
        return


def capture_memory_liveness_snapshot(phase: str, iteration: Optional[int] = None) -> None:
    """Capture a raw CUDA memory snapshot and compact memory stats for one phase."""
    if not memory_liveness_enabled() or not torch.cuda.is_available():
        return

    rank, world_size = _distributed_rank_world()
    if not _rank_selected(rank):
        return

    key = f"{phase}:{iteration}"
    if key in _CAPTURED_KEYS:
        return
    _CAPTURED_KEYS.add(key)

    _configure_memory_history()
    snapshot_dir = _snapshot_dir()
    phase_safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in phase)
    iter_label = "init" if iteration is None else f"iter{iteration:06d}"
    prefix = f"{phase_safe}_{iter_label}_rank{rank:05d}_of{world_size:05d}"
    snapshot_path = snapshot_dir / f"{prefix}.pickle"
    stats_path = snapshot_dir / f"{prefix}.stats.json"

    payload: Dict[str, Any] = {
        "phase": phase,
        "iteration": iteration,
        "rank": rank,
        "world_size": world_size,
        "timestamp": time.time(),
        "device": torch.cuda.current_device(),
        "snapshot_path": str(snapshot_path),
        "stats_path": str(stats_path),
        "status": "ok",
    }

    try:
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        payload.update(
            {
                "memory_allocated": torch.cuda.memory_allocated(),
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
                "memory_stats": _jsonable(stats),
            }
        )
        snapshot = torch.cuda.memory_snapshot()
        with snapshot_path.open("wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = repr(exc)
        payload["traceback"] = traceback.format_exc()

    with stats_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"[Rank {rank}] memory liveness snapshot {phase}"
        f" iteration={iteration} status={payload['status']} stats={stats_path} snapshot={snapshot_path}",
        flush=True,
    )
