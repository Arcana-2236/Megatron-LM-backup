# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Select a CoLA layer spec and run the unchanged GPT training entry point."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cola-attn-rank", type=int, default=None)
    parser.add_argument("--cola-mlp-rank", type=int, default=None)
    parser.add_argument("--spec", nargs=2)
    cola, remaining = parser.parse_known_args()
    if cola.spec is not None:
        parser.error("pretrain_cola.py selects its own --spec")
    if any(rank is not None and rank <= 0 for rank in (cola.cola_attn_rank, cola.cola_mlp_rank)):
        parser.error("CoLA ranks must be positive")

    from examples.cola import cola_model

    cola_model.layer_spec = cola_model.get_cola_layer_spec(cola.cola_attn_rank, cola.cola_mlp_rank)
    entry = root / "pretrain_gpt.py"
    sys.argv = [str(entry), *remaining, "--spec", "examples.cola.cola_model", "layer_spec"]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
