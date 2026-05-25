#!/usr/bin/env python3
import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["tiny", "1b", "3b"], action="append")
    parser.add_argument("--model-impl", choices=["baseline", "cola"], action="append")
    parser.add_argument("--strategy", choices=["baseline", "distopt", "fsdp"], action="append")
    parser.add_argument("--offload", choices=["0", "1"], action="append")
    parser.add_argument("--cuda-graph", choices=["0", "1"], action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_one = root / "benchmarks" / "cola_memory" / "run_one.sh"
    model_sizes = args.model_size or ["1b", "3b"]
    model_impls = args.model_impl or ["baseline", "cola"]
    strategies = args.strategy or ["baseline", "distopt", "fsdp"]
    offloads = args.offload or ["0", "1"]
    cuda_graphs = args.cuda_graph or ["0", "1"]

    failures = 0
    for model_size, model_impl, strategy, offload, cuda_graph in itertools.product(
        model_sizes, model_impls, strategies, offloads, cuda_graphs
    ):
        env = os.environ.copy()
        env.update(
            {
                "ROOT_DIR": str(root),
                "MODEL_SIZE": model_size,
                "MODEL_IMPL": model_impl,
                "STRATEGY": strategy,
                "OFFLOAD": offload,
                "CUDA_GRAPH": cuda_graph,
            }
        )
        if args.results_root:
            env["RESULTS_ROOT"] = args.results_root
        label = f"{model_size} {model_impl} {strategy} offload={offload} cuda_graph={cuda_graph}"
        print(f"==> {label}", flush=True)
        if args.dry_run:
            print(" ".join([str(run_one)]), flush=True)
            continue
        result = subprocess.run([str(run_one)], env=env, cwd=root)
        if result.returncode != 0:
            failures += 1
            print(f"FAILED ({result.returncode}): {label}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
