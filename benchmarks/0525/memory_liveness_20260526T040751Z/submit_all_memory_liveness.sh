#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/zhengyangwang/offloading/ATC-Megatron}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmarks/0525/memory_liveness_20260526T040751Z}
PBS_SCRIPT="$RESULT_DIR/run_one_memory_liveness_polaris.pbs"
SUBMITTED="$RESULT_DIR/submitted_jobs.csv"

write_header=1
if [ -s "$SUBMITTED" ]; then
  write_header=0
fi
if [ "$write_header" = "1" ]; then
  printf 'model_impl,runtime_mode,model_env,strategy_env,offload_env,job_id\n' > "$SUBMITTED"
fi

submit_row() {
  local model_impl="$1"
  local runtime_mode="$2"
  local model_env="$3"
  local strategy_env="$4"
  local offload_env="$5"
  local job_id
  job_id=$(qsub -v ROOT_DIR="$ROOT_DIR",RESULT_DIR="$RESULT_DIR",MODEL_IMPL="$model_env",STRATEGY="$strategy_env",OFFLOAD="$offload_env",TRAIN_ITERS=12,MEMORY_SNAPSHOT_RANKS=0 "$PBS_SCRIPT")
  printf '%s,%s,%s,%s,%s,%s\n' "$model_impl" "$runtime_mode" "$model_env" "$strategy_env" "$offload_env" "$job_id" >> "$SUBMITTED"
  echo "$model_impl $runtime_mode $job_id"
}

submit_row FullRank DistOpt baseline distopt 0
submit_row CoLA DistOpt cola distopt 0
submit_row FullRank DistOpt+offload baseline distopt 1
submit_row CoLA DistOpt+offload cola distopt 1
submit_row FullRank FSDP baseline fsdp 0
submit_row CoLA FSDP cola fsdp 0
submit_row FullRank FSDP+offload baseline fsdp 1
submit_row CoLA FSDP+offload cola fsdp 1
