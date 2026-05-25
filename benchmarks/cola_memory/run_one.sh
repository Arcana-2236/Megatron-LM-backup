#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/zhengyangwang/offloading/ATC-Megatron}
cd "$ROOT_DIR"
ulimit -c 0 || true

MODEL_SIZE=${MODEL_SIZE:-1b}        # 1b | 3b | tiny
MODEL_IMPL=${MODEL_IMPL:-baseline}  # baseline | cola
STRATEGY=${STRATEGY:-baseline}      # baseline | distopt | fsdp
FSDP_IMPL=${FSDP_IMPL:-megatron}    # megatron | torch
OFFLOAD=${OFFLOAD:-0}               # 0 | 1
CUDA_GRAPH=${CUDA_GRAPH:-0}         # 0 | 1
TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-1}
TIMING_LOG_OPTION=${TIMING_LOG_OPTION:-minmax}
TRAIN_ITERS=${TRAIN_ITERS:-20}
WARMUP_ITERS=${WARMUP_ITERS:-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
PYTHON=${PYTHON:-/home/zhengyangwang/.conda/envs/dspeed_env/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-${ROOT_DIR}/benchmarks/cola_memory/results}
RUN_TIMEOUT_SECONDS=${RUN_TIMEOUT_SECONDS:-600}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${MODEL_SIZE}_${MODEL_IMPL}_${STRATEGY}_offload${OFFLOAD}_cg${CUDA_GRAPH}}
PROFILE_NSYS=${PROFILE_NSYS:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-7}
PROFILE_STEP_END=${PROFILE_STEP_END:-10}
PROFILE_RANKS=${PROFILE_RANKS:-0}

export TMPDIR=${TMPDIR:-/var/tmp/${USER}/megatron_cola_tmp}
EAGLE_PROJECT_DIR=${EAGLE_PROJECT_DIR:-/eagle/TensorCompress/${USER}}
if [ -d "$EAGLE_PROJECT_DIR" ] && [ -w "$EAGLE_PROJECT_DIR" ]; then
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$EAGLE_PROJECT_DIR/cache/pip}
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$EAGLE_PROJECT_DIR/cache/uv}
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$EAGLE_PROJECT_DIR/cache/torch_extensions}
elif [ -d "/eagle/${USER}" ] && [ -w "/eagle/${USER}" ]; then
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/eagle/${USER}/cache/pip}
  export UV_CACHE_DIR=${UV_CACHE_DIR:-/eagle/${USER}/cache/uv}
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/eagle/${USER}/cache/torch_extensions}
else
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${TMPDIR}/pip}
  export UV_CACHE_DIR=${UV_CACHE_DIR:-${TMPDIR}/uv}
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${TMPDIR}/torch_extensions}
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$RESULTS_ROOT"

case "$MODEL_SIZE" in
  tiny)
    NUM_LAYERS=${NUM_LAYERS:-2}
    HIDDEN_SIZE=${HIDDEN_SIZE:-128}
    FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-352}
    NUM_HEADS=${NUM_HEADS:-4}
    SEQ_LENGTH=${SEQ_LENGTH:-128}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
    ;;
  1b)
    NUM_LAYERS=${NUM_LAYERS:-24}
    HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
    FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-5504}
    NUM_HEADS=${NUM_HEADS:-32}
    SEQ_LENGTH=${SEQ_LENGTH:-1024}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
    ;;
  3b)
    NUM_LAYERS=${NUM_LAYERS:-24}
    HIDDEN_SIZE=${HIDDEN_SIZE:-3200}
    FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-8640}
    NUM_HEADS=${NUM_HEADS:-32}
    SEQ_LENGTH=${SEQ_LENGTH:-1024}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
    ;;
  *)
    echo "Unknown MODEL_SIZE=$MODEL_SIZE" >&2
    exit 2
    ;;
esac

MLP_RANK=${MLP_RANK:-$((HIDDEN_SIZE / 4))}
ATTN_RANK=${ATTN_RANK:-$((HIDDEN_SIZE / 4))}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LOG_DIR="$RESULTS_ROOT/logs"
META_DIR="$RESULTS_ROOT/meta"
PROFILE_DIR="$RESULTS_ROOT/profiles"
mkdir -p "$LOG_DIR" "$META_DIR" "$PROFILE_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
META_FILE="$META_DIR/${RUN_ID}.json"
CMD_FILE="$META_DIR/${RUN_ID}.cmd"
NSYS_OUTPUT=${NSYS_OUTPUT:-$PROFILE_DIR/${RUN_ID}}

COMMON_ARGS=(
  pretrain_gpt.py
  --model-impl "$MODEL_IMPL"
  --transformer-impl local
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --mlp-rank "$MLP_RANK"
  --attn-rank "$ATTN_RANK"
  --num-attention-heads "$NUM_HEADS"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$SEQ_LENGTH"
  --train-iters "$TRAIN_ITERS"
  --lr-warmup-iters "$LR_WARMUP_ITERS"
  --lr 3e-4
  --lr-decay-style cosine
  --min-lr 3e-5
  --weight-decay 0.1
  --clip-grad 1
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.95
  --log-interval 1
  --log-memory-interval 1
  --save-interval 100000
  --eval-interval 100000
  --eval-iters 0
  --bf16
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 31980
  --split 1000,0,0
  --distributed-backend nccl
  --attention-dropout 0
  --hidden-dropout 0
  --position-embedding-type rope
  --no-rope-fusion
  --no-persist-layer-norm
  --untie-embeddings-and-output-weights
  --swiglu
  --normalization RMSNorm
  --disable-bias-linear
)

if [ "$CUDA_GRAPH" = "1" ] && [ "$TIMING_LOG_LEVEL" != "0" ]; then
  echo "CUDA_GRAPH=1: disabling timing-log-level because CUDA graph capture cannot include timer sync/barrier calls."
  TIMING_LOG_LEVEL=0
fi

if [ "$TIMING_LOG_LEVEL" != "0" ]; then
  COMMON_ARGS+=(--timing-log-level "$TIMING_LOG_LEVEL" --timing-log-option "$TIMING_LOG_OPTION")
fi

STRATEGY_ARGS=()
case "$STRATEGY" in
  baseline)
    ;;
  distopt)
    STRATEGY_ARGS+=(--use-distributed-optimizer)
    ;;
  fsdp)
    if [ "${CUDA_DEVICE_MAX_CONNECTIONS:-}" = "1" ]; then
      echo "STRATEGY=fsdp: unsetting CUDA_DEVICE_MAX_CONNECTIONS=1 because FSDP requires it to be unset or >1."
      unset CUDA_DEVICE_MAX_CONNECTIONS
    fi
    case "$FSDP_IMPL" in
      megatron)
        STRATEGY_ARGS+=(
          --use-megatron-fsdp
          --use-distributed-optimizer
          --data-parallel-sharding-strategy optim_grads_params
          --ckpt-format fsdp_dtensor
          --no-gradient-accumulation-fusion
        )
        if [ "${FSDP_USE_PRECISION_AWARE_OPTIMIZER:-0}" = "1" ]; then
          STRATEGY_ARGS+=(--use-precision-aware-optimizer)
        fi
        if [ "${FSDP_GRAD_REDUCE_IN_BF16:-0}" = "1" ]; then
          STRATEGY_ARGS+=(--grad-reduce-in-bf16)
        fi
        if [ "${FSDP_USE_NCCL_UB:-0}" = "1" ]; then
          STRATEGY_ARGS+=(--use-nccl-ub --fsdp-manual-registration)
        fi
        if [ "${FSDP_DOUBLE_BUFFER:-0}" = "1" ]; then
          STRATEGY_ARGS+=(--fsdp-double-buffer)
        fi
        if [ "${FSDP_INIT_MODEL_WITH_META_DEVICE:-0}" = "1" ]; then
          STRATEGY_ARGS+=(--init-model-with-meta-device)
        fi
        if [ "${FSDP_USE_TORCH_OPTIMIZER:-0}" = "1" ]; then
          export MEGATRON_FSDP_USE_TORCH_OPTIMIZER=1
        fi
        ;;
      torch)
        STRATEGY_ARGS+=(
          --use-torch-fsdp2
          --ckpt-format torch_dist
          --no-gradient-accumulation-fusion
        )
        ;;
      *)
        echo "Unknown FSDP_IMPL=$FSDP_IMPL" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown STRATEGY=$STRATEGY" >&2
    exit 2
    ;;
esac

if [ "$OFFLOAD" = "1" ]; then
  STRATEGY_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
  if [ "$STRATEGY" != "fsdp" ] || [ "$FSDP_IMPL" != "torch" ]; then
    if [[ " ${STRATEGY_ARGS[*]} " != *" --use-distributed-optimizer "* ]]; then
      STRATEGY_ARGS+=(--use-distributed-optimizer)
    fi
  fi
fi

if [ "$CUDA_GRAPH" = "1" ]; then
  STRATEGY_ARGS+=(--cuda-graph-impl full_iteration --no-check-for-nan-in-loss-and-grad)
fi

if [ "$PROFILE_NSYS" = "1" ]; then
  STRATEGY_ARGS+=(
    --profile
    --profile-step-start "$PROFILE_STEP_START"
    --profile-step-end "$PROFILE_STEP_END"
    --profile-ranks "$PROFILE_RANKS"
  )
fi

LAUNCH=(
  "$PYTHON" -m torch.distributed.run
  --nnodes "$NNODES"
  --nproc-per-node "$GPUS_PER_NODE"
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
)
if [ "$PROFILE_NSYS" = "1" ]; then
  NSYS_CMD=(
    nsys profile
    --sample=none
    --cpuctxsw=none
    --trace=cuda,nvtx,osrt
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-graph-trace=graph
    --force-overwrite=true
    --export=sqlite
    -o "$NSYS_OUTPUT"
  )
  CMD=("${NSYS_CMD[@]}" "${LAUNCH[@]}" "${COMMON_ARGS[@]}" "${STRATEGY_ARGS[@]}")
else
  CMD=("${LAUNCH[@]}" "${COMMON_ARGS[@]}" "${STRATEGY_ARGS[@]}")
fi

cat > "$META_FILE" <<JSON
{
  "run_id": "$RUN_ID",
  "model_size": "$MODEL_SIZE",
  "model_impl": "$MODEL_IMPL",
  "strategy": "$STRATEGY",
  "fsdp_impl": "$FSDP_IMPL",
  "offload": "$OFFLOAD",
  "cuda_graph": "$CUDA_GRAPH",
  "train_iters": "$TRAIN_ITERS",
  "warmup_iters": "$WARMUP_ITERS",
  "run_timeout_seconds": "$RUN_TIMEOUT_SECONDS",
  "gpus_per_node": "$GPUS_PER_NODE",
  "nnodes": "$NNODES",
  "hidden_size": "$HIDDEN_SIZE",
  "ffn_hidden_size": "$FFN_HIDDEN_SIZE",
  "num_layers": "$NUM_LAYERS",
  "num_heads": "$NUM_HEADS",
  "seq_length": "$SEQ_LENGTH",
  "mlp_rank": "$MLP_RANK",
  "attn_rank": "$ATTN_RANK",
  "python": "$PYTHON",
  "cuda_launch_blocking": "${CUDA_LAUNCH_BLOCKING:-}",
  "torch_nccl_async_error_handling": "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}",
  "cuda_device_max_connections": "${CUDA_DEVICE_MAX_CONNECTIONS:-}",
  "fsdp_use_precision_aware_optimizer": "${FSDP_USE_PRECISION_AWARE_OPTIMIZER:-0}",
  "fsdp_grad_reduce_in_bf16": "${FSDP_GRAD_REDUCE_IN_BF16:-0}",
  "fsdp_use_nccl_ub": "${FSDP_USE_NCCL_UB:-0}",
  "fsdp_double_buffer": "${FSDP_DOUBLE_BUFFER:-0}",
  "fsdp_init_model_with_meta_device": "${FSDP_INIT_MODEL_WITH_META_DEVICE:-0}",
  "fsdp_use_torch_optimizer": "${FSDP_USE_TORCH_OPTIMIZER:-0}",
  "profile_nsys": "$PROFILE_NSYS",
  "profile_step_start": "$PROFILE_STEP_START",
  "profile_step_end": "$PROFILE_STEP_END",
  "profile_ranks": "$PROFILE_RANKS",
  "nsys_output": "$NSYS_OUTPUT",
  "log_file": "$LOG_FILE"
}
JSON
printf '%q ' "${CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

echo "RUN_ID=$RUN_ID"
echo "LOG_FILE=$LOG_FILE"
echo "META_FILE=$META_FILE"
echo "CMD_FILE=$CMD_FILE"
if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

set +e
timeout --kill-after=30s "$RUN_TIMEOUT_SECONDS" "${CMD[@]}" > "$LOG_FILE" 2>&1
status=$?
set -e
if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
  echo "RUN_TIMEOUT after ${RUN_TIMEOUT_SECONDS}s" >> "$LOG_FILE"
  echo "RUN_TIMEOUT after ${RUN_TIMEOUT_SECONDS}s"
fi
exit "$status"
