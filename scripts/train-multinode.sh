#!/usr/bin/env bash
set -euo pipefail

export RDZV_PORT="$((RDZV_BASE_PORT + REPLICA_ID))"
export TORCHFT_MANAGER_PORT="$((TORCHFT_MANAGER_BASE_PORT + (REPLICA_ID * NNODES) + NODE_RANK))"
# Kubernetes allocates 4 physical GPUs per pod. Inside the pod they are addressed as local ordinals 0..3.

case "${REPLICA_ID}" in
        0)
        export HIP_VISIBLE_DEVICES="0,1,2,3"
        ;;
        1)
        export HIP_VISIBLE_DEVICES="4,5,6,7"
        ;;
    esac
export MASTER_PORT="${RDZV_PORT}"
export RUN_SUFFIX="r${REPLICA_ID}"
export TRAIN_OUTPUT_DIR="./outputs/${RUN_NAME}/${RUN_SUFFIX}"
export TRAIN_LOG="${TRAIN_OUTPUT_DIR}/trainer-node${NODE_RANK}.log"

mkdir -p "$(dirname "${TORCHTITAN_CONFIG_PATH}")" ./assets/hf "${TRAIN_OUTPUT_DIR}"

cat <<EOF
TorchFT trainer environment
    replica_id=${REPLICA_ID}
    group_size=${GROUP_SIZE}
    node_rank=${NODE_RANK}
    nnodes=${NNODES}
    ngpu=${NGPU}
    rocr_visible_devices=${ROCR_VISIBLE_DEVICES:-unset}
    hip_visible_devices=${HIP_VISIBLE_DEVICES:-unset}
    cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}
    rdzv_host_rank=${RDZV_HOST_RANK}
    master_addr=${MASTER_ADDR}
    rdzv_port=${RDZV_PORT}
    master_port=${MASTER_PORT}
    torchft_manager_port=${TORCHFT_MANAGER_PORT}
    node_addr=${NODE_ADDR:-unknown}
    pod=${POD_NAME:-unknown}
    job=${JOB_NAME:-unknown}
EOF

pip install torchft-nightly -q

if [ ! -f ./assets/hf/Qwen3-32B/tokenizer.json ]; then
    python scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-32B \
    --local_dir ./assets/hf \
    --assets tokenizer
fi

set +e
torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_id="${RUN_NAME}_replica_${REPLICA_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    -m torchtitan.experiments.ft.train \
    --job.config_file "${TORCHTITAN_CONFIG_PATH}" \
    --job.dump_folder "${TRAIN_OUTPUT_DIR}" \
    --fault_tolerance.enable \
    --fault_tolerance.replica_id="${REPLICA_ID}" \
    --fault_tolerance.group_size="${GROUP_SIZE}" \
    --fault_tolerance.min_replica_size=1 \
    --fault_tolerance.process_group=gloo \
    --fault_tolerance.process_group_timeout_ms=120000 \
    --training.steps="${STEPS}" \
    2>&1 | tee "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}
set -e

find "${TRAIN_OUTPUT_DIR}" -maxdepth 6 -print | sort | tee "${TRAIN_OUTPUT_DIR}/output-listing-node${NODE_RANK}.txt"
exit "${train_status}"