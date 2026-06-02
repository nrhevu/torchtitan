#!/usr/bin/bash
# Chạy trên node vu (replica 0 + lighthouse).
# Chạy script này TRƯỚC, sau đó mới chạy 03_run_son.sh trên node son.
set -euxo pipefail

IMAGE="${IMAGE:-torchtitan:rocm7.2.4}"
VU_ADDR="${VU_ADDR:-10.130.0.13}"
LIGHTHOUSE_PORT="${LIGHTHOUSE_PORT:-29510}"
RDZV_PORT="${RDZV_PORT:-29500}"
TORCHFT_MANAGER_PORT="${TORCHFT_MANAGER_PORT:-29520}"
NGPU="${NGPU:-4}"
ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-4,5,6,7}"
STEPS="${STEPS:-10000}"
RUN_NAME="${RUN_NAME:-qwen3_ft}"
CONTAINER_NAME="${CONTAINER_NAME:-torchtitan_ft_r0_vu_${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/nfs_server/vunguyen13/outputs}"
ASSET_ROOT="${ASSET_ROOT:-/scratch/nfs_server/vunguyen13/ServicePackage/torchtitan/assets/hf}"

mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"

# ── Lighthouse ─────────────────────────────────────────────────────────────
docker rm -f torchft_lighthouse 2>/dev/null || true

docker run -d \
    --name torchft_lighthouse \
    --network host \
    --restart unless-stopped \
    "${IMAGE}" \
    bash -lc "pip uninstall torchft torchft-nightly -y >/tmp/torchft_uninstall.log 2>&1 || true; \
              pip install torchft-nightly -q && \
              torchft_lighthouse --join_timeout_ms 1000000 --quorum_tick_ms 10000 --min_replicas 1 --bind [::]:${LIGHTHOUSE_PORT}"

echo "Lighthouse started at http://${VU_ADDR}:${LIGHTHOUSE_PORT}, waiting 8s..."
sleep 8

# ── Training replica 0 ──────────────────────────────────────────────────────
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --hostname "${VU_ADDR}" \
    --network host \
    --ipc host \
    --shm-size 16g \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e NODE_ADDR="${VU_ADDR}" \
    -e RDZV_PORT="${RDZV_PORT}" \
    -e NGPU="${NGPU}" \
    -e STEPS="${STEPS}" \
    -e RUN_NAME="${RUN_NAME}" \
    -e TORCHFT_LIGHTHOUSE="http://${VU_ADDR}:${LIGHTHOUSE_PORT}" \
    -e TORCHFT_MANAGER_PORT="${TORCHFT_MANAGER_PORT}" \
    -e TORCHFT_TIMEOUT_SEC=120 \
    -e TORCHFT_CONNECT_TIMEOUT_SEC=120 \
    -e TORCHFT_QUORUM_TIMEOUT_SEC=120 \
    -e MASTER_ADDR="${VU_ADDR}" \
    -e MASTER_PORT="${RDZV_PORT}" \
    -e NCCL_SOCKET_IFNAME=eth1 \
    -e GLOO_SOCKET_IFNAME=eth1 \
    -e NCCL_P2P_DISABLE=1 \
    -e RUST_LOG=info \
    -e RUST_BACKTRACE=1 \
    -e PYTORCH_ALLOC_CONF="expandable_segments:True" \
    -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" \
    -v /scratch/nfs_server/vunguyen13/tokenized_data:/workspace/torchtitan/tokenized_data:ro \
    -v "${ASSET_ROOT}:/workspace/torchtitan/assets/hf:ro" \
    -v "${OUTPUT_ROOT}:/workspace/torchtitan/outputs" \
    -w /workspace/torchtitan \
    "${IMAGE}" \
    bash -lc '
pip install torchft-nightly -q

cat > /tmp/qwen3_ft.toml <<TOML
[job]
dump_folder = "./outputs/${RUN_NAME}/r0_vu"
description = "Qwen 3 0.6B tokenized dataset training"

[profiling]
enable_profiling = false

[metrics]
log_freq = 1
enable_tensorboard = false

[model]
name = "qwen3"
flavor = "0.6B"
hf_assets_path = "./assets/hf/Qwen3-0.6B"

[optimizer]
name = "AdamW"
lr = 3e-4
eps = 1e-8

[lr_scheduler]
warmup_steps = 1

[training]
local_batch_size = 1
seq_len = 128
max_norm = 1.0
steps = 10000
dataset = "tokenized_disk"
dataset_path = "./tokenized_data"

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
fsdp_reshard_after_forward = "default"
tensor_parallel_degree = 1
context_parallel_degree = 1

[checkpoint]
enable = false
folder = "checkpoint"
interval = 500
export_dtype = "float16"
async_mode = "disabled"

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"

[compile]
enable = false
components = ["model", "loss"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = false
precompute_float8_dynamic_scale_for_fsdp = false
filter_fqns = ["output"]
TOML

torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_endpoint="${NODE_ADDR}:${RDZV_PORT}" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    -m torchtitan.experiments.ft.train \
    --job.config_file /tmp/qwen3_ft.toml \
    --job.dump_folder "./outputs/${RUN_NAME}/r0_vu" \
    --fault_tolerance.enable \
    --fault_tolerance.replica_id=0 \
    --fault_tolerance.group_size=2 \
    --fault_tolerance.min_replica_size=1 \
    --fault_tolerance.process_group=gloo \
    --fault_tolerance.process_group_timeout_ms=120000 \
    --training.steps="${STEPS}"
'

docker rm -f torchft_lighthouse 2>/dev/null || true
