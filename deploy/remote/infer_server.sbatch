#!/bin/bash
#SBATCH --job-name=legalrag_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Adapt the partition above to your cluster.
# On-demand vLLM inference server for LegalRAG, sized to release the GPU
# on its own instead of sitting idle (shared GPUs should not sit
# idle). PORT and IDLE_TIMEOUT_SECONDS come in via `sbatch --export`.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL="${LEGALRAG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-18080}"
IDLE_TIMEOUT="${IDLE_TIMEOUT_SECONDS:-600}"
VLLM_LOG="logs/vllm-server-${SLURM_JOB_ID}.log"

source "${LEGALRAG_CONDA_SH:-/opt/miniforge3/etc/profile.d/conda.sh}"
conda activate "${LEGALRAG_CONDA_ENV:-legalrag-vllm}"

echo "=== SLURM JOB INFO ==="
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "IDLE_TIMEOUT=${IDLE_TIMEOUT}s"
nvidia-smi
echo "==============================================="

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name legalrag \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Stopping vLLM (pid $VLLM_PID) and releasing the GPU"
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
  exit 0
}
trap cleanup TERM INT

echo "Waiting for vLLM to come up on 127.0.0.1:${PORT} ..."
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process exited during startup, check ${VLLM_LOG}"
    exit 1
  fi
  if curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/v1/models"; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  echo "vLLM did not become ready in time, check ${VLLM_LOG}"
  cleanup
fi
echo "vLLM is ready."

# Idle-shutdown watchdog. vLLM logs periodic throughput stats even with
# zero traffic, so log mtime can't signal idleness; the cumulative
# request counter in /metrics can.
prev_count=-1
last_change=$(date +%s)
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process exited, ending job"
    exit 1
  fi

  count=$(curl -s -m 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
    | awk '/^vllm:request_success_total/ {sum += $2} END {print sum + 0}')
  now=$(date +%s)
  if [[ -n "$count" && "$count" != "$prev_count" ]]; then
    prev_count="$count"
    last_change="$now"
  fi

  idle=$(( now - last_change ))
  if (( idle > IDLE_TIMEOUT )); then
    echo "No new requests for ${idle}s (limit ${IDLE_TIMEOUT}s); shutting down to free the GPU"
    cleanup
  fi
  sleep 30
done
