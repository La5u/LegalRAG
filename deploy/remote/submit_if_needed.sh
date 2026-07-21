#!/bin/bash
# Idempotent Slurm submission: reuses an already queued/running job instead
# of stacking duplicates if called again before the server is needed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

JOB_NAME=legalrag_infer
PORT="${1:-18080}"

existing="$(squeue -h -u "$USER" -n "$JOB_NAME" -o '%i %T' | head -1)"
if [[ -n "$existing" ]]; then
  echo "Already queued/running: $existing" >&2
  echo "$existing" | awk '{print $1}'
  exit 0
fi

job_id="$(sbatch --parsable --export=ALL,PORT="$PORT" infer_server.sbatch)"
echo "Submitted job $job_id" >&2
echo "$job_id"
