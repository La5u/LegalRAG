#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source .venv/bin/activate
export LEGALRAG_GENERATION_MODEL=qwen2.5:1.5b-instruct
export LEGALRAG_JUDGE_MODEL=qwen2.5:1.5b-instruct
export LEGALRAG_CONTEXT_WINDOW=2048
export LEGALRAG_MAX_OUTPUT_TOKENS=160
export LEGALRAG_TOP_K=3
export LEGALRAG_OLLAMA_KEEP_ALIVE=1m
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
exec python -m evaluation.evaluator \
  --questions evaluation/questions.jsonl \
  --results evaluation/deepseek_full_results.jsonl \
  --strategies vector,graph,neuro_symbolic \
  --concurrency 1
