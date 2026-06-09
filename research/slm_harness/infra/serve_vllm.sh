#!/usr/bin/env bash
# Serve a model with vLLM (OpenAI-compatible). Used for the teacher and students.
#
# Usage:
#   serve_vllm.sh teacher   # serve the large open teacher (tensor-parallel over GPUs)
#   serve_vllm.sh students  # serve Qwen3-4B + 1.7B base WITH LoRA adapters enabled
#
# Config via env (see infra/secrets.example.env / run_all.sh):
#   TEACHER_MODEL, TEACHER_PORT, TEACHER_TP
#   STUDENT4B_MODEL, STUDENT17B_MODEL, STUDENT_PORT
#   ADAPTER4B_PATH, ADAPTER17B_PATH   (optional; enables LoRA serving)
#   VLLM_API_KEY                      (optional; default "EMPTY")
set -euo pipefail

ROLE="${1:?usage: serve_vllm.sh teacher|students}"
API_KEY="${VLLM_API_KEY:-EMPTY}"

ngpu() { nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '; }

case "$ROLE" in
  teacher)
    MODEL="${TEACHER_MODEL:?set TEACHER_MODEL, e.g. meta-llama/Llama-3.3-70B-Instruct}"
    PORT="${TEACHER_PORT:-8001}"
    TP="${TEACHER_TP:-$(ngpu)}"
    echo "[serve] teacher $MODEL  tp=$TP  port=$PORT"
    exec vllm serve "$MODEL" \
      --served-model-name teacher \
      --tensor-parallel-size "$TP" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.92 \
      --api-key "$API_KEY" \
      --port "$PORT"
    ;;
  students)
    # Serve ONE student base, optionally with ONE LoRA adapter, so the same script
    # handles the 4B headline and the 1.7B ablation passes.
    MODEL="${STUDENT_BASE:-${STUDENT4B_MODEL:-Qwen/Qwen3-4B-Instruct-2507}}"
    PORT="${STUDENT_PORT:-8002}"
    LORA_ARGS=()
    if [[ -n "${ADAPTER_PATH:-}" && -d "${ADAPTER_PATH:-/nonexistent}" ]]; then
      LORA_ARGS=(--enable-lora --max-lora-rank 64 \
                 --lora-modules "${ADAPTER_NAME:-navsearch}=${ADAPTER_PATH}")
    fi
    echo "[serve] students base=$MODEL port=$PORT adapter=${ADAPTER_NAME:-none}@${ADAPTER_PATH:-none}"
    exec vllm serve "$MODEL" \
      --served-model-name "$MODEL" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --api-key "$API_KEY" \
      --port "$PORT" \
      "${LORA_ARGS[@]}"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 2;;
esac
