#!/usr/bin/env bash
# Serve a model with vLLM (OpenAI-compatible). V100/Volta-safe.
#
#   serve_vllm.sh teacher        # serve $TEACHER_MODEL tensor-parallel over GPUs
#   serve_vllm.sh model          # serve $SERVE_MODEL as $SERVE_NAME on $SERVE_PORT
#
# On Volta (V100) set VLLM_DTYPE=half (fp16; bf16 is unsupported) and we force the
# xformers attention backend (no FlashAttention on sm70). LoRA is NOT served live on
# V100 (vLLM multi-LoRA needs sm80+); instead we merge the adapter and serve it here
# via the generic `model` role.
set -euo pipefail

ROLE="${1:?usage: serve_vllm.sh teacher|model}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
DTYPE="${VLLM_DTYPE:-auto}"
MAXLEN="${MAX_MODEL_LEN:-8192}"

# Volta has no FlashAttention; force a compatible backend when running fp16.
if [[ "$DTYPE" == "half" || "$DTYPE" == "float16" ]]; then
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
fi

ngpu() { nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '; }

case "$ROLE" in
  teacher)
    MODEL="${TEACHER_MODEL:?set TEACHER_MODEL}"
    PORT="${TEACHER_PORT:-8001}"
    TP="${TEACHER_TP:-$(ngpu)}"
    echo "[serve] teacher $MODEL tp=$TP port=$PORT dtype=$DTYPE maxlen=$MAXLEN backend=${VLLM_ATTENTION_BACKEND:-default}"
    exec vllm serve "$MODEL" \
      --served-model-name teacher \
      --tensor-parallel-size "$TP" \
      --dtype "$DTYPE" \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.90 \
      --enforce-eager \
      --api-key "$API_KEY" \
      --port "$PORT"
    ;;
  model)
    MODEL="${SERVE_MODEL:?set SERVE_MODEL (path or HF id)}"
    NAME="${SERVE_NAME:-$MODEL}"
    PORT="${SERVE_PORT:-8002}"
    TP="${SERVE_TP:-1}"
    echo "[serve] model $MODEL as '$NAME' tp=$TP port=$PORT dtype=$DTYPE maxlen=$MAXLEN backend=${VLLM_ATTENTION_BACKEND:-default}"
    exec vllm serve "$MODEL" \
      --served-model-name "$NAME" \
      --tensor-parallel-size "$TP" \
      --dtype "$DTYPE" \
      --max-model-len "$MAXLEN" \
      --gpu-memory-utilization 0.90 \
      --enforce-eager \
      --api-key "$API_KEY" \
      --port "$PORT"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 2;;
esac
