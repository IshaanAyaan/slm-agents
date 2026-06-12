#!/usr/bin/env bash
# Phase-2 turnkey pipeline: sub-500M subroutine specialists on one GPU pod.
# Idempotent stage markers allow restart after interruption:
#   setup -> datagen -> sft_export -> dryrun -> train -> eval -> floor -> DONE
# No vLLM, no teacher, no API keys: data is oracle-labeled, eval is local HF.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-/root/slm-agents}"
RES="$REPO/slm_harness/results/subroutines"
STAGEDIR="$RES/.stages"
PY="$REPO/.venv/bin/python"
LOG="${LOG:-/root/run_phase2.log}"

SIZES="${SIZES:-smollm2-135m,smollm2-360m,qwen2.5-0.5b,qwen2.5-1.5b}"
SUBS="${SUBS:-action_router,evidence_judge,json_repair,path_normalizer,read_span_selector,search_hit_ranker,search_query_gen,trace_localizer}"
N_TRAIN="${N_TRAIN:-2000}"; N_VAL="${N_VAL:-150}"; N_TEST="${N_TEST:-300}"
MAX_TEST="${MAX_TEST:-250}"
GPU_USD_HR="${GPU_USD_HR:-3.19}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
stage_done(){ [[ -f "$STAGEDIR/$1.done" ]]; }
mark(){ mkdir -p "$STAGEDIR"; touch "$STAGEDIR/$1.done"; log "STAGE $1 DONE"; }

hf_for(){ "$PY" - "$1" <<'EOF'
import sys
from slm_harness.subroutines.registry import MODEL_GRID
print(MODEL_GRID[sys.argv[1]]["hf_id"])
EOF
}

setup(){
  stage_done setup && return
  cd "$REPO"
  python3 -m venv .venv 2>/dev/null || true
  $PY -m pip install -q -U pip setuptools wheel
  $PY -m pip install -q -e ".[dev]"
  # Same coherent cu124 stack the Phase-1 run validated, minus vLLM.
  $PY -m pip install -q "torch==2.5.1" "transformers==4.48.3" "trl==0.14.0" \
      "peft==0.14.0" "datasets>=2.19" "accelerate>=0.34"
  $PY -c "import torch; x=torch.zeros(8, device='cuda'); print('cuda OK', torch.cuda.get_device_name(0))"
  mark setup
}

datagen(){
  stage_done datagen && return
  cd "$REPO"
  $PY -m slm_harness.scripts.gen_subroutine_data \
    --out "$RES/data" --repos-cache /root/repos \
    --train "$N_TRAIN" --val "$N_VAL" --test "$N_TEST" 2>&1 | tee -a "$LOG"
  mark datagen
}

sft_export(){
  stage_done sft_export && return
  cd "$REPO"
  $PY -m slm_harness.training.build_subroutine_sft \
    --data "$RES/data" --out "$RES/sft" 2>&1 | tee -a "$LOG"
  mark sft_export
}

dryrun(){
  stage_done dryrun && return
  cd "$REPO"
  local first_sub; first_sub="${SUBS%%,*}"
  $PY -m slm_harness.training.train_subroutine_sft \
    --base-model "$(hf_for smollm2-135m)" \
    --train "$RES/sft/$first_sub/train.jsonl" --val "$RES/sft/$first_sub/val.jsonl" \
    --out "$RES/models/_dryrun" --epochs 0.02 --max-examples 64 2>&1 | tee -a "$LOG"
  rm -rf "$RES/models/_dryrun"
  mark dryrun
}

train_one(){
  local sub="$1" size="$2"
  local out="$RES/models/$sub/$size"
  [[ -f "$out/train_meta.json" ]] && { log "skip train $sub/$size"; return; }
  local epochs=3 bs=32 ga=1
  case "$size" in
    qwen2.5-0.5b) bs=16; ga=2 ;;
    qwen2.5-1.5b) epochs=2; bs=8; ga=2 ;;
  esac
  # Long-window subroutines (1.5k-2k tokens/example): the fp32-upcast LM-head
  # logits dominate memory (seq x bs x vocab x 4B; Qwen's 152k vocab is worst),
  # so shrink bs at constant tokens/step via grad accumulation. Gradient
  # checkpointing in the trainer handles the activation side.
  case "$sub" in
    read_span_selector|evidence_judge)
      bs=8; ga=4
      [[ "$size" == "qwen2.5-1.5b" ]] && { bs=2; ga=8; }
      ;;
    json_repair|trace_localizer)
      [[ "$size" == "qwen2.5-1.5b" ]] && { bs=4; ga=4; }
      ;;
  esac
  log "train $sub/$size (epochs=$epochs bs=$bs ga=$ga)"
  $PY -m slm_harness.training.train_subroutine_sft \
    --base-model "$(hf_for "$size")" \
    --train "$RES/sft/$sub/train.jsonl" --val "$RES/sft/$sub/val.jsonl" \
    --out "$out" --epochs "$epochs" --batch-size "$bs" --grad-accum "$ga" \
    2>&1 | tee -a "$LOG"
}

train_all(){
  stage_done train && return
  cd "$REPO"
  IFS=',' read -ra sizes <<< "$SIZES"; IFS=',' read -ra subs <<< "$SUBS"
  for size in "${sizes[@]}"; do
    for sub in "${subs[@]}"; do train_one "$sub" "$size"; done
  done
  mark train
}

eval_all(){
  stage_done eval && return
  cd "$REPO"
  IFS=',' read -ra sizes <<< "$SIZES"
  for size in "${sizes[@]}"; do
    local bs=64; [[ "$size" == "qwen2.5-1.5b" ]] && bs=32
    $PY -m slm_harness.evals.subroutine_eval \
      --data "$RES/data" --models-root "$RES/models" --out "$RES/eval" \
      --sizes "$size" --conditions ft,base --subroutines "$SUBS" \
      --max-test "$MAX_TEST" --batch-size "$bs" 2>&1 | tee -a "$LOG"
  done
  mark eval
}

floor(){
  stage_done floor && return
  cd "$REPO"
  $PY -m slm_harness.evals.parameter_floor \
    --eval "$RES/eval" --data "$RES/data" --out "$RES" \
    --gpu-usd-hr "$GPU_USD_HR" --figures 2>&1 | tee -a "$LOG"
  mark floor
}

main(){
  log "=== run_phase2 start (SIZES=$SIZES) ==="
  setup; datagen; sft_export; dryrun; train_all; eval_all; floor
  log "=== run_phase2 DONE ==="
}
main "$@"
