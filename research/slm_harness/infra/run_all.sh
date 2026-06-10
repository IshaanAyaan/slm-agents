#!/usr/bin/env bash
# Turnkey, GPU-frugal, resumable orchestration for the self-hosted (zero-API) run.
# V100/Volta-safe: fp16 serving, pinned vLLM, LoRA merged offline (no live multi-LoRA).
#
#   bash research/slm_harness/infra/run_all.sh [stage]
#
# Stages: preflight setup genbench teacher train students_a students_b report
# (default: all, in order; pass one stage name to resume)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
RES="research/slm_harness/results/real"
LOGS="$RES/logs"; mkdir -p "$LOGS"
BENCH="research/slm_harness/tasks/realbench"
CK="research/slm_harness/training/checkpoints"

[[ -f "$HERE/config.env" ]] && source "$HERE/config.env" || { echo "create $HERE/config.env (copy config.env.example)"; exit 1; }
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
# Auto-pick serving dtype if unset: fp16 on pre-Ampere (no bf16), else auto (bf16).
if [[ -z "${VLLM_DTYPE:-}" ]]; then
  _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  if [[ -n "$_cc" && "$_cc" -lt 80 ]]; then export VLLM_DTYPE=half; else export VLLM_DTYPE=auto; fi
fi
export TEACHER_TP="${TEACHER_TP:-$NUM_GPUS}"
TEACHER_URL="http://localhost:${TEACHER_PORT:-8001}/v1"
STUDENT_URL="http://localhost:${STUDENT_PORT:-8002}/v1"
PY="python"
# Use the project-local venv when it exists (isolates us from the base image's pins).
[[ -x "$REPO/.venv/bin/python" ]] && { export PATH="$REPO/.venv/bin:$PATH"; hash -r 2>/dev/null || true; }

log(){ echo "[$(date +%H:%M:%S)] $*"; }

gpu_cc(){ # compute capability of GPU0, e.g. "7.0"
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
}

wait_health(){ # url timeout_s [server_pid]
  # Confirm it is really our vLLM OpenAI server (a JSON model list), not some other
  # process (e.g. nginx) squatting on the port. Abort early if the server died.
  local url="$1" t="${2:-3600}" pid="${3:-}" i=0
  local auth="Authorization: Bearer ${VLLM_API_KEY:-EMPTY}"
  until curl -fsS -H "$auth" "$url/models" 2>/dev/null | grep -q '"object"'; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: serving process ($pid) exited before becoming ready"; return 1
    fi
    sleep 10; i=$((i+10))
    if [[ $i -ge $t ]]; then echo "timeout waiting for $url"; return 1; fi
    if (( i % 120 == 0 )); then log "still waiting for $url (${i}s; model may be downloading)"; fi
  done; log "endpoint healthy (verified vLLM): $url"
}

serve_bg(){ bash "$HERE/serve_vllm.sh" "$1" >"$2" 2>&1 & echo $!; }
stop_pid(){ [[ -n "${1:-}" ]] && kill "$1" 2>/dev/null || true; sleep 8; }

# ----------------------------------------------------------------------------
stage_preflight(){
  log "preflight: GPUs=$NUM_GPUS compute_cap=$(gpu_cc)"
  nvidia-smi -L || { echo "no GPUs"; exit 1; }
  df -h "$REPO" | tail -1
  local free_gb; free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  [[ "$free_gb" -ge 60 ]] || log "WARN: <60GB free; clear ~/.cache/huggingface if downloads fail"
}

stage_setup(){
  # Build a CLEAN virtualenv so RunPod's preinstalled, mutually-conflicting packages
  # (e.g. a cu12.9 torch, a transformers pin) can't corrupt our stack. One coherent
  # cu124 set that runs on the pod's 12.8 driver; trl 0.14 (unlike 0.12) accepts the
  # transformers that vLLM 0.7.3 requires, so a single-pass resolve succeeds.
  log "setup: clean virtualenv + pinned stack (compute_cap=$(gpu_cc))"
  if [[ ! -x "$REPO/.venv/bin/python" ]]; then
    python3 -m venv "$REPO/.venv" || { python3 -m pip install --user -q virtualenv && python3 -m virtualenv "$REPO/.venv"; }
  fi
  export PATH="$REPO/.venv/bin:$PATH"; hash -r 2>/dev/null || true; PY="python"
  $PY -m pip install -q -U pip setuptools wheel
  $PY -m pip install -q -e ".[dev]"
  local cc vllm_pin; cc=$(gpu_cc)
  if [[ "${cc%%.*}" -lt 8 ]]; then vllm_pin="vllm==0.6.6.post1"; else vllm_pin="vllm==0.7.3"; fi
  $PY -m pip install -q "torch==2.5.1" "$vllm_pin" "transformers==4.48.3" \
      "trl==0.14.0" "peft==0.14.0" "datasets>=2.19" "accelerate>=0.34" matplotlib
  $PY -c "import torch,transformers,vllm,trl,peft; print('torch',torch.__version__,'tf',transformers.__version__,'vllm',vllm.__version__,'trl',trl.__version__,'cuda_ok',torch.cuda.is_available())"
  $PY -m pytest research/slm_harness/tests/ -q
}

stage_genbench(){
  log "genbench: $BENCH_REPOS"
  rm -f "$BENCH/run_train.json" "$BENCH/run_test.json"
  for spec in $BENCH_REPOS; do
    local id rest path split
    id="${spec%%:*}"; rest="${spec#*:}"; path="${rest%:*}"; split="${rest##*:}"
    if [[ "$path" == auto ]]; then  # id:auto:module:split form
      :
    fi
    # support 'auto:<module>' -> installed package dir
    if [[ "$path" == auto:* || "$path" == auto ]]; then
      local mod="${path#auto:}"; [[ "$mod" == auto ]] && mod="$id"
      path=$($PY - "$mod" <<'P' 2>/dev/null || true
import importlib,os,sys
m=importlib.import_module(sys.argv[1]); print(os.path.dirname(m.__file__))
P
)
    fi
    [[ "$path" = /* ]] || path="$REPO/$path"
    [[ -d "$path" ]] || { log "skip $id (missing: $path)"; continue; }
    local out n
    if [[ "$split" == "train" ]]; then out="$BENCH/run_train.json"; n="${TRAIN_MAX_TASKS:-250}"
    else out="$BENCH/run_test.json"; n="${TEST_MAX_TASKS:-120}"; fi
    $PY "$BENCH/generate_tasks.py" --repo "$path" --repo-id "$id" --split "$split" \
        --max-tasks "$n" --out "$out" --append
  done
  for f in run_train run_test; do
    [[ -f "$BENCH/$f.json" ]] || { echo "ERROR: $BENCH/$f.json missing (no repos found for that split)"; exit 1; }
    log "$f: $($PY -c "import json;print(len(json.load(open('$BENCH/$f.json'))['tasks']))") tasks"
  done
}

measure_price(){ # base_url served_name num_gpus out.json
  $PY -m research.slm_harness.infra.cost_model --base-url "$1" --model "$2" \
      --num-gpus "$3" --gpu-hourly-usd "${GPU_HOURLY_USD:-2.5}" --out "$4" \
      || log "WARN: price measurement failed for $2 (will use fallback pricing)"
}

stage_teacher(){
  pkill -f "vllm serve" 2>/dev/null || true; sleep 3
  rm -rf "$RES/teacher" "$RES/eval_teacher" "$RES/distillation"  # clean re-runs
  log "teacher: serve $TEACHER_MODEL (tp=${TEACHER_TP:-auto}) on port ${TEACHER_PORT:-8001}"
  local PID; PID=$(serve_bg teacher "$LOGS/teacher.log")
  trap "stop_pid $PID" RETURN
  wait_health "$TEACHER_URL" 5400 "$PID" || { echo '--- teacher.log tail ---'; tail -30 "$LOGS/teacher.log"; exit 1; }
  measure_price "$TEACHER_URL" teacher "${TEACHER_TP:-$NUM_GPUS}" "$RES/price_teacher.json"
  log "teacher: distillation data-gen (C1,C6 on train, ${TRAIN_SEEDS:-2} seeds)"
  $PY -m research.slm_harness.infra.gen_teacher_data \
      --tasks "$BENCH/run_train.json" --teacher-url "$TEACHER_URL" \
      --teacher-pricing "$RES/price_teacher.json" \
      --seeds "${TRAIN_SEEDS:-2}" --out "$RES/teacher" --distill-out "$RES/distillation"
  log "teacher: eval C1,C6 on test (${SEEDS:-3} seeds)"
  $PY -m research.slm_harness.infra.run_conditions \
      --tasks "$BENCH/run_test.json" --split test --conditions C1 C6 \
      --teacher-url "$TEACHER_URL" --teacher-pricing "$RES/price_teacher.json" \
      --seeds "${SEEDS:-3}" --out "$RES/eval_teacher"
}

slug(){ basename "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.\n' '-'; }

train_one(){ # base_model out_dir
  local base="$1" out="$2" t0=$SECONDS
  local D="$RES/distillation"
  [[ -s "$D/train.jsonl" ]] || { echo "ERROR: no distillation data at $D/train.jsonl"; exit 1; }
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( ${TRAIN_GPUS:-1} - 1 ))) \
    $PY research/slm_harness/training/train_lora.py --base-model "$base" \
      --train "$D/train.jsonl" --val "$D/val.jsonl" --out "$out" \
      --lora-rank 64 --lora-alpha 128 --epochs 3
  echo $(( SECONDS - t0 )) > "$out/.train_seconds"
  $PY research/slm_harness/training/merge_lora.py --base "$base" \
      --adapter "$out" --out "${out%-lora}-merged" --dtype float16
}

stage_train(){
  log "train: LoRA on distillation data (fp16-aware), then merge for Volta serving"
  train_one "$STUDENT_A_MODEL" "$CK/$(slug "$STUDENT_A_MODEL")-navsearch-lora"
  if [[ "${RUN_ABLATION:-1}" == "1" ]]; then
    train_one "$STUDENT_B_MODEL" "$CK/$(slug "$STUDENT_B_MODEL")-navsearch-lora"
  fi
}

ft_cost(){ # seconds_file gpus
  $PY - "$(cat "$1" 2>/dev/null || echo 0)" "$2" "${GPU_HOURLY_USD:-2.5}" <<'P'
import sys; s,g,r=(float(x) for x in sys.argv[1:4]); print(f"{s/3600.0*g*r:.4f}")
P
}

eval_student(){ # base_model variant_tag
  local base="$1" tag="$2"
  local lora="$CK/$(slug "$base")-navsearch-lora" merged
  merged="${lora%-lora}-merged"
  [[ -d "$merged" ]] || { echo "ERROR: merged model missing: $merged (run 'train')"; exit 1; }
  mkdir -p "$RES/eval_$tag" "$RES/eval_${tag}_base" "$RES/eval_${tag}_ft"
  rm -f "$RES/eval_${tag}_base/runs.jsonl" "$RES/eval_${tag}_ft/runs.jsonl"  # clean re-runs
  local fc; fc=$(ft_cost "$lora/.train_seconds" "${TRAIN_GPUS:-1}")
  log "[$tag] fine-tune cost: \$$fc"

  # Phase 1: base student -> C2 (generic), C3 (custom)
  pkill -f "vllm serve" 2>/dev/null || true; sleep 3
  log "[$tag] serving BASE $base for C2,C3 on port ${STUDENT_PORT:-8002}"
  export SERVE_MODEL="$base" SERVE_NAME="$base" SERVE_PORT="${STUDENT_PORT:-8002}" SERVE_TP=1
  local PID; PID=$(serve_bg model "$LOGS/student_${tag}_base.log")
  wait_health "$STUDENT_URL" 3600 "$PID" || { tail -30 "$LOGS/student_${tag}_base.log"; stop_pid $PID; exit 1; }
  measure_price "$STUDENT_URL" "$base" 1 "$RES/eval_$tag/price_base.json" || true
  $PY -m research.slm_harness.infra.run_conditions \
      --experiment-id "real-$tag" --tasks "$BENCH/run_test.json" --split test \
      --conditions C2 C3 \
      --student-url "$STUDENT_URL" --student-model "$base" --adapter-name "navsearch-$tag" \
      --student-base-pricing "$RES/eval_$tag/price_base.json" \
      --teacher-url "$TEACHER_URL" --teacher-pricing "$RES/price_teacher.json" \
      --seeds "${SEEDS:-3}" --out "$RES/eval_${tag}_base"
  stop_pid $PID

  # Phase 2: merged fine-tuned student -> C4 (generic), C5 (custom)
  pkill -f "vllm serve" 2>/dev/null || true; sleep 3
  log "[$tag] serving MERGED $merged as navsearch-$tag for C4,C5 on port ${STUDENT_PORT:-8002}"
  export SERVE_MODEL="$merged" SERVE_NAME="navsearch-$tag" SERVE_PORT="${STUDENT_PORT:-8002}" SERVE_TP=1
  PID=$(serve_bg model "$LOGS/student_${tag}_ft.log")
  wait_health "$STUDENT_URL" 3600 "$PID" || { tail -30 "$LOGS/student_${tag}_ft.log"; stop_pid $PID; exit 1; }
  measure_price "$STUDENT_URL" "navsearch-$tag" 1 "$RES/eval_$tag/price_ft.json" || true
  $PY -m research.slm_harness.infra.run_conditions \
      --experiment-id "real-$tag" --tasks "$BENCH/run_test.json" --split test \
      --conditions C4 C5 \
      --student-url "$STUDENT_URL" --student-model "$base" --adapter-name "navsearch-$tag" \
      --student-ft-pricing "$RES/eval_$tag/price_ft.json" \
      --teacher-url "$TEACHER_URL" --teacher-pricing "$RES/price_teacher.json" \
      --finetune-cost "$fc" --seeds "${SEEDS:-3}" --out "$RES/eval_${tag}_ft"
  stop_pid $PID
  echo "$fc" > "$RES/eval_${tag}_ft/.ft_cost" 2>/dev/null || { mkdir -p "$RES/eval_${tag}_ft"; echo "$fc" > "$RES/eval_${tag}_ft/.ft_cost"; }
}

stage_students_a(){ mkdir -p "$RES/eval_a"; eval_student "$STUDENT_A_MODEL" a; }
stage_students_b(){
  [[ "${RUN_ABLATION:-1}" == "1" ]] || { log "skip ablation"; return; }
  mkdir -p "$RES/eval_b"; eval_student "$STUDENT_B_MODEL" b
}

stage_report(){
  log "report: merge runs, metrics, figures, white paper"
  for tag in a b; do
    [[ -f "$RES/eval_${tag}_base/runs.jsonl" && -f "$RES/eval_${tag}_ft/runs.jsonl" ]] || continue
    local merged="$RES/runs_$tag.jsonl"
    cat "$RES/eval_teacher/runs.jsonl" "$RES/eval_${tag}_base/runs.jsonl" \
        "$RES/eval_${tag}_ft/runs.jsonl" > "$merged"
    local fc; fc=$(cat "$RES/eval_${tag}_ft/.ft_cost" 2>/dev/null || echo 0)
    $PY research/slm_harness/scripts/compute_metrics.py --runs "$merged" \
        --finetune-cost "$fc" --out "$RES/metrics_$tag.json"
    $PY research/slm_harness/scripts/make_figures.py --runs "$merged" --real \
        --finetune-cost "$fc" --out "$RES/figures_$tag"
  done
  $PY research/slm_harness/paper/fill_paper.py \
      --headline "$RES/metrics_a.json" --ablation "$RES/metrics_b.json" \
      --figures "$RES/figures_a" --out "research/slm_harness/paper/white_paper.md" || true
  log "DONE. metrics: $RES/metrics_a.json ; paper: research/slm_harness/paper/white_paper.md"
}

STAGE="${1:-all}"
run(){ log ">>> stage $1"; "stage_$1"; }
case "$STAGE" in
  all) for s in preflight setup genbench teacher train students_a students_b report; do run "$s"; done;;
  *) run "$STAGE";;
esac
