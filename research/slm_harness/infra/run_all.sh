#!/usr/bin/env bash
# Turnkey, GPU-frugal, resumable orchestration for the self-hosted (zero-API) run.
#
#   bash research/slm_harness/infra/run_all.sh [stage]
#
# Stages (run in order; pass one to resume): preflight setup genbench teacher train
#   students_4b students_17b report  (default: all)
#
# Only one model family is resident at a time, so this works even on a modest GPU count.
# Edit research/slm_harness/infra/config.env first (copy from config.env.example).
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
TEACHER_URL="http://localhost:${TEACHER_PORT:-8001}/v1"
STUDENT_URL="http://localhost:${STUDENT_PORT:-8002}/v1"
PY="python"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

wait_health(){ # url, timeout_s
  local url="$1" t="${2:-1800}" i=0
  until curl -fsS "${url%/v1}/health" >/dev/null 2>&1 || curl -fsS "$url/models" >/dev/null 2>&1; do
    sleep 5; i=$((i+5)); [[ $i -ge $t ]] && { echo "timeout waiting for $url"; return 1; }
  done; log "endpoint healthy: $url"
}

serve_bg(){ # role logfile -> echoes PID
  bash "$HERE/serve_vllm.sh" "$1" >"$2" 2>&1 & echo $!
}
stop(){ [[ -n "${1:-}" ]] && kill "$1" 2>/dev/null || true; wait "${1:-}" 2>/dev/null || true; }

# ----------------------------------------------------------------------------
stage_preflight(){
  log "preflight: GPUs=$NUM_GPUS"
  nvidia-smi -L || { echo "no GPUs"; exit 1; }
  command -v vllm >/dev/null || echo "WARN: vllm not on PATH yet (setup stage installs it)"
  df -h "$REPO" | tail -1
  [[ "$NUM_GPUS" -ge 1 ]] || { echo "need >=1 GPU"; exit 1; }
}

stage_setup(){
  log "setup: installing deps"
  $PY -m pip install -q -U pip
  $PY -m pip install -q -e ".[dev]"
  $PY -m pip install -q "vllm>=0.6.3" "transformers>=4.51" "peft>=0.11" "trl>=0.9" \
      "datasets>=2.19" "accelerate>=0.30" matplotlib
  $PY -m pytest research/slm_harness/tests/ -q
}

stage_genbench(){
  log "genbench: generating real tasks over $BENCH_REPOS"
  rm -f "$BENCH/run_train.json" "$BENCH/run_test.json"
  for spec in $BENCH_REPOS; do
    id="${spec%%:*}"; rest="${spec#*:}"; path="${rest%:*}"; split="${rest##*:}"
    [[ -d "$path" ]] || { log "skip $id (missing $path)"; continue; }
    if [[ "$split" == "train" ]]; then out="$BENCH/run_train.json"; n="${TRAIN_MAX_TASKS:-250}"
    else out="$BENCH/run_test.json"; n="${TEST_MAX_TASKS:-120}"; fi
    $PY "$BENCH/generate_tasks.py" --repo "$path" --repo-id "$id" --split "$split" \
        --max-tasks "$n" --out "$out" --workspace-rel --append
  done
  log "genbench done: $(grep -c task_id "$BENCH/run_train.json" 2>/dev/null || echo 0) train tokens"
}

measure_price(){ # base_url model num_gpus out.json
  $PY -m research.slm_harness.infra.cost_model --base-url "$1" --model "$2" \
      --num-gpus "$3" --gpu-hourly-usd "${GPU_HOURLY_USD:-2.5}" --out "$4" || true
}

stage_teacher(){
  log "teacher stage: serve $TEACHER_MODEL"
  PID=$(serve_bg teacher "$LOGS/teacher.log"); trap "stop $PID" RETURN
  wait_health "$TEACHER_URL" 3600
  measure_price "$TEACHER_URL" teacher "$NUM_GPUS" "$RES/price_teacher.json"
  log "teacher: generating distillation data (C1,C6 on train)"
  $PY -m research.slm_harness.infra.gen_teacher_data \
      --tasks "$BENCH/run_train.json" --teacher-url "$TEACHER_URL" \
      --teacher-pricing "$RES/price_teacher.json" \
      --seeds "${TRAIN_SEEDS:-3}" --out "$RES/teacher" --distill-out "$RES/distillation"
  log "teacher: evaluating C1,C6 on test"
  $PY -m research.slm_harness.infra.run_conditions \
      --tasks "$BENCH/run_test.json" --split test --conditions C1 C6 \
      --teacher-url "$TEACHER_URL" --teacher-pricing "$RES/price_teacher.json" \
      --seeds "${SEEDS:-5}" --out "$RES/eval_teacher"
}

train_one(){ # base_model train.jsonl val.jsonl out_dir
  local t0=$SECONDS
  $PY research/slm_harness/training/train_lora.py --base-model "$1" \
      --train "$2" --val "$3" --out "$4" --lora-rank 64 --lora-alpha 128 --epochs 3
  echo $(( SECONDS - t0 )) > "$4/.train_seconds"
}

stage_train(){
  log "train stage: LoRA on distillation data"
  D="$RES/distillation"
  train_one "$STUDENT4B_MODEL" "$D/train.jsonl" "$D/val.jsonl" "$CK/qwen3-4b-navsearch-lora"
  if [[ "${RUN_17B_ABLATION:-1}" == "1" ]]; then
    train_one "$STUDENT17B_MODEL" "$D/train.jsonl" "$D/val.jsonl" "$CK/qwen3-1.7b-navsearch-lora"
  fi
}

ft_cost(){ # train_seconds_file -> usd
  local s; s=$(cat "$1" 2>/dev/null || echo 0)
  $PY - "$s" "$NUM_GPUS" "${GPU_HOURLY_USD:-2.5}" <<'PY'
import sys; s,g,r=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); print(f"{s/3600.0*g*r:.4f}")
PY
}

eval_students(){ # base_model adapter_path adapter_name expid outdir
  local base="$1" apath="$2" aname="$3" expid="$4" outdir="$5"
  mkdir -p "$outdir"
  export STUDENT_BASE="$base" ADAPTER_PATH="$apath" ADAPTER_NAME="$aname"
  local PID; PID=$(serve_bg students "$LOGS/students_$expid.log"); trap "stop $PID" RETURN
  wait_health "$STUDENT_URL" 1800
  measure_price "$STUDENT_URL" "$base" "$NUM_GPUS" "$outdir/price_base.json"
  measure_price "$STUDENT_URL" "$aname" "$NUM_GPUS" "$outdir/price_ft.json"
  local fc; fc=$(ft_cost "$apath/.train_seconds")
  log "$expid: fine-tune cost = \$$fc ; evaluating C2,C3,C4,C5 on test"
  $PY -m research.slm_harness.infra.run_conditions \
      --experiment-id "$expid" --tasks "$BENCH/run_test.json" --split test \
      --conditions C2 C3 C4 C5 \
      --student-url "$STUDENT_URL" --student-model "$base" --adapter-name "$aname" \
      --student-base-pricing "$outdir/price_base.json" \
      --student-ft-pricing "$outdir/price_ft.json" \
      --finetune-cost "$fc" --seeds "${SEEDS:-5}" --out "$outdir"
  echo "$fc" > "$outdir/.ft_cost"
}

stage_students_4b(){
  log "students stage (4B headline)"
  eval_students "$STUDENT4B_MODEL" "$CK/qwen3-4b-navsearch-lora" navsearch-4b real-4b "$RES/eval_4b"
}
stage_students_17b(){
  [[ "${RUN_17B_ABLATION:-1}" == "1" ]] || { log "skip 1.7B ablation"; return; }
  log "students stage (1.7B ablation)"
  eval_students "$STUDENT17B_MODEL" "$CK/qwen3-1.7b-navsearch-lora" navsearch-17b real-17b "$RES/eval_17b"
}

stage_report(){
  log "report stage: merge runs, metrics, figures, paper"
  for variant in 4b 17b; do
    ev="$RES/eval_$variant"; [[ -d "$ev" ]] || continue
    merged="$RES/runs_$variant.jsonl"
    cat "$RES/eval_teacher/runs.jsonl" "$ev/runs.jsonl" > "$merged" 2>/dev/null || cp "$ev/runs.jsonl" "$merged"
    fc=$(cat "$ev/.ft_cost" 2>/dev/null || echo 0)
    $PY research/slm_harness/scripts/compute_metrics.py --runs "$merged" --finetune-cost "$fc" \
        --out "$RES/metrics_$variant.json"
    $PY research/slm_harness/scripts/make_figures.py --runs "$merged" --real \
        --finetune-cost "$fc" --out "$RES/figures_$variant"
  done
  $PY research/slm_harness/paper/fill_paper.py \
      --headline "$RES/metrics_4b.json" --ablation "$RES/metrics_17b.json" \
      --figures "$RES/figures_4b" --out "research/slm_harness/paper/white_paper.md" || true
  log "DONE. See $RES/metrics_4b.json and research/slm_harness/paper/white_paper.md"
}

STAGE="${1:-all}"
run(){ log ">>> stage $1"; "stage_$1"; }
case "$STAGE" in
  all) for s in preflight setup genbench teacher train students_4b students_17b report; do run "$s"; done;;
  *) run "$STAGE";;
esac
