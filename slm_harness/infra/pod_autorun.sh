#!/usr/bin/env bash
# Autopilot: run the pipeline on the pod and use the GitHub repo as a control bus so
# the agent can drive it without a live shell.
#
# Each cycle it: (1) adopts the latest code from main (any fix the agent pushed),
# (2) runs run_all.sh, (3) publishes the log tail + metrics + serving logs to the
# 'pod-status' branch (force-pushed; never collides with main), (4) on failure waits
# a few minutes for the agent to push a fix, then retries. Stops on success, on
# `touch STOP_AUTORUN`, or after MAX_CYCLES.
#
# Launch (in the pod's Jupyter terminal), pasting your token once:
#   cd /workspace/slm-harness
#   GIT_TOKEN=YOUR_GITHUB_PAT nohup bash slm_harness/infra/pod_autorun.sh \
#       > autopilot.log 2>&1 &
#   tail -f autopilot.log
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
STATUS_DIR="slm_harness/_status"
MAX_CYCLES="${MAX_CYCLES:-12}"
RETRY_SLEEP="${RETRY_SLEEP:-180}"
: "${GIT_TOKEN:?set GIT_TOKEN=your_github_pat}"
URL="https://x-access-token:${GIT_TOKEN}@github.com/IshaanAyaan/slm-agents.git"

git config user.email "pod-autorun@slm-agents" 2>/dev/null || true
git config user.name  "pod-autorun" 2>/dev/null || true

stamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }

publish(){ # message
  mkdir -p "$STATUS_DIR"
  { echo "# autopilot status @ $(stamp)"; echo "$1"; } > "$STATUS_DIR/STATUS.md"
  tail -n 400 runall.log    > "$STATUS_DIR/runall.tail.log"    2>/dev/null || true
  tail -n 200 autopilot.log > "$STATUS_DIR/autopilot.tail.log" 2>/dev/null || true
  cp slm_harness/results/real/metrics_*.json "$STATUS_DIR/" 2>/dev/null || true
  for f in slm_harness/results/real/logs/*.log; do
    [[ -f "$f" ]] && tail -n 120 "$f" > "$STATUS_DIR/$(basename "$f").tail" 2>/dev/null || true
  done
  git add -f "$STATUS_DIR" >/dev/null 2>&1 || true
  git commit -q -m "status: $1" >/dev/null 2>&1 || true
  git push -qf "$URL" HEAD:pod-status >/dev/null 2>&1 \
    && echo "[autopilot] published status to pod-status branch" \
    || echo "[autopilot] WARN: status push failed (check token)"
}

echo "[autopilot] start; repo=$REPO max_cycles=$MAX_CYCLES"
for ((i=1; i<=MAX_CYCLES; i++)); do
  [[ -f STOP_AUTORUN ]] && { echo "[autopilot] STOP_AUTORUN present; exiting"; break; }
  echo "[autopilot] cycle $i: pulling latest code from main"
  git fetch -q "$URL" main 2>/dev/null && git reset -q --hard FETCH_HEAD 2>/dev/null || true

  echo "[autopilot] cycle $i: running pipeline ($(stamp))"
  bash slm_harness/infra/run_all.sh > runall.log 2>&1
  code=$?
  echo "[autopilot] cycle $i: run_all exited $code"
  publish "cycle $i exited $code"

  if [[ $code -eq 0 ]]; then
    echo "[autopilot] SUCCESS — pipeline finished. Results published."
    break
  fi
  echo "[autopilot] cycle $i failed; waiting ${RETRY_SLEEP}s for a pushed fix, then retrying"
  for ((s=0; s<RETRY_SLEEP; s+=15)); do [[ -f STOP_AUTORUN ]] && break; sleep 15; done
done
echo "[autopilot] done @ $(stamp)"
