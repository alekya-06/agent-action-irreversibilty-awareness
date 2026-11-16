#!/usr/bin/env bash
# overnight.sh: run R3 + AB4 on the DGX with no manual steps.
# Steps: wait for conda env -> check files -> download models -> patch job script
#        -> tests -> sbatch -> wait for the job (retry up to 3 times) -> archive results.
# Start (from anywhere):  nohup bash ~/dnn-proj/overnight.sh > ~/overnight.log 2>&1 &
# Stop everything:        pkill -f overnight.sh ; scancel -u $USER
set -uo pipefail

# ---- settings (edit here only) ------------------------------------------------
ENV_NAME=rq3b
PART=gpu_student
GRES=gpu:a100_3g.20gb:1
PROJ=$HOME/dnn-proj
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
MAX_ATTEMPTS=3
WAIT_ENV_MINUTES=480
QWEN=Qwen/Qwen2.5-7B-Instruct
LLAMA=meta-llama/Llama-3.1-8B-Instruct
# -------------------------------------------------------------------------------

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "FATAL: $*"; exit 1; }

mkdir -p "$HF_HOME"
grep -q "HF_HOME=" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

cd "$PROJ" || die "folder $PROJ does not exist"
log "project: $PROJ | HF_HOME: $HF_HOME | partition: $PART | gres: $GRES"

# 1. Update the repository (R3 files are copied by scp and are not affected).
GIT_TERMINAL_PROMPT=0 git pull --ff-only >/dev/null 2>&1 && log "git pull OK" || log "git pull skipped or failed (continuing)"

# 2. Move decision_points.jsonl into place if scp left it in requirements/.
if [[ -f requirements/decision_points.jsonl ]]; then
  mkdir -p results/rq3 && mv requirements/decision_points.jsonl results/rq3/
  log "moved decision_points.jsonl to results/rq3/"
fi

# 3. Check that every file the job needs exists. Stop now if not (before any long wait).
missing=0
for f in requirements/requirement3/agent_loop.py requirements/requirement3/run_rq3.py \
         requirements/requirement3/analyze_rq3.py requirements/requirement3/decision_points.py \
         requirements/requirement3/test_rq3.py requirements/requirement3/run_rq3_dgx.sh \
         requirements/prompts.py data/agent_tasks/tasks_v1.jsonl \
         results/rq2/k_Qwen2.5-7B-Instruct.json results/rq2/k_Llama-3.1-8B-Instruct.json \
         results/rq3/decision_points.jsonl; do
  [[ -f "$f" ]] || { log "MISSING: $f"; missing=1; }
done
[[ -d src/sandbox_v2 ]] || { log "MISSING: src/sandbox_v2"; missing=1; }
(( missing == 0 )) || die "copy the missing files from the Mac (scp step) or run git pull, then start again"

# 4. Find the conda environment and wait until it is complete.
if [[ -z "${PY:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base); else CONDA_BASE=$HOME/miniconda3; fi
  PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi
log "python: $PY"
ready=0
for ((i = 0; i < WAIT_ENV_MINUTES; i++)); do
  if [[ -x "$PY" ]] && "$PY" -c "import torch, transformers, numpy, pytest, huggingface_hub" 2>/dev/null \
     && ! pgrep -u "$(id -un)" -f "pip install" >/dev/null; then ready=1; break; fi
  (( i % 10 == 0 )) && log "waiting for conda env $ENV_NAME to finish installing..."
  sleep 60
done
(( ready == 1 )) || die "conda env $ENV_NAME is not ready after $WAIT_ENV_MINUTES minutes"
log "conda env ready"

# 5. Hugging Face token (needed for Llama). Put the token in ~/.hf_token (chmod 600).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then HF_TOKEN=$(<"$HOME/.hf_token"); fi
export HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] || log "WARNING: no HF token. Qwen will download. Llama will fail."

# 6. Download the models on the login node (compute nodes may have no internet).
#    The patterns skip the 16 GB 'original/*.pth' files in the Llama repository.
dl() {  # dl <model> [local]   -> "local" only checks the local cache
  local m=$1 mode=${2:-net}
  "$PY" - "$m" "$mode" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
m, mode = sys.argv[1], sys.argv[2]
snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
                  local_files_only=(mode == "local"))
PYEOF
}
have_model() { dl "$1" local >/dev/null 2>&1; }
get_model() {
  have_model "$1" && { log "$1 already in cache"; return 0; }
  for a in 1 2 3 4 5; do
    log "downloading $1 (attempt $a of 5)"
    dl "$1" && have_model "$1" && { log "$1 downloaded"; return 0; }
    sleep 30
  done
  return 1
}
MODELS=""
get_model "$QWEN"  && MODELS="qwen"  || log "WARNING: Qwen download failed"
get_model "$LLAMA" && MODELS="$MODELS llama" || log "WARNING: Llama download failed (licence not accepted, or token missing?)"
MODELS=$(echo $MODELS)
[[ -n "$MODELS" ]] || die "no model could be downloaded"
log "models to run: $MODELS"
du -sh "$HF_HOME"/hub/models--* 2>/dev/null || true

# 7. Patch the job script (safe to run many times).
F=requirements/requirement3/run_rq3_dgx.sh
sed -i "s/^#SBATCH --gres=.*/#SBATCH --gres=$GRES/" "$F"
if grep -q "^#SBATCH --partition=" "$F"; then
  sed -i "s/^#SBATCH --partition=.*/#SBATCH --partition=$PART/" "$F"
else
  sed -i "/^#SBATCH --gres=/a #SBATCH --partition=$PART" "$F"
fi
if ! grep -q "GPU memory GiB" "$F"; then
  cat > /tmp/memcheck_$$.txt <<'MEM'
"$PY" -c "import torch; m=torch.cuda.get_device_properties(0).total_memory/2**30; print('GPU memory GiB', round(m,1)); assert m>=18, 'GPU slice too small'"
MEM
  sed -i "/assert torch.cuda.is_available()/r /tmp/memcheck_$$.txt" "$F"
  rm -f /tmp/memcheck_$$.txt
fi
bash -n "$F" || die "job script has a syntax error"
grep -n -E "^#SBATCH|GPU memory GiB" "$F"

# 8. Tests on the login node (CPU only, a few seconds).
"$PY" -m pytest -q requirements/requirement3/test_rq3.py || die "tests failed"

# 9. Submit, wait, retry. The job uses --resume, so a retry continues where it stopped.
mkdir -p results
attempt=1
ok=0
while (( attempt <= MAX_ATTEMPTS )); do
  # If a job named rq3 already exists (for example after a second start of this script), wait for it.
  JID=$(squeue -h -u "$(id -un)" -n rq3 -o %i 2>/dev/null | head -n 1)
  if [[ -n "$JID" ]]; then
    log "job $JID (name rq3) already exists: waiting for it, no new job"
  else
    JID=$(PY="$PY" MODELS="$MODELS" HF_HUB_OFFLINE=1 sbatch --parsable "$F" | cut -d';' -f1)
    [[ "$JID" =~ ^[0-9]+$ ]] || die "sbatch failed: '$JID'"
    log "submitted job $JID (attempt $attempt of $MAX_ATTEMPTS)"
  fi
  while true; do
    out=$(squeue -h -j "$JID" -o %T 2>&1); rc=$?
    if (( rc == 0 )); then
      [[ -z "$out" ]] && break
      [[ "$out" != "${last:-}" ]] && { log "job $JID state: $out"; last=$out; }
    else
      [[ "$out" == *"Invalid job id"* ]] && break
    fi
    sleep 120
  done
  LOG=results/rq3_dgx_${JID}.log
  if [[ -f "$LOG" ]] && grep -q "^== analysis" "$LOG" && ! grep -q "^!! " "$LOG" && [[ -f results/rq3/summary.md ]]; then
    ok=1; break
  fi
  log "job $JID did not finish cleanly. Last lines of $LOG:"
  tail -n 25 "$LOG" 2>/dev/null || log "(no log file)"
  attempt=$((attempt + 1))
done

# 10. Archive the results (the brief requires all agent logs).
ARCH=$HOME/rq3_results_$(date +%F_%H%M).tar.gz
tar czf "$ARCH" results/rq3 results/rq3_dgx_*.log 2>/dev/null || true
if (( ok == 1 )); then
  log "DONE. Summary: results/rq3/summary.md | Archive: $ARCH"
  head -n 40 results/rq3/summary.md
else
  log "FINISHED WITH PROBLEMS after $MAX_ATTEMPTS attempts. See results/rq3_dgx_*.log | Archive: $ARCH"
  exit 1
fi
