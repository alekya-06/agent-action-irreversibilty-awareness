#!/usr/bin/env bash
# R3 + AB4 on the DGX: 2 models x 2 prompt modes x 120 task rows, then the analysis.
#
# Each (model, prompt mode) is a separate Python process, so only one model is ever on
# the GPU and the GPU is freed when the process exits. Safe to re-run: --resume skips
# finished runs and the script continues where it stopped.
#
# Usage (from the project root, on the DGX):
#   bash requirements/requirement3/run_rq3_dgx.sh                  # everything
#   GPU=3 bash requirements/requirement3/run_rq3_dgx.sh            # pick a GPU
#   MODELS="qwen" MODES="neutral" bash requirements/requirement3/run_rq3_dgx.sh
#   nohup bash requirements/requirement3/run_rq3_dgx.sh > results/rq3_dgx.log 2>&1 &
# Under Slurm: sbatch requirements/requirement3/run_rq3_dgx.sh  (edit the #SBATCH lines)
#
#SBATCH --job-name=rq3
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --partition=gpu_student
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/rq3_dgx_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
PY=${PY:-python}
MODELS=${MODELS:-"qwen llama"}
MODES=${MODES:-"neutral careful"}
if [[ -z "${SLURM_JOB_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES=${GPU:-0}; fi
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS

"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
"$PY" -c "import torch; m=torch.cuda.get_device_properties(0).total_memory/2**30; print('GPU memory GiB', round(m,1)); assert m>=18, 'GPU slice too small'"

echo "== tests (fake models, no GPU)"
"$PY" -m pytest -q requirements/requirement3/test_rq3.py

echo "== pre-registered decision points (no model)"
"$PY" requirements/requirement3/decision_points.py

status=0
for model in $MODELS; do
  for mode in $MODES; do
    echo "== $model / $mode  ($(date))"
    "$PY" requirements/requirement3/run_rq3.py --model "$model" --prompt-mode "$mode" --seed 0 --resume || {
      echo "!! $model / $mode had failed runs; re-run this script to retry them"; status=1; }
  done
done

echo "== analysis"
"$PY" requirements/requirement3/analyze_rq3.py
exit $status
