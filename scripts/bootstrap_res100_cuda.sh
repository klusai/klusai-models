#!/usr/bin/env bash
# Run ONLY on an already provisioned Ubuntu CUDA box. Does not provision, publish, or upload.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: bootstrap_res100_cuda.sh MODEL_GIT_REF BENCH_GIT_REF DATASETS_GIT_REF [WORK_DIR]' >&2
  exit 2
fi
model_ref=$1
bench_ref=$2
data_ref=$3
work_dir=${4:-"$PWD/res100-wave"}
command -v git >/dev/null
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi # Existing driver required; this script never provisions a GPU.
if pgrep -af '[p]ython.*(train.py|train_)'; then
  echo 'Another training process is live. Queue this wave and rerun after it exits.' >&2
  exit 1
fi
mkdir -p "$work_dir"
cd "$work_dir"
work_dir=$PWD
# Ubuntu prerequisites: git, python3, python3-venv, ca-certificates; working NVIDIA driver.
python3 -m venv .bootstrap
.bootstrap/bin/pip install 'uv>=0.8,<1'
.bootstrap/bin/uv python install 3.13
for repo in klusai-models europriv-bench klusai-datasets; do
  if [[ ! -d "$repo/.git" ]]; then
    git clone "https://github.com/klusai/$repo.git" "$repo"
  fi
  git -C "$repo" diff --exit-code
  git -C "$repo" diff --cached --exit-code
done
for pair in "klusai-models:$model_ref" "europriv-bench:$bench_ref" "klusai-datasets:$data_ref"; do
  repo=${pair%%:*}
  ref=${pair#*:}
  git -C "$repo" fetch origin "$ref"
  git -C "$repo" checkout --detach FETCH_HEAD
done
cd klusai-models
"$work_dir/.bootstrap/bin/uv" venv --python 3.13 --allow-existing .venv-moe
"$work_dir/.bootstrap/bin/uv" pip install --python .venv-moe/bin/python -e ../europriv-bench -e '.[moe,dev]'
# Only the TAB curation module is needed; avoid its VCS dependency overriding the pinned harness.
export PYTHONPATH="$PWD:$work_dir/klusai-datasets${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p runs/res100-logs
.venv-moe/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA wheel/driver unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$work_dir/.bootstrap/bin/uv" pip freeze --python .venv-moe/bin/python > runs/res100-logs/environment.txt
for repo in klusai-models europriv-bench klusai-datasets; do
  git -C "$work_dir/$repo" rev-parse HEAD >> runs/res100-logs/git-refs.txt
done
# Preserve the prepared corpus across retries. Review manifest.json before starting another wave.
if ! .venv-moe/bin/python -c 'import hashlib,json,pathlib; p=pathlib.Path("runs/res100-data"); m=json.loads((p/"manifest.json").read_text()); assert m["complete"]; assert hashlib.sha256((p/"train.jsonl").read_bytes()).hexdigest() == m["outputs"]["train"]["sha256"]' 2>/dev/null; then
  .venv-moe/bin/python scripts/prepare_res100_data.py --output-dir runs/res100-data \
    2>&1 | tee runs/res100-logs/prepare.log
fi
# Sequential seeds; no competing jobs on this GPU. A second invocation resumes any saved seed.
for seed in 0 1 2; do
  output="runs/res100-seed$seed"
  resume=()
  if .venv-moe/bin/python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"training-manifest.json").read_text()); assert m["completed_steps"] > 0; assert (p/"merged/config.json").is_file(); assert (p/"merged/model.safetensors").is_file()' "$output" 2>/dev/null; then
    echo "Seed $seed already exported; skipping."
    continue
  fi
  if [[ -f "$output/training-manifest.json" ]]; then
    resume=(--resume latest)
  fi
  .venv-moe/bin/python scripts/train.py moe-finetune \
    --base openai/privacy-filter --revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b \
    --dataset runs/res100-data/train.jsonl --out klusai/kp-deid-moe-res100 \
    --backend cuda --device cuda --epochs 3 --lr 0.0001 --batch-size 4 --lora-rank 8 \
    --max-length 256 --gradient-accumulation-steps 4 --save-steps 250 --seed "$seed" \
    --output-dir "$output" "${resume[@]}" 2>&1 | tee -a "runs/res100-logs/seed$seed.log"
done
