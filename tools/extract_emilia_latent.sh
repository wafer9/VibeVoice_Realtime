#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional first positional argument: en or zh. Keep --languages compatible.
if [[ "${1:-}" == en || "${1:-}" == zh ]]; then
  extract_language="$1"
  shift
  set -- "$@" --languages "${extract_language}"
fi
extract_python="${EXTRACT_PYTHON:-/data/joe/anaconda3/envs/vibe/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"
if [[ ! "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'PROCESSES_PER_GPU must be a positive integer' >&2
  exit 2
fi

# Check before torchrun, so inaccessible GPUs produce a clear error and no workers.
extract_gpu_count="$("${extract_python}" -c 'import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable; CPU fallback is disabled")
print(torch.cuda.device_count())')"
extract_process_count=$((extract_gpu_count * PROCESSES_PER_GPU))
if [[ "${extract_process_count}" -eq 1 ]]; then
  exec "${extract_python}" "${project_dir}/tools/extract_emilia_latent.py" "$@"
fi
exec "${extract_python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${extract_process_count}" \
  "${project_dir}/tools/extract_emilia_latent.py" "$@"
