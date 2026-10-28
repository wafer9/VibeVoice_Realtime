#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$NPROC_PER_NODE" \
        -m twinlakes.bin.train "$@"
else
    exec "$PYTHON_BIN" -m twinlakes.bin.train "$@"
fi
