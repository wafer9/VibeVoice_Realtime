#!/usr/bin/env bash
# Run manually after extraction; retain local files and existing CFS features.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  en) sync_language=EN ;;
  zh) sync_language=ZH ;;
  *) echo 'Usage: bash tools/sync_emilia_latent.sh en|zh' >&2; exit 2 ;;
esac
sync_local_root="${EXTRACT_LOCAL_DIR:-/data/joe/data/latent}"
sync_cfs_root="${EXTRACT_CFS_DIR:-${project_dir}/data/Emilia/latent}"
sync_python="${EXTRACT_PYTHON:-/data/joe/anaconda3/envs/vibe/bin/python}"
export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -d "${sync_local_root}/${sync_language}" || ! -f "${sync_local_root}/_codec.json" ]]; then
  echo "Missing local ${sync_language} cache or codec metadata: ${sync_local_root}" >&2
  exit 1
fi
"${sync_python}" - "${sync_local_root}" "${sync_cfs_root}" <<'PY'
import json
from pathlib import Path
import sys
from tools.extract_emilia_latent import ensure_metadata
source, destination = map(Path, sys.argv[1:])
if source.resolve() == destination.resolve():
    raise SystemExit('Source and destination are the same directory')
ensure_metadata(destination, json.loads((source / '_codec.json').read_text()))
PY
mkdir -p "${sync_cfs_root}/${sync_language}"
exec rsync -a --ignore-existing --info=progress2 --exclude='*.partial' -- \
  "${sync_local_root}/${sync_language}/" "${sync_cfs_root}/${sync_language}/"
