"""Audit stratified samples of Emilia prompt pairs; never changes source data.

SIM files are sampled in evenly spaced byte windows, then metadata is scanned
fully to join both ends by ID. Percentages describe this sample, not a census.
No waveform decoding, ASR verification, or global duplicate detection is done.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np


def sample_lines(path, windows, rows_per_window):
    size = path.stat().st_size
    seen = set()
    with path.open("rb") as stream:
        for window in range(windows):
            offset = int(size * window / windows)
            stream.seek(offset)
            if offset:
                stream.readline()  # Skip the partial line at this byte offset.
            stop = int(size * (window + 1) / windows)
            for _ in range(rows_per_window):
                position = stream.tell()
                if position >= stop:
                    break
                line = stream.readline()
                if not line:
                    break
                if position not in seen:
                    seen.add(position)
                    yield line


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def summary(values):
    array = np.array([v for v in values if v is not None], dtype=float)
    if not len(array):
        return {"count": 0}
    return {"count": len(array), "mean": float(array.mean()),
            "quantiles": dict(zip(["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"],
                                  np.quantile(array, [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]).tolist()))}


def normalized_text(text):
    return "".join(c for c in str(text).casefold() if c.isalnum())


def audit_language(task):
    directory, language, windows, rows_per_window = task
    directory = Path(directory)
    sim_path = directory / f"{language}_sim.list"
    metadata_path = directory / f"{language}.jsonl"
    pairs, wanted, issues = [], set(), Counter()
    for line in sample_lines(sim_path, windows, rows_per_window):
        fields = line.decode().strip().split()
        if len(fields) != 3 or finite(fields[2]) is None:
            issues["invalid_sampled_sim_line"] += 1
            continue
        target, prompt, score = fields
        pairs.append((target, prompt, float(score)))
        wanted.update((target.encode(), prompt.encode()))
    if not pairs:
        raise ValueError(f"No valid sampled pairs in {sim_path}")

    # Only retain metadata needed by sampled pairs, rather than building an
    # all-corpus in-memory ID map. Fast path matches this corpus's JSON format;
    # other layouts use the JSON parser to extract the ID correctly.
    metadata, scanned = {}, 0
    prefix = b'{"id": "'
    with metadata_path.open("rb") as stream:
        for scanned, line in enumerate(stream, 1):
            if scanned % 2_000_000 == 0:
                print(f"{language}: scanned {scanned:,} metadata rows", flush=True)
            if line.startswith(prefix):
                identifier = line[len(prefix):line.find(b'"', len(prefix))]
                if identifier not in wanted:
                    continue
                row = json.loads(line)
            else:
                row = json.loads(line)
                identifier = str(row.get("id", "")).encode()
                if identifier not in wanted:
                    continue
            key = identifier.decode()
            if key in metadata:
                issues["duplicate_metadata_id_among_requested_ids"] += 1
            metadata[key] = row
    print(f"{language}: joined {len(metadata):,}/{len(wanted):,} requested IDs, scanned {scanned:,} metadata rows", flush=True)

    sampled_metadata = [json.loads(line) for line in sample_lines(metadata_path, windows, rows_per_window)]
    independent = {"rows": len(sampled_metadata),
                   "duration": summary(finite(row.get("duration")) for row in sampled_metadata),
                   "dnsmos": summary(finite(row.get("dnsmos")) for row in sampled_metadata)}
    grid = {(sim, mos): {"pairs": 0, "target_hours_in_sample": 0.0}
            for sim in [.7, .75, .8, .85, .9, .95] for mos in [0, 3, 3.2, 3.4, 3.5]}
    joined, examples = [], []
    for target_id, prompt_id, sim in pairs:
        target, prompt = metadata.get(target_id), metadata.get(prompt_id)
        failures = []
        if not -1 <= sim <= 1:
            failures.append("sim_out_of_range")
        if target_id == prompt_id:
            failures.append("self_pair")
            if sim == 0:
                issues["self_pair_with_sim_zero"] += 1
        if target_id.rsplit("_W", 1)[0] != prompt_id.rsplit("_W", 1)[0]:
            failures.append("speaker_id_mismatch")
        if target is None or prompt is None:
            failures.append("missing_target_or_prompt_metadata")
        else:
            duration, prompt_duration = finite(target.get("duration")), finite(prompt.get("duration"))
            mos, prompt_mos = finite(target.get("dnsmos")), finite(prompt.get("dnsmos"))
            same_text = bool(normalized_text(target.get("text", ""))) and normalized_text(target.get("text", "")) == normalized_text(prompt.get("text", ""))
            joined.append({"duration": duration, "prompt_duration": prompt_duration,
                           "mos": mos, "prompt_mos": prompt_mos, "same_text": same_text})
            if target.get("wav") == prompt.get("wav"):
                failures.append("same_wav_path")
            if not str(target.get("text", "")).strip() or not normalized_text(target.get("text", "")):
                failures.append("empty_target_text")
            if duration is None or not 1 <= duration <= 20:
                failures.append("target_duration_outside_1_20")
            if prompt_duration is None or not 3 <= prompt_duration <= 30:
                failures.append("prompt_duration_outside_3_30")
            if mos is None or prompt_mos is None:
                failures.append("missing_dnsmos")
            if same_text:
                issues["same_normalized_text"] += 1
                if target_id != prompt_id:
                    issues["same_normalized_text_different_ids"] += 1
                if target_id != prompt_id and len(examples) < 8:
                    examples.append({"target": target_id, "prompt": prompt_id, "sim": sim,
                                     "text": target["text"]})
            if not failures:
                for (sim_min, mos_min), cell in grid.items():
                    if sim >= sim_min and min(mos, prompt_mos) >= mos_min:
                        cell["pairs"] += 1
                        cell["target_hours_in_sample"] += duration / 3600
        issues.update(failures)

    n = len(pairs)
    return {"language": language, "sample_pairs": n,
            "sample_source_groups": len({pair[0].split("_S", 1)[0] for pair in pairs}),
            "sample_speaker_ids": len({pair[0].rsplit("_W", 1)[0] for pair in pairs}),
            "metadata_total_lines_scanned": scanned, "requested_metadata_ids": len(wanted),
            "matched_metadata_ids": len(metadata), "joined_pairs": len(joined),
            "similarity": summary(pair[2] for pair in pairs),
            "sim_only_retention": {str(threshold): sum(pair[2] >= threshold for pair in pairs) / n for threshold in [.7, .75, .8, .85, .9, .95, .98, .99]},
            "target_duration": summary(row["duration"] for row in joined),
            "prompt_duration": summary(row["prompt_duration"] for row in joined),
            "target_dnsmos": summary(row["mos"] for row in joined),
            "prompt_dnsmos": summary(row["prompt_mos"] for row in joined),
            "independent_metadata_sample": independent, "issue_counts_overlapping": dict(issues),
            "same_text_examples": examples,
            "retention_grid": [{"sim_min": sim, "both_dnsmos_min": mos,
                                "retained_fraction": value["pairs"] / n, **value}
                               for (sim, mos), value in grid.items()],
            "inputs": [{"path": str(path.resolve()), "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns} for path in [sim_path, metadata_path]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/Emilia"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--rows-per-window", type=int, default=512)
    args = parser.parse_args()
    if args.windows <= 0 or args.rows_per_window <= 0:
        parser.error("Sample sizes must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        languages = list(pool.map(audit_language, [(str(args.directory), lang, args.windows, args.rows_per_window) for lang in ["en", "zh"]]))
    report = {"sampling": "Deterministic evenly spaced byte windows; SIM percentages and joint retention are sample estimates, NOT full-corpus counts/hours. Metadata is fully scanned for sampled-ID joins only.",
              "windows": args.windows, "rows_per_window": args.rows_per_window,
              "grid_rules": "ID differs; ID speaker matches; paths differ; nonempty target text; target duration 1-20s; prompt duration 3-30s; finite SIM in [-1,1]; both DNSMOS present. Same text is flagged, NOT excluded in grid.",
              "unchecked": ["waveform existence/decoding/quality", "ASR consistency", "global ID uniqueness", "audio duplicates or overlap", "true speaker identity and cross-source leakage"],
              "elapsed_seconds": time.monotonic() - started, "languages": languages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
