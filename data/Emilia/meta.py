"""
1）按照sim > 0.8 过滤en.jsonl zh.jsonl, sim 来自en_sim.list, zh_sim.list
2）按照wer < 0.1 过滤en.jsonl zh.jsonl, wer 来自/nfs-speech-cfs/wangzhou/data/tts/emilia/en_qwen3asr.wer 
                                              /nfs-speech-cfs/wangzhou/data/tts/emilia/zh_qwen3asr.wer
    en 直接使用en_qwen3asr.wer 中qwen3asr的识别结果，中文使用Emilia 的原始paraformer的识别结果
3）输出到train.jsonl, 新增prompt_wav， 和prompt_text, prompt_wav 是en_sim.list, zh_sim.list第二列的id，prompt_text 是该id对应的文本
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time


DEFAULT_WER_DIR = Path("/nfs-speech-cfs/wangzhou/data/tts/emilia")
MISSING = object()
METADATA_SEEN = object()


def log_progress(language, phase, rows, started):
    if rows % 1_000_000 == 0:
        print(json.dumps({"language": language, "phase": phase, "rows": rows,
                          "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)


def file_identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def filter_language(language, metadata_path, sim_path, wer_path, output,
                    sim_threshold=0.8, wer_threshold=0.1):
    """Join by ID, keeping one language's qualifying IDs in memory.

    Candidate values contain [prompt ID, WER state]. WER state transitions
    from None (missing), to False (rejected) or True/ASR text (accepted), then
    METADATA_SEEN. Prompt text uses the same language-specific source as target
    text, but the prompt itself is not subjected to the target's score gates.
    """
    started = time.monotonic()
    counts = Counter()
    candidates = {}
    prompt_texts = {}
    with open(sim_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            counts["sim_rows"] += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{sim_path}:{line_number}: expected target_id, prompt_id, SIM")
            identifier, prompt_id, raw_score = fields
            score = float(raw_score)
            if not math.isfinite(score) or not -1 <= score <= 1:
                counts["sim_invalid_score"] += 1
            elif score > sim_threshold:
                if identifier in candidates:
                    raise ValueError(f"{sim_path}:{line_number}: duplicate qualifying SIM ID {identifier}")
                candidates[identifier] = [prompt_id, None]
                if language == "en":
                    prompt_texts[prompt_id] = None
                counts["sim_pass"] += 1
            else:
                counts["sim_rejected"] += 1
            log_progress(language, "sim", line_number, started)

    # cal_wer.py writes: ID, WER, raw_reference, raw_hypothesis, INS, DEL, SUB.
    with open(wer_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            counts["wer_rows"] += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 7:
                raise ValueError(f"{wer_path}:{line_number}: expected 7 TSV columns")
            identifier, raw_wer, _, hypothesis, _, _, _ = fields
            score = float(raw_wer)
            valid = math.isfinite(score) and score >= 0
            passes = valid and score < wer_threshold
            counts["wer_pass" if passes else "wer_rejected"] += 1
            if not valid:
                counts["wer_invalid_score"] += 1
            if language == "en" and identifier in prompt_texts:
                if prompt_texts[identifier] is not None:
                    raise ValueError(f"{wer_path}:{line_number}: duplicate prompt WER ID {identifier}")
                prompt_texts[identifier] = hypothesis
            entry = candidates.get(identifier)
            if entry is not None:
                if entry[1] is not None:
                    raise ValueError(f"{wer_path}:{line_number}: duplicate WER ID {identifier}")
                entry[1] = (hypothesis if language == "en" else True) if passes else False
                counts["sim_pass_wer_matched"] += 1
                if passes:
                    counts["both_scores_pass"] += 1
            log_progress(language, "wer", line_number, started)
    counts["sim_pass_missing_wer"] = counts["sim_pass"] - counts["sim_pass_wer_matched"]

    if language == "zh":
        prompt_texts = {entry[0]: None for entry in candidates.values()
                        if entry[1] is not None and entry[1] is not False}
        with open(metadata_path, "rb") as source:
            prefix = b'{"id": "'
            for line_number, line in enumerate(source, 1):
                # Avoid decoding every metadata object during this lookup pass.
                if line.startswith(prefix):
                    identifier = line[len(prefix):line.find(b'"', len(prefix))].decode("utf-8")
                    row = None
                else:
                    row = json.loads(line)
                    identifier = row["id"]
                if identifier in prompt_texts:
                    if prompt_texts[identifier] is not None:
                        raise ValueError(f"{metadata_path}:{line_number}: duplicate prompt metadata ID {identifier}")
                    row = row if row is not None else json.loads(line)
                    if not isinstance(row["text"], str):
                        raise TypeError(f"Invalid prompt text for {identifier}")
                    prompt_texts[identifier] = row["text"]
                log_progress(language, "prompt_text", line_number, started)
    counts["prompt_lookup_ids"] = len(prompt_texts)
    counts["prompt_lookup_missing_text"] = sum(text is None for text in prompt_texts.values())

    input_seconds = accepted_seconds = 0.0
    digest = hashlib.sha256()
    output_bytes = 0
    with open(metadata_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            counts["metadata_rows"] += 1
            row = json.loads(line)
            if not isinstance(row, dict) or not {"id", "text", "duration"} <= row.keys():
                raise ValueError(f"{metadata_path}:{line_number}: missing metadata fields")
            identifier = row["id"]
            if not identifier.startswith(language.upper() + "_") or row.get("language", language) != language:
                raise ValueError(f"{metadata_path}:{line_number}: unexpected language for {identifier}")
            duration = float(row["duration"])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{metadata_path}:{line_number}: invalid duration")
            input_seconds += duration
            entry = candidates.get(identifier)
            if entry is None:
                counts["metadata_without_passing_sim"] += 1
            else:
                prompt_id, state = entry
                if state is METADATA_SEEN:
                    raise ValueError(f"{metadata_path}:{line_number}: duplicate joined metadata ID {identifier}")
                entry[1] = METADATA_SEEN
                if state is None:
                    counts["metadata_missing_wer"] += 1
                elif state is False:
                    counts["metadata_wer_rejected"] += 1
                else:
                    counts["both_scores_pass_metadata_matched"] += 1
                    prompt_text = prompt_texts.get(prompt_id)
                    if prompt_text is None:
                        counts["metadata_missing_prompt_text"] += 1
                        log_progress(language, "metadata", line_number, started)
                        continue
                    if language == "en":
                        if not isinstance(state, str):
                            raise TypeError(f"Missing English ASR text for {identifier}")
                        counts["english_text_changed"] += int(row["text"] != state)
                        row["text"] = state  # Raw Qwen3-ASR hypothesis, not normalized scoring text.
                    row["prompt_wav"] = prompt_id  # User requested the SIM column-2 ID, not a path.
                    row["prompt_text"] = prompt_text
                    if not str(row["text"]).strip():
                        counts["accepted_empty_text"] += 1
                    if not prompt_text.strip():
                        counts["accepted_empty_prompt_text"] += 1
                    encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                    output.write(encoded)
                    digest.update(encoded)
                    output_bytes += len(encoded)
                    counts["accepted"] += 1
                    accepted_seconds += duration
            log_progress(language, "metadata", line_number, started)

    counts["both_scores_pass_missing_metadata"] = counts["both_scores_pass"] - counts["both_scores_pass_metadata_matched"]
    if counts["metadata_rows"] != (counts["metadata_without_passing_sim"] + counts["metadata_missing_wer"]
                                  + counts["metadata_wer_rejected"] + counts["metadata_missing_prompt_text"] + counts["accepted"]):
        raise AssertionError("Metadata count accounting failed")
    report = {"language": language, "counts": dict(counts),
              "input_hours": input_seconds / 3600, "accepted_hours": accepted_seconds / 3600,
              "retained_row_fraction": counts["accepted"] / counts["metadata_rows"] if counts["metadata_rows"] else 0,
              "retained_hour_fraction": accepted_seconds / input_seconds if input_seconds else 0,
              "output_bytes": output_bytes, "output_language_block_sha256": digest.hexdigest(),
              "elapsed_seconds": time.monotonic() - started}
    print(json.dumps({"completed_language": language, **report}, ensure_ascii=False), flush=True)
    del candidates, prompt_texts
    gc.collect()
    return report


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=directory)
    parser.add_argument("--wer-dir", type=Path, default=DEFAULT_WER_DIR)
    parser.add_argument("--output", type=Path, help="Default: <data-dir>/train.jsonl")
    parser.add_argument("--sim-threshold", type=float, default=0.8, help="Strictly greater than")
    parser.add_argument("--wer-threshold", type=float, default=0.1, help="Strictly less than")
    args = parser.parse_args()
    if not math.isfinite(args.sim_threshold) or not -1 <= args.sim_threshold <= 1:
        parser.error("SIM threshold must be finite and within [-1, 1]")
    if not math.isfinite(args.wer_threshold) or args.wer_threshold <= 0:
        parser.error("WER threshold must be finite and positive")
    output_path = (args.output or args.data_dir / "train.jsonl").resolve()
    stats_path = output_path.with_suffix(".stats.json")
    partial = output_path.with_name(output_path.name + ".partial")
    stats_partial = stats_path.with_name(stats_path.name + ".partial")
    for path in [output_path, stats_path, partial, stats_partial]:
        if path.exists():
            raise FileExistsError(f"Choose a new output path; already exists: {path}")
    inputs = [(language, args.data_dir / f"{language}.jsonl", args.data_dir / f"{language}_sim.list",
               args.wer_dir / f"{language}_qwen3asr.wer") for language in ["en", "zh"]]
    identities = [file_identity(path) for item in inputs for path in item[1:]]
    report = {"complete": False, "inputs": identities,
              "rules": {"sim": f"> {args.sim_threshold}", "wer": f"< {args.wer_threshold}",
                        "english_text": "raw Qwen3-ASR hypothesis, WER TSV column 4",
                        "chinese_text": "original Emilia metadata text",
                        "output_schema": "Original metadata plus prompt_wav (SIM column-2 ID) and prompt_text",
                        "prompt_text": "English: raw Qwen3-ASR hypothesis; Chinese: original Emilia metadata text",
                        "prompt_score_filters": "none; score gates apply to target ID only",
                        "missing_prompt_text": "exclude pair and count separately",
                        "order": "en then zh; original metadata order within language",
                        "extra_quality_filters": "none", "invalid_scores": "nonfinite/out-of-range scores rejected",
                        "duplicate_checks": "qualifying SIM IDs; WER and metadata IDs joined to qualifying SIM IDs"},
              "languages": [], "output": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with partial.open("xb", buffering=8 * 1024 * 1024) as output:
            for language, metadata_path, sim_path, wer_path in inputs:
                report["languages"].append(filter_language(language, metadata_path, sim_path, wer_path,
                                                         output, args.sim_threshold, args.wer_threshold))
                stats_partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output.flush()
            os.fsync(output.fileno())
        if identities != [file_identity(path) for item in inputs for path in item[1:]]:
            raise RuntimeError("Input files changed during filtering")
        report["totals"] = {"accepted": sum(row["counts"].get("accepted", 0) for row in report["languages"]),
                            "accepted_hours": sum(row["accepted_hours"] for row in report["languages"]),
                            "output_bytes": partial.stat().st_size}
        if report["totals"]["output_bytes"] != sum(row["output_bytes"] for row in report["languages"]):
            raise AssertionError("Output byte count mismatch")
        report["elapsed_seconds"] = time.monotonic() - started
        os.replace(partial, output_path)
        report["complete"] = True
        stats_partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(stats_partial, stats_path)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        stats_partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"output": str(output_path), "stats": str(stats_path), **report["totals"]}), flush=True)


if __name__ == "__main__":
    main()
