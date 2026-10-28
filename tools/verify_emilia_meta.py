"""Verify complete filtered-file hashes/counts and sampled joins against sources."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def samples_from_block(stream, start, size, windows=64, rows_per_window=8):
    result = {}
    for window in range(windows):
        offset = start + size * window // windows
        end = start + size * (window + 1) // windows
        stream.seek(offset)
        if offset > start:
            stream.readline()
        for _ in range(rows_per_window):
            if stream.tell() >= end:
                break
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            result[row["id"]] = row
    return result


def verify_language(language, rows, paths, sim_min, wer_max):
    metadata_path, sim_path, wer_path = paths
    targets = {identifier.encode(): row for identifier, row in rows.items()}
    prompts = {row["prompt_wav"].encode() for row in rows.values()}
    sim_found, wer_found = set(), set()
    hypotheses = {}
    with open(sim_path, "rb") as source:
        for line in source:
            identifier = line.partition(b"\t")[0]
            if identifier in targets:
                fields = line.decode("utf-8").rstrip("\r\n").split("\t")
                score = float(fields[2])
                if not math.isfinite(score) or not sim_min < score <= 1 or fields[1] != targets[identifier]["prompt_wav"]:
                    raise AssertionError(f"SIM/prompt mismatch: {identifier!r}")
                if identifier in sim_found:
                    raise AssertionError("Duplicate sampled SIM ID")
                sim_found.add(identifier)
    print(f"{language}: sampled target SIM and prompt IDs checked", flush=True)
    with open(wer_path, "rb") as source:
        for line in source:
            identifier = line.partition(b"\t")[0]
            if identifier in targets or (language == "en" and identifier in prompts):
                fields = line.decode("utf-8").rstrip("\r\n").split("\t")
                if identifier in targets:
                    if not 0 <= float(fields[1]) < wer_max:
                        raise AssertionError(f"Target WER mismatch: {identifier!r}")
                    if identifier in wer_found:
                        raise AssertionError("Duplicate sampled WER ID")
                    wer_found.add(identifier)
                    if language == "en" and targets[identifier]["text"] != fields[3]:
                        raise AssertionError(f"English target transcript mismatch: {identifier!r}")
                if language == "en" and identifier in prompts:
                    hypotheses[identifier] = fields[3]
    print(f"{language}: sampled target WER and ASR text checked", flush=True)
    originals = {}
    wanted = targets.keys() | prompts
    with open(metadata_path, "rb") as source:
        prefix = b'{"id": "'
        for line in source:
            if line.startswith(prefix):
                identifier = line[len(prefix):line.find(b'"', len(prefix))]
                row = None
            else:
                row = json.loads(line)
                identifier = row["id"].encode()
            if identifier in wanted:
                if identifier in originals:
                    raise AssertionError("Duplicate sampled metadata ID")
                originals[identifier] = row if row is not None else json.loads(line)
    if sim_found != set(targets) or wer_found != set(targets) or set(originals) != wanted:
        raise AssertionError(f"Missing sampled source IDs for {language}")
    for identifier, actual in targets.items():
        prompt_id = actual["prompt_wav"].encode()
        prompt_text = hypotheses[prompt_id] if language == "en" else originals[prompt_id]["text"]
        expected = {**originals[identifier], "prompt_wav": actual["prompt_wav"], "prompt_text": prompt_text}
        if language == "en":
            expected["text"] = actual["text"]  # Independently compared to WER column 4 above.
        if actual != expected:
            raise AssertionError(f"Output metadata/prompt text mismatch: {identifier!r}")
    print(f"{language}: original metadata and reference text checked ({len(rows)} output rows)", flush=True)
    return {"language": language, "sampled_output_rows": len(rows), "sampled_prompt_ids": len(prompts),
            "sim_and_prompt_id_match": True, "target_wer_pass": True,
            "target_and_prompt_text_match": True, "other_metadata_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/Emilia/train.jsonl"))
    args = parser.parse_args()
    started = time.monotonic()
    stats = json.loads(args.manifest.with_suffix(".stats.json").read_text())
    if not stats["complete"]:
        raise AssertionError("Filtering has not completed")
    output_path = args.manifest.with_suffix(".verify.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    for identity in stats["inputs"]:
        stat = Path(identity["path"]).stat()
        if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise AssertionError(f"Changed input: {identity['path']}")
    if args.manifest.stat().st_size != stats["totals"]["output_bytes"]:
        raise AssertionError("Wrong manifest size")
    blocks, samples, offset = [], [], 0
    with args.manifest.open("rb") as stream:
        for language in stats["languages"]:
            digest, lines, remaining = hashlib.sha256(), 0, language["output_bytes"]
            stream.seek(offset)
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError("Truncated manifest")
                digest.update(chunk)
                lines += chunk.count(b"\n")
                remaining -= len(chunk)
            if digest.hexdigest() != language["output_language_block_sha256"] or lines != language["counts"]["accepted"]:
                raise AssertionError("Manifest block hash/count mismatch")
            samples.append(samples_from_block(stream, offset, language["output_bytes"]))
            blocks.append({"language": language["language"], "rows": lines, "sha256_verified": True})
            offset += language["output_bytes"]
    print("Full manifest hashes and row counts checked", flush=True)
    joins = []
    for i, (language, rows) in enumerate(zip(stats["languages"], samples)):
        paths = [row["path"] for row in stats["inputs"][3 * i:3 * i + 3]]
        joins.append(verify_language(language["language"], rows, paths,
                                     float(stats["rules"]["sim"].split()[-1]),
                                     float(stats["rules"]["wer"].split()[-1])))
    report = {"passed": True, "manifest": str(args.manifest.resolve()), "full_file_blocks": blocks,
              "sample_method": "64 equally spaced byte windows per language block, 8 complete output rows/window",
              "source_join_checks": joins, "elapsed_seconds": time.monotonic() - started,
              "filter_code_sha256": hashlib.sha256(Path("data/Emilia/meta.py").read_bytes()).hexdigest(),
              "unchecked": ["audio waveform quality", "speaker identity", "speaker/source split leakage"]}
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Verified: {output_path}", flush=True)


if __name__ == "__main__":
    main()
