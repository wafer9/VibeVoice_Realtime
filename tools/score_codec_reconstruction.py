"""Score saved codec reconstructions with the local Seed-TTS WavLM speaker verifier."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


def load_verifier(eval_root, checkpoint):
    sys.path.insert(0, str(eval_root / "s3prl"))
    from s3prl.upstream.wavlm.expert import UpstreamExpert

    path = eval_root / "thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py"
    spec = importlib.util.spec_from_file_location("codec_validation_ecapa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.UpstreamExpert = UpstreamExpert
    # Use an explicit local checkpoint rather than torch.hub's downloading path.
    original_load = torch.load

    def local_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    try:
        torch.load = local_load
        model = module.ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large",
                                       config_path=str(eval_root / "wavlm_large.pt"))
    finally:
        torch.load = original_load
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    # The training-only speaker classification projection is not part of
    # inference. Require every embedding-model parameter to match strictly.
    weights = dict(state["model"])
    weights.pop("loss_calculator.projection.weight", None)
    result = model.load_state_dict(weights, strict=True)
    print("SPEAKER_MODEL", str(result), flush=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/codec_validation"))
    parser.add_argument("--eval-root", type=Path, default=Path("/nfs-speech-cfs/wangzhou/s2s/data/seed-tts-eval"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/nfs-speech-cfs/wangzhou/.cache/models/wavlm_large_finetune.pth"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing SIM values when only adding a new reconstruction variant")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report_path = args.directory / "report.json"
    report = json.loads(report_path.read_text())
    if not report["samples"] or not all("stream_finite" in row for row in report["samples"]):
        raise RuntimeError("Reconstruction must finish before scoring")
    model = load_verifier(args.eval_root, args.checkpoint)
    report["speaker_verifier"] = {"architecture": "WavLM-Large + ECAPA-TDNN", "checkpoint": str(args.checkpoint),
                                  "load_state_dict": "strict=True; all inference keys matched; omitted training-only loss_calculator.projection.weight", "sampling_rate": 16000,
                                  "metric": "cosine similarity, original normalized audio vs reconstruction"}
    kinds = ["source_decoder", "realtime_decoder", "realtime_stream"]
    if all((args.directory / (r["name"] + "_affine_decoder.wav")).exists() for r in report["samples"]):
        kinds.append("affine_decoder")
    originals = {}
    with torch.inference_mode():
        for row in report["samples"]:
            embeddings = {}
            for kind in ["input"] + kinds:
                if args.reuse_existing and kind != "input" and "sim_" + kind in row:
                    continue
                audio, sr = sf.read(args.directory / (row["name"] + "_" + kind + ".wav"), dtype="float32")
                divisor = math.gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(np.float32)
                embeddings[kind] = F.normalize(model(torch.from_numpy(audio.copy())[None]), dim=-1)
            originals[row["name"]] = embeddings["input"]
            for kind in kinds:
                if kind in embeddings:
                    row["sim_" + kind] = float((embeddings["input"] * embeddings[kind]).sum())
            print("SIM", row["name"], {k: v for k, v in row.items() if k.startswith("sim_")}, flush=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = {}
    for language in ("en", "zh", "all"):
        rows = [r for r in report["samples"] if language == "all" or r["language"] == language]
        summary[language] = {"count": len(rows)}
        for kind in kinds:
            values = [r["sim_" + kind] for r in rows]
            summary[language]["sim_" + kind] = {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
        # Different prompt recordings are a sanity control, not a labeled SV benchmark.
        pairs = [(originals[x["name"]], originals[y["name"]]) for i, x in enumerate(rows) for y in rows[i+1:]]
        summary[language]["different_recording_control_mean"] = float(np.mean([float((x*y).sum()) for x, y in pairs])) if pairs else None
    report["summary"] = summary
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
