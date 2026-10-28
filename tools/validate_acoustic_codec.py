"""Check a TwinLakes VibeVoice codec against the Realtime checkpoint on CPU/GPU."""
import argparse
import gc
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from scipy.signal import resample_poly


def header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length)), length + 8


def checkpoint_index(root):
    result = {}
    for path in sorted(root.glob("*.safetensors")):
        entries, offset = header(path)
        for key, entry in entries.items():
            if key != "__metadata__":
                result[key] = (path, offset, entry)
    return result


def tensor_hash(entry):
    path, offset, item = entry
    start, end = item["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(offset + start)
        remaining = end - start
        while remaining:
            chunk = f.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise EOFError(path)
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def get_tensor(index, key):
    with safe_open(str(index[key][0]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def write_report(out, report):
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg"))
    parser.add_argument("--candidate", type=Path, default=Path("/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B"))
    parser.add_argument("--realtime", type=Path, default=Path("pretrain_model"))
    parser.add_argument("--output", type=Path, default=Path("outputs/codec_validation"))
    parser.add_argument("--per-language", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.candidate / "config.json").read_text())
    new = json.loads((args.realtime / "config.json").read_text())
    a, b = old["acoustic_tokenizer_config"], new["acoustic_tokenizer_config"]
    left, right = checkpoint_index(args.candidate), checkpoint_index(args.realtime)
    prefix = "model.acoustic_tokenizer."
    decoder_keys = sorted(k for k in right if k.startswith(prefix + "decoder."))
    mismatches, equal_keys, shape_errors = [], [], []
    for key in decoder_keys:
        if key not in left or any(left[key][2][field] != right[key][2][field] for field in ("shape", "dtype")):
            shape_errors.append(key)
        elif tensor_hash(left[key]) == tensor_hash(right[key]):
            equal_keys.append(key)
        else:
            x, y = get_tensor(left, key).float(), get_tensor(right, key).float()
            mismatches.append({"key": key, "max_abs_diff": float((x-y).abs().max()),
                               "rms_diff": float((x-y).square().mean().sqrt())})
    report = {
        "source": str(args.source), "candidate": str(args.candidate.resolve()),
        "realtime": str(args.realtime.resolve()), "device": args.device,
        "config_differences": {k: [a.get(k), b.get(k)] for k in sorted(set(a)|set(b)) if a.get(k) != b.get(k)},
        "decoder_tensors": len(decoder_keys), "decoder_bitwise_equal_tensors": len(equal_keys),
        "decoder_shape_errors": shape_errors, "decoder_weight_differences": mismatches,
        "decoder_parameters": sum(math.prod(right[k][2]["shape"]) for k in decoder_keys),
        "candidate_encoder_tensors": sum(k.startswith(prefix + "encoder.") for k in left),
        "realtime_encoder_tensors": sum(k.startswith(prefix + "encoder.") for k in right),
        "normalization": "mono, 24kHz, -25 dBFS RMS with clipping prevention",
        "latent_mode": "encoder mean (no posterior sampling)",
        "samples": [],
    }
    for label, index in (("candidate", left), ("realtime", right)):
        report[label + "_latent_scaling"] = {
            k: float(get_tensor(index, "model." + k)) for k in ("speech_scaling_factor", "speech_bias_factor")}
    write_report(args.output, report)
    print("STRUCTURAL", json.dumps({k: v for k, v in report.items() if k != "decoder_weight_differences"}), flush=True)
    if shape_errors:
        raise RuntimeError("Decoder architecture mismatch; see report")

    sys.path.insert(0, str(args.source))
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel, VibeVoiceTokenizerStreamingCache
    config = VibeVoiceAcousticTokenizerConfig(**a)
    # This implementation calls .item() on constructor temporaries, so it
    # cannot be initialized on the meta device.
    codec = VibeVoiceAcousticTokenizerModel(config)
    state = {k[len(prefix):]: get_tensor(left, k).float() for k in left if k.startswith(prefix)}
    result = codec.load_state_dict(state, strict=True, assign=True)
    # fix_std is a nonpersistent buffer and is not provided by the checkpoint.
    codec.fix_std = torch.tensor(config.fix_std)
    codec = codec.to(args.device).eval()
    del state
    gc.collect()
    report["codec_load"] = str(result)

    samples = []
    for language in ("en", "zh"):
        seen = set()
        for line in (args.source / "data/seedtts_testset" / (language + ".list")).read_text().splitlines():
            item = json.loads(line)
            path = item["prompt_wav"]
            if path in seen or not Path(path).exists():
                continue
            seen.add(path)
            samples.append((language, path))
            if len(seen) >= args.per_language:
                break

    scale = report["realtime_latent_scaling"]["speech_scaling_factor"]
    bias = report["realtime_latent_scaling"]["speech_bias_factor"]
    latent_records = []
    for language, path in samples:
        start = time.monotonic()
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 24000:
            divisor = math.gcd(sr, 24000)
            audio = resample_poly(audio, 24000 // divisor, sr // divisor).astype(np.float32)
        audio = audio[:int(24000 * args.max_seconds)]
        audio *= 10 ** (-25/20) / (np.sqrt(np.mean(audio**2)) + 1e-6)
        audio /= max(1., float(np.abs(audio).max()) + 1e-6)
        name = language + "_" + Path(path).stem
        sf.write(args.output / (name + "_input.wav"), audio, 24000, subtype="FLOAT")
        wav = torch.from_numpy(audio.copy())[None, None].to(args.device)
        with torch.inference_mode():
            latent = codec.encode(wav).mean
            restored = (latent + bias) * scale / scale - bias
            recon = codec.decode(restored.transpose(1, 2).contiguous()).flatten().cpu().numpy()
        sf.write(args.output / (name + "_source_decoder.wav"), recon[:len(audio)], 24000, subtype="FLOAT")
        data = {"name": name, "language": language, "source_wav": path,
                "seconds": len(audio)/24000, "latent_shape": list(latent.shape),
                "latent_mean": float(latent.mean()), "latent_std": float(latent.std()),
                "scaling_roundtrip_max_abs": float((latent-restored).abs().max()),
                "decoded_samples": len(recon), "finite": bool(np.isfinite(recon).all()),
                "source_decoder_snr_db": float(10*np.log10((np.mean(audio**2)+1e-12)/(np.mean((audio-recon[:len(audio)])**2)+1e-12))),
                "source_seconds_elapsed": time.monotonic()-start}
        latent_records.append((latent.detach().cpu(), audio, name))
        report["samples"].append(data)
        write_report(args.output, report)
        print("SOURCE_RECON", json.dumps(data), flush=True)

    decoder_state = {k[len(prefix + "decoder."):]: get_tensor(right, k).float() for k in decoder_keys}
    codec.decoder.load_state_dict(decoder_state, strict=True, assign=True)
    codec.decoder.to(args.device).eval()
    del decoder_state
    for row, (latent, audio, name) in zip(report["samples"], latent_records):
        start = time.monotonic()
        latent = latent.to(args.device)
        with torch.inference_mode():
            recon = codec.decode(latent.transpose(1, 2).contiguous()).flatten().cpu().numpy()
            # Exercise the incremental decoder with cache and six latent frames per chunk.
            cache = VibeVoiceTokenizerStreamingCache()
            chunks = [codec.decode(latent[:, i:i+6].transpose(1, 2).contiguous(), cache=cache,
                                   sample_indices=torch.tensor([0], device=args.device), use_cache=True)
                      for i in range(0, latent.shape[1], 6)]
            streamed = torch.cat(chunks, dim=-1).flatten().cpu().numpy()
        sf.write(args.output / (name + "_realtime_decoder.wav"), recon[:len(audio)], 24000, subtype="FLOAT")
        sf.write(args.output / (name + "_realtime_stream.wav"), streamed[:len(audio)], 24000, subtype="FLOAT")
        source, _ = sf.read(args.output / (name + "_source_decoder.wav"), dtype="float32")
        row.update({"realtime_decoder_snr_db": float(10*np.log10((np.mean(audio**2)+1e-12)/(np.mean((audio-recon[:len(audio)])**2)+1e-12))),
                    "source_vs_realtime_decoder_max_abs": float(np.max(np.abs(source-recon[:len(audio)]))),
                    "stream_vs_full_max_abs": float(np.max(np.abs(streamed-recon))),
                    "stream_vs_full_rms_diff": float(np.sqrt(np.mean((streamed-recon)**2))),
                    "realtime_seconds_elapsed": time.monotonic()-start,
                    "realtime_finite": bool(np.isfinite(recon).all()), "stream_finite": bool(np.isfinite(streamed).all())})
        write_report(args.output, report)
        print("REALTIME_RECON", json.dumps(row), flush=True)
    print("COMPLETE", str(args.output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
