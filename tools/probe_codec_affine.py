"""Test whether preserving old normalized latents resolves cross-codec mismatch."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from validate_acoustic_codec import checkpoint_index, get_tensor


def main():
    torch.set_num_threads(8)
    out = Path("outputs/codec_validation")
    report = json.loads((out / "report.json").read_text())
    sys.path.insert(0, report["source"])
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel

    root = Path(report["candidate"])
    config = json.loads((root / "config.json").read_text())["acoustic_tokenizer_config"]
    codec = VibeVoiceAcousticTokenizerModel(VibeVoiceAcousticTokenizerConfig(**config)).eval()
    source = checkpoint_index(root)
    prefix = "model.acoustic_tokenizer."
    state = {k[len(prefix):]: get_tensor(source, k).float() for k in source if k.startswith(prefix)}
    codec.load_state_dict(state, strict=True, assign=True)
    del state
    latents = []
    with torch.inference_mode():
        for row in report["samples"]:
            audio, _ = sf.read(out / (row["name"] + "_input.wav"), dtype="float32")
            latent = codec.encode(torch.from_numpy(audio.copy())[None, None]).mean
            latents.append((latent, len(audio)))
    realtime = checkpoint_index(Path(report["realtime"]))
    state = {k[len(prefix + "decoder."):]: get_tensor(realtime, k).float()
             for k in realtime if k.startswith(prefix + "decoder.")}
    codec.decoder.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    old = report["candidate_latent_scaling"]
    new = report["realtime_latent_scaling"]
    report["affine_probe"] = {
        "formula": "raw_realtime = (raw_source + bias_source) * scale_source / scale_realtime - bias_realtime",
        "purpose": "preserve normalized diffusion latent, rather than raw codec latent"}
    with torch.inference_mode():
        for row, (latent, length) in zip(report["samples"], latents):
            mapped = (latent + old["speech_bias_factor"]) * old["speech_scaling_factor"] / new["speech_scaling_factor"] - new["speech_bias_factor"]
            recon = codec.decode(mapped.transpose(1, 2).contiguous()).flatten().numpy()[:length]
            sf.write(out / (row["name"] + "_affine_decoder.wav"), recon, 24000, subtype="FLOAT")
            print("AFFINE_DECODE", row["name"], bool(np.isfinite(recon).all()), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
