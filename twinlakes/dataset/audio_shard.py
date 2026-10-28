"""Raw paired-audio TAR protocol; codec encoding is still required for training."""
import json
from pathlib import Path
import re

AUDIO_SHARD_FORMAT = "emilia_audio_pair_v1"
AUDIO_FORMATS = {"flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"}
EMILIA_ID = re.compile(r"(EN|ZH)_B[0-9]+_S[0-9]+_W[0-9]+")


def resolve_audio_path(value, language, roots):
    if not isinstance(value, str) or not value:
        raise ValueError("Audio path/ID must be a nonempty string")
    path = Path(value)
    if EMILIA_ID.fullmatch(value):
        if value.split("_")[0].lower() != language:
            raise ValueError(f"Prompt ID language differs from target: {value}")
        speaker = value.rsplit("_W", 1)[0]
        source = speaker.split("_S", 1)[0]
        path = Path(source) / speaker / "mp3" / (value + ".mp3")
    if not path.is_absolute():
        path = Path(roots[language]) / path
    if path.suffix[1:].lower() not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio extension: {path}")
    return path


def decode_audio_pair(sample):
    """Map an existing TwinLakes TAR DataPipe sample to metadata + audio bytes.

    `.json`, the target's real extension, and `.prompt` all have the same key.
    The opaque `.prompt` extension keeps the original rfind('.') grouping valid.
    Bytes are deliberately not decoded, normalized, or treated as codec latents.
    """
    row = json.loads(sample["json"])
    if row.get("shard_format") != AUDIO_SHARD_FORMAT or row["id"] != sample["key"]:
        raise ValueError("Audio shard format or sample key mismatch")
    formats, members = row["audio_formats"], row["audio_members"]
    if any(formats[name] not in AUDIO_FORMATS for name in ["target", "prompt"]):
        raise ValueError("Unsupported shard audio format")
    if members != {"target": f"{row['id']}.{formats['target']}", "prompt": f"{row['id']}.prompt"}:
        raise ValueError("Audio member names differ from metadata")
    target, prompt = sample["wav"], sample["prompt"]
    if not isinstance(target, bytes) or not isinstance(prompt, bytes) or not target or not prompt:
        raise ValueError("Missing or empty audio payload")
    return {**row, "target_audio_bytes": target, "prompt_audio_bytes": prompt}
