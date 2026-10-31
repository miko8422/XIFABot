"""Configuration: TOML file for knobs, the repo-root .env for secrets."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]  # packages/maid_voice -> repo root
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"


@dataclass
class VoiceProfile:
    """Either a clone (ref_audio + ref_text) or a fine-tuned speaker (speaker, usually with its own model)."""
    name: str
    ref_audio: Path | None = None
    ref_text: str = ""
    language: str = "japanese"
    speaker: str = ""   # CustomVoice speaker name; no reference audio is sent
    model: str = ""     # overrides [tts].model for this voice


@dataclass
class Config:
    raw: dict
    voices: dict[str, VoiceProfile] = field(default_factory=dict)

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]

    @property
    def voice(self) -> VoiceProfile:
        return self.voices[self.raw["tts"]["voice_profile"]]

    def secret(self, env_name: str) -> str:
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"missing environment variable {env_name} (expected in .env)")
        return value


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    # .env lives at the repo root; a package-local .env is still honored
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    voices = {}
    for name, v in raw.get("voices", {}).items():
        # Clone profiles may refer to private reference files that are not
        # present in a clean checkout. They should not prevent an installed
        # speaker profile (for example the fine-tuned default) from starting.
        ref_audio = PROJECT_ROOT / v["ref_audio"] if "ref_audio" in v else None
        ref_text_path = PROJECT_ROOT / v["ref_text_file"] if "ref_text_file" in v else None
        if ref_audio is not None and ref_text_path is not None and not (ref_audio.exists() and ref_text_path.exists()):
            continue
        voices[name] = VoiceProfile(
            name=name,
            ref_audio=ref_audio,
            ref_text=ref_text_path.read_text(encoding="utf-8").strip() if ref_text_path is not None else "",
            language=v.get("language", "japanese"),
            speaker=v.get("speaker", ""),
            model=v.get("model", ""),
        )
    return Config(raw=raw, voices=voices)
