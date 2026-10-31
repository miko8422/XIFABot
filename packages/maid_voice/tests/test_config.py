from maid_voice.config import load_config


def test_missing_private_clone_profile_does_not_block_installed_speaker(tmp_path):
    config = tmp_path / "demo.toml"
    config.write_text(
        """
[voices.natsume_ft]
speaker = "natsume"

[voices.private_clone]
ref_audio = "voices/private/ref.wav"
ref_text_file = "voices/private/ref.txt"
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert "natsume_ft" in loaded.voices
    assert "private_clone" not in loaded.voices
