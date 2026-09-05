from pathlib import Path

from agent_harness.config import load_config


def test_environment_overrides_yaml(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "target:\n  base_url: https://yaml.example\n  chat_path: /yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESS_BASE_URL", "https://env.example")
    monkeypatch.setenv("HARNESS_CHAT_PATH", "/env")

    config = load_config(config_path)

    assert config.target.base_url == "https://env.example"
    assert config.target.chat_path == "/env"
