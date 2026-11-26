"""Isolated Pi settings with explicitly configured web access credentials."""
import json
from pathlib import Path


def prepare_agent_state(config: dict, directory: Path) -> Path:
    state = directory / "pi-state"
    source = config.get("runtime", {}).get("web_config_path")
    if source is not None:
        payload = Path(source).read_bytes()
        if not isinstance(json.loads(payload), dict):
            raise ValueError("web_config_path must contain a JSON object")
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        state.chmod(0o700)
        target = state / "web-search.json"
        target.touch(mode=0o600, exist_ok=True)
        target.chmod(0o600)
        target.write_bytes(payload)
    return state
