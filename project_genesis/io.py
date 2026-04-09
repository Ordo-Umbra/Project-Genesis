import json
from pathlib import Path

import numpy as np

from .config import EngineConfig


def save_snapshot(
    path: str | Path,
    field: np.ndarray,
    config: EngineConfig,
    history: list[dict[str, float | int | str]],
    agents: list[dict],
    run_metadata: dict,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        field=field,
        config_json=json.dumps(config.to_dict()),
        history_json=json.dumps(history),
        agents_json=json.dumps(agents),
        run_metadata_json=json.dumps(run_metadata),
    )
    return target


def load_snapshot(
    path: str | Path,
) -> tuple[np.ndarray, EngineConfig, list[dict[str, float | int | str]], list[dict], dict]:
    snapshot = np.load(Path(path), allow_pickle=False)
    field = snapshot["field"]
    config = EngineConfig.from_dict(json.loads(str(snapshot["config_json"])))
    history = json.loads(str(snapshot["history_json"]))
    agents = json.loads(str(snapshot["agents_json"])) if "agents_json" in snapshot.files else []
    run_metadata = (
        json.loads(str(snapshot["run_metadata_json"]))
        if "run_metadata_json" in snapshot.files
        else {}
    )
    return field, config, history, agents, run_metadata
