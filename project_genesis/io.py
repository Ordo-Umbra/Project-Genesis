import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EngineConfig
from .memory_corpus import MemoryCorpus


def save_snapshot(
    path: str | Path,
    field: np.ndarray,
    config: EngineConfig,
    history: list[dict[str, float | int | str]],
    agents: list[dict],
    run_metadata: dict,
    *,
    memory_corpus: MemoryCorpus | None = None,
    stability_map: np.ndarray | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "field": field,
        "config_json": json.dumps(config.to_dict()),
        "history_json": json.dumps(history),
        "agents_json": json.dumps(agents),
        "run_metadata_json": json.dumps(run_metadata),
    }

    if memory_corpus is not None:
        arrays["corpus_json"] = json.dumps(memory_corpus.to_dict())
    if stability_map is not None:
        arrays["stability_map"] = stability_map

    np.savez_compressed(target, **arrays)
    return target


def load_snapshot(
    path: str | Path,
) -> tuple[
    np.ndarray,
    EngineConfig,
    list[dict[str, float | int | str]],
    list[dict],
    dict,
    MemoryCorpus | None,
    np.ndarray | None,
]:
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

    corpus: MemoryCorpus | None = None
    if "corpus_json" in snapshot.files:
        corpus = MemoryCorpus.from_dict(json.loads(str(snapshot["corpus_json"])))

    stability_map: np.ndarray | None = None
    if "stability_map" in snapshot.files:
        stability_map = snapshot["stability_map"]

    return field, config, history, agents, run_metadata, corpus, stability_map
