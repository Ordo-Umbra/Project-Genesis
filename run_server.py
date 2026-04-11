#!/usr/bin/env python3
"""Headless entry point for running the Project Genesis simulation server.

Usage::

    python run_server.py --world-size 64 --save-interval 100 --port 8765

The script:

1. Initialises (or loads) a :class:`GenesisEngine`.
2. Optionally starts a WebSocket server for remote monitoring / control.
3. Enters a ``while True`` loop calling ``engine.step()`` until terminated.
4. Traps **SIGINT** / **SIGTERM** to auto-save before exiting.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Ensure the repo root is on ``sys.path`` so ``project_genesis`` is importable
# when the script is invoked directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from project_genesis import EngineConfig, GenesisEngine  # noqa: E402
from project_genesis.network_server import NetworkServer  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Project Genesis as a headless simulation server.",
    )
    p.add_argument("--world-size", type=int, default=32, help="Cubic world edge length.")
    p.add_argument("--dt", type=float, default=0.01, help="Timestep per evolution step.")
    p.add_argument("--beta", type=float, default=0.09, help="Complexity coupling β.")
    p.add_argument("--gravity", type=float, default=0.22, help="Gravity / damping G.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for determinism.")
    p.add_argument("--agent-count", type=int, default=0, help="Number of initial agents.")
    p.add_argument(
        "--agent-goal",
        choices=("density", "explore", "s_functional"),
        default="density",
    )
    p.add_argument("--save-interval", type=int, default=100, help="Auto-save every N steps.")
    p.add_argument(
        "--save-dir",
        default="server_snapshots",
        help="Directory for periodic snapshots.",
    )
    p.add_argument("--resume", default=None, help="Path to a .npz snapshot to resume from.")
    p.add_argument("--config", default=None, help="Path to a JSON configuration file.")
    p.add_argument("--port", type=int, default=8765, help="WebSocket server port (0 to disable).")
    p.add_argument("--host", default="0.0.0.0", help="WebSocket server bind address.")
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many steps (0 = run forever).",
    )
    return p


def _save_state(engine: GenesisEngine, save_dir: Path, label: str = "auto") -> Path:
    """Persist the current engine state with a timestamped filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"snapshot_{label}_{ts}.npz"
    engine.save(path)
    return path


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ----- Load or build config from optional JSON file ----- #
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            file_cfg = json.load(fh)
        config = EngineConfig.from_dict(file_cfg)
    else:
        config = EngineConfig(
            chunk_size=args.world_size,
            beta=args.beta,
            gravity=args.gravity,
            seed=args.seed,
            agent_count=args.agent_count,
            agent_goal=args.agent_goal,
            default_dt=args.dt,
        )

    # ----- Initialize engine ----- #
    if args.resume:
        print(f"Resuming from {args.resume}")
        engine = GenesisEngine.load(args.resume)
    else:
        engine = GenesisEngine(config=config)

    save_dir = Path(args.save_dir)

    # ----- Start WebSocket server ----- #
    ws_server: NetworkServer | None = None
    if args.port > 0:
        ws_server = NetworkServer(engine, host=args.host, port=args.port)
        ws_server.start()
        print(f"WebSocket server listening on ws://{args.host}:{args.port}")

    # ----- Graceful shutdown handler ----- #
    shutdown_requested = False

    def _on_signal(signum: int, _frame: object) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] Saving state before exit…")
        shutdown_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ----- Main simulation loop ----- #
    step = 0
    dt = args.dt
    save_interval = max(1, args.save_interval)
    max_steps = args.max_steps

    print(f"Starting headless simulation (world={engine.chunk_size}³, dt={dt}, save every {save_interval} steps)")

    try:
        while not shutdown_requested:
            engine.step(dt)
            step += 1

            if step % save_interval == 0:
                path = _save_state(engine, save_dir)
                summary = engine.get_world_summary()
                print(
                    f"[step {step}] saved → {path}  "
                    f"S={summary['s_functional'].get('s_increment', 0):.6f}"
                )

            if 0 < max_steps <= step:
                print(f"Reached max-steps ({max_steps}). Stopping.")
                break
    finally:
        path = _save_state(engine, save_dir, label="final")
        print(f"Final state saved → {path}")
        if ws_server is not None:
            ws_server.stop()


if __name__ == "__main__":
    main()
