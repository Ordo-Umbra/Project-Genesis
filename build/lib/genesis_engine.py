import argparse
import json
from pathlib import Path

from project_genesis import EngineConfig, GenesisEngine
from project_genesis.render import write_voxel_slice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Project Genesis URP terrain sandbox.")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk edge length for the cubic field.")
    parser.add_argument("--steps", type=int, default=50, help="Number of field evolution steps to simulate.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time delta applied at each evolution step.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for primordial noise.")
    parser.add_argument("--beta", type=float, default=0.09, help="Complexity coupling used by the field update.")
    parser.add_argument("--gravity", type=float, default=0.22, help="Coherence/gravity damping term.")
    parser.add_argument("--agent-count", type=int, default=0, help="Number of agents to spawn for a fresh run.")
    parser.add_argument(
        "--agent-goal",
        choices=("density", "explore", "s_functional"),
        default="density",
        help="Default decision policy applied to spawned agents.",
    )
    parser.add_argument(
        "--agent-explore-probability",
        type=float,
        default=0.2,
        help="Per-step probability that an agent ignores its goal and explores randomly.",
    )
    parser.add_argument(
        "--agent-interaction-radius",
        type=int,
        default=2,
        help="Distance at which agents report nearby peers.",
    )
    parser.add_argument(
        "--agent-influence",
        type=float,
        default=0.0,
        help="Amount of local field reinforcement applied by agents after moving.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=5,
        help="Write metrics and voxel slice snapshots at this step interval.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/latest_run",
        help="Directory that receives metrics, slices, and the saved engine snapshot.",
    )
    parser.add_argument("--resume", default=None, help="Optional .npz snapshot path to resume from.")
    return parser


def build_run_summary(
    engine: GenesisEngine,
    *,
    output_dir: Path,
    record_every: int,
    requested_steps: int,
    resumed_from: str | None,
) -> dict[str, object]:
    final_metrics = engine.summarize_state()
    return {
        "config": engine.config.to_dict(),
        "history_length": len(engine.history),
        "final_step": engine.history[-1]["step"] if engine.history else 0,
        "record_every": record_every,
        "requested_steps": requested_steps,
        "resumed_from": resumed_from,
        "artifacts_dir": str(output_dir),
        "final_metrics": final_metrics,
        "run_metadata": engine.run_metadata,
    }


def export_artifacts(
    output_dir: Path,
    engine: GenesisEngine,
    *,
    record_every: int,
    requested_steps: int,
    resumed_from: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = output_dir / "slices"
    final_slices_dir = output_dir / "final_slices"

    final_metrics = engine.summarize_state()
    run_summary = build_run_summary(
        engine,
        output_dir=output_dir,
        record_every=record_every,
        requested_steps=requested_steps,
        resumed_from=resumed_from,
    )
    (output_dir / "config.json").write_text(
        json.dumps(engine.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_history.json").write_text(
        json.dumps(engine.history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "agent_timelines.json").write_text(
        json.dumps(engine.agent_timelines(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    voxels = engine.quantize_to_voxels()
    write_voxel_slice(output_dir / "final_slice_z.txt", voxels, axis="z")
    for axis in ("x", "y", "z"):
        write_voxel_slice(final_slices_dir / f"final_slice_{axis}.txt", voxels, axis=axis)

    for snapshot in engine.history:
        step = int(snapshot["step"])
        (slices_dir / f"step_{step:04d}_z.txt").parent.mkdir(parents=True, exist_ok=True)
        (slices_dir / f"step_{step:04d}_z.txt").write_text(str(snapshot["slice_z"]) + "\n", encoding="utf-8")

    engine.save(output_dir / "engine_snapshot.npz")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.resume:
        engine = GenesisEngine.load(args.resume)
        if args.agent_count > len(engine.agents):
            engine.populate_agents(args.agent_count - len(engine.agents), goal=args.agent_goal)
        engine.run_metadata["resumed_from"] = str(args.resume)
    else:
        engine = GenesisEngine(
            config=EngineConfig(
                chunk_size=args.chunk_size,
                beta=args.beta,
                gravity=args.gravity,
                agent_count=args.agent_count,
                agent_goal=args.agent_goal,
                agent_explore_probability=args.agent_explore_probability,
                agent_interaction_radius=args.agent_interaction_radius,
                agent_influence=args.agent_influence,
                seed=args.seed,
                default_steps=args.steps,
                default_dt=args.dt,
            )
        )

    engine.evolve_field(steps=args.steps, dt=args.dt, record_every=args.record_every)
    export_artifacts(
        output_dir,
        engine,
        record_every=args.record_every,
        requested_steps=args.steps,
        resumed_from=args.resume,
    )

    print(json.dumps(engine.summarize_state(), indent=2, sort_keys=True))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
