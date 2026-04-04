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


def export_artifacts(output_dir: Path, engine: GenesisEngine) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = output_dir / "slices"

    final_metrics = engine.summarize_state()
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
    write_voxel_slice(output_dir / "final_slice_z.txt", engine.quantize_to_voxels(), axis="z")

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
    else:
        engine = GenesisEngine(
            config=EngineConfig(
                chunk_size=args.chunk_size,
                beta=args.beta,
                gravity=args.gravity,
                seed=args.seed,
                default_steps=args.steps,
                default_dt=args.dt,
            )
        )

    engine.evolve_field(steps=args.steps, dt=args.dt, record_every=args.record_every)
    export_artifacts(output_dir, engine)

    print(json.dumps(engine.summarize_state(), indent=2, sort_keys=True))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
