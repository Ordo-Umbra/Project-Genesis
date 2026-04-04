from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EngineConfig:
    chunk_size: int = 32
    beta: float = 0.09
    gravity: float = 0.22
    void_threshold: float = 0.15
    air_threshold: float = 0.3
    soil_threshold: float = 0.6
    bedrock_threshold: float = 0.8
    seed: int | None = None
    default_steps: int = 50
    default_dt: float = 0.01

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float | int | None]) -> "EngineConfig":
        return cls(**data)
