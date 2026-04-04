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

    def __post_init__(self) -> None:
        if not (self.void_threshold < self.air_threshold < self.soil_threshold < self.bedrock_threshold):
            raise ValueError(
                "Thresholds must be strictly ordered: "
                "void_threshold < air_threshold < soil_threshold < bedrock_threshold"
            )

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float | int | None]) -> "EngineConfig":
        return cls(**data)
