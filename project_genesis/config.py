from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EngineConfig:
    chunk_size: int = 32
    beta: float = 0.09
    gravity: float = 0.22
    agent_count: int = 0
    agent_goal: str = "density"
    agent_explore_probability: float = 0.2
    agent_interaction_radius: int = 2
    agent_influence: float = 0.0
    void_threshold: float = 0.15
    air_threshold: float = 0.3
    soil_threshold: float = 0.6
    bedrock_threshold: float = 0.8
    seed: int | None = None
    default_steps: int = 50
    default_dt: float = 0.01
    use_coherence_potential: bool = False
    poisson_iterations: int = 30
    use_integration_functional: bool = False
    integration_radius: int = 2
    integration_decay: float = 1.0
    integration_weight: float = 0.01
    enable_memory_corpus: bool = False
    corpus_max_size: int = 50
    corpus_min_stability: int = 5
    corpus_min_local_s: float = 0.01
    corpus_patch_scales: str = "4,8,16"
    corpus_compose_probability: float = 0.15

    def __post_init__(self) -> None:
        if not (self.void_threshold < self.air_threshold < self.soil_threshold < self.bedrock_threshold):
            raise ValueError(
                "Thresholds must be strictly ordered: "
                "void_threshold < air_threshold < soil_threshold < bedrock_threshold"
            )
        if self.agent_count < 0:
            raise ValueError("agent_count must be non-negative")
        if self.agent_goal not in {"density", "explore", "s_functional"}:
            raise ValueError("agent_goal must be one of: density, explore, s_functional")
        if not 0.0 <= self.agent_explore_probability <= 1.0:
            raise ValueError("agent_explore_probability must be between 0 and 1")
        if self.agent_interaction_radius < 0:
            raise ValueError("agent_interaction_radius must be non-negative")
        if self.agent_influence < 0.0:
            raise ValueError("agent_influence must be non-negative")
        if self.poisson_iterations <= 0:
            raise ValueError("poisson_iterations must be > 0")
        if self.integration_radius <= 0:
            raise ValueError("integration_radius must be > 0")
        if self.integration_decay <= 0.0:
            raise ValueError("integration_decay must be > 0.0")
        if self.integration_weight < 0.0:
            raise ValueError("integration_weight must be >= 0.0")
        if self.corpus_max_size <= 0:
            raise ValueError("corpus_max_size must be > 0")
        if self.corpus_min_stability < 0:
            raise ValueError("corpus_min_stability must be >= 0")
        if self.corpus_min_local_s < 0.0:
            raise ValueError("corpus_min_local_s must be >= 0.0")
        if not all(
            int(s) > 0 for s in self.corpus_patch_scales.split(",") if s.strip()
        ):
            raise ValueError("corpus_patch_scales must be comma-separated positive integers")
        if not 0.0 <= self.corpus_compose_probability <= 1.0:
            raise ValueError("corpus_compose_probability must be between 0 and 1")

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float | int | None]) -> "EngineConfig":
        return cls(**data)
