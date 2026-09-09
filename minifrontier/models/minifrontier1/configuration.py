"""Versioned MiniFrontier1 native fusion configuration (independent of its sources)."""

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

from minifrontier.strict_types import validate_dataclass_payload


@dataclass
class MF1VisionConfig:
    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    output_size: int = 512
    gradient_checkpointing: bool = True


@dataclass
class MiniFrontier1Config:
    model_version: str = "1.0-reference-v1"
    vocab_size: int = 32768
    hidden_size: int = 512
    num_hidden_layers: int = 16
    attention_schedule: tuple[str, ...] = (
        "kda",
        "kda",
        "kda",
        "csa4",
        "kda",
        "kda",
        "kda",
        "qsa_mla",
        "kda",
        "kda",
        "kda",
        "csa4",
        "kda",
        "kda",
        "kda",
        "qsa_mla",
    )
    hc_count: int = 4
    hc_lowrank: int = 64
    num_attention_heads: int = 8
    kda_head_dim: int = 64
    conv_size: int = 4
    kda_backend: str = "auto"
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    mrope_sections: tuple[int, ...] = (4, 6, 6)
    rope_theta: float = 10000.0
    csa_head_dim: int = 64
    csa_rope_dim: int = 16
    index_heads: int = 4
    index_dim: int = 32
    index_rope_dim: int = 16
    block_size: int = 4
    top_blocks: int = 64
    window_size: int = 128
    protected_media_tokens: int = 1024
    protect_media: bool = True
    query_chunk_size: int = 64
    index_query_count: int = 64
    indexer_loss_coef: float = 0.01
    num_experts: int = 32
    num_experts_per_token: int = 4
    routed_expert_hidden_size: int = 256
    moe_intermediate_size: int = 256
    shared_intermediate_size: int = 768
    activation: str = "situ"
    routed_scale: float = 1.0
    lookup_layer: int = 2  # Public, one-based; converted only in the decoder constructor.
    lookup_enabled: bool = True
    lookup_rows: int = 32768
    lookup_dim: int = 64
    lookup_seed: int = 1729
    mtp_enabled: bool = True
    mtp_loss_coef: float = 0.1
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    image_token_id: int = 7
    control_token_count: int = 24
    forbidden_action_ids: tuple[int, ...] = (0, 1, 7)
    vision_config: MF1VisionConfig = field(default_factory=MF1VisionConfig)

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = MF1VisionConfig(**self.vision_config)
        for name in ("attention_schedule", "mrope_sections", "forbidden_action_ids"):
            setattr(self, name, tuple(getattr(self, name)))
        validate_dataclass_payload(type(self), asdict(self))
        if len(self.attention_schedule) != self.num_hidden_layers or not set(
            self.attention_schedule
        ) <= {"kda", "csa4", "qsa_mla"}:
            raise ValueError("attention schedule must match decoder depth and supported paths")
        if self.hc_count != 4 or not 0 < self.num_experts_per_token < self.num_experts:
            raise ValueError("MF1 requires four streams and 0 < top-k < experts")
        if len(self.mrope_sections) != 3 or 2 * sum(self.mrope_sections) != self.qk_rope_head_dim:
            raise ValueError("mRoPE sections count frequency pairs, not real dimensions")
        if self.block_size != 4 or not 1 <= self.lookup_layer <= self.num_hidden_layers:
            raise ValueError("MF1 uses four-token blocks and a one-based lookup layer")
        if (
            self.csa_rope_dim % 2
            or self.csa_rope_dim > self.csa_head_dim
            or self.index_rope_dim % 2
            or self.index_rope_dim > self.index_dim
        ):
            raise ValueError("invalid core/index rotary dimensions")
        if not 0 < self.protected_media_tokens < self.max_position_embeddings:
            raise ValueError("media budget must leave context for text and answers")
        if (
            self.vision_config.output_size != self.hidden_size
            or self.vision_config.temporal_patch_size != 2
            or self.vision_config.spatial_merge_size != 2
        ):
            raise ValueError("vision merger width or temporal/spatial patch contract differs")
        if self.vision_config.hidden_size % (4 * self.vision_config.num_heads):
            raise ValueError("vision head width must support axial RoPE")
        if self.kda_backend not in {"auto", "reference"} or self.activation not in {
            "situ",
            "swiglu",
        }:
            raise ValueError("unsupported numerical backend or expert activation")
        for name, value in asdict(self).items():
            if type(value) is int and name not in {"pad_token_id", "lookup_seed"} and value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            len({self.pad_token_id, self.eos_token_id, self.image_token_id}) != 3
            or not 0 <= self.pad_token_id < self.control_token_count < self.vocab_size
        ):
            raise ValueError("invalid reserved vocabulary")

    def upstream_config(self):
        return SimpleNamespace(**asdict(self))

    @classmethod
    def tiny(cls, vocab_size=320):
        return cls(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=4,
            attention_schedule=("kda", "csa4", "kda", "qsa_mla"),
            hc_lowrank=8,
            num_attention_heads=2,
            kda_head_dim=8,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            mrope_sections=(2, 1, 1),
            v_head_dim=8,
            csa_head_dim=16,
            csa_rope_dim=8,
            index_heads=2,
            index_dim=8,
            index_rope_dim=4,
            num_experts=4,
            num_experts_per_token=2,
            routed_expert_hidden_size=16,
            moe_intermediate_size=16,
            shared_intermediate_size=48,
            lookup_rows=64,
            lookup_dim=8,
            window_size=8,
            top_blocks=2,
            protected_media_tokens=64,
            max_position_embeddings=512,
            index_query_count=8,
            gradient_checkpointing=False,
            vision_config=MF1VisionConfig(
                depth=1,
                hidden_size=24,
                num_heads=2,
                intermediate_size=48,
                patch_size=4,
                output_size=32,
                gradient_checkpointing=False,
            ),
        )
