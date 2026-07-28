from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from embodied_vla.language import VOCAB
from embodied_vla.proprioception import uses_end_effector_position


@dataclass(frozen=True)
class TinyVLAConfig:
    image_size: int = 64
    patch_size: int = 8
    proprio_dim: int = 12
    action_dim: int = 5
    action_horizon: int = 8
    language_length: int = 16
    vocabulary_size: int = len(VOCAB)
    phase_count: int = 7
    model_dim: int = 128
    attention_heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    feedforward_dim: int = 384
    dropout: float = 0.1
    action_head: Literal["deterministic", "flow_matching"] = "deterministic"
    flow_matching_steps: int = 8
    grounding_action_conditioning: bool = False
    grounding_coordinate_refinement: bool = False
    high_resolution_grounding: bool = False
    world_grounding: bool = False
    world_grounding_action_conditioning: bool = False
    phase_action_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.flow_matching_steps <= 0:
            raise ValueError("flow_matching_steps must be positive")
        uses_end_effector_position(self.proprio_dim)
        if self.world_grounding_action_conditioning and not self.world_grounding:
            raise ValueError(
                "world_grounding_action_conditioning requires world_grounding"
            )

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def vision_token_count(self) -> int:
        return self.grid_size**2

    @property
    def high_resolution_grounding_grid_size(self) -> int:
        return self.image_size // 4


@dataclass
class TinyVLAOutput:
    action_chunk: Tensor
    phase_logits: Tensor
    grounding_coordinates: Tensor
    grounding_heatmaps: Tensor
    grounding_world_positions: Tensor | None = None
    flow_velocity: Tensor | None = None
    flow_target: Tensor | None = None


class TinyVLA(nn.Module):
    """Small multimodal Transformer with continuous action queries.

    This is intentionally not presented as a foundation model. It is a
    from-scratch VLA used to expose tokenization, fusion, grounding, action
    chunking, and closed-loop deployment on CPU-scale experiments.
    """

    def __init__(self, config: TinyVLAConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.model_dim
        self.patch_embedding = nn.Conv2d(
            3,
            dimension,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.vision_position = nn.Parameter(
            torch.zeros(1, config.vision_token_count, dimension)
        )
        self.language_embedding = nn.Embedding(
            config.vocabulary_size,
            dimension,
            padding_idx=VOCAB["<pad>"],
        )
        self.language_position = nn.Parameter(
            torch.zeros(1, config.language_length, dimension)
        )
        self.proprio_projection = nn.Sequential(
            nn.Linear(config.proprio_dim, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.task_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.modality_embedding = nn.Embedding(4, dimension)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(dimension),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(dimension),
        )
        self.action_queries = nn.Parameter(
            torch.zeros(1, config.action_horizon, dimension)
        )
        self.action_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, config.action_dim),
        )
        self.flow_action_projection = nn.Linear(config.action_dim, dimension)
        self.flow_time_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.flow_velocity_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, config.action_dim),
        )
        self.phase_head = nn.Linear(dimension, config.phase_count)

        self.grounding_queries = nn.Parameter(torch.zeros(1, 2, dimension))
        self.grounding_attention = nn.MultiheadAttention(
            dimension,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        patch_centers = self._make_patch_centers(config.grid_size)
        self.register_buffer("patch_centers", patch_centers, persistent=False)
        self.grounding_action_projection = (
            nn.Sequential(
                nn.Linear(config.proprio_dim + 4, dimension),
                nn.LayerNorm(dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            if config.grounding_action_conditioning
            else None
        )
        self.grounding_coordinate_refiner = (
            nn.Sequential(
                nn.Linear(dimension + 2, dimension),
                nn.GELU(),
                nn.Linear(dimension, 2),
            )
            if config.grounding_coordinate_refinement
            else None
        )
        high_resolution_hidden = max(32, dimension // 2)
        self.high_resolution_grounding_stem = (
            nn.Sequential(
                nn.Conv2d(3, high_resolution_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    high_resolution_hidden,
                    high_resolution_hidden,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    high_resolution_hidden,
                    dimension,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.GELU(),
            )
            if config.high_resolution_grounding
            else None
        )
        self.high_resolution_grounding_key = (
            nn.Conv2d(dimension, dimension, kernel_size=1)
            if config.high_resolution_grounding
            else None
        )
        self.high_resolution_grounding_query = (
            nn.Linear(dimension, dimension)
            if config.high_resolution_grounding
            else None
        )
        self.high_resolution_grounding_gate = (
            nn.Parameter(torch.zeros(()))
            if config.high_resolution_grounding
            else None
        )
        high_resolution_patch_centers = self._make_patch_centers(
            config.high_resolution_grounding_grid_size
        )
        self.register_buffer(
            "high_resolution_patch_centers",
            high_resolution_patch_centers,
            persistent=False,
        )
        self.world_grounding_head = (
            nn.Sequential(
                nn.Linear(dimension + 2, dimension),
                nn.GELU(),
                nn.Linear(dimension, 3),
            )
            if config.world_grounding
            else None
        )
        self.world_grounding_action_projection = (
            nn.Sequential(
                nn.Linear(config.proprio_dim + 6, dimension),
                nn.LayerNorm(dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            if config.world_grounding_action_conditioning
            else None
        )
        self.phase_action_projection = (
            nn.Sequential(
                nn.Linear(config.phase_count, dimension),
                nn.GELU(),
                nn.Linear(dimension, dimension),
            )
            if config.phase_action_conditioning
            else None
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.task_token, std=0.02)
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        nn.init.trunc_normal_(self.grounding_queries, std=0.02)
        nn.init.trunc_normal_(self.vision_position, std=0.02)
        nn.init.trunc_normal_(self.language_position, std=0.02)
        if self.grounding_action_projection is not None:
            nn.init.zeros_(self.grounding_action_projection[-1].weight)
            nn.init.zeros_(self.grounding_action_projection[-1].bias)
        if self.grounding_coordinate_refiner is not None:
            nn.init.zeros_(self.grounding_coordinate_refiner[-1].weight)
            nn.init.zeros_(self.grounding_coordinate_refiner[-1].bias)
        if self.world_grounding_action_projection is not None:
            nn.init.zeros_(self.world_grounding_action_projection[-1].weight)
            nn.init.zeros_(self.world_grounding_action_projection[-1].bias)
        if self.phase_action_projection is not None:
            nn.init.zeros_(self.phase_action_projection[-1].weight)
            nn.init.zeros_(self.phase_action_projection[-1].bias)

    def forward(
        self,
        rgb: Tensor,
        proprio: Tensor,
        language: Tensor,
        language_mask: Tensor,
        action_targets: Tensor | None = None,
        flow_noise: Tensor | None = None,
        flow_time: Tensor | None = None,
    ) -> TinyVLAOutput:
        batch_size = rgb.shape[0]
        if rgb.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(
                f"expected {self.config.image_size}x{self.config.image_size} RGB, "
                f"got {tuple(rgb.shape[-2:])}"
            )
        vision = self.patch_embedding(rgb).flatten(2).transpose(1, 2)
        vision = vision + self.vision_position
        language_tokens = self.language_embedding(language) + self.language_position
        proprio_token = self.proprio_projection(proprio).unsqueeze(1)
        task_token = self.task_token.expand(batch_size, -1, -1)

        task_token = task_token + self.modality_embedding.weight[0]
        proprio_token = proprio_token + self.modality_embedding.weight[1]
        language_tokens = language_tokens + self.modality_embedding.weight[2]
        vision = vision + self.modality_embedding.weight[3]
        tokens = torch.cat((task_token, proprio_token, language_tokens, vision), dim=1)

        fixed_tokens = torch.zeros(
            (batch_size, 2),
            dtype=torch.bool,
            device=rgb.device,
        )
        vision_mask = torch.zeros(
            (batch_size, self.config.vision_token_count),
            dtype=torch.bool,
            device=rgb.device,
        )
        padding_mask = torch.cat(
            (fixed_tokens, ~language_mask.bool(), vision_mask),
            dim=1,
        )
        memory = self.encoder(tokens, src_key_padding_mask=padding_mask)
        task_features = memory[:, 0]
        vision_start = 2 + self.config.language_length
        vision_features = memory[:, vision_start:]

        grounding_queries = self.grounding_queries.expand(batch_size, -1, -1)
        grounding_queries = grounding_queries + task_features.unsqueeze(1)
        grounded_features, attention_weights = self.grounding_attention(
            grounding_queries,
            vision_features,
            vision_features,
            need_weights=True,
            average_attn_weights=True,
        )
        attention_weights = attention_weights.clamp_min(0.0)
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(attention_weights.dtype).eps)
        coarse_grounding_coordinates = attention_weights @ self.patch_centers.to(
            attention_weights.dtype
        )
        grounding_coordinates = coarse_grounding_coordinates
        if self.grounding_coordinate_refiner is not None:
            refinement_input = torch.cat(
                (grounded_features, coarse_grounding_coordinates),
                dim=-1,
            )
            coordinate_delta = torch.tanh(
                self.grounding_coordinate_refiner(refinement_input)
            ) / self.config.grid_size
            grounding_coordinates = torch.clamp(
                coarse_grounding_coordinates + coordinate_delta,
                0.0,
                1.0,
            )
        heatmaps = attention_weights.view(
            batch_size,
            2,
            self.config.grid_size,
            self.config.grid_size,
        )
        if self.high_resolution_grounding_stem is not None:
            if (
                self.high_resolution_grounding_key is None
                or self.high_resolution_grounding_query is None
                or self.high_resolution_grounding_gate is None
            ):
                raise RuntimeError("high-resolution grounding modules are incomplete")
            high_resolution_features = self.high_resolution_grounding_stem(rgb)
            high_resolution_keys = self.high_resolution_grounding_key(
                high_resolution_features
            ).flatten(2).transpose(1, 2)
            high_resolution_queries = self.high_resolution_grounding_query(
                grounded_features
            )
            high_resolution_logits = torch.einsum(
                "bqd,bnd->bqn",
                high_resolution_queries,
                high_resolution_keys,
            ) / math.sqrt(self.config.model_dim)
            high_resolution_attention = high_resolution_logits.softmax(dim=-1)
            high_resolution_coordinates = (
                high_resolution_attention
                @ self.high_resolution_patch_centers.to(
                    high_resolution_attention.dtype
                )
            )
            blend = torch.tanh(self.high_resolution_grounding_gate)
            grounding_coordinates = torch.clamp(
                grounding_coordinates
                + blend * (
                    high_resolution_coordinates - grounding_coordinates
                ),
                0.0,
                1.0,
            )
            high_resolution_grid_size = (
                self.config.high_resolution_grounding_grid_size
            )
            heatmaps = high_resolution_attention.view(
                batch_size,
                2,
                high_resolution_grid_size,
                high_resolution_grid_size,
            )
        grounding_world_positions = None
        if self.world_grounding_head is not None:
            world_grounding_input = torch.cat(
                (grounded_features, grounding_coordinates),
                dim=-1,
            )
            grounding_world_positions = torch.tanh(
                self.world_grounding_head(world_grounding_input)
            )

        phase_logits = self.phase_head(task_features)
        action_context = task_features
        if self.grounding_action_projection is not None:
            spatial_state = torch.cat(
                (proprio, grounding_coordinates.flatten(start_dim=1)),
                dim=-1,
            )
            action_context = action_context + self.grounding_action_projection(
                spatial_state
            )
        if self.world_grounding_action_projection is not None:
            if grounding_world_positions is None:
                raise RuntimeError(
                    "world-grounding action path requires world position predictions"
                )
            geometry_state = torch.cat(
                (proprio, grounding_world_positions.flatten(start_dim=1)),
                dim=-1,
            )
            action_context = (
                action_context
                + self.world_grounding_action_projection(geometry_state)
            )
        if self.phase_action_projection is not None:
            action_context = action_context + self.phase_action_projection(
                phase_logits.softmax(dim=-1)
            )

        flow_velocity = None
        flow_target = None
        if self.config.action_head == "deterministic":
            action_queries = self.action_queries.expand(batch_size, -1, -1)
            action_queries = action_queries + action_context.unsqueeze(1)
            decoded_actions = self.action_decoder(
                action_queries,
                memory,
                memory_key_padding_mask=padding_mask,
            )
            action_chunk = torch.tanh(self.action_head(decoded_actions))
        elif action_targets is not None:
            if action_targets.shape != (
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
            ):
                raise ValueError(
                    "action_targets must have shape "
                    f"[batch, {self.config.action_horizon}, {self.config.action_dim}]"
                )
            if flow_noise is None:
                flow_noise = torch.randn_like(action_targets)
            if flow_time is None:
                flow_time = torch.rand(
                    batch_size,
                    device=action_targets.device,
                    dtype=action_targets.dtype,
                )
            flow_time = flow_time.reshape(batch_size)
            interpolation_time = flow_time[:, None, None]
            noisy_actions = (
                interpolation_time * flow_noise
                + (1.0 - interpolation_time) * action_targets
            )
            flow_target = flow_noise - action_targets
            flow_velocity = self._decode_flow_velocity(
                noisy_actions,
                flow_time,
                action_context,
                memory,
                padding_mask,
            )
            action_chunk = torch.clamp(
                noisy_actions - interpolation_time * flow_velocity,
                -1.0,
                1.0,
            )
        else:
            action_chunk = self._sample_flow_actions(
                action_context,
                memory,
                padding_mask,
            )
        return TinyVLAOutput(
            action_chunk=action_chunk,
            phase_logits=phase_logits,
            grounding_coordinates=grounding_coordinates,
            grounding_heatmaps=heatmaps,
            grounding_world_positions=grounding_world_positions,
            flow_velocity=flow_velocity,
            flow_target=flow_target,
        )

    def _decode_flow_velocity(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        task_features: Tensor,
        memory: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch_size = noisy_actions.shape[0]
        action_features = self.flow_action_projection(noisy_actions)
        time_features = self.flow_time_projection(
            self._sinusoidal_time_embedding(time, self.config.model_dim)
        ).unsqueeze(1)
        queries = (
            action_features
            + self.action_queries.expand(batch_size, -1, -1)
            + time_features
            + task_features.unsqueeze(1)
        )
        decoded = self.action_decoder(
            queries,
            memory,
            memory_key_padding_mask=padding_mask,
        )
        return self.flow_velocity_head(decoded)

    def _sample_flow_actions(
        self,
        task_features: Tensor,
        memory: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        batch_size = memory.shape[0]
        actions = torch.randn(
            (
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
            ),
            device=memory.device,
            dtype=memory.dtype,
        )
        step_size = 1.0 / self.config.flow_matching_steps
        for step_index in range(self.config.flow_matching_steps, 0, -1):
            time = torch.full(
                (batch_size,),
                step_index * step_size,
                device=memory.device,
                dtype=memory.dtype,
            )
            velocity = self._decode_flow_velocity(
                actions,
                time,
                task_features,
                memory,
                padding_mask,
            )
            actions = actions - step_size * velocity
        return torch.clamp(actions, -1.0, 1.0)

    @staticmethod
    def _sinusoidal_time_embedding(time: Tensor, dimension: int) -> Tensor:
        half_dimension = dimension // 2
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1_000.0),
                half_dimension,
                device=time.device,
                dtype=time.dtype,
            )
        )
        angles = 2.0 * math.pi * time.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if dimension % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    @staticmethod
    def _make_patch_centers(grid_size: int) -> Tensor:
        coordinates = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        y_coordinates, x_coordinates = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        return torch.stack((x_coordinates.flatten(), y_coordinates.flatten()), dim=-1)
