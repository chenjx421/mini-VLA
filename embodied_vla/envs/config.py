from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ObservationMode = Literal["state", "multimodal"]
TaskLevel = Literal["reach", "pick", "pick_place"]
GraspMode = Literal["contact", "contact_assisted"]


@dataclass(frozen=True)
class SOArmEnvConfig:
    observation_mode: ObservationMode = "state"
    task_level: TaskLevel = "pick_place"
    grasp_mode: GraspMode = "contact_assisted"
    image_size: int = 64
    max_episode_steps: int = 250
    physics_timestep: float = 0.002
    control_substeps: int = 10
    cartesian_action_scale: float = 0.018
    wrist_action_scale: float = 0.08
    ik_damping: float = 0.04
    success_radius: float = 0.045
    table_height: float = 0.0
    cube_half_size: float = 0.015
    pregrasp_jaw: float = 0.30
    open_jaw: float = 0.65
    closed_jaw: float = -0.174
    render_camera: str = "front"
    domain_randomization: bool = False
    include_end_effector_position_in_proprio: bool = False

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.control_substeps <= 0:
            raise ValueError("control_substeps must be positive")
        if self.cartesian_action_scale <= 0:
            raise ValueError("cartesian_action_scale must be positive")
        if self.closed_jaw >= self.open_jaw:
            raise ValueError("closed_jaw must be smaller than open_jaw")
        if not self.closed_jaw < self.pregrasp_jaw < self.open_jaw:
            raise ValueError("pregrasp_jaw must lie between closed_jaw and open_jaw")
