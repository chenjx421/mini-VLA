from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class StateActorCritic(nn.Module):
    """Gaussian actor and value critic sharing a state encoder."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        initial_log_std: float = -0.7,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def distribution_and_value(self, observation: Tensor) -> tuple[Normal, Tensor]:
        features = self.encoder(observation)
        mean = self.actor_mean(features)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), self.value_head(features).squeeze(-1)

    def sample(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        distribution, value = self.distribution_and_value(observation)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = self._squashed_log_probability(
            distribution,
            raw_action,
            action,
        )
        return action, raw_action, log_probability, value

    def evaluate_raw_action(
        self,
        observation: Tensor,
        raw_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, value = self.distribution_and_value(observation)
        action = torch.tanh(raw_action)
        log_probability = self._squashed_log_probability(
            distribution,
            raw_action,
            action,
        )
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value

    @torch.no_grad()
    def act(self, observation: Tensor, *, deterministic: bool = True) -> Tensor:
        distribution, _ = self.distribution_and_value(observation)
        raw_action = distribution.mean if deterministic else distribution.sample()
        return torch.tanh(raw_action)

    @staticmethod
    def _squashed_log_probability(
        distribution: Normal,
        raw_action: Tensor,
        action: Tensor,
    ) -> Tensor:
        gaussian_log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        log_jacobian = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        return gaussian_log_probability - log_jacobian
