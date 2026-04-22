"""SocialSociety — Society with leaky embodiment feedback loop.

Thin subclass of Society that feeds each agent's internal state back to
the SocialGridEnv after each step. This closes the loop:

    Agent selects action
    → SocialGridEnv returns 21-dim obs (nearest other's PREVIOUS tick state)
    → Agent processes with same WorldModel
    → SocialSociety calls update_agent_state()
    → That state becomes visible to others NEXT tick

No changes to Agent, WorldModel, or any primitive.
"""

from __future__ import annotations

from ..agent.core import Agent
from ..env.social import SocialGridEnv
from .bridge import SignalBridge
from .core import Society


class SocialSociety(Society):
    """Society that feeds agent vitality/surprise back to a SocialGridEnv.

    Args:
        agents: The individual agents.
        env: Must be a SocialGridEnv (not plain SharedGridEnv).
        bridge: Converts aggregate state to society-level signals.
        **kwargs: Passed to Society.
    """

    def __init__(
        self,
        agents: list[Agent],
        env: SocialGridEnv,
        bridge: SignalBridge,
        **kwargs,
    ) -> None:
        self._social_env = env
        super().__init__(agents=agents, env=env, bridge=bridge, **kwargs)

    def _step_all_agents(self) -> None:
        """Step all agents, then feed leaked state back to environment."""
        alive_indices = [
            i for i, a in enumerate(self.agents) if a.vitality.alive
        ]
        self._rng.shuffle(alive_indices)

        for idx in alive_indices:
            agent = self.agents[idx]
            action = agent.select_action(self.env.action_space)
            obs, energy_delta, _done = self.env.step_agent(idx, action)
            agent.step_with_action(obs, energy_delta, action)

            # Feed leaked state back for social perception
            self._social_env.update_agent_state(
                idx,
                vitality=agent.vitality.energy,
                surprise=agent.world_model.last_surprise,
            )
