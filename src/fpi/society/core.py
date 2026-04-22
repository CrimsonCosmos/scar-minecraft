"""Society — the meta-agent. Recursive intelligence at the collective level.

A Society is to Agents what a Brain is to Neurons:
- It SENSES by observing aggregate population statistics
- It DISTINGUISHES patterns in collective behavior
- It PREDICTS collective outcomes using the same WorldModel an Agent uses
- It has VITALITY (collective health under entropy)
- It learns VALENCE (which collective states correlate with survival)
- It ACTS by modifying the agents' environment (resource allocation)

Individual agents do not know the Society exists. They perceive only
their local environment. The Society's intelligence is emergent —
invisible and incomprehensible from the level below.

This is composable: Society has the same shape as Agent (WorldModel +
Vitality + Valence + sense-predict-act loop). A Collective of Societies
could form a Civilization using the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives.vitality import Vitality
from ..primitives.valence import Valence
from ..world_model.model import WorldModel
from ..agent.core import Agent
from ..env.shared import SharedGridEnv
from .bridge import SignalBridge


@dataclass
class SocietyStepResult:
    """The outcome of a single society tick."""

    observation: object  # Signal (the collective state)
    surprise: float
    num_alive: int
    mean_vitality: float
    action: int | None
    collective_vitality: float
    collective_vitality_delta: float
    tick: int


class Society:
    """A collective intelligence that emerges from observing agent behavior.

    Uses the SAME primitives as Agent: WorldModel, Vitality, Valence.
    The only difference is what "sense" and "act" mean at this scale.

    Args:
        agents: The "neurons" — individual agents in the society.
        env: The shared environment all agents inhabit.
        bridge: Converts aggregate agent state to society-level signals.
        similarity_threshold: For the society's pattern recognition.
        exploration_base: Society's exploration vs exploitation balance.
        valence_learning_rate: How fast the society learns what's good/bad.
        max_patterns: Society's brain capacity.
        max_associations: Society's connection capacity.
        association_decay_rate: How fast unused society-level associations decay.
        seed: Random seed.
    """

    def __init__(
        self,
        agents: list[Agent],
        env: SharedGridEnv,
        bridge: SignalBridge,
        similarity_threshold: float = 0.7,
        exploration_base: float = 0.15,
        valence_learning_rate: float = 0.3,
        max_patterns: int | None = 20,
        max_associations: int | None = 60,
        association_decay_rate: float = 0.005,
        seed: int | None = None,
    ) -> None:
        self.agents = agents
        self.env = env
        self.bridge = bridge

        # The SAME primitives as Agent — same code, higher scale
        self.world_model = WorldModel(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
            max_associations=max_associations,
            association_decay_rate=association_decay_rate,
        )
        self.vitality = Vitality(entropy_rate=0.005)  # Societies are more stable
        self.valence = Valence(learning_rate=valence_learning_rate)
        self._exploration_base = exploration_base
        self._rng = np.random.default_rng(seed)
        self._tick = 0
        self.history: list[SocietyStepResult] = []

        # Register all agents in the shared environment
        for i, agent in enumerate(agents):
            # Spread agents across the grid
            spread = env.grid_size // (len(agents) + 1)
            pos = spread * (i + 1)
            env.register_agent(i, position=pos)

    # ------------------------------------------------------------------
    # The sense-predict-act loop — structurally identical to Agent
    # ------------------------------------------------------------------

    def step(self) -> SocietyStepResult:
        """One tick of the society.

        This is structurally identical to Agent.step_with_action():
        1. Sub-entities act (agents step in shared env)
        2. Sense aggregate state
        3. Compute vitality delta
        4. Update vitality (restore/spend + entropy)
        5. WorldModel observes → surprise → update associations
        6. Update valence (pattern ↔ vitality delta)
        7. Select and apply action
        8. Record action outcome
        """
        # 1. Each agent acts in the shared environment
        self._step_all_agents()

        # Advance environment clock (regenerate resources)
        self.env.tick()

        # 2. SENSE: encode aggregate state
        observation = self._sense()

        # 3. Compute collective vitality delta
        vitality_before = self.vitality.energy
        collective_delta = self._compute_collective_vitality_delta()

        # 4. Apply to society's vitality
        if collective_delta > 0:
            self.vitality.restore(collective_delta)
        elif collective_delta < 0:
            self.vitality.spend(abs(collective_delta))
        self.vitality.tick()

        # Maintenance cost (thinking costs energy at every scale)
        maintenance_cost = self.world_model.tick()
        if maintenance_cost > 0:
            self.vitality.spend(maintenance_cost)

        vitality_after = self.vitality.energy
        actual_delta = vitality_after - vitality_before

        # 5. WorldModel observes the collective signal
        surprise = self.world_model.observe(observation)

        # 6. Update valence
        if self.world_model.current_pattern is not None:
            self.valence.update(
                self.world_model.current_pattern.pattern_id,
                actual_delta,
            )

        # 7. Select and apply action
        action = self._select_action()
        self.env.set_regen_bias(action)

        # 8. Record action outcome
        if self.world_model.current_pattern is not None:
            self.world_model.record_action_outcome(
                action,
                self.world_model.current_pattern,
                actual_delta,
            )

        # Build result
        alive_agents = [a for a in self.agents if a.vitality.alive]
        result = SocietyStepResult(
            observation=observation,
            surprise=surprise,
            num_alive=len(alive_agents),
            mean_vitality=(
                sum(a.vitality.energy for a in alive_agents) / len(alive_agents)
                if alive_agents else 0.0
            ),
            action=action,
            collective_vitality=self.vitality.energy,
            collective_vitality_delta=actual_delta,
            tick=self._tick,
        )
        self.history.append(result)
        self._tick += 1
        return result

    def run_episode(self, max_steps: int = 300) -> list[SocietyStepResult]:
        """Run a full episode of the society.

        Returns when done, or when the society dies (all agents dead or
        society vitality = 0).
        """
        episode_results: list[SocietyStepResult] = []

        for _ in range(max_steps):
            if not self.vitality.alive:
                break

            # Check if any agents are still alive
            alive = any(a.vitality.alive for a in self.agents)
            if not alive:
                break

            result = self.step()
            episode_results.append(result)

            if self.env.step_count >= self.env._max_steps:
                break

        return episode_results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _step_all_agents(self) -> None:
        """Each agent selects an action and steps in the shared environment.

        Agent order is shuffled each tick to prevent positional bias.
        """
        alive_indices = [
            i for i, a in enumerate(self.agents) if a.vitality.alive
        ]
        self._rng.shuffle(alive_indices)

        for idx in alive_indices:
            agent = self.agents[idx]
            action = agent.select_action(self.env.action_space)
            obs, energy_delta, _done = self.env.step_agent(idx, action)
            agent.step_with_action(obs, energy_delta, action)

    def _sense(self):
        """Encode aggregate agent state as a signal.

        The society doesn't see individual neurons — it sees population
        activity patterns.
        """
        positions: dict[int, int] = {}
        vitalities: dict[int, float] = {}
        for i, agent in enumerate(self.agents):
            if agent.vitality.alive:
                positions[i] = self.env.agent_positions.get(i, 0)
                vitalities[i] = agent.vitality.energy

        return self.bridge.encode(positions, vitalities, self._tick)

    def _compute_collective_vitality_delta(self) -> float:
        """How much energy the society gains/loses from agent states.

        Society thrives when agents thrive. Society weakens when agents die.
        This is not designed — it's the same thermodynamic principle:
        the society's substrate (agents) must persist for the society to persist.
        """
        alive = [a for a in self.agents if a.vitality.alive]
        if not alive:
            return -0.2  # Catastrophic — all neurons dead

        alive_fraction = len(alive) / len(self.agents)
        mean_vitality = sum(a.vitality.energy for a in alive) / len(alive)

        # Positive when agents are thriving, negative when struggling
        return (alive_fraction * mean_vitality - 0.3) * 0.15

    def _select_action(self) -> int:
        """Choose a society-level action.

        This is LITERALLY the same algorithm as Agent.select_action().
        Same scoring: predicted_vitality * confidence + exploration bonus.
        The only difference is what the actions mean at this scale.
        """
        action_space = [0, 1, 2]

        current = self.world_model.current_pattern
        if current is None:
            return int(self._rng.choice(action_space))

        scores: list[float] = []
        for action in action_space:
            predicted_vitality = self.world_model.predict_action_vitality(action)
            prediction = self.world_model.predict_action_outcome(action)

            if predicted_vitality is not None and prediction is not None:
                _pattern, confidence = prediction
                score = predicted_vitality * confidence
                uncertainty = 1.0 - confidence
                score += uncertainty * 0.05 * (1.0 - self.vitality.urgency)
                scores.append(score)
            elif prediction is not None:
                pattern, confidence = prediction
                v = self.valence.get(pattern.pattern_id)
                score = v * confidence
                uncertainty = 1.0 - confidence
                score += uncertainty * 0.05 * (1.0 - self.vitality.urgency)
                scores.append(score)
            else:
                scores.append(0.05 * (1.0 - self.vitality.urgency))

        explore_prob = self._exploration_base * (1.0 - self.vitality.urgency)
        if self._rng.random() < explore_prob:
            return int(self._rng.choice(action_space))

        best_idx = int(np.argmax(scores))
        return action_space[best_idx]

    # ------------------------------------------------------------------
    # Properties (same shape as Agent)
    # ------------------------------------------------------------------

    @property
    def pattern_count(self) -> int:
        return self.world_model.memory.pattern_count

    @property
    def association_count(self) -> int:
        return self.world_model.memory.association_count

    @property
    def average_surprise(self) -> float:
        return self.world_model.average_surprise
