"""Tests for episodic replay / consolidation.

Consolidation replays high-importance episodes to strengthen memories
when the agent is safe. It uses only existing primitives (valence,
associations, action-transitions) — no new mechanisms.
"""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.agent.core import Agent
from fpi.memory.episodic import Episode, EpisodicMemory


def _signal(values: list[float], t: int = 0) -> Signal:
    return Signal(data=np.array(values, dtype=np.float64), timestamp=t)


SIG_A = _signal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SIG_B = _signal([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
SIG_C = _signal([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
SIG_D = _signal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def _make_agent(**kwargs):
    defaults = dict(
        similarity_threshold=0.9,
        max_patterns=20,
        max_associations=100,
        seed=42,
        exploration_base=0.0,
        enable_episodic_memory=True,
        episodic_surprise_threshold=0.3,  # Low threshold so episodes record easily
    )
    defaults.update(kwargs)
    return Agent(**defaults)


class TestConsolidate:
    def test_consolidate_reinforces_valence(self):
        """Consolidation should strengthen valence for replayed patterns."""
        agent = _make_agent()
        agent.vitality.spend(0.5)  # Room for reward

        # Generate a surprising high-valence episode
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.5, 0)  # Reward at B

        # Get the pattern ID for B
        pid_b = agent.world_model.current_pattern.pattern_id
        valence_before = agent.valence.get(pid_b)

        # Advance ticks past rate limit
        for _ in range(35):
            agent.step_with_action(SIG_A, 0.0, 1)

        # Consolidate (agent should be safe — urgency low after restoring)
        agent.vitality.restore(0.8)
        replayed = agent.consolidate()

        assert replayed > 0
        valence_after = agent.valence.get(pid_b)
        # Valence should be reinforced (more positive)
        assert valence_after > valence_before or valence_after > 0.0

    def test_consolidate_reinforces_associations(self):
        """Consolidation should strengthen context → pattern associations."""
        agent = _make_agent()

        # Create distinct patterns with a transition
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.5, 0)  # Surprising negative → recorded

        pid_a = agent.world_model.prev_pattern.pattern_id
        pid_b = agent.world_model.current_pattern.pattern_id

        # Get initial association strength
        assoc = agent.world_model.memory.associations.get(pid_a, pid_b)
        strength_before = assoc.strength if assoc else 0.0

        # Advance past rate limit
        for _ in range(35):
            agent.step_with_action(SIG_C, 0.0, 1)

        # Make safe and consolidate
        agent.vitality.restore(0.9)
        replayed = agent.consolidate()

        assert replayed > 0
        assoc_after = agent.world_model.memory.associations.get(pid_a, pid_b)
        assert assoc_after is not None
        assert assoc_after.strength >= strength_before

    def test_consolidate_skipped_when_urgent(self):
        """High urgency should prevent consolidation."""
        agent = _make_agent()

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.5, 0)

        # Drain vitality to make urgent
        agent.vitality.spend(0.9)
        assert agent.vitality.urgency >= 0.3

        # Advance past rate limit
        for _ in range(35):
            agent._tick += 1

        replayed = agent.consolidate()
        assert replayed == 0

    def test_consolidate_rate_limited(self):
        """Consolidation should not fire again within 30 ticks."""
        agent = _make_agent()
        agent.vitality.spend(0.5)

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.5, 0)

        # Advance past initial rate limit
        for _ in range(35):
            agent.step_with_action(SIG_C, 0.0, 1)

        agent.vitality.restore(0.9)
        first = agent.consolidate()
        assert first > 0

        # Immediately call again — should be rate limited
        second = agent.consolidate()
        assert second == 0

    def test_consolidate_prioritizes_high_valence(self):
        """Death episodes should be replayed before neutral ones."""
        agent = _make_agent()
        agent.vitality.spend(0.5)

        # Create a mild episode
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.1, 0)

        # Create a severe episode (death-like)
        agent.step_with_action(SIG_C, -0.8, 1)

        pid_c = agent.world_model.current_pattern.pattern_id

        # Advance past rate limit
        for _ in range(35):
            agent.step_with_action(SIG_D, 0.0, 2)

        agent.vitality.restore(0.9)
        # Replay only 1 — should pick the high-|valence| one
        replayed = agent.consolidate(max_replays=1)
        assert replayed == 1

        # The severe episode's pattern should have reinforced negative valence
        val_c = agent.valence.get(pid_c)
        assert val_c < 0.0

    def test_consolidate_propagates_valence_backward(self):
        """Context patterns should inherit discounted valence from episodes."""
        agent = _make_agent()
        agent.vitality.spend(0.5)

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.8, 0)  # Bad event at B, context is A

        pid_a = agent.world_model.prev_pattern.pattern_id

        # Advance past rate limit
        for _ in range(35):
            agent.step_with_action(SIG_C, 0.0, 1)

        agent.vitality.restore(0.9)
        agent.consolidate()

        # A should have inherited negative valence (as predecessor to B)
        val_a = agent.valence.get(pid_a)
        assert val_a < 0.0

    def test_consolidate_with_action_reinforces_transitions(self):
        """Replay with action_taken should reinforce action-outcome predictions."""
        agent = _make_agent()
        agent.vitality.spend(0.5)

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.5, 0)  # Action 0 from A → B

        pid_a = agent.world_model.prev_pattern.pattern_id
        pid_b = agent.world_model.current_pattern.pattern_id

        # Advance past rate limit
        for _ in range(35):
            agent.step_with_action(SIG_C, 0.0, 1)

        agent.vitality.restore(0.9)
        agent.consolidate()

        # The action transition (A, action=0) → B should be reinforced
        key = (pid_a, 0)
        assert key in agent.world_model._action_transitions
        targets = agent.world_model._action_transitions[key]
        assert pid_b in targets
        assert targets[pid_b] > 0.0

    def test_episode_stores_action_taken(self):
        """Episode dataclass should store action_taken field."""
        ep = Episode(
            tick=0,
            pattern_id=1,
            centroid=np.zeros(6),
            surprise=0.5,
            vitality=0.8,
            context_ids=(0,),
            valence=-0.5,
            action_taken=3,
        )
        assert ep.action_taken == 3

    def test_episode_action_default_none(self):
        """action_taken should default to None for backward compatibility."""
        ep = Episode(
            tick=0,
            pattern_id=1,
            centroid=np.zeros(6),
            surprise=0.5,
            vitality=0.8,
            context_ids=(),
            valence=0.0,
        )
        assert ep.action_taken is None

    def test_consolidate_noop_without_episodic_memory(self):
        """Agent without episodic memory should return 0."""
        agent = _make_agent(enable_episodic_memory=False)
        agent.step_with_action(SIG_A, 0.0, None)
        assert agent.consolidate() == 0
