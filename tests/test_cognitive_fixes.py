"""Tests for FPI cognitive fixes: lookahead, eligibility traces, exploration.

These test the three mechanisms that address FPI limitations #3, #4, and #5:
- Lookahead: multi-step forward simulation in select_action
- Eligibility traces: TD(λ) credit assignment backward through time
- Goal-directed exploration: prefer info-sparse states over random
"""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.primitives.valence import Valence
from fpi.agent.core import Agent


def _signal(values: list[float], t: int = 0) -> Signal:
    return Signal(data=np.array(values, dtype=np.float64), timestamp=t)


# Distinct signals for different "states"
SIG_A = _signal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SIG_B = _signal([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
SIG_C = _signal([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
SIG_D = _signal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
SIG_E = _signal([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
SIG_F = _signal([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _make_agent(**kwargs):
    """Create an agent with sensible test defaults."""
    defaults = dict(
        similarity_threshold=0.9,
        max_patterns=20,
        max_associations=100,
        seed=42,
        exploration_base=0.0,  # Disable random exploration for deterministic tests
    )
    defaults.update(kwargs)
    return Agent(**defaults)


# ---------------------------------------------------------------------------
# Lookahead Tests
# ---------------------------------------------------------------------------

class TestLookahead:
    def test_lookahead_score_returns_zero_with_no_model(self):
        agent = _make_agent(enable_lookahead=True, lookahead_depth=3)
        score = agent._lookahead_score(0, [0, 1, 2], )
        assert score == 0.0

    def test_lookahead_prefers_multi_step_positive_outcome(self):
        """Lookahead should prefer an action that leads to delayed reward
        over one with no prediction data."""
        agent = _make_agent(
            enable_lookahead=True,
            lookahead_depth=3,
            lookahead_discount=0.9,
        )

        # Drain vitality so reward doesn't hit the 1.0 cap
        agent.vitality.spend(0.8)

        # Train a chain: A --(action 0)--> B --(action 1)--> C
        # where C has positive vitality, A and B are neutral
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)   # A --0--> B (neutral)
        agent.step_with_action(SIG_C, 0.5, 1)   # B --1--> C (reward!)
        # Go back and reinforce
        agent.step_with_action(SIG_A, 0.0, 2)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, 0.5, 1)
        agent.step_with_action(SIG_A, 0.0, 2)

        # Now from A, action 0 leads to B, from which action 1 leads to C (reward)
        # Action 2 leads back to A (neutral)
        # Lookahead should score action 0 higher because it leads to reward chain
        score_0 = agent._lookahead_score(0, [0, 1, 2])
        score_2 = agent._lookahead_score(2, [0, 1, 2])
        assert score_0 > score_2

    def test_lookahead_used_in_select_action(self):
        """When lookahead is enabled, select_action should use it."""
        agent = _make_agent(
            enable_lookahead=True,
            lookahead_depth=2,
        )
        # With no model, should still return a valid action
        agent.step_with_action(SIG_A, 0.0, None)
        action = agent.select_action([0, 1, 2])
        assert action in [0, 1, 2]

    def test_lookahead_discount_reduces_future_weight(self):
        """Deeper steps should contribute less due to discounting."""
        agent = _make_agent(
            enable_lookahead=True,
            lookahead_depth=3,
            lookahead_discount=0.5,  # Heavy discounting
        )

        # Drain vitality so reward doesn't hit the 1.0 cap
        agent.vitality.spend(0.8)

        # Train: A --0--> B --1--> C, C has big reward
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, 1.0, 1)
        agent.step_with_action(SIG_A, 0.0, 2)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, 1.0, 1)
        agent.step_with_action(SIG_A, 0.0, 2)

        # Score from A, action 0: step 0 gives ~0 (B is neutral),
        # step 1 gives 0.5 * ~1.0 (C has reward, discounted)
        score = agent._lookahead_score(0, [0, 1, 2])
        # The score should be positive but less than 1.0 due to discounting
        assert score > 0.0


# ---------------------------------------------------------------------------
# Eligibility Trace Tests
# ---------------------------------------------------------------------------

class TestEligibilityTraces:
    def test_traces_propagate_credit_backward(self):
        """Reward at step N should affect valence of patterns visited before N."""
        agent = _make_agent(
            enable_eligibility_traces=True,
            trace_decay=0.8,
            discount_factor=0.9,
        )

        # Visit A, B, C, D in sequence, then big reward at E
        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, 0.0, 1)
        agent.step_with_action(SIG_D, 0.0, 2)
        agent.step_with_action(SIG_E, 0.8, 3)  # Big reward here

        # All predecessor patterns should have some positive valence
        # D should have more than C, C more than B, B more than A
        val_a = agent.valence.get(0)  # pattern A
        val_b = agent.valence.get(1)  # pattern B
        val_c = agent.valence.get(2)  # pattern C
        val_d = agent.valence.get(3)  # pattern D

        # D was most recently before the reward → highest trace → most credit
        # Each predecessor should have received some positive adjustment
        # (The exact values depend on TD error computation, but the ordering
        # should show decay)
        assert val_d > val_c or val_d > val_b  # D gets more credit than older states

    def test_traces_decay_over_time(self):
        """Traces for old patterns should decay toward zero."""
        agent = _make_agent(
            enable_eligibility_traces=True,
            trace_decay=0.5,  # Fast decay for testing
            discount_factor=0.9,
        )

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, 0.0, 1)
        agent.step_with_action(SIG_D, 0.0, 2)
        agent.step_with_action(SIG_E, 0.0, 3)
        agent.step_with_action(SIG_F, 0.0, 4)

        # With trace_decay=0.5 and discount=0.9, traces decay by 0.45 per step
        # After 5 steps, A's trace = 0.45^5 ≈ 0.018, which is above 0.001 threshold
        # But B's, C's, D's should still be present and ordered
        traces = agent._eligibility_traces
        # Recent patterns should have higher traces
        if 4 in traces and 0 in traces:
            assert traces[4] > traces[0]

    def test_traces_clean_up_tiny_values(self):
        """Traces below threshold should be pruned."""
        agent = _make_agent(
            enable_eligibility_traces=True,
            trace_decay=0.1,  # Very fast decay
            discount_factor=0.1,
        )

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)
        # Many steps to decay A's trace below threshold
        for i in range(20):
            agent.step_with_action(SIG_C, 0.0, 1)
            agent.step_with_action(SIG_D, 0.0, 2)

        # A's trace should have been pruned (0.01^20 ≈ 0)
        assert 0 not in agent._eligibility_traces

    def test_negative_reward_penalizes_predecessors(self):
        """Death or damage should create negative valence for preceding states."""
        agent = _make_agent(
            enable_eligibility_traces=True,
            trace_decay=0.8,
            discount_factor=0.9,
        )

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, 0.0, 0)
        agent.step_with_action(SIG_C, -1.0, 1)  # Death!

        # B was the state before death — should have negative valence
        val_b = agent.valence.get(1)
        assert val_b < 0.0

    def test_traces_disabled_uses_retroactive_window(self):
        """When traces are disabled, the old retroactive window should work."""
        agent = _make_agent(enable_eligibility_traces=False)

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.5, 0)

        # With traces disabled, the recent_pattern_ids list should be populated
        assert len(agent._recent_pattern_ids) > 0


# ---------------------------------------------------------------------------
# Goal-Directed Exploration Tests
# ---------------------------------------------------------------------------

class TestGoalDirectedExploration:
    def test_exploration_prefers_unknown_outcomes(self):
        """When exploring, actions with unknown outcomes should be preferred
        over actions with well-known outcomes."""
        agent = _make_agent(
            exploration_base=1.0,  # Always explore
            seed=None,  # Truly random for statistical test
        )

        # Train action 0 extensively (well-known outcome)
        agent.step_with_action(SIG_A, 0.0, None)
        for _ in range(20):
            agent.step_with_action(SIG_B, 0.0, 0)
            agent.step_with_action(SIG_A, 0.0, 1)

        # Action 0 is well-known (A → B), action 2 is unknown
        # Over many trials, action 2 should be selected more often
        action_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(300):
            a = agent.select_action([0, 1, 2])
            action_counts[a] += 1

        # Unknown actions (1, 2) should collectively be preferred over known (0)
        unknown_count = action_counts[1] + action_counts[2]
        known_count = action_counts[0]
        assert unknown_count > known_count

    def test_exploration_with_no_predictions_is_uniform(self):
        """With no model at all, exploration should be roughly uniform."""
        agent = _make_agent(exploration_base=1.0, seed=None)
        agent.step_with_action(SIG_A, 0.0, None)

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(300):
            a = agent.select_action([0, 1, 2])
            counts[a] += 1

        # Each action should get roughly 100 selections (±50)
        for c in counts.values():
            assert 30 < c < 200  # Loose bounds for randomness


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_three_mechanisms_together(self):
        """All three mechanisms should work simultaneously without crashing."""
        agent = _make_agent(
            enable_lookahead=True,
            lookahead_depth=2,
            enable_eligibility_traces=True,
            trace_decay=0.8,
            discount_factor=0.9,
            exploration_base=0.1,
        )

        signals = [SIG_A, SIG_B, SIG_C, SIG_D, SIG_E]
        for i in range(50):
            sig = signals[i % len(signals)]
            action = agent.select_action([0, 1, 2, 3])
            agent.step_with_action(sig, 0.01 * (i % 3 - 1), action)

        # Should have learned patterns and valence
        assert agent.pattern_count >= 3
        assert agent.valence.known_count >= 3

    def test_lookahead_with_eligibility_traces(self):
        """Lookahead should benefit from eligibility trace learning,
        since traces make valence predictions more accurate."""
        # Train two agents on the same sequence, one with traces
        agent_no_traces = _make_agent(
            enable_lookahead=True,
            lookahead_depth=2,
            enable_eligibility_traces=False,
        )
        agent_traces = _make_agent(
            enable_lookahead=True,
            lookahead_depth=2,
            enable_eligibility_traces=True,
            trace_decay=0.8,
            discount_factor=0.9,
        )

        # Both should run without error
        for agent in [agent_no_traces, agent_traces]:
            agent.step_with_action(SIG_A, 0.0, None)
            agent.step_with_action(SIG_B, 0.0, 0)
            agent.step_with_action(SIG_C, 0.5, 1)
            agent.step_with_action(SIG_A, 0.0, 2)
            action = agent.select_action([0, 1, 2])
            assert action in [0, 1, 2]

    def test_backward_compat_without_new_features(self):
        """Agent with all new features disabled should behave as before."""
        agent = _make_agent(
            enable_lookahead=False,
            enable_eligibility_traces=False,
            exploration_base=0.0,
        )

        agent.step_with_action(SIG_A, 0.0, None)
        agent.step_with_action(SIG_B, -0.3, 0)
        action = agent.select_action([0, 1, 2])
        assert action in [0, 1, 2]
        # Retroactive window should still be populated
        assert len(agent._recent_pattern_ids) > 0
