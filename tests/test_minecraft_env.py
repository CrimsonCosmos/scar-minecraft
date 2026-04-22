"""Tests for MinecraftEnv with mock bridge.

Tests the environment interface, energy delta computation, and death handling
without requiring a running Minecraft server.
"""

import pytest

from fpi.minecraft.env import MinecraftEnv, compute_energy_delta
from fpi.minecraft.actions import PHASE_1_ACTIONS, PHASE_2_ACTIONS


# ---------------------------------------------------------------------------
# Test compute_energy_delta
# ---------------------------------------------------------------------------

class TestEnergyDelta:
    def test_no_change(self):
        prev = {"health": 20.0, "food": 20.0, "xp_points": 0, "alive": True}
        curr = {"health": 20.0, "food": 20.0, "xp_points": 0, "alive": True}
        delta = compute_energy_delta(prev, curr)
        # Only idle cost
        assert delta == pytest.approx(-0.001)

    def test_health_loss(self):
        prev = {"health": 20.0, "food": 20.0, "xp_points": 0, "alive": True}
        curr = {"health": 15.0, "food": 20.0, "xp_points": 0, "alive": True}
        delta = compute_energy_delta(prev, curr)
        # health_change = -5/20 = -0.25
        # delta = -0.25 * 0.5 + -0.25 * 0.3 - 0.001 = -0.125 - 0.075 - 0.001
        expected = -0.125 - 0.075 - 0.001
        assert delta == pytest.approx(expected)

    def test_health_gain(self):
        prev = {"health": 10.0, "food": 20.0, "xp_points": 0, "alive": True}
        curr = {"health": 15.0, "food": 20.0, "xp_points": 0, "alive": True}
        delta = compute_energy_delta(prev, curr)
        # health_change = 5/20 = 0.25
        # No extra damage penalty (positive change)
        expected = 0.25 * 0.5 - 0.001
        assert delta == pytest.approx(expected)

    def test_food_loss(self):
        prev = {"health": 20.0, "food": 20.0, "xp_points": 0, "alive": True}
        curr = {"health": 20.0, "food": 15.0, "xp_points": 0, "alive": True}
        delta = compute_energy_delta(prev, curr)
        # food_change = -5/20 = -0.25
        expected = -0.25 * 0.1 - 0.001
        assert delta == pytest.approx(expected)

    def test_hit_landed(self):
        prev = {"health": 20.0, "food": 20.0, "alive": True}
        curr = {"health": 20.0, "food": 20.0, "alive": True, "hit_landed": True}
        delta = compute_energy_delta(prev, curr)
        # hit bonus (0.1) + idle cost
        expected = 0.1 - 0.001
        assert delta == pytest.approx(expected)

    def test_kill_does_not_affect_energy_delta(self):
        """Kills are tracked for stats but don't contribute to energy_delta."""
        prev = {"health": 20.0, "food": 20.0, "alive": True}
        curr = {"health": 20.0, "food": 20.0, "alive": True, "kills": 1}
        delta = compute_energy_delta(prev, curr)
        # Only idle cost — kills don't give energy
        expected = -0.001
        assert delta == pytest.approx(expected)

    def test_hit_with_kill(self):
        """Hit is what matters for energy, kill is just stats."""
        prev = {"health": 20.0, "food": 20.0, "alive": True}
        curr = {"health": 20.0, "food": 20.0, "alive": True, "kills": 1, "hit_landed": True}
        delta = compute_energy_delta(prev, curr)
        # Only hit bonus + idle cost (kill doesn't add to delta)
        expected = 0.1 - 0.001
        assert delta == pytest.approx(expected)

    def test_death_is_catastrophic(self):
        prev = {"health": 5.0, "food": 10.0, "xp_points": 0, "alive": True}
        curr = {"health": 0.0, "food": 0.0, "xp_points": 0, "alive": False}
        delta = compute_energy_delta(prev, curr)
        assert delta == -1.0

    def test_damage_penalty_is_extra(self):
        """Damage should hurt MORE than health gain helps (asymmetric)."""
        prev_base = {"health": 15.0, "food": 20.0, "xp_points": 0, "alive": True}
        # Lose 5 health
        loss = compute_energy_delta(
            prev_base,
            {"health": 10.0, "food": 20.0, "xp_points": 0, "alive": True},
        )
        # Gain 5 health
        gain = compute_energy_delta(
            prev_base,
            {"health": 20.0, "food": 20.0, "xp_points": 0, "alive": True},
        )
        # Damage penalty makes loss magnitude > gain magnitude
        assert abs(loss) > abs(gain)


# ---------------------------------------------------------------------------
# Test MinecraftEnv interface
# ---------------------------------------------------------------------------

class TestMinecraftEnvInterface:
    def test_phase_1_action_space(self):
        env = MinecraftEnv(phase=1)
        assert env.action_space == PHASE_1_ACTIONS
        assert len(env.action_space) == 13

    def test_phase_2_action_space(self):
        env = MinecraftEnv(phase=2)
        assert env.action_space == PHASE_2_ACTIONS
        assert len(env.action_space) == 18

    def test_initial_death_count_is_zero(self):
        env = MinecraftEnv()
        assert env.death_count == 0

    def test_initial_step_count_is_zero(self):
        env = MinecraftEnv()
        assert env.step_count == 0

    def test_all_actions_have_names(self):
        from fpi.minecraft.actions import ACTION_NAMES
        for action_id in PHASE_2_ACTIONS:
            assert action_id in ACTION_NAMES, f"Action {action_id} has no name"
