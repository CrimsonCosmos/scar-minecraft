"""Tests for ContextualSurvivalEnv — resources shift positions over time."""

from fpi.env.base import ContextualSurvivalEnv


class TestContextualSurvivalEnv:
    def test_initial_phase_uses_first_resource_set(self) -> None:
        """During phase 0, resources should be at the first set."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=5, resource_sets=[[2], [8]], max_steps=100
        )
        env.reset()
        assert env._resource_positions == [2]

    def test_phase_switches_resource_positions(self) -> None:
        """After phase_length steps, resources should switch."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=3, resource_sets=[[2], [8]], max_steps=100
        )
        env.reset()

        # Steps 0, 1, 2 are phase 0 (resources at [2])
        for _ in range(3):
            env.step(1)  # stay
        # Now step_count=3, phase=1 -> resources at [8]
        assert env._resource_positions == [8]

    def test_phase_cycles_back(self) -> None:
        """Phases should cycle: 0, 1, 0, 1, ..."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=2, resource_sets=[[1], [9]], max_steps=100
        )
        env.reset()

        # Phase 0: steps 0-1
        env.step(1)
        env.step(1)
        # Phase 1: steps 2-3
        assert env._resource_positions == [9]
        env.step(1)
        env.step(1)
        # Phase 0 again: steps 4-5
        assert env._resource_positions == [1]

    def test_resource_at_correct_position_per_phase(self) -> None:
        """Agent should get resources only at the current phase's positions."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=5,
            resource_sets=[[0], [9]],
            max_steps=100,
            resource_value=0.5,
        )
        env.reset()

        # Move agent to position 0 (left edge)
        env._position = 0
        _, delta, _ = env.step(1)  # stay at 0, phase 0, resources at [0]
        assert delta > 0  # Should get resource

        # Advance to phase 1 (resources at [9])
        for _ in range(4):
            env.step(1)
        # Now in phase 1
        env._position = 0
        _, delta2, _ = env.step(1)  # stay at 0, but resources now at [9]
        assert delta2 < 0  # Should NOT get resource (only stay cost)

    def test_reset_restores_first_phase(self) -> None:
        """Reset should go back to phase 0."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=2, resource_sets=[[1], [9]], max_steps=100
        )
        env.reset()
        env.step(1)
        env.step(1)
        env.step(1)
        # Now in phase 1
        assert env._resource_positions == [9]

        env.reset()
        assert env._resource_positions == [1]

    def test_three_resource_sets(self) -> None:
        """Should support more than two resource sets."""
        env = ContextualSurvivalEnv(
            grid_size=10, phase_length=2,
            resource_sets=[[1], [5], [9]],
            max_steps=100,
        )
        env.reset()

        env.step(1)
        env.step(1)  # phase 1
        assert env._resource_positions == [5]
        env.step(1)
        env.step(1)  # phase 2
        assert env._resource_positions == [9]
        env.step(1)
        env.step(1)  # phase 0 again
        assert env._resource_positions == [1]

    def test_inherits_action_space(self) -> None:
        """Should have same action space as SurvivalEnv."""
        env = ContextualSurvivalEnv(grid_size=10)
        assert env.action_space == [0, 1, 2]

    def test_default_resource_sets(self) -> None:
        """Default resource sets should be [[2,4], [6,8]]."""
        env = ContextualSurvivalEnv(grid_size=10)
        env.reset()
        assert env._resource_sets == [[2, 4], [6, 8]]
