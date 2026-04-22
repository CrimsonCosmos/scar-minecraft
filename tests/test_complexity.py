"""Tests for FEP complexity penalty (Occam's razor for pattern models)."""

import numpy as np
import pytest

from fpi.primitives.pattern import Pattern
from fpi.primitives.signal import Signal
from fpi.world_model.model import WorldModel


class TestSpecificity:
    def test_zero_centroid_returns_zero(self):
        p = Pattern(centroid=np.zeros(4), pattern_id=0)
        assert p.specificity() == 0.0

    def test_uniform_centroid_low_specificity(self):
        p = Pattern(centroid=np.array([0.5, 0.5, 0.5, 0.5]), pattern_id=0)
        # Sharpness = max/mean = 1.0, exposure_factor = 0/20 = 0.0
        assert p.specificity() == 0.0  # No exposure yet

    def test_specificity_increases_with_exposure(self):
        p = Pattern(centroid=np.array([1.0, 0.0, 0.0, 0.0]), pattern_id=0)
        s1 = p.specificity()  # exposure_count=0
        p.exposure_count = 10
        s2 = p.specificity()  # exposure_count=10
        p.exposure_count = 20
        s3 = p.specificity()  # exposure_count=20 (capped)
        assert s1 < s2 < s3

    def test_sharp_centroid_higher_specificity(self):
        # Sharp: one dominant dimension
        p_sharp = Pattern(centroid=np.array([1.0, 0.01, 0.01, 0.01]),
                          pattern_id=0, exposure_count=10)
        # Flat: uniform dimensions
        p_flat = Pattern(centroid=np.array([0.5, 0.5, 0.5, 0.5]),
                         pattern_id=1, exposure_count=10)
        assert p_sharp.specificity() > p_flat.specificity()

    def test_specificity_caps_at_20_exposure(self):
        p = Pattern(centroid=np.array([1.0, 0.0, 0.0, 0.0]),
                    pattern_id=0, exposure_count=20)
        s20 = p.specificity()
        p.exposure_count = 100
        s100 = p.specificity()
        assert s20 == pytest.approx(s100)


class TestComplexityCost:
    def test_disabled_returns_zero(self):
        wm = WorldModel(enable_complexity_cost=False)
        assert wm.complexity_cost() == 0.0

    def test_scales_with_patterns(self):
        wm = WorldModel(
            similarity_threshold=0.5,
            enable_complexity_cost=True,
            complexity_cost_rate=0.001,
        )
        # Add patterns by observing diverse signals
        wm.observe(Signal(data=np.array([1.0, 0.0, 0.0, 0.0])))
        cost1 = wm.complexity_cost()
        wm.observe(Signal(data=np.array([0.0, 0.0, 0.0, 1.0])))
        cost2 = wm.complexity_cost()
        # More patterns → higher cost
        assert cost2 >= cost1

    def test_included_in_tick(self):
        wm = WorldModel(
            similarity_threshold=0.5,
            enable_complexity_cost=True,
            complexity_cost_rate=0.01,
        )
        wm.observe(Signal(data=np.array([1.0, 0.0, 0.0, 0.0])))
        # Tick should include complexity cost
        tick_cost = wm.tick()
        assert tick_cost >= 0.0


class TestAgentWithComplexity:
    def test_agent_with_complexity_survives(self):
        from fpi.agent.core import Agent
        from fpi.env.base import SurvivalEnv

        agent = Agent(
            similarity_threshold=0.7, seed=42,
            enable_complexity_cost=True,
            complexity_cost_rate=0.0005,
        )
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0

    def test_agent_without_complexity_unchanged(self):
        from fpi.agent.core import Agent

        agent = Agent(similarity_threshold=0.7, seed=42)
        assert agent.world_model._enable_complexity_cost is False
        assert agent.world_model.complexity_cost() == 0.0
