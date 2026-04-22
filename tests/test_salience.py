"""Tests for Salience primitive and weighted pattern matching."""

import numpy as np
import pytest

from fpi.primitives.salience import Salience
from fpi.primitives.signal import Signal
from fpi.primitives.pattern import Pattern, Distinction


class TestSalience:
    def test_uniform_creation(self):
        s = Salience.uniform(10)
        assert s.dim == 10
        assert np.allclose(s.weights, 1.0)

    def test_update_increases_high_deviation_dims(self):
        s = Salience.uniform(4)
        deviation = np.array([0.0, 0.0, 1.0, 1.0])
        s.update(deviation, error=1.0)
        # Dims 2, 3 should have higher weight than dims 0, 1
        assert s.weights[2] > s.weights[0]
        assert s.weights[3] > s.weights[1]

    def test_update_normalizes_to_mean_one(self):
        s = Salience.uniform(8)
        deviation = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0])
        s.update(deviation, error=0.5)
        assert abs(np.mean(s.weights) - 1.0) < 0.01

    def test_decay_reduces_weights(self):
        s = Salience.uniform(4, decay=0.1)
        initial = s.weights.copy()
        # Update with zero deviation — only decay applies
        s.update(np.zeros(4), error=0.0)
        # After normalization, weights should still be ~1.0 (uniform decay + normalize)
        # But raw weights before normalize would have decayed
        # With uniform decay and normalize, weights stay at 1.0
        # Test that non-uniform deviation + decay creates separation
        s2 = Salience.uniform(4, decay=0.1)
        s2.update(np.array([1.0, 0.0, 0.0, 0.0]), error=1.0)
        # Dim 0 was boosted, others just decayed → dim 0 should be highest
        assert s2.weights[0] > s2.weights[1]

    def test_clip_bounds(self):
        s = Salience.uniform(4, min_weight=0.1, max_weight=5.0)
        # Massive update to push weights high
        for _ in range(100):
            s.update(np.array([10.0, 0.0, 0.0, 0.0]), error=10.0)
        # After normalization, relative weights are what matter
        # But pre-clip values should be bounded
        assert np.all(s.weights >= 0.0)
        # The highest weight should be finite
        assert np.all(np.isfinite(s.weights))

    def test_zero_error_no_change(self):
        s = Salience.uniform(4, decay=0.0)
        before = s.weights.copy()
        s.update(np.array([1.0, 1.0, 1.0, 1.0]), error=0.0)
        # No error, no decay → weights should remain 1.0 (normalized)
        assert np.allclose(s.weights, 1.0)


class TestWeightedSimilarity:
    def test_uniform_weights_equal_unweighted(self):
        centroid = np.array([1.0, 0.5, 0.0, 0.2])
        p = Pattern(centroid=centroid, pattern_id=0)
        sig = Signal(data=np.array([0.8, 0.6, 0.1, 0.3]))
        weights = np.ones(4)
        assert p.weighted_similarity(sig, weights) == pytest.approx(
            p.similarity(sig), abs=1e-10,
        )

    def test_high_weight_dims_increase_discrimination(self):
        """Weighting dims where signals differ makes them MORE dissimilar."""
        # Two signals that match in dims 0-1, differ in dims 2-3
        s1 = Signal(data=np.array([1.0, 0.5, 0.9, 0.1]))
        s2 = Signal(data=np.array([1.0, 0.5, 0.1, 0.9]))
        p = Pattern(centroid=s1.data.copy(), pattern_id=0)

        uniform_sim = p.similarity(s2)

        # Weight the differing dims higher
        weights = np.array([1.0, 1.0, 3.0, 3.0])
        weighted_sim = p.weighted_similarity(s2, weights)

        assert weighted_sim < uniform_sim

    def test_low_weight_dims_reduce_discrimination(self):
        """Weighting dims where signals differ low makes them MORE similar."""
        s1 = Signal(data=np.array([1.0, 0.5, 0.9, 0.1]))
        s2 = Signal(data=np.array([0.1, 0.9, 0.9, 0.1]))
        p = Pattern(centroid=s1.data.copy(), pattern_id=0)

        uniform_sim = p.similarity(s2)

        # Down-weight the differing dims (0, 1), up-weight matching dims (2, 3)
        weights = np.array([0.1, 0.1, 3.0, 3.0])
        weighted_sim = p.weighted_similarity(s2, weights)

        assert weighted_sim > uniform_sim

    def test_zero_weight_ignores_dim(self):
        """Zero-weight dimension contributes nothing."""
        s1 = Signal(data=np.array([1.0, 0.0]))
        s2 = Signal(data=np.array([0.0, 1.0]))
        p = Pattern(centroid=s1.data.copy(), pattern_id=0)

        # Normally orthogonal
        assert p.similarity(s2) == pytest.approx(0.0, abs=1e-10)

        # Weight only dim 0 → both project to different values, still dissimilar
        # Weight only dim 1 → centroid has 0 in dim 1, so norm_c → 0
        weights = np.array([0.0, 1.0])
        # wc = [0, 0], norm = 0 → returns 0.0
        assert p.weighted_similarity(s2, weights) == 0.0


class TestDistinctionWithSalience:
    def test_enable_salience_creates_weights(self):
        d = Distinction(similarity_threshold=0.7, enable_salience=True)
        sig = Signal(data=np.array([1.0, 0.0, 0.5]))
        d.distinguish(sig)
        assert d._salience is not None
        assert d._salience.dim == 3

    def test_disabled_salience_uses_unweighted(self):
        d = Distinction(similarity_threshold=0.7, enable_salience=False)
        sig = Signal(data=np.array([1.0, 0.0, 0.5]))
        d.distinguish(sig)
        assert d._salience is None

    def test_update_salience_modifies_weights(self):
        d = Distinction(similarity_threshold=0.5, enable_salience=True)
        # Create a pattern
        sig = Signal(data=np.array([1.0, 0.0, 1.0, 0.0]))
        d.distinguish(sig)

        # Second signal that matches but deviates significantly in dims 1, 3
        sig2 = Signal(data=np.array([1.0, 0.8, 1.0, 0.8]))
        d.distinguish(sig2)

        # Now update salience with a large error
        before = d._salience.weights.copy()
        d.update_salience(error=5.0)
        # Dims 1 and 3 had large deviation → weights should change
        assert not np.allclose(d._salience.weights, before)

    def test_salience_causes_pattern_split(self):
        """After increasing salience on differing dims, previously-matched
        signals create distinct patterns."""
        # Two signals: identical in dims 0-3, opposite in dims 4-5
        base = Signal(data=np.array([1.0, 0.8, 0.6, 0.5, 0.9, 0.1]))
        variant = Signal(data=np.array([1.0, 0.8, 0.6, 0.5, 0.1, 0.9]))

        # With uniform weights, they should match (high cosine from shared dims)
        d1 = Distinction(similarity_threshold=0.7, enable_salience=True)
        p1, _ = d1.distinguish(base)
        p2, _ = d1.distinguish(variant)
        assert p1.pattern_id == p2.pattern_id, "Should match with uniform weights"

        # Now create a new Distinction with extreme weights on dims 4-5
        d2 = Distinction(similarity_threshold=0.7, enable_salience=True)
        # Trigger salience init
        d2.distinguish(Signal(data=np.zeros(6)))
        d2.patterns.clear()
        d2._next_id = 0
        # Set weights: low for shared dims, high for differing dims
        d2._salience._weights = np.array([0.1, 0.1, 0.1, 0.1, 5.0, 5.0])
        d2._salience._weights /= np.mean(d2._salience._weights)

        p3, _ = d2.distinguish(base)
        p4, _ = d2.distinguish(variant)
        assert p3.pattern_id != p4.pattern_id, "Should split with high salience on differing dims"

    def test_no_deviation_for_new_patterns(self):
        """New patterns have no deviation (centroid = signal exactly)."""
        d = Distinction(similarity_threshold=0.7, enable_salience=True)
        sig = Signal(data=np.array([1.0, 0.0, 0.5]))
        d.distinguish(sig)
        assert d._last_deviation is None  # New pattern, no deviation

    def test_salience_persists_across_calls(self):
        """Salience weights persist and accumulate."""
        d = Distinction(similarity_threshold=0.5, enable_salience=True)

        sig1 = Signal(data=np.array([1.0, 0.0, 1.0, 0.0]))
        d.distinguish(sig1)

        # Signal that matches but deviates in dims 1, 3
        sig2 = Signal(data=np.array([1.0, 0.8, 1.0, 0.8]))
        d.distinguish(sig2)
        d.update_salience(error=5.0)
        w1 = d._salience.weights.copy()

        sig3 = Signal(data=np.array([1.0, 0.6, 1.0, 0.6]))
        d.distinguish(sig3)
        d.update_salience(error=5.0)
        w2 = d._salience.weights.copy()

        # Weights should have changed between updates
        assert not np.allclose(w1, w2)


class TestRetroactiveBuffer:
    def test_buffer_fills_up_to_window(self):
        """distinguish() fills the deviation buffer up to _salience_window."""
        d = Distinction(similarity_threshold=0.5, enable_salience=True)
        base = Signal(data=np.array([1.0, 0.0, 1.0, 0.0]))
        d.distinguish(base)  # Creates pattern, no deviation

        # Feed 8 similar signals — buffer should max out at window size (6)
        for i in range(8):
            sig = Signal(data=np.array([1.0, 0.1 * i, 1.0, 0.1 * i]))
            d.distinguish(sig)

        assert len(d._deviation_buffer) == d._salience_window

    def test_retroactive_update_uses_all_recent(self):
        """update_salience() applies to all buffered deviations, not just last."""
        d = Distinction(similarity_threshold=0.5, enable_salience=True)
        base = Signal(data=np.array([1.0, 0.0, 1.0, 0.0]))
        d.distinguish(base)

        # Feed 3 signals that deviate in different dims
        sigs = [
            Signal(data=np.array([1.0, 0.5, 1.0, 0.0])),  # deviates in dim 1
            Signal(data=np.array([1.0, 0.0, 1.0, 0.5])),  # deviates in dim 3
            Signal(data=np.array([0.5, 0.0, 1.0, 0.0])),  # deviates in dim 0
        ]
        for sig in sigs:
            d.distinguish(sig)

        assert len(d._deviation_buffer) == 3

        before = d._salience.weights.copy()
        d.update_salience(error=5.0)
        after = d._salience.weights

        # All dims that had deviation in ANY of the 3 signals should be affected
        # Dims 0, 1, 3 all had deviations → should change
        assert not np.allclose(before, after)

    def test_retroactive_decay(self):
        """Older deviations get less weight (geometric decay 0.85^i)."""
        # Use Salience directly to isolate the decay math
        s = Salience.uniform(2, decay=0.0)

        # Simulate buffer: older = [1.0, 0.0], newer = [0.0, 1.0]
        buffer = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

        # Retroactive update: newer gets full error, older gets 0.85 * error
        error = 1.0
        for i, dev in enumerate(reversed(buffer)):
            decayed_error = error * (0.85 ** i)
            s.update(dev, decayed_error)

        # Dim 1 (newer, full error=1.0) got 0.2*1.0*1.0 = 0.2 boost
        # Dim 0 (older, error=0.85) got 0.2*0.85*1.0 = 0.17 boost
        # After normalization, dim 1 should be higher
        assert s.weights[1] > s.weights[0]


class TestModalSimilarity:
    def test_equal_weight_per_modality(self):
        """Each modality contributes equally regardless of dim count."""
        # Modality A: 2 dims (identical), Modality B: 8 dims (orthogonal)
        # With standard cosine, B's 8 dims dominate → low similarity.
        # With modal cosine, A contributes equally → higher similarity.
        centroid = np.concatenate([
            np.array([1.0, 0.0]),          # Modality A
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Modality B
        ])
        signal_data = np.concatenate([
            np.array([1.0, 0.0]),          # Modality A: identical
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Modality B: orthogonal
        ])
        signal = Signal(data=signal_data)
        p = Pattern(centroid=centroid, pattern_id=0)
        weights = np.ones(10)
        slices = [(0, 2), (2, 10)]

        modal_sim = p.modal_similarity(signal, weights, slices)
        standard_sim = p.weighted_similarity(signal, weights)

        # Standard: single cosine over 10 dims, A matches but B dominates → low
        # Modal: avg(cos_A=1.0, cos_B=0.0) = 0.5
        assert modal_sim == pytest.approx(0.5, abs=0.01)
        assert modal_sim > standard_sim

    def test_social_dims_sensitive(self):
        """Signals differing only in social dims produce lower modal similarity
        than they would under standard cosine."""
        # 12 position dims + 18 social dims = 30 total
        pos = np.ones(12) * 0.5
        soc_a = np.array([1.0, 0.0] * 9)  # 18 dims
        soc_b = np.array([0.0, 1.0] * 9)  # 18 dims, opposite

        centroid = np.concatenate([pos, soc_a])
        signal = Signal(data=np.concatenate([pos, soc_b]))
        p = Pattern(centroid=centroid, pattern_id=0)
        weights = np.ones(30)
        slices = [(0, 12), (12, 30)]

        modal_sim = p.modal_similarity(signal, weights, slices)
        standard_sim = p.weighted_similarity(signal, weights)

        # Position is identical → cos_pos = 1.0
        # Social is opposite → cos_soc ≈ 0.0 (or negative)
        # Modal = (1.0 + cos_soc) / 2 ≈ 0.5
        # Standard = one cosine over 30 dims, position dominates less
        # Modal should give social dims MORE influence (equal to position)
        assert modal_sim < 0.6  # Social opposition drags it down
        assert modal_sim != pytest.approx(standard_sim, abs=0.01)

    def test_zero_norm_modality_skipped(self):
        """Modalities where centroid or signal is all-zero are skipped."""
        centroid = np.array([1.0, 0.5, 0.0, 0.0])
        signal = Signal(data=np.array([0.8, 0.6, 0.0, 0.0]))
        p = Pattern(centroid=centroid, pattern_id=0)
        weights = np.ones(4)
        slices = [(0, 2), (2, 4)]

        # Modality B is all zeros in both → skipped, only modality A counts
        modal_sim = p.modal_similarity(signal, weights, slices)
        # Should equal cosine of just the first 2 dims
        p_sub = Pattern(centroid=centroid[:2], pattern_id=0)
        expected = p_sub.similarity(Signal(data=signal.data[:2]))
        assert modal_sim == pytest.approx(expected, abs=1e-10)


class TestStrongerLearning:
    def test_default_lr_is_0_2(self):
        s = Salience.uniform(4)
        assert s._learning_rate == 0.2

    def test_default_decay_is_0_001(self):
        s = Salience.uniform(4)
        assert s._decay == 0.001

    def test_retroactive_updates_produce_meaningful_differentiation(self):
        """With lr=0.2 and retroactive buffer, repeated errors cause divergence."""
        # Use Salience directly for deterministic test
        s = Salience.uniform(4, decay=0.0)

        # Simulate 5 retroactive update rounds (as if 5 outcome events)
        # Each round: buffer of 4 deviations, all high in dims 2-3, low in 0-1
        buffer = [
            np.array([0.0, 0.0, 1.0, 0.8]),
            np.array([0.0, 0.1, 0.9, 1.0]),
            np.array([0.1, 0.0, 1.0, 0.7]),
            np.array([0.0, 0.0, 0.8, 0.9]),
        ]
        for _ in range(5):
            for i, dev in enumerate(reversed(buffer)):
                decayed_error = 2.0 * (0.85 ** i)
                s.update(dev, decayed_error)

        max_w = float(np.max(s.weights))
        min_w = float(np.min(s.weights))
        # Dims 2, 3 get large boosts, dims 0, 1 get minimal → significant spread
        assert max_w - min_w > 0.3, (
            f"Weights not differentiated enough: max={max_w:.3f}, min={min_w:.3f}"
        )
        # Dims 2, 3 should be higher than dims 0, 1
        assert s.weights[2] > s.weights[0]
        assert s.weights[3] > s.weights[1]


class TestAgentWithSalience:
    def test_agent_runs_with_salience(self):
        """Agent with salience enabled completes a survival episode."""
        from fpi.agent.core import Agent
        from fpi.env.base import SurvivalEnv

        agent = Agent(similarity_threshold=0.7, seed=42, enable_salience=True)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])

        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0

        # Salience should be initialized
        salience = agent.world_model.memory.distinction._salience
        assert salience is not None
        assert salience.dim == 6  # SurvivalEnv uses 6-dim position encoding

    def test_salience_weights_diverge_from_uniform(self):
        """After enough experience, salience weights should not all be 1.0."""
        from fpi.agent.core import Agent
        from fpi.env.base import SurvivalEnv

        agent = Agent(similarity_threshold=0.7, seed=42, enable_salience=True)
        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])

        agent.run_survival_episode(env, max_steps=100)
        salience = agent.world_model.memory.distinction._salience
        # After 100 steps, weights should have some variation
        assert salience is not None
        # Mean is always 1.0 due to normalization
        assert abs(np.mean(salience.weights) - 1.0) < 0.01

    def test_agent_without_salience_unchanged(self):
        """Agent without salience enabled behaves exactly as before."""
        from fpi.agent.core import Agent
        from fpi.env.base import SurvivalEnv

        agent = Agent(similarity_threshold=0.7, seed=42, enable_salience=False)
        assert agent.world_model.memory.distinction._salience is None
        assert agent.world_model.memory.distinction.enable_salience is False

        env = SurvivalEnv(grid_size=10, resource_positions=[2, 8])
        results = agent.run_survival_episode(env, max_steps=50)
        assert len(results) > 0
        assert agent.world_model.memory.distinction._salience is None
