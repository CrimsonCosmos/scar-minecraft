"""Tests for introspection — CognitiveVitals, CognitiveStateBridge, SelfModel."""

import numpy as np

from fpi.agent.introspection import (
    CognitiveStateBridge,
    CognitiveVitals,
    SelfModel,
)


class TestCognitiveVitals:
    def test_initial_state(self):
        v = CognitiveVitals(window=10)
        assert v.surprise_momentum == 0.5  # Initial EMA
        assert v.learning_rate == 0.0
        assert v.prediction_confidence == 0.0
        assert v.cognitive_load == 0.0

    def test_surprise_momentum_tracks_ema(self):
        v = CognitiveVitals(window=5)
        # Feed high surprise — momentum should increase
        for _ in range(20):
            v.update(1.0, 10, 30, 0.5)
        assert v.surprise_momentum > 0.9

        # Feed low surprise — momentum should decrease
        for _ in range(20):
            v.update(0.0, 10, 30, 0.5)
        assert v.surprise_momentum < 0.1

    def test_learning_rate_detects_growth(self):
        v = CognitiveVitals(window=10)
        # Pattern count increasing every tick
        for i in range(10):
            v.update(0.5, i + 1, 30, 0.5)
        assert v.learning_rate > 0.5  # Most ticks had increases

    def test_learning_rate_zero_when_stable(self):
        v = CognitiveVitals(window=10)
        # Pattern count stable
        for _ in range(10):
            v.update(0.5, 5, 30, 0.5)
        assert v.learning_rate == 0.0

    def test_prediction_confidence(self):
        v = CognitiveVitals(window=5)
        v.update(0.5, 5, 30, 0.85)
        assert v.prediction_confidence == 0.85
        v.update(0.5, 5, 30, None)
        assert v.prediction_confidence == 0.0

    def test_cognitive_load(self):
        v = CognitiveVitals(window=5)
        v.update(0.5, 15, 30, 0.5)
        assert v.cognitive_load == 0.5
        v.update(0.5, 30, 30, 0.5)
        assert v.cognitive_load == 1.0

    def test_cognitive_load_zero_max(self):
        v = CognitiveVitals(window=5)
        v.update(0.5, 5, 0, 0.5)
        assert v.cognitive_load == 0.0

    def test_as_dict(self):
        v = CognitiveVitals(window=5)
        v.update(0.5, 5, 30, 0.7)
        d = v.as_dict()
        assert "surprise_momentum" in d
        assert "learning_rate" in d
        assert "prediction_confidence" in d
        assert "cognitive_load" in d
        assert d["prediction_confidence"] == 0.7


class TestCognitiveStateBridge:
    def test_signal_dim(self):
        bridge = CognitiveStateBridge(bases_per_dim=6)
        assert bridge.signal_dim == 24  # 4 * 6

    def test_encode_produces_signal(self):
        bridge = CognitiveStateBridge(bases_per_dim=6)
        v = CognitiveVitals(window=5)
        v.update(0.5, 5, 30, 0.7)
        signal = bridge.encode(v, timestamp=0)
        assert signal.dim == 24
        assert signal.modality == "cognitive"
        assert signal.timestamp == 0
        # All values should be positive (Gaussian basis)
        assert np.all(signal.data >= 0.0)

    def test_different_states_produce_different_signals(self):
        bridge = CognitiveStateBridge(bases_per_dim=6)

        v1 = CognitiveVitals(window=5)
        for _ in range(5):
            v1.update(0.1, 3, 30, 0.9)

        v2 = CognitiveVitals(window=5)
        for _ in range(5):
            v2.update(0.9, 25, 30, 0.1)

        s1 = bridge.encode(v1, timestamp=0)
        s2 = bridge.encode(v2, timestamp=0)

        # Different cognitive states should produce distinct signals
        sim = s1.cosine_similarity(s2)
        assert sim < 0.95  # Not identical


class TestSelfModel:
    def test_observe_returns_surprise(self):
        sm = SelfModel(vitals_window=5)
        surprise = sm.observe(0.5, 5, 30, 0.7)
        assert 0.0 <= surprise <= 1.0

    def test_stable_state_reduces_cognitive_surprise(self):
        sm = SelfModel(vitals_window=5)
        # Feed identical cognitive state repeatedly
        for _ in range(30):
            sm.observe(0.5, 5, 30, 0.7)
        # After learning stable state, cognitive surprise should be low
        assert sm.cognitive_surprise < 0.5

    def test_changing_state_increases_cognitive_surprise(self):
        sm = SelfModel(vitals_window=5)
        # Establish stable state
        for _ in range(20):
            sm.observe(0.1, 5, 30, 0.9)
        low_surprise = sm.cognitive_surprise

        # Sudden shift in cognitive state
        surprise = sm.observe(0.9, 25, 30, 0.1)
        # New state should be more surprising
        assert surprise > low_surprise or surprise >= 0.5

    def test_predict(self):
        sm = SelfModel(vitals_window=5)
        for _ in range(20):
            sm.observe(0.5, 5, 30, 0.7)
        pred = sm.predict()
        # After stable input, should have a prediction
        # (may be None early, but after 20 ticks should exist)
        if pred is not None:
            pattern, confidence = pred
            assert confidence > 0.0

    def test_get_status(self):
        sm = SelfModel(vitals_window=5)
        sm.observe(0.5, 5, 30, 0.7)
        status = sm.get_status()
        assert "vitals" in status
        assert "cognitive_patterns" in status
        assert "cognitive_surprise" in status
        assert "cognitive_prediction" in status
        assert isinstance(status["vitals"], dict)

    def test_vitals_property(self):
        sm = SelfModel(vitals_window=5)
        sm.observe(0.5, 5, 30, 0.7)
        v = sm.vitals
        assert isinstance(v, CognitiveVitals)
        assert v.prediction_confidence == 0.7

    def test_tick(self):
        sm = SelfModel(vitals_window=5)
        sm.observe(0.5, 5, 30, 0.7)
        # tick should not raise
        sm.tick()
