"""Self-model — the agent observes its own cognitive state.

The agent's own cognitive process becomes another data stream, processed
by the same primitives (Signal, Distinction, Association, Prediction).
This is self-similar intelligence: the agent watches itself the same way
it watches the world.

Three components:
- CognitiveVitals: derives 4 interoceptive metrics from raw agent state.
- CognitiveStateBridge: encodes vitals as a Signal via Gaussian basis.
- SelfModel: a WorldModel observing the cognitive stream.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..primitives.pattern import Pattern
from ..primitives.signal import Signal
from ..primitives.valence import Valence
from ..world_model.model import WorldModel


class CognitiveVitals:
    """Derived interoceptive signals from raw agent state.

    Computes 4 metrics each tick:
    - surprise_momentum: EMA of surprise — is surprise trending up or down?
    - learning_rate: fraction of recent ticks with new pattern acquisition.
    - prediction_confidence: strength of current prediction.
    - cognitive_load: active patterns / max_patterns.

    Args:
        window: Number of recent ticks to consider.
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._alpha = 2.0 / (window + 1)

        # Ring buffers for windowed computations
        self._surprise_history: deque[float] = deque(maxlen=window)
        self._pattern_count_history: deque[int] = deque(maxlen=window)

        # EMA state
        self._surprise_ema: float = 0.5
        self._prediction_conf: float = 0.0
        self._cognitive_load_val: float = 0.0
        self._tick_count: int = 0

    def update(
        self,
        surprise: float,
        pattern_count: int,
        max_patterns: int,
        prediction_confidence: float | None,
    ) -> None:
        """Update all vitals from the latest tick's raw state."""
        self._surprise_history.append(surprise)
        self._pattern_count_history.append(pattern_count)

        # Surprise momentum: EMA
        self._surprise_ema = (
            self._alpha * surprise + (1.0 - self._alpha) * self._surprise_ema
        )

        # Prediction confidence
        self._prediction_conf = prediction_confidence if prediction_confidence is not None else 0.0

        # Cognitive load
        if max_patterns > 0:
            self._cognitive_load_val = min(1.0, pattern_count / max_patterns)
        else:
            self._cognitive_load_val = 0.0

        self._tick_count += 1

    @property
    def surprise_momentum(self) -> float:
        """EMA of surprise — trending up means world getting chaotic."""
        return self._surprise_ema

    @property
    def learning_rate(self) -> float:
        """Fraction of recent ticks where pattern_count increased."""
        counts = list(self._pattern_count_history)
        if len(counts) < 2:
            return 0.0
        increases = sum(
            1 for i in range(1, len(counts)) if counts[i] > counts[i - 1]
        )
        return increases / (len(counts) - 1)

    @property
    def prediction_confidence(self) -> float:
        """Strength of the current prediction."""
        return self._prediction_conf

    @property
    def cognitive_load(self) -> float:
        """Active patterns / max_patterns."""
        return self._cognitive_load_val

    def as_dict(self) -> dict[str, float]:
        """All vitals as a dict."""
        return {
            "surprise_momentum": self.surprise_momentum,
            "learning_rate": self.learning_rate,
            "prediction_confidence": self.prediction_confidence,
            "cognitive_load": self.cognitive_load,
        }


class CognitiveStateBridge:
    """Gaussian basis encoding of cognitive metrics into a Signal.

    Analog of WatcherBridge but for self-observation.
    4 metrics * bases_per_dim = signal dimensionality.

    Args:
        bases_per_dim: Gaussian bases per cognitive metric.
    """

    def __init__(self, bases_per_dim: int = 6) -> None:
        self._bases_per_dim = bases_per_dim
        self._centers = np.linspace(0.0, 1.0, bases_per_dim)
        self._sigma = max(0.01, 1.0 / bases_per_dim)

    @property
    def signal_dim(self) -> int:
        """Total signal dimensionality: 4 metrics * bases_per_dim."""
        return 4 * self._bases_per_dim

    def encode(self, vitals: CognitiveVitals, timestamp: int) -> Signal:
        """Encode CognitiveVitals as a Signal."""
        metrics = [
            np.clip(vitals.surprise_momentum, 0.0, 1.0),
            np.clip(vitals.learning_rate, 0.0, 1.0),
            np.clip(vitals.prediction_confidence, 0.0, 1.0),
            np.clip(vitals.cognitive_load, 0.0, 1.0),
        ]
        parts: list[np.ndarray] = []
        for val in metrics:
            basis = np.exp(
                -((val - self._centers) ** 2) / (2 * self._sigma**2)
            )
            parts.append(basis)

        data = np.concatenate(parts).astype(np.float64)
        return Signal(data=data, timestamp=timestamp, modality="cognitive")


class SelfModel:
    """A WorldModel observing the agent's own cognitive stream.

    Composes CognitiveVitals + CognitiveStateBridge + WorldModel.

    Data flow:
        Agent observes world -> surprise, pattern_count, prediction
                             |
                    CognitiveVitals.update()
                             |
                    CognitiveStateBridge.encode() -> Signal
                             |
                    WorldModel.observe(signal) -> cognitive_surprise

    Args:
        vitals_window: Window size for CognitiveVitals.
        bases_per_dim: Gaussian bases per cognitive metric.
        similarity_threshold: For the internal WorldModel.
        max_patterns: Capacity of the internal WorldModel.
        max_associations: Association limit.
    """

    def __init__(
        self,
        vitals_window: int = 20,
        bases_per_dim: int = 6,
        similarity_threshold: float = 0.7,
        max_patterns: int = 15,
        max_associations: int = 40,
    ) -> None:
        self._vitals = CognitiveVitals(window=vitals_window)
        self._bridge = CognitiveStateBridge(bases_per_dim=bases_per_dim)
        self._world_model = WorldModel(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
            max_associations=max_associations,
        )
        # Cognitive valence: learned evaluation of cognitive states.
        # "Being confused tends to precede bad outcomes" = negative valence.
        # "Being confident tends to precede good outcomes" = positive valence.
        self._valence = Valence(learning_rate=0.2)
        self._cognitive_surprise: float = 1.0
        self._tick = 0

    def observe(
        self,
        surprise: float,
        pattern_count: int,
        max_patterns: int,
        prediction_confidence: float | None,
        actual_delta: float = 0.0,
    ) -> float:
        """Update cognitive vitals, encode, feed to internal WorldModel.

        Returns cognitive surprise — how unexpected the agent's OWN STATE is.

        The actual_delta from the environment is used to update cognitive
        valence: the agent learns which cognitive states tend to precede
        good or bad outcomes. This replaces hardcoded modulations with
        learned ones.
        """
        self._vitals.update(surprise, pattern_count, max_patterns, prediction_confidence)
        signal = self._bridge.encode(self._vitals, timestamp=self._tick)
        self._cognitive_surprise = self._world_model.observe(signal)

        # Update cognitive valence: "this cognitive state correlates with this outcome"
        if self._world_model.current_pattern is not None:
            self._valence.update(
                self._world_model.current_pattern.pattern_id,
                actual_delta * 0.5,  # Weaker than direct (it's metacognitive)
            )

        self._tick += 1
        return self._cognitive_surprise

    def predict(self) -> tuple[Pattern, float] | None:
        """What cognitive state does the agent expect next?"""
        return self._world_model.predict()

    @property
    def cognitive_surprise(self) -> float:
        """How unexpected was the agent's most recent cognitive state?"""
        return self._cognitive_surprise

    @property
    def cognitive_valence(self) -> float:
        """Learned valence of the current cognitive state.

        Positive = this cognitive mode tends to precede good outcomes (exploit).
        Negative = this cognitive mode tends to precede bad outcomes (explore).
        """
        if self._world_model.current_pattern is None:
            return 0.0
        return self._valence.get(self._world_model.current_pattern.pattern_id)

    @property
    def vitals(self) -> CognitiveVitals:
        """Access the underlying CognitiveVitals."""
        return self._vitals

    def get_status(self) -> dict:
        """Vitals + cognitive pattern count + cognitive surprise + prediction."""
        pred = self._world_model.predict()
        return {
            "vitals": self._vitals.as_dict(),
            "cognitive_patterns": len(self._world_model.memory.distinction.patterns),
            "cognitive_surprise": self._cognitive_surprise,
            "cognitive_prediction": pred[0].pattern_id if pred else None,
            "cognitive_prediction_confidence": pred[1] if pred else None,
        }

    def tick(self) -> None:
        """Advance the internal WorldModel one tick."""
        self._world_model.tick()
