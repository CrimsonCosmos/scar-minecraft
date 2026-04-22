"""World Model — the predictive engine.

The world model wraps associative memory to provide higher-level operations:
predict the next signal, measure surprise, and update the model based on
prediction error. This is where "understanding" lives.

In Phase 2, the world model also learns action-conditional predictions:
"from state S, if I take action A, I expect to end up in state S' with
vitality change V." This is what enables volition — the agent can simulate
different futures and choose.
"""

from __future__ import annotations

import numpy as np

from ..primitives.signal import Signal
from ..primitives.pattern import Pattern
from ..memory.associative import AssociativeMemory


class WorldModel:
    """Predictive model that learns temporal structure in signal streams.

    The world model:
    1. Observes signals and categorizes them into patterns.
    2. Builds associations between consecutive patterns.
    3. Uses those associations to predict what comes next.
    4. Measures surprise when predictions are wrong.
    5. Updates itself based on surprise.
    6. Learns action-conditional predictions: (state, action) → (outcome, vitality).

    Attributes:
        memory: The underlying associative memory.
        current_pattern: The pattern matched by the most recent observation.
        prev_pattern: The pattern before the most recent observation.
        last_prediction: The pattern predicted before the latest observation.
        last_surprise: Surprise value from the most recent observation.
        total_surprise: Cumulative surprise across the model's lifetime.
        observation_count: Total observations processed.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_patterns: int | None = None,
        max_associations: int | None = None,
        association_decay_rate: float = 0.0,
        maintenance_cost_per_pattern: float = 0.0,
        maintenance_cost_per_association: float = 0.0,
        enable_salience: bool = False,
        modality_slices: list[tuple[int, int]] | None = None,
        enable_complexity_cost: bool = False,
        complexity_cost_rate: float = 0.0005,
        enable_compositional: bool = False,
        patterns_per_modality: int = 15,
        modality_thresholds: list[float] | None = None,
        composite_similarity_threshold: float = 0.5,
    ) -> None:
        self.memory = AssociativeMemory(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
            max_associations=max_associations,
            association_decay_rate=association_decay_rate,
            maintenance_cost_per_pattern=maintenance_cost_per_pattern,
            maintenance_cost_per_association=maintenance_cost_per_association,
            enable_salience=enable_salience,
            modality_slices=modality_slices,
            enable_compositional=enable_compositional,
            patterns_per_modality=patterns_per_modality,
            modality_thresholds=modality_thresholds,
        )
        self._enable_complexity_cost = enable_complexity_cost
        self._complexity_cost_rate = complexity_cost_rate
        self.current_pattern: Pattern | None = None
        self.prev_pattern: Pattern | None = None
        self.last_prediction: tuple[Pattern, float] | None = None
        self.last_surprise: float = 1.0
        self.total_surprise: float = 0.0
        self.observation_count: int = 0

        # Action-conditional predictions:
        # (pattern_id, action) → {target_pattern_id: strength}
        self._action_transitions: dict[tuple[int, int], dict[int, float]] = {}
        # (pattern_id, action) → average vitality delta
        self._action_vitality: dict[tuple[int, int], float] = {}

        # Composite similarity fallback: when exact lookup fails, generalize
        # from the most similar composite that has data for this action.
        # Same principle as Distinction (similarity-based matching) applied
        # one level up to composite patterns.
        self._composite_sim_threshold = composite_similarity_threshold

    def _find_similar_transitions(
        self, pattern_id: int, action: int,
    ) -> dict[int, float] | None:
        """Fallback: find transitions from the most similar composite pattern.

        When the exact (pattern_id, action) has no recorded transitions, search
        for a composite with a similar centroid that does. Returns that composite's
        transitions discounted by similarity, or None if nothing is close enough.

        This is the same principle as Distinction's similarity-based matching,
        applied one level up: generalize from known exemplars to novel situations
        proportional to their similarity (Shepard's universal law).
        """
        # Find the target pattern's centroid
        target_centroid = None
        for p in self.memory.distinction.patterns:
            if p.pattern_id == pattern_id:
                target_centroid = p.centroid
                break
        if target_centroid is None:
            return None

        target_norm = np.linalg.norm(target_centroid)
        if target_norm == 0.0:
            return None

        best_sim = -1.0
        best_targets: dict[int, float] | None = None

        for p in self.memory.distinction.patterns:
            if p.pattern_id == pattern_id:
                continue
            key = (p.pattern_id, action)
            if key not in self._action_transitions:
                continue
            p_norm = np.linalg.norm(p.centroid)
            if p_norm == 0.0:
                continue
            sim = float(np.dot(target_centroid, p.centroid) / (target_norm * p_norm))
            if sim > best_sim:
                best_sim = sim
                best_targets = self._action_transitions[key]

        if best_targets is None or best_sim < self._composite_sim_threshold:
            return None

        # Discount transition strengths by similarity
        return {pid: strength * best_sim for pid, strength in best_targets.items()}

    def _find_similar_vitality(
        self, pattern_id: int, action: int,
    ) -> float | None:
        """Fallback: find vitality prediction from the most similar composite."""
        target_centroid = None
        for p in self.memory.distinction.patterns:
            if p.pattern_id == pattern_id:
                target_centroid = p.centroid
                break
        if target_centroid is None:
            return None

        target_norm = np.linalg.norm(target_centroid)
        if target_norm == 0.0:
            return None

        best_sim = -1.0
        best_vitality: float | None = None

        for p in self.memory.distinction.patterns:
            if p.pattern_id == pattern_id:
                continue
            key = (p.pattern_id, action)
            if key not in self._action_vitality:
                continue
            p_norm = np.linalg.norm(p.centroid)
            if p_norm == 0.0:
                continue
            sim = float(np.dot(target_centroid, p.centroid) / (target_norm * p_norm))
            if sim > best_sim:
                best_sim = sim
                best_vitality = self._action_vitality[key]

        if best_vitality is None or best_sim < self._composite_sim_threshold:
            return None

        return best_vitality * best_sim

    def observe(self, signal: Signal, last_action: int | None = None) -> float:
        """Process a new signal. Returns the surprise value.

        This is the main entry point — one call per timestep. It:
        1. Categorizes the signal into a pattern
        2. Compares reality (this pattern) to the last prediction
        3. Updates associations based on whether the prediction was right
        4. Generates a new prediction for the next timestep

        Args:
            signal: The observation to process.
            last_action: Action taken before this observation. When provided,
                uses action-conditional prediction for surprise instead of
                unconditional temporal associations. This is critical for
                controllable environments where the agent's action determines
                what comes next.
        """
        # Track previous pattern before updating
        self.prev_pattern = self.current_pattern

        # Categorize the incoming signal
        new_pattern = self.memory.store(signal)

        # Use action-conditional prediction if available
        if last_action is not None and self.prev_pattern is not None:
            action_pred = self._get_action_prediction(
                self.prev_pattern.pattern_id, last_action
            )
            if action_pred is not None:
                self.last_prediction = action_pred

        # Measure surprise against prediction
        surprise = self._compute_surprise(new_pattern)
        self.last_surprise = surprise
        self.total_surprise += surprise
        self.observation_count += 1

        # Update associations based on prediction accuracy
        if self.prev_pattern is not None:
            if surprise < 0.5:
                # Prediction was roughly right — reinforce
                self.memory.associate(
                    self.prev_pattern.pattern_id,
                    new_pattern.pattern_id,
                    reinforce=True,
                )
            else:
                # Prediction was wrong — weaken wrong association, reinforce correct one
                if self.last_prediction is not None:
                    self.memory.weaken(
                        self.prev_pattern.pattern_id,
                        self.last_prediction[0].pattern_id,
                    )
                self.memory.associate(
                    self.prev_pattern.pattern_id,
                    new_pattern.pattern_id,
                    reinforce=True,
                )

        # Generate prediction for the next timestep
        self.last_prediction = self.memory.predict_next(new_pattern.pattern_id)
        self.current_pattern = new_pattern

        return surprise

    def _get_action_prediction(
        self, pattern_id: int, action: int,
    ) -> tuple[Pattern, float] | None:
        """Get action-conditional prediction for surprise computation."""
        key = (pattern_id, action)
        targets = self._action_transitions.get(key)
        if not targets:
            targets = self._find_similar_transitions(pattern_id, action)
        if not targets:
            return None

        best_pid = max(targets, key=targets.get)  # type: ignore[arg-type]
        best_strength = targets[best_pid]

        for p in self.memory.distinction.patterns:
            if p.pattern_id == best_pid:
                return p, best_strength
        return None

    def record_action_outcome(
        self, action: int, resulting_pattern: Pattern | None, vitality_delta: float,
    ) -> None:
        """Record what happened when we took an action from the previous state.

        Learns: "from pattern P, taking action A, I ended up in pattern Q
        with vitality change V."
        """
        if self.prev_pattern is None or resulting_pattern is None:
            return

        key = (self.prev_pattern.pattern_id, action)

        # Update transition strengths
        targets = self._action_transitions.setdefault(key, {})
        pid = resulting_pattern.pattern_id
        targets[pid] = targets.get(pid, 0.0) + 0.1 * (1.0 - targets.get(pid, 0.0))

        # Update vitality expectation (exponential moving average)
        if key in self._action_vitality:
            self._action_vitality[key] = (
                0.7 * self._action_vitality[key] + 0.3 * vitality_delta
            )
        else:
            self._action_vitality[key] = vitality_delta

    def predict_action_outcome(self, action: int) -> tuple[Pattern, float] | None:
        """Predict the outcome of taking an action from the current state.

        Returns (predicted_pattern, confidence) or None if no data.
        """
        if self.current_pattern is None:
            return None
        key = (self.current_pattern.pattern_id, action)
        targets = self._action_transitions.get(key)
        if not targets:
            targets = self._find_similar_transitions(
                self.current_pattern.pattern_id, action,
            )
        if not targets:
            return None

        best_pid = max(targets, key=targets.get)  # type: ignore[arg-type]
        best_strength = targets[best_pid]

        for p in self.memory.distinction.patterns:
            if p.pattern_id == best_pid:
                return p, best_strength
        return None

    def predict_action_vitality(self, action: int) -> float | None:
        """Predict the expected vitality change from taking an action.

        Returns expected vitality delta, or None if no data.
        """
        if self.current_pattern is None:
            return None
        key = (self.current_pattern.pattern_id, action)
        result = self._action_vitality.get(key)
        if result is None:
            result = self._find_similar_vitality(
                self.current_pattern.pattern_id, action,
            )
        return result

    def predict_from(
        self, pattern_id: int, action: int,
    ) -> tuple[Pattern, float] | None:
        """Predict outcome from any pattern, not just current.

        Like predict_action_outcome but takes an explicit pattern_id.
        Stateless: does not read or modify current_pattern.
        """
        key = (pattern_id, action)
        targets = self._action_transitions.get(key)
        if not targets:
            targets = self._find_similar_transitions(pattern_id, action)
        if not targets:
            return None

        best_pid = max(targets, key=targets.get)  # type: ignore[arg-type]
        best_strength = targets[best_pid]

        for p in self.memory.distinction.patterns:
            if p.pattern_id == best_pid:
                return p, best_strength
        return None

    def predict_vitality_from(
        self, pattern_id: int, action: int,
    ) -> float | None:
        """Predict vitality delta from any pattern, not just current.

        Like predict_action_vitality but takes an explicit pattern_id.
        """
        key = (pattern_id, action)
        result = self._action_vitality.get(key)
        if result is None:
            result = self._find_similar_vitality(pattern_id, action)
        return result

    def simulate_trajectory(
        self, start_pattern_id: int, actions: list[int],
    ) -> tuple[float, list[int]]:
        """Simulate a sequence of actions from a starting pattern.

        Returns (total_predicted_vitality, list_of_pattern_ids).
        Stateless: does NOT mutate current_pattern or any model state.
        This is forward simulation — the agent hallucinating possible futures.
        """
        total_vitality = 0.0
        pattern_ids = [start_pattern_id]
        current_pid = start_pattern_id

        for action in actions:
            vit = self.predict_vitality_from(current_pid, action)
            if vit is not None:
                total_vitality += vit

            pred = self.predict_from(current_pid, action)
            if pred is not None:
                current_pid = pred[0].pattern_id
            pattern_ids.append(current_pid)

        return total_vitality, pattern_ids

    def update_salience(
        self, actual_delta: float, action_taken: int | None,
    ) -> None:
        """Compute error signal and update salience weights.

        Uses vitality prediction error when available (primary), or
        surprise * |actual_delta| as a bootstrap fallback.
        """
        if not self.memory.distinction.enable_salience:
            return

        # Primary: vitality prediction error from prev state
        if action_taken is not None and self.prev_pattern is not None:
            key = (self.prev_pattern.pattern_id, action_taken)
            predicted = self._action_vitality.get(key)
            if predicted is not None:
                error = abs(actual_delta - predicted)
            else:
                # Fallback: surprise weighted by outcome magnitude
                error = self.last_surprise * abs(actual_delta) if actual_delta != 0 else 0.0
        else:
            error = self.last_surprise * abs(actual_delta) if actual_delta != 0 else 0.0

        self.memory.update_salience(error)

    def predict(self) -> tuple[Pattern, float] | None:
        """Get the current prediction for the next signal.

        Returns (predicted_pattern, confidence) or None if no prediction.
        """
        return self.last_prediction

    def surprise(self, predicted_pattern: Pattern | None, actual_signal: Signal) -> float:
        """Compute surprise: how different is reality from prediction?

        Returns a value in [0, 1]:
        - 0.0 = perfect prediction (no surprise)
        - 1.0 = maximum surprise (completely unexpected)
        """
        if predicted_pattern is None:
            return 1.0  # No prediction = maximum surprise

        sim = predicted_pattern.similarity(actual_signal)
        return 1.0 - max(0.0, sim)

    def _compute_surprise(self, actual_pattern: Pattern) -> float:
        """Internal surprise computation using current prediction state."""
        if self.last_prediction is None:
            return 1.0

        predicted_pattern, confidence = self.last_prediction
        if predicted_pattern.pattern_id == actual_pattern.pattern_id:
            return 0.0  # Exact pattern match

        # Compute based on centroid similarity
        norm_p = np.linalg.norm(predicted_pattern.centroid)
        norm_a = np.linalg.norm(actual_pattern.centroid)
        if norm_p == 0.0 or norm_a == 0.0:
            return 1.0
        sim = float(np.dot(predicted_pattern.centroid, actual_pattern.centroid) / (norm_p * norm_a))
        return 1.0 - max(0.0, sim)

    @property
    def average_surprise(self) -> float:
        """Average surprise per observation."""
        if self.observation_count == 0:
            return 1.0
        return self.total_surprise / self.observation_count

    def reset_stats(self) -> None:
        """Reset surprise tracking (e.g. between episodes)."""
        self.total_surprise = 0.0
        self.observation_count = 0

    def complexity_cost(self) -> float:
        """Model complexity = sum of pattern specificities.

        FEP: agent pays for model complexity, not just prediction error.
        Specific, well-worn patterns are expensive. This creates Occam's
        razor pressure — prefer simpler models that still predict well.
        """
        if not self._enable_complexity_cost:
            return 0.0
        total = sum(
            p.specificity() for p in self.memory.distinction.patterns
        )
        return total * self._complexity_cost_rate

    def tick(self) -> float:
        """Advance one tick: decay, prune, compute maintenance cost.

        Returns total maintenance cost (energy drain for having a brain).
        Also cleans up action-transition data for evicted patterns.
        """
        cost = self.memory.tick()
        cost += self.complexity_cost()

        # Clean action data for patterns that no longer exist
        live_ids = {p.pattern_id for p in self.memory.distinction.patterns}
        dead_keys = [
            k for k in self._action_transitions
            if k[0] not in live_ids
        ]
        for k in dead_keys:
            del self._action_transitions[k]
            self._action_vitality.pop(k, None)

        # Also clean target references within surviving transitions
        for key, targets in list(self._action_transitions.items()):
            dead_targets = [pid for pid in targets if pid not in live_ids]
            for pid in dead_targets:
                del targets[pid]
            if not targets:
                del self._action_transitions[key]
                self._action_vitality.pop(key, None)

        return cost
