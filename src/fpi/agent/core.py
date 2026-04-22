"""Agent — the core sense-predict-act loop.

The Agent is the central coordinator. Each tick it:
1. Senses (receives a signal from the environment)
2. Distinguishes (matches the signal to a pattern)
3. Predicts (what pattern comes next?)
4. Compares (prediction vs. reality = surprise)
5. Updates (strengthen/weaken associations)
6. Acts (selects action based on predicted vitality outcomes)

In Phase 2, the agent has VITALITY — finite energy under entropy — and
must ACT to survive. Action selection is driven by internal state (vitality)
and learned valence (which patterns correlate with energy gain/loss).
This is volition: action from within, not from external prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING

from ..primitives.affect import AffectState, AffectStream
from ..primitives.attention import AttentionGate
from ..primitives.signal import Signal
from ..memory.working_memory import WorkingMemory
from ..memory.predictive_hierarchy import PredictiveHierarchy
from ..memory.episodic import Episode, EpisodicMemory
from ..memory.sequence import SequenceMemory
from ..memory.temporal import TemporalHierarchy
from .options import OptionDiscovery, OptionExecutor
from .internal_society import InternalSociety, ActionContext
from ..primitives.vitality import Vitality
from ..primitives.valence import Valence
from ..world_model.model import WorldModel
from ..env.base import Environment, SurvivalEnv

if TYPE_CHECKING:
    from .introspection import SelfModel


@dataclass
class StepResult:
    """The outcome of a single agent step."""

    observation: Signal
    surprise: float
    predicted_correctly: bool
    tick: int
    action: int | None = None
    vitality: float = 1.0
    vitality_delta: float = 0.0


class Agent:
    """The sense-predict-act loop incarnated.

    The agent has:
    - A world model that learns to predict signals and action outcomes.
    - Vitality: finite energy that depletes under entropy.
    - Valence: learned mapping from patterns to vitality-change correlation.

    Volition emerges from: vitality (why act) + valence (what's good/bad) +
    world model (what will happen if I do X) → action selection.

    Attributes:
        world_model: The predictive engine that learns temporal structure.
        vitality: Finite energy — the ground of all motivation.
        valence: Learned pattern evaluations — what correlates with survival.
        history: Log of per-step results.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        seed: int | None = None,
        exploration_base: float = 0.15,
        valence_learning_rate: float = 0.3,
        max_patterns: int | None = None,
        max_associations: int | None = None,
        association_decay_rate: float = 0.0,
        maintenance_cost_per_pattern: float = 0.0,
        maintenance_cost_per_association: float = 0.0,
        self_model: SelfModel | None = None,
        enable_salience: bool = False,
        modality_slices: list[tuple[int, int]] | None = None,
        enable_complexity_cost: bool = False,
        complexity_cost_rate: float = 0.0005,
        enable_affect: bool = False,
        enable_attention: bool = False,
        enable_working_memory: bool = False,
        enable_options: bool = False,
        internal_society: InternalSociety | None = None,
        enable_predictive_hierarchy: bool = False,
        ph_num_levels: int = 3,
        ph_surprise_weight: float = 0.3,
        enable_episodic_memory: bool = False,
        episodic_capacity: int = 50,
        episodic_surprise_threshold: float = 0.7,
        enable_sequence_memory: bool = False,
        sequence_window: int = 3,
        temporal_scales: tuple[int, ...] | None = None,
        th_surprise_weight: float = 0.2,
        enable_compositional: bool = False,
        patterns_per_modality: int = 15,
        modality_thresholds: list[float] | None = None,
        composite_similarity_threshold: float = 0.5,
        enable_lookahead: bool = False,
        lookahead_depth: int = 3,
        lookahead_discount: float = 0.9,
        enable_eligibility_traces: bool = False,
        trace_decay: float = 0.8,
        discount_factor: float = 0.9,
        enable_abstraction: bool = False,
    ) -> None:
        self.world_model = WorldModel(
            similarity_threshold=similarity_threshold,
            max_patterns=max_patterns,
            max_associations=max_associations,
            association_decay_rate=association_decay_rate,
            maintenance_cost_per_pattern=maintenance_cost_per_pattern,
            maintenance_cost_per_association=maintenance_cost_per_association,
            enable_salience=enable_salience,
            modality_slices=modality_slices,
            enable_complexity_cost=enable_complexity_cost,
            complexity_cost_rate=complexity_cost_rate,
            enable_compositional=enable_compositional,
            patterns_per_modality=patterns_per_modality,
            modality_thresholds=modality_thresholds,
            composite_similarity_threshold=composite_similarity_threshold,
        )
        self.vitality = Vitality()
        self.valence = Valence(learning_rate=valence_learning_rate)
        self._exploration_base = exploration_base
        self._self_model = self_model
        self._affect: AffectStream | None = AffectStream() if enable_affect else None

        # Attention gate (Global Workspace — selective channel competition)
        self._attention: AttentionGate | None = None
        if enable_attention and modality_slices:
            self._attention = AttentionGate(capacity=2, suppress_factor=0.1)

        # Working memory (Baddeley — capacity-limited active buffer)
        self._working_memory: WorkingMemory | None = None
        if enable_working_memory:
            self._working_memory = WorkingMemory()

        # Options (Sutton — temporal abstraction / habits)
        self._option_discovery: OptionDiscovery | None = None
        self._option_executor: OptionExecutor | None = None
        if enable_options:
            self._option_discovery = OptionDiscovery()
            self._option_executor = OptionExecutor()

        # Internal Society (Minsky — competing specialists)
        self._internal_society = internal_society

        # Predictive Hierarchy (Rao & Ballard — hierarchical predictive coding)
        self._predictive_hierarchy: PredictiveHierarchy | None = None
        self._ph_surprise_weight = ph_surprise_weight
        self._base_similarity_threshold = similarity_threshold
        if enable_predictive_hierarchy:
            self._predictive_hierarchy = PredictiveHierarchy(
                num_levels=ph_num_levels,
                base_threshold=similarity_threshold,
                base_max_patterns=max_patterns or 20,
            )

        # Episodic memory (Tulving — high-surprise event snapshots)
        self._episodic: EpisodicMemory | None = None
        self._episodic_surprise_threshold = episodic_surprise_threshold
        if enable_episodic_memory:
            self._episodic = EpisodicMemory(capacity=episodic_capacity)

        # Level 2: hierarchical pattern composition
        # TemporalHierarchy supersedes SequenceMemory (same as Watcher)
        self._sequence_memory: SequenceMemory | None = None
        self._temporal_hierarchy: TemporalHierarchy | None = None
        self._th_surprise_weight = th_surprise_weight
        if temporal_scales is not None:
            self._temporal_hierarchy = TemporalHierarchy(
                scales=temporal_scales,
                similarity_threshold=similarity_threshold,
                max_patterns_per_scale=max_patterns or 15,
            )
        elif enable_sequence_memory:
            self._sequence_memory = SequenceMemory(
                window_size=sequence_window,
                similarity_threshold=similarity_threshold,
                max_patterns=max_patterns or 20,
            )

        self.history: list[StepResult] = []
        self._tick = 0
        self._rng = np.random.default_rng(seed)
        self._recent_pattern_ids: list[int] = []
        self._retroactive_window = 6

        # Lookahead: multi-step forward simulation for action selection
        self._enable_lookahead = enable_lookahead
        self._lookahead_depth = lookahead_depth
        self._lookahead_discount = lookahead_discount

        # Eligibility traces: temporal credit assignment (TD(λ))
        self._enable_eligibility_traces = enable_eligibility_traces
        self._trace_decay = trace_decay
        self._discount_factor = discount_factor
        self._eligibility_traces: dict[int, float] = {} if enable_eligibility_traces else {}
        self._last_consolidation_tick: int = 0

        # Abstraction layer: patterns-of-patterns (recursive self-similarity)
        self._abstraction: AbstractionLayer | None = None
        if enable_abstraction:
            from ..memory.abstraction import AbstractionLayer
            self._abstraction = AbstractionLayer()

    # ------------------------------------------------------------------
    # Phase 1 interface (backward compatible — passive observation)
    # ------------------------------------------------------------------

    def step(self, observation: Signal) -> StepResult:
        """Process one observation from the environment (passive mode).

        Returns a StepResult with the surprise value and whether the
        prediction was correct.
        """
        surprise = self.world_model.observe(observation)
        predicted_correctly = surprise == 0.0

        result = StepResult(
            observation=observation,
            surprise=surprise,
            predicted_correctly=predicted_correctly,
            tick=self._tick,
        )
        self.history.append(result)
        self._tick += 1
        return result

    def run_episode(self, env: Environment, max_steps: int = 1000) -> list[StepResult]:
        """Run a full episode in a passive environment.

        Returns the list of step results for this episode.
        """
        obs = env.reset()
        episode_results: list[StepResult] = []

        result = self.step(obs)
        episode_results.append(result)

        for _ in range(max_steps - 1):
            obs, _reward, done = env.step()
            result = self.step(obs)
            episode_results.append(result)
            if done:
                break

        return episode_results

    # ------------------------------------------------------------------
    # Phase 2 interface — volition (active survival)
    # ------------------------------------------------------------------

    def _lookahead_score(
        self, first_action: int, action_space: list[int],
        depth: int | None = None,
    ) -> float:
        """Score an action by simulating a greedy rollout from its outcome.

        For the candidate first_action, predicts the outcome pattern, then
        greedily selects the best follow-up action for ``depth`` steps,
        accumulating discounted predicted vitality.

        This is mental rehearsal: the agent imagines possible futures.

        Args:
            depth: Override lookahead depth (used by adaptive lookahead for
                   large action spaces). Defaults to ``self._lookahead_depth``.
        """
        if self.world_model.current_pattern is None:
            return 0.0

        effective_depth = depth if depth is not None else self._lookahead_depth
        current_pid = self.world_model.current_pattern.pattern_id
        total = 0.0
        discount = self._lookahead_discount

        # Step 0: score the candidate action
        vit = self.world_model.predict_vitality_from(current_pid, first_action)
        if vit is not None:
            total += vit

        pred = self.world_model.predict_from(current_pid, first_action)
        if pred is None:
            # Phase 13: Creative generalization — when specific prediction is
            # unavailable, use abstract category valence as a rough estimate.
            # "I've never tried this exact action here, but situations LIKE
            # this generally have positive/negative outcomes."
            if self._abstraction is not None and self.world_model.current_pattern is not None:
                meta_val = self._abstraction.meta_valence_for(
                    self.world_model.current_pattern.centroid
                )
                total += meta_val * 0.3 * self._lookahead_discount  # Discounted for uncertainty
            return total

        sim_pid = pred[0].pattern_id

        # Steps 1..depth-1: greedy rollout
        for step in range(1, effective_depth):
            best_vit: float | None = None
            best_next_pid: int | None = None
            for a in action_space:
                v = self.world_model.predict_vitality_from(sim_pid, a)
                if v is not None and (best_vit is None or v > best_vit):
                    best_vit = v
                    p = self.world_model.predict_from(sim_pid, a)
                    if p is not None:
                        best_next_pid = p[0].pattern_id
            if best_vit is not None:
                total += (discount ** step) * best_vit
            if best_next_pid is not None:
                sim_pid = best_next_pid
            else:
                break  # No predictions available
        return total

    def select_action(self, action_space: list[int]) -> int:
        """Choose an action based on predicted outcomes and valence.

        This IS volition: action driven by internal state, not external prompts.

        Strategy:
        1. For each action, predict outcome and expected vitality change.
        2. Score by: predicted valence * confidence + exploration bonus.
        3. Exploration bonus scales with (1 - urgency): explore when safe,
           exploit when dying.
        4. With some probability, explore randomly (decreases with urgency).
        """
        if not action_space:
            return -1

        # --- Options: if an option is active, follow it ---
        if self._option_executor is not None:
            option_action = self._option_executor.next_action()
            if option_action is not None:
                return option_action
            # Try to initiate a new option from current state
            if self.world_model.current_pattern is not None:
                initiated = self._option_executor.should_initiate(
                    self.world_model.current_pattern.pattern_id,
                    self.vitality.urgency,
                )
                if initiated is not None:
                    option_action = self._option_executor.next_action()
                    if option_action is not None:
                        return option_action

        # --- Internal Society: delegate to competing specialists ---
        if self._internal_society is not None:
            wm_contents = None
            if self._working_memory is not None:
                wm_contents = self._working_memory.contents()
            affect_state = None
            if self._affect is not None:
                affect_state = self._affect.current_state
            episodic_recalls = None
            if self._episodic is not None and self.world_model.current_pattern is not None:
                episodic_recalls = self._episodic.recall(
                    self.world_model.current_pattern.centroid, k=3,
                )
            sequence_prediction = None
            if self._sequence_memory is not None:
                sequence_prediction = self._sequence_memory.predict_constituent_ids()
            temporal_predictions = None
            if self._temporal_hierarchy is not None:
                th_preds = self._temporal_hierarchy.predict()
                temporal_predictions = {
                    scale: (pred[0].constituent_ids, pred[1])
                    for scale, pred in th_preds.items() if pred is not None
                }
            self_model_vitals = None
            if self._self_model is not None:
                self_model_vitals = self._self_model.vitals.as_dict()
            context = ActionContext(
                current_pattern=self.world_model.current_pattern,
                vitality=self.vitality,
                valence=self.valence,
                world_model=self.world_model,
                action_space=action_space,
                rng=self._rng,
                working_memory_contents=wm_contents,
                affect_state=affect_state,
                episodic_recalls=episodic_recalls,
                sequence_prediction=sequence_prediction,
                temporal_predictions=temporal_predictions,
                self_model_vitals=self_model_vitals,
            )
            return self._internal_society.select_action(context)

        # --- Fallback: inline scoring (original behavior) ---
        current = self.world_model.current_pattern
        if current is None:
            # No model yet — explore randomly
            return int(self._rng.choice(action_space))

        # Cognitive modulations from self-model (all default to neutral)
        # Phase 12: LEARNED modulations from cognitive valence replace hardcoded formulas.
        # Positive cognitive valence = "this mode works" → exploit (boost confidence).
        # Negative cognitive valence = "this mode precedes failure" → explore more.
        chaotic_discount = 1.0
        load_discount = 1.0
        learning_explore_boost = 1.0
        if self._self_model is not None:
            vitals = self._self_model.vitals
            cog_val = self._self_model.cognitive_valence

            # Learned modulation: positive cog_val → trust predictions more
            # negative cog_val → discount predictions, explore more
            chaotic_discount = 1.0 - 0.5 * vitals.surprise_momentum * (1.0 - cog_val)
            load_discount = 1.0 - vitals.cognitive_load * 0.3 * (1.0 - cog_val)
            learning_explore_boost = 1.0 + vitals.learning_rate * (1.0 - cog_val)

        # Episodic memory bias: "I've been somewhere like this before"
        episodic_bias = 0.0
        if self._episodic is not None and current is not None:
            recalled = self._episodic.recall(current.centroid, k=3)
            if recalled:
                avg_ep_valence = sum(ep.valence for ep in recalled) / len(recalled)
                episodic_bias = avg_ep_valence * 0.2

        # Abstraction bias: category-level valence for generalization
        abstraction_bias = 0.0
        if self._abstraction is not None and current is not None:
            abstraction_bias = self._abstraction.meta_valence_for(current.centroid) * 0.3

        # Adaptive lookahead: reduce depth for large action spaces (factored).
        # 168 actions × depth 5 is too expensive; depth 2 keeps it fast.
        effective_depth = self._lookahead_depth
        if len(action_space) > 100:
            effective_depth = min(effective_depth, 2)
        elif len(action_space) > 50:
            effective_depth = min(effective_depth, 3)

        scores: list[float] = []
        for action in action_space:
            # --- Lookahead: multi-step forward simulation ---
            if self._enable_lookahead:
                la_score = self._lookahead_score(action, action_space, depth=effective_depth)
                # Modulate by confidence if we have a prediction
                prediction = self.world_model.predict_action_outcome(action)
                if prediction is not None:
                    _pattern, confidence = prediction
                    confidence_effective = confidence * load_discount
                    score = la_score * confidence_effective * chaotic_discount
                    uncertainty = 1.0 - confidence_effective
                    explore_bonus = uncertainty * 0.05 * (1.0 - self.vitality.urgency)
                    score += explore_bonus * learning_explore_boost
                else:
                    # Unknown outcome — exploration bonus
                    score = 0.05 * (1.0 - self.vitality.urgency) * learning_explore_boost
                scores.append(score + episodic_bias + abstraction_bias)
                continue

            # --- Original 1-step scoring (backward compatible) ---
            predicted_vitality = self.world_model.predict_action_vitality(action)
            prediction = self.world_model.predict_action_outcome(action)

            if predicted_vitality is not None and prediction is not None:
                _pattern, confidence = prediction
                confidence_effective = confidence * load_discount
                score = predicted_vitality * confidence_effective * chaotic_discount
                uncertainty = 1.0 - confidence_effective
                explore_bonus = uncertainty * 0.05 * (1.0 - self.vitality.urgency)
                score += explore_bonus * learning_explore_boost
                scores.append(score + episodic_bias + abstraction_bias)
            elif prediction is not None:
                pattern, confidence = prediction
                confidence_effective = confidence * load_discount
                v = self.valence.get(pattern.pattern_id)
                score = v * confidence_effective * chaotic_discount
                uncertainty = 1.0 - confidence_effective
                explore_bonus = uncertainty * 0.05 * (1.0 - self.vitality.urgency)
                score += explore_bonus * learning_explore_boost
                scores.append(score + episodic_bias + abstraction_bias)
            else:
                explore_bonus = 0.05 * (1.0 - self.vitality.urgency)
                scores.append(explore_bonus * learning_explore_boost + episodic_bias + abstraction_bias)

        # With some probability, explore instead of exploit
        explore_prob = self._exploration_base * (1.0 - self.vitality.urgency)
        if self._self_model is not None:
            explore_prob *= (1.0 - self._self_model.vitals.surprise_momentum)
        if self._rng.random() < explore_prob:
            # --- Goal-directed exploration: prefer info-sparse states ---
            explore_scores = np.ones(len(action_space), dtype=np.float64)
            for i, action in enumerate(action_space):
                pred = self.world_model.predict_action_outcome(action)
                if pred is not None:
                    pattern, _conf = pred
                    # Less exposure = more interesting
                    explore_scores[i] = 1.0 / (1.0 + pattern.exposure_count * 0.1)
                    # Unknown valence = extra interesting
                    if not self.valence.is_known(pattern.pattern_id):
                        explore_scores[i] *= 2.0
                # Unknown outcome stays at 1.0 (maximally interesting)
            # Normalize to probabilities and sample
            total = explore_scores.sum()
            if total > 0:
                explore_scores /= total
            else:
                explore_scores[:] = 1.0 / len(action_space)
            return int(self._rng.choice(action_space, p=explore_scores))

        # Choose the highest-scoring action
        best_idx = int(np.argmax(scores))
        return action_space[best_idx]

    def _compute_attention_priorities(
        self, slices: list[tuple[int, int]],
    ) -> list[float]:
        """Per-modality priority for attention gating.

        WM bias: if working memory holds a high-valence item, boost the
        modality whose centroid energy matches it.
        """
        priorities = [1.0] * len(slices)
        if self._working_memory is not None:
            best = self._working_memory.best_item()
            if best is not None and best.centroid is not None:
                for i, (start, end) in enumerate(slices):
                    if end <= len(best.centroid):
                        energy = float(np.sum(np.abs(best.centroid[start:end])))
                        priorities[i] += energy * 0.5
        return priorities

    def consolidate(self, max_replays: int = 5) -> int:
        """Replay important episodes to strengthen memories.

        Only runs when safe (low urgency) and rate-limited. Replays
        reinforce valence, associations, and action-outcome predictions
        without modifying the agent's current perceptual state.

        Returns count of episodes replayed.
        """
        if self._episodic is None or self._episodic.count == 0:
            return 0
        if self.vitality.urgency >= 0.3:
            return 0
        if self._tick - self._last_consolidation_tick < 30:
            return 0

        self._last_consolidation_tick = self._tick
        replay_weight = 0.3

        # Prioritize high-|valence| episodes (deaths, rewards)
        episodes = self._episodic.get_recent(self._episodic.count)
        episodes.sort(key=lambda e: abs(e.valence), reverse=True)
        episodes = episodes[:max_replays]

        replayed = 0
        for ep in episodes:
            # Reinforce this pattern's valence
            self.valence.update(ep.pattern_id, ep.valence * replay_weight)

            # Reinforce predecessor associations + propagate valence backward
            for ctx_id in ep.context_ids:
                self.world_model.memory.associate(
                    ctx_id, ep.pattern_id, reinforce=True,
                )
                self.valence.update(ctx_id, ep.valence * replay_weight * 0.5)

            # Reinforce action-outcome prediction (helps lookahead)
            if ep.action_taken is not None and ep.context_ids:
                key = (ep.context_ids[0], ep.action_taken)
                targets = self.world_model._action_transitions.setdefault(key, {})
                pid = ep.pattern_id
                targets[pid] = (
                    targets.get(pid, 0.0) + 0.05 * (1.0 - targets.get(pid, 0.0))
                )

            replayed += 1
        return replayed

    def step_with_action(
        self,
        observation: Signal,
        energy_delta: float,
        action_taken: int | None,
    ) -> StepResult:
        """Process one observation + vitality update (active mode).

        This is the Phase 2 step function used in survival environments.
        In Phase 3, also applies maintenance cost from world_model.tick().
        """
        # Track vitality before changes
        vitality_before = self.vitality.energy

        # Apply energy delta from environment
        if energy_delta > 0:
            self.vitality.restore(energy_delta)
        elif energy_delta < 0:
            self.vitality.spend(abs(energy_delta))

        # Apply entropy (existence costs energy)
        self.vitality.tick()

        # Apply maintenance cost — thinking costs energy (Phase 3)
        maintenance_cost = self.world_model.tick()
        if maintenance_cost > 0:
            self.vitality.spend(maintenance_cost)

        vitality_after = self.vitality.energy
        actual_delta = vitality_after - vitality_before

        # --- Predictive Hierarchy: top-down bias (before observe) ---
        if self._predictive_hierarchy is not None:
            if self._predictive_hierarchy.num_levels > 1:
                bias = self._predictive_hierarchy._top_down_bias(0)
                self.world_model.memory.distinction.similarity_threshold = max(
                    0.3, self._base_similarity_threshold - bias * 0.5,
                )

        # --- Attention gate: filter observation before pattern matching ---
        if self._attention is not None:
            slices = self.world_model.memory.distinction.modality_slices
            if slices:
                priorities = self._compute_attention_priorities(slices)
                arousal = 0.0
                if self._affect is not None:
                    arousal = self._affect.current_state.arousal
                observation = self._attention.gate(
                    observation, slices, priorities, arousal,
                )

        # Observe the world (learn patterns, associations, predictions)
        # Pass last_action so world model uses action-conditional prediction for surprise
        surprise = self.world_model.observe(observation, last_action=action_taken)

        # --- Predictive Hierarchy: observe + blend surprise + restore threshold ---
        if self._predictive_hierarchy is not None:
            self._predictive_hierarchy.observe(observation, tick=self._tick)
            ph_agg = self._predictive_hierarchy.aggregate_surprise()
            w = self._ph_surprise_weight
            surprise = (1.0 - w) * surprise + w * ph_agg
            self._predictive_hierarchy.tick()
            # Restore base threshold (bias is per-tick, not cumulative)
            self.world_model.memory.distinction.similarity_threshold = (
                self._base_similarity_threshold
            )

        predicted_correctly = surprise == 0.0

        # Feed self-model with cognitive state from this observation
        if self._self_model is not None:
            pred = self.world_model.last_prediction
            pred_conf = pred[1] if pred is not None else None
            max_pats = self.world_model.memory.distinction.max_patterns or 30
            self._self_model.observe(
                surprise=surprise,
                pattern_count=self.pattern_count,
                max_patterns=max_pats,
                prediction_confidence=pred_conf,
                actual_delta=actual_delta,
            )

        # Update valence: associate current pattern with vitality change
        if self.world_model.current_pattern is not None:
            self.valence.update(
                self.world_model.current_pattern.pattern_id,
                actual_delta,
            )

        # --- Abstraction layer: categorize pattern into meta-pattern ---
        if self._abstraction is not None and self.world_model.current_pattern is not None:
            self._abstraction.observe(
                self.world_model.current_pattern.centroid, actual_delta,
            )

        # --- Working Memory: hold high-valence or high-surprise patterns ---
        if self._working_memory is not None and self.world_model.current_pattern is not None:
            pattern = self.world_model.current_pattern
            v = self.valence.get(pattern.pattern_id)
            if abs(v) > 0.1 or surprise >= 0.5:
                self._working_memory.hold(pattern.pattern_id, pattern.centroid, v)
            elif self._working_memory.contains(pattern.pattern_id):
                self._working_memory.refresh(pattern.pattern_id)

        # --- Temporal credit assignment ---
        if self._enable_eligibility_traces and self.world_model.prev_pattern is not None:
            # TD(λ): eligibility traces propagate credit backward through time
            prev_pid = self.world_model.prev_pattern.pattern_id
            current_pid = self.world_model.current_pattern.pattern_id if self.world_model.current_pattern else prev_pid

            # TD error: how much better/worse was this transition than expected?
            td_error = (
                actual_delta
                + self._discount_factor * self.valence.get(current_pid)
                - self.valence.get(prev_pid)
            )

            # Decay all existing traces
            dead_traces = []
            for pid in self._eligibility_traces:
                self._eligibility_traces[pid] *= self._discount_factor * self._trace_decay
                if self._eligibility_traces[pid] < 0.001:
                    dead_traces.append(pid)
            for pid in dead_traces:
                del self._eligibility_traces[pid]

            # Refresh trace for the state we just left
            self._eligibility_traces[prev_pid] = 1.0

            # Propagate TD error through all traced states
            for pid, trace in self._eligibility_traces.items():
                self.valence.update(pid, td_error * trace)

        elif self.world_model.current_pattern is not None:
            # Fallback: original retroactive window (backward compat)
            self._recent_pattern_ids.append(
                self.world_model.current_pattern.pattern_id,
            )
            if len(self._recent_pattern_ids) > self._retroactive_window:
                self._recent_pattern_ids = self._recent_pattern_ids[
                    -self._retroactive_window :
                ]

            if surprise >= 0.5 and actual_delta < -0.01 and len(self._recent_pattern_ids) > 1:
                predecessors = self._recent_pattern_ids[:-1]
                self.valence.adjust_retroactive(
                    predecessors, outcome_delta=actual_delta, strength=0.1,
                )
            elif surprise < 0.2 and actual_delta > 0.01 and len(self._recent_pattern_ids) > 1:
                predecessors = self._recent_pattern_ids[:-1]
                self.valence.adjust_retroactive(
                    predecessors, outcome_delta=actual_delta, strength=0.05,
                )

        # Record action-outcome in world model
        if action_taken is not None:
            self.world_model.record_action_outcome(
                action_taken,
                self.world_model.current_pattern,
                actual_delta,
            )

        # --- Option discovery: observe action stream and discover habits ---
        if self._option_discovery is not None and action_taken is not None:
            if self.world_model.current_pattern is not None:
                v = self.valence.get(self.world_model.current_pattern.pattern_id)
                self._option_discovery.observe(
                    self.world_model.current_pattern.pattern_id, action_taken, v,
                )
                new_options = self._option_discovery.discover()
                if self._option_executor is not None:
                    for opt in new_options:
                        self._option_executor.add_option(opt)

        # --- Option termination on bad outcome ---
        if (self._option_executor is not None
                and self._option_executor.active_option is not None
                and actual_delta < -0.05):
            self._option_executor.terminate(success=False)

        # Update salience weights (learn which dimensions predict outcomes)
        self.world_model.update_salience(actual_delta, action_taken)

        # --- Level 2: feed Level 1 pattern to temporal hierarchy or sequence memory ---
        if self.world_model.current_pattern is not None:
            if self._temporal_hierarchy is not None:
                th_surprises = self._temporal_hierarchy.observe(
                    self.world_model.current_pattern,
                )
                self._temporal_hierarchy.tick()
                available = [s for s in th_surprises.values() if s is not None]
                if available:
                    th_agg = sum(available) / len(available)
                    w = self._th_surprise_weight
                    surprise = (1.0 - w) * surprise + w * th_agg
            elif self._sequence_memory is not None:
                self._sequence_memory.observe(self.world_model.current_pattern)
                self._sequence_memory.tick()

        # Update affect state (emotion = pattern in valence x arousal space)
        if self._affect is not None:
            arousal = surprise * self.vitality.urgency
            self._affect.observe(
                AffectState(valence=actual_delta, arousal=arousal),
                timestamp=self._tick,
            )
            self._affect.tick()

        # --- Episodic memory: record high-surprise moments ---
        if (self._episodic is not None
                and self.world_model.current_pattern is not None
                and surprise >= self._episodic_surprise_threshold):
            prev = self.world_model.prev_pattern
            context_ids: tuple[int, ...] = ()
            if prev is not None:
                context_ids = (prev.pattern_id,)
            self._episodic.record(Episode(
                tick=self._tick,
                pattern_id=self.world_model.current_pattern.pattern_id,
                centroid=self.world_model.current_pattern.centroid.copy(),
                surprise=surprise,
                vitality=self.vitality.energy,
                context_ids=context_ids,
                valence=actual_delta,
                action_taken=action_taken,
            ))

        # --- Working Memory: tick (decay + maintenance cost) ---
        if self._working_memory is not None:
            wm_cost = self._working_memory.tick()
            if wm_cost > 0:
                self.vitality.spend(wm_cost)

        result = StepResult(
            observation=observation,
            surprise=surprise,
            predicted_correctly=predicted_correctly,
            tick=self._tick,
            action=action_taken,
            vitality=self.vitality.energy,
            vitality_delta=actual_delta,
        )
        self.history.append(result)
        self._tick += 1
        return result

    def run_survival_episode(
        self, env: SurvivalEnv, max_steps: int = 200,
    ) -> list[StepResult]:
        """Run an episode where the agent must survive.

        Returns when done, or when the agent dies (vitality = 0).
        """
        obs = env.reset()
        self.vitality = Vitality()  # Fresh energy each episode
        episode_results: list[StepResult] = []

        # First step: observe, no action yet
        result = self.step_with_action(obs, 0.0, None)
        episode_results.append(result)

        for _ in range(max_steps - 1):
            if not self.vitality.alive:
                break

            # Choose action based on internal state + learned model
            action = self.select_action(env.action_space)

            # Act and observe result
            obs, energy_delta, done = env.step(action)

            # Process result
            result = self.step_with_action(obs, energy_delta, action)
            episode_results.append(result)

            if done:
                break

        return episode_results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def average_surprise(self) -> float:
        return self.world_model.average_surprise

    @property
    def pattern_count(self) -> int:
        return self.world_model.memory.pattern_count

    @property
    def association_count(self) -> int:
        return self.world_model.memory.association_count
