"""Watcher — one Agent watching one data stream.

The core integration between real-world data and the intelligence.
A Watcher wraps a Stream + Encoder + Agent and runs the observe-predict-
surprise loop. The agent doesn't act — it only observes and predicts.

Energy model: surprise costs energy, prediction restores energy.
The agent "thrives" when it can predict its stream and "struggles"
when the stream is unpredictable. This is the same thermodynamic
principle: prediction = order = sustaining; chaos = depleting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..agent.core import Agent
from ..agent.introspection import SelfModel
from ..memory.episodic import Episode, EpisodicMemory
from ..memory.sequence import SequenceMemory
from ..memory.temporal import TemporalHierarchy
from .encoder import AutoEncoder
from .insight import Insight, InsightKind, InsightLevel
from .stream import Stream


@dataclass
class WatcherStepResult:
    """Result of one watcher observation."""

    surprise: float
    pattern_id: int
    is_new_pattern: bool
    vitality: float
    vitality_delta: float
    tick: int


class Watcher:
    """One Agent watching one Stream.

    Args:
        stream: The data source being watched.
        encoder: Converts raw strings to Signals. Defaults to AutoEncoder.
        surprise_threshold: Surprise above this generates an Insight.
        agent_kwargs: Override Agent constructor parameters.
    """

    def __init__(
        self,
        stream: Stream,
        encoder: AutoEncoder | None = None,
        surprise_threshold: float = 0.5,
        agent_kwargs: dict | None = None,
        report_interval: int = 0,
        sequence_window: int = 0,
        temporal_scales: tuple[int, ...] | None = None,
        episodic_capacity: int = 50,
        introspection: bool = False,
    ) -> None:
        self.stream = stream
        self.encoder = encoder or AutoEncoder()
        self.name = stream.name
        self.surprise_threshold = surprise_threshold
        self._report_interval = report_interval

        kwargs = {
            "similarity_threshold": 0.7,
            "max_patterns": 30,
            "max_associations": 100,
            "association_decay_rate": 0.005,
        }
        if agent_kwargs:
            kwargs.update(agent_kwargs)
        self.agent = Agent(**kwargs)

        # Level 2: hierarchical pattern composition (opt-in)
        # temporal_scales takes precedence over sequence_window
        self._sequence_memory: SequenceMemory | None = None
        self._temporal_hierarchy: TemporalHierarchy | None = None
        if temporal_scales is not None:
            self._temporal_hierarchy = TemporalHierarchy(scales=temporal_scales)
        elif sequence_window > 0:
            self._sequence_memory = SequenceMemory(window_size=sequence_window)

        # Episodic memory — records high-surprise events
        self._episodic: EpisodicMemory | None = None
        if self._temporal_hierarchy is not None or self._sequence_memory is not None:
            self._episodic = EpisodicMemory(capacity=episodic_capacity)

        # Self-model (introspection) — observes own cognitive state
        self._self_model: SelfModel | None = None
        if introspection:
            self._self_model = SelfModel()

        # Energy model parameters
        self._surprise_energy_cost = 0.03
        self._prediction_energy_gain = 0.02
        self._surprise_cost_threshold = 0.5
        self._episodic_surprise_threshold = 0.7

        # Temporal credit assignment
        self._recent_pattern_ids: list[int] = []
        self._retroactive_window = 6

        self._tick = 0

    def poll_and_observe(self) -> list[Insight]:
        """Poll stream for new data, feed to agent, return any insights."""
        raw_values = self.stream.poll()
        insights: list[Insight] = []

        for raw in raw_values:
            result = self._observe_one(raw)
            if result.surprise >= self.surprise_threshold:
                level = (
                    InsightLevel.ALERT
                    if result.surprise > 0.8
                    else InsightLevel.ANOMALY
                )
                # Enrich with valence and prediction
                valence_val = None
                predicted_next = None
                cp = self.agent.world_model.current_pattern
                if cp is not None:
                    valence_val = self.agent.valence.get(cp.pattern_id)
                pred = self.agent.world_model.last_prediction
                if pred is not None:
                    predicted_next = pred[0].pattern_id

                insight = Insight(
                    stream_name=self.name,
                    tick=result.tick,
                    timestamp_seconds=time.time(),
                    surprise=result.surprise,
                    level=level,
                    raw_value=raw,
                    pattern_id=result.pattern_id,
                    is_new_pattern=result.is_new_pattern,
                    message=self._format_message(result, raw),
                    vitality=result.vitality,
                    valence=valence_val,
                    predicted_next=predicted_next,
                )
                insights.append(insight)

        # Check for periodic status report
        if (
            self._report_interval > 0
            and self._tick > 0
            and self._tick % self._report_interval == 0
        ):
            insights.append(self._generate_status_report())

        return insights

    def _observe_one(self, raw: str) -> WatcherStepResult:
        """Feed one raw value through encoder -> agent -> energy model.

        Structurally identical to Agent.step_with_action() but without
        the action selection loop.
        """
        signal = self.encoder.encode(raw, timestamp=self._tick)

        patterns_before = self.agent.pattern_count
        vitality_before = self.agent.vitality.energy

        # Core observation
        surprise = self.agent.world_model.observe(signal)

        # Energy model: surprise costs, prediction restores
        if surprise >= self._surprise_cost_threshold:
            cost = self._surprise_energy_cost * surprise
            self.agent.vitality.spend(cost)
        else:
            gain = self._prediction_energy_gain * (1.0 - surprise)
            self.agent.vitality.restore(gain)

        # Entropy
        self.agent.vitality.tick()

        # Maintenance cost
        maintenance = self.agent.world_model.tick()
        if maintenance > 0:
            self.agent.vitality.spend(maintenance)

        vitality_after = self.agent.vitality.energy
        vitality_delta = vitality_after - vitality_before

        # Update valence
        current_pattern = self.agent.world_model.current_pattern
        if current_pattern is not None:
            self.agent.valence.update(current_pattern.pattern_id, vitality_delta)

        # Temporal credit assignment: retroactively adjust predecessors
        if current_pattern is not None:
            self._recent_pattern_ids.append(current_pattern.pattern_id)
            if len(self._recent_pattern_ids) > self._retroactive_window:
                self._recent_pattern_ids = self._recent_pattern_ids[
                    -self._retroactive_window :
                ]

            # On bad outcome (high surprise + vitality loss), penalize predecessors
            if surprise >= 0.5 and vitality_delta < -0.01 and len(self._recent_pattern_ids) > 1:
                predecessors = self._recent_pattern_ids[:-1]
                self.agent.valence.adjust_retroactive(
                    predecessors, outcome_delta=vitality_delta, strength=0.1,
                )
            # On good outcome (low surprise + vitality gain), boost predecessors (weaker)
            elif surprise < 0.2 and vitality_delta > 0.01 and len(self._recent_pattern_ids) > 1:
                predecessors = self._recent_pattern_ids[:-1]
                self.agent.valence.adjust_retroactive(
                    predecessors, outcome_delta=vitality_delta, strength=0.05,
                )

        # Feed Level 1 pattern to temporal hierarchy or sequence memory
        if current_pattern is not None:
            if self._temporal_hierarchy is not None:
                self._temporal_hierarchy.observe(current_pattern)
            elif self._sequence_memory is not None:
                self._sequence_memory.observe(current_pattern)

        # Episodic memory — record high-surprise moments
        if (
            self._episodic is not None
            and current_pattern is not None
            and surprise >= self._episodic_surprise_threshold
        ):
            prev = self.agent.world_model.prev_pattern
            context_ids: tuple[int, ...] = ()
            if prev is not None:
                context_ids = (prev.pattern_id,)
            episode = Episode(
                tick=self._tick,
                pattern_id=current_pattern.pattern_id,
                centroid=current_pattern.centroid.copy(),
                surprise=surprise,
                vitality=self.agent.vitality.energy,
                context_ids=context_ids,
                valence=vitality_delta,
            )
            self._episodic.record(episode)

        # Self-model (introspection) — observe own cognitive state
        if self._self_model is not None:
            pred = self.agent.world_model.last_prediction
            pred_conf = pred[1] if pred is not None else None
            max_pats = self.agent.world_model.memory.distinction.max_patterns or 30
            self._self_model.observe(
                surprise=surprise,
                pattern_count=self.agent.pattern_count,
                max_patterns=max_pats,
                prediction_confidence=pred_conf,
            )

        is_new = self.agent.pattern_count > patterns_before

        result = WatcherStepResult(
            surprise=surprise,
            pattern_id=current_pattern.pattern_id if current_pattern else -1,
            is_new_pattern=is_new,
            vitality=self.agent.vitality.energy,
            vitality_delta=vitality_delta,
            tick=self._tick,
        )
        self._tick += 1
        return result

    def get_status(self) -> dict:
        """Return a structured summary of all learned knowledge.

        Inspects the agent's internal state and returns everything it has
        learned: patterns, associations, valence, predictions, vitality.
        """
        mem = self.agent.world_model.memory
        patterns = []
        for p in mem.distinction.patterns:
            patterns.append({
                "pattern_id": p.pattern_id,
                "exposure_count": p.exposure_count,
                "valence": self.agent.valence.get(p.pattern_id),
                "fitness": p.fitness(mem.distinction._current_tick),
            })

        associations = []
        for a in mem.associations._by_key.values():
            associations.append({
                "source_id": a.source_id,
                "target_id": a.target_id,
                "strength": a.strength,
            })

        strongest = sorted(
            associations, key=lambda a: a["strength"], reverse=True
        )[:5]

        pred = self.agent.world_model.last_prediction
        current_prediction = None
        prediction_confidence = None
        if pred is not None:
            current_prediction = pred[0].pattern_id
            prediction_confidence = pred[1]

        positive_patterns = sum(1 for p in patterns if p["valence"] > 0.001)
        negative_patterns = sum(1 for p in patterns if p["valence"] < -0.001)

        # Sequence memory info (Level 2) — single scale
        seq_info = None
        if self._sequence_memory is not None:
            seq_info = {
                "sequence_patterns_learned": self._sequence_memory.pattern_count,
                "sequence_associations": self._sequence_memory.association_count,
                "current_sequence": None,
                "predicted_next_sequence": None,
            }
            if self._sequence_memory.current_sequence is not None:
                seq_info["current_sequence"] = list(
                    self._sequence_memory.current_sequence.constituent_ids
                )
            pred_ids = self._sequence_memory.predict_constituent_ids()
            if pred_ids is not None:
                seq_info["predicted_next_sequence"] = list(pred_ids)

        # Temporal hierarchy info (multi-scale)
        temporal_info = None
        if self._temporal_hierarchy is not None:
            temporal_info = self._temporal_hierarchy.get_status()

        # Episodic memory info
        episodic_info = None
        if self._episodic is not None:
            recent = self._episodic.get_recent(3)
            episodic_info = {
                "episode_count": self._episodic.count,
                "recent_episodes": [
                    {
                        "tick": ep.tick,
                        "pattern_id": ep.pattern_id,
                        "surprise": ep.surprise,
                        "valence": ep.valence,
                    }
                    for ep in recent
                ],
            }

        # Self-model info
        self_model_info = None
        if self._self_model is not None:
            self_model_info = self._self_model.get_status()

        return {
            "tick": self._tick,
            "vitality": self.agent.vitality.energy,
            "alive": self.agent.vitality.alive,
            "patterns_learned": len(patterns),
            "positive_patterns": positive_patterns,
            "negative_patterns": negative_patterns,
            "neutral_patterns": len(patterns) - positive_patterns - negative_patterns,
            "patterns": patterns,
            "associations": associations,
            "strongest_associations": strongest,
            "current_prediction": current_prediction,
            "prediction_confidence": prediction_confidence,
            "average_surprise": self.agent.world_model.average_surprise,
            "sequences": seq_info,
            "temporal": temporal_info,
            "episodic": episodic_info,
            "self_model": self_model_info,
        }

    def _generate_status_report(self) -> Insight:
        """Generate a periodic status report as an INFO-level Insight."""
        status = self.get_status()

        parts = [f"Tick {status['tick']}"]
        parts.append(
            f"{status['patterns_learned']} patterns "
            f"({status['positive_patterns']}+/{status['negative_patterns']}-)"
        )
        parts.append(f"vitality: {status['vitality']:.2f}")
        parts.append(f"avg surprise: {status['average_surprise']:.2f}")

        if status["strongest_associations"]:
            top = status["strongest_associations"][0]
            parts.append(
                f"strongest: #{top['source_id']}->"
                f"#{top['target_id']} ({top['strength']:.2f})"
            )

        if status["current_prediction"] is not None:
            parts.append(
                f"predicting #{status['current_prediction']} "
                f"({status['prediction_confidence']:.2f})"
            )

        if status.get("sequences") and status["sequences"]["sequence_patterns_learned"] > 0:
            si = status["sequences"]
            parts.append(f"{si['sequence_patterns_learned']} sequences")
            if si["predicted_next_sequence"]:
                ids = si["predicted_next_sequence"]
                parts.append(
                    f"next seq: [{', '.join(f'#{x}' for x in ids)}]"
                )

        if status.get("temporal"):
            scales_info = status["temporal"]["scales"]
            scale_parts = []
            for scale, info in scales_info.items():
                scale_parts.append(f"w{scale}:{info['pattern_count']}p")
            parts.append(f"temporal [{', '.join(scale_parts)}]")

        if status.get("episodic"):
            parts.append(f"{status['episodic']['episode_count']} episodes")

        if status.get("self_model"):
            sm = status["self_model"]
            v = sm["vitals"]
            parts.append(
                f"cognitive: surprise={sm['cognitive_surprise']:.2f} "
                f"momentum={v['surprise_momentum']:.2f} "
                f"load={v['cognitive_load']:.2f}"
            )

        message = " | ".join(parts)
        cp = self.agent.world_model.current_pattern

        return Insight(
            stream_name=self.name,
            tick=self._tick,
            timestamp_seconds=time.time(),
            surprise=self.agent.world_model.last_surprise,
            level=InsightLevel.INFO,
            raw_value="",
            pattern_id=cp.pattern_id if cp else -1,
            is_new_pattern=False,
            message=message,
            vitality=self.agent.vitality.energy,
            kind=InsightKind.STATUS_REPORT,
        )

    def _format_message(self, result: WatcherStepResult, raw: str) -> str:
        """Generate a human-readable insight message."""
        truncated = raw[:60]
        if result.is_new_pattern:
            return (
                f"New pattern detected: '{truncated}' "
                f"(surprise: {result.surprise:.2f})"
            )
        return (
            f"Unexpected value: '{truncated}' "
            f"(surprise: {result.surprise:.2f}, pattern #{result.pattern_id})"
        )
