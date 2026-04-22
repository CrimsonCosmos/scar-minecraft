"""Internal Society — competing specialists for action selection.

Implements Minsky's Society of Mind (1986): the mind is a society of
agents with narrow expertise. Conflicts are resolved by priority-weighted
voting. This is INTERNAL (competing sub-processes within one agent),
not the existing EXTERNAL Society (multiple agents in a shared env).

The existing select_action() already has proto-specialists:
- Exploration bonus → Explorer
- Valence-driven scoring → Exploiter
- Urgency discount → Guardian

We formalize these as first-class competing agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..primitives.affect import AffectState
from ..primitives.pattern import Pattern
from ..primitives.vitality import Vitality
from ..primitives.valence import Valence
from ..world_model.model import WorldModel


@dataclass(slots=True)
class ActionContext:
    """Shared context passed to all specialists."""

    current_pattern: Pattern | None
    vitality: Vitality
    valence: Valence
    world_model: WorldModel
    action_space: list[int]
    rng: np.random.Generator
    working_memory_contents: list | None = None
    affect_state: AffectState | None = None
    episodic_recalls: list | None = None
    sequence_prediction: tuple[int, ...] | None = None
    temporal_predictions: dict | None = None
    self_model_vitals: dict | None = None


@dataclass(slots=True)
class ActionProposal:
    """A specialist's proposed action."""

    action: int
    priority: float
    source: str


class Specialist:
    """Base class for internal action-proposing agents."""

    def __init__(self, name: str, base_priority: float = 1.0) -> None:
        self.name = name
        self.base_priority = base_priority

    def propose(self, context: ActionContext) -> ActionProposal:
        raise NotImplementedError


class Explorer(Specialist):
    """Curiosity-driven: prefers actions with unknown outcomes."""

    def __init__(self, base_priority: float = 0.8) -> None:
        super().__init__("explorer", base_priority)

    def propose(self, context: ActionContext) -> ActionProposal:
        if context.current_pattern is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, self.base_priority, self.name)

        # Prefer actions with unknown outcomes (no prediction data)
        unknown_actions = []
        for a in context.action_space:
            pred = context.world_model.predict_action_vitality(a)
            if pred is None:
                unknown_actions.append(a)

        if unknown_actions:
            action = int(context.rng.choice(unknown_actions))
        else:
            action = int(context.rng.choice(context.action_space))

        # Priority scales inversely with urgency (explore when safe)
        priority = self.base_priority * (1.0 - context.vitality.urgency)
        return ActionProposal(action, priority, self.name)


class Exploiter(Specialist):
    """Valence-driven: prefers actions with highest predicted reward."""

    def __init__(self, base_priority: float = 1.0) -> None:
        super().__init__("exploiter", base_priority)

    def propose(self, context: ActionContext) -> ActionProposal:
        if context.current_pattern is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, 0.1, self.name)

        best_action = context.action_space[0]
        best_score = -float("inf")

        for a in context.action_space:
            pred_v = context.world_model.predict_action_vitality(a)
            pred = context.world_model.predict_action_outcome(a)

            if pred_v is not None and pred is not None:
                _, conf = pred
                score = pred_v * conf
            elif pred is not None:
                pattern, conf = pred
                v = context.valence.get(pattern.pattern_id)
                score = v * conf
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_action = a

        # Priority scales with urgency (exploit when desperate)
        priority = self.base_priority * (0.5 + 0.5 * context.vitality.urgency)
        return ActionProposal(best_action, priority, self.name)


class Guardian(Specialist):
    """Threat-averse: avoids actions associated with vitality loss."""

    def __init__(self, base_priority: float = 0.9) -> None:
        super().__init__("guardian", base_priority)

    def propose(self, context: ActionContext) -> ActionProposal:
        if context.current_pattern is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, 0.1, self.name)

        # Find the least-dangerous action
        safest_action = context.action_space[0]
        safest_score = -float("inf")

        for a in context.action_space:
            pred_v = context.world_model.predict_action_vitality(a)
            if pred_v is not None:
                # Guardian only cares about avoiding loss
                score = pred_v if pred_v < 0 else 0.0
                score = -abs(score)  # Less negative = safer
            else:
                score = 0.0  # Unknown is neutral

            if score > safest_score:
                safest_score = score
                safest_action = a

        # Priority scales with urgency (guardian speaks loudest when dying)
        priority = self.base_priority * context.vitality.urgency
        return ActionProposal(safest_action, priority, self.name)


class Planner(Specialist):
    """Multi-step lookahead: evaluates action sequences via simulation.

    Uses WorldModel.simulate_trajectory() to score different first-actions
    by their multi-step consequences. Lookahead depth scales inversely
    with urgency — plan when safe, react when dying.
    """

    def __init__(self, base_priority: float = 0.7, max_depth: int = 3) -> None:
        super().__init__("planner", base_priority)
        self._max_depth = max_depth

    def propose(self, context: ActionContext) -> ActionProposal:
        if context.current_pattern is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, 0.1, self.name)

        # Lookahead depth scales inversely with urgency
        depth = max(1, int(self._max_depth * (1.0 - context.vitality.urgency)))

        best_action = context.action_space[0]
        best_value = -float("inf")

        for first_action in context.action_space:
            # Simulate repeating this action for `depth` steps
            actions = [first_action] * depth
            total_vit, _ = context.world_model.simulate_trajectory(
                context.current_pattern.pattern_id, actions,
            )
            if total_vit > best_value:
                best_value = total_vit
                best_action = first_action

        # Priority: medium-high when safe (planning is a luxury)
        priority = self.base_priority * (1.0 - 0.5 * context.vitality.urgency)
        return ActionProposal(best_action, priority, self.name)


class SocialModeler(Specialist):
    """Theory of mind: predicts entity behavior using own world model.

    When entities are nearby (entity modality has non-zero patterns), the
    agent simulates what IT would do in a similar situation, then acts
    accordingly. This is intersubjectivity through self-simulation.

    Strategy: if entity is approaching (entity signal increasing), predict
    their likely trajectory using own model, then take evasive/preparatory
    action BEFORE contact.
    """

    def __init__(self, base_priority: float = 0.6) -> None:
        super().__init__("social_modeler", base_priority)

    def propose(self, context: ActionContext) -> ActionProposal:
        if context.current_pattern is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, 0.0, self.name)

        # Check entity proximity via valence of entity-related patterns.
        # If current pattern has negative valence AND entities are present
        # (detected by high exposure_count on the pattern), evade.
        current_val = context.valence.get(context.current_pattern.pattern_id)

        # Simulate: "if I were them, what would I do next?"
        # Use own world model predictions as a proxy for entity behavior.
        pred = context.world_model.predict()
        if pred is None:
            action = int(context.rng.choice(context.action_space))
            return ActionProposal(action, 0.0, self.name)

        predicted_pattern, confidence = pred
        predicted_val = context.valence.get(predicted_pattern.pattern_id)

        # If predicted next state has negative valence, the "other" is
        # likely approaching danger — take evasive action (pick safest).
        if predicted_val < -0.1 and confidence > 0.3:
            # Find action that leads away from the predicted bad state
            best_action = context.action_space[0]
            best_escape = -float("inf")
            for a in context.action_space:
                outcome = context.world_model.predict_action_outcome(a)
                if outcome is not None:
                    out_pattern, _ = outcome
                    out_val = context.valence.get(out_pattern.pattern_id)
                    if out_val > best_escape:
                        best_escape = out_val
                        best_action = a
                else:
                    # Unknown = potentially safe
                    if 0.0 > best_escape:
                        best_escape = 0.0
                        best_action = a

            # Priority scales with threat level and confidence
            priority = self.base_priority * abs(predicted_val) * confidence
            return ActionProposal(best_action, priority, self.name)

        # No social threat detected
        action = int(context.rng.choice(context.action_space))
        return ActionProposal(action, 0.0, self.name)


class InternalSociety:
    """Mediates between competing specialists via priority-weighted vote.

    Each specialist proposes an action with a priority. The society
    resolves conflicts by stochastic selection proportional to priority
    (softmax), not hard argmax. This allows minority specialists to
    occasionally win, preserving behavioral diversity.

    Args:
        specialists: List of Specialist instances. Defaults to
            [Explorer, Exploiter, Guardian].
        seed: RNG seed for stochastic selection.
    """

    def __init__(
        self,
        specialists: list[Specialist] | None = None,
        seed: int | None = None,
    ) -> None:
        if specialists is None:
            specialists = [Explorer(), Exploiter(), Guardian(), Planner(), SocialModeler()]
        self._specialists = specialists
        self._rng = np.random.default_rng(seed)
        self._last_proposals: list[ActionProposal] = []

    @property
    def specialists(self) -> list[Specialist]:
        return self._specialists

    @property
    def last_proposals(self) -> list[ActionProposal]:
        return self._last_proposals

    def select_action(self, context: ActionContext) -> int:
        """Collect proposals, resolve by priority-weighted stochastic vote."""
        proposals = [s.propose(context) for s in self._specialists]
        self._last_proposals = proposals

        if not proposals:
            return int(context.rng.choice(context.action_space))

        # Filter out zero-priority proposals
        valid = [p for p in proposals if p.priority > 0]
        if not valid:
            return int(context.rng.choice(context.action_space))

        # Metacognitive modulation from SelfModel
        if context.self_model_vitals is not None:
            vitals = context.self_model_vitals
            for p in valid:
                if p.source == "guardian":
                    p.priority *= (1.0 + 0.3 * vitals["surprise_momentum"])
                elif p.source == "explorer":
                    p.priority *= (1.0 + vitals["learning_rate"])

        # Stochastic selection proportional to priority
        priorities = np.array([p.priority for p in valid])
        probs = priorities / priorities.sum()
        idx = int(self._rng.choice(len(valid), p=probs))
        return valid[idx].action
