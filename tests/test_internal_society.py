"""Tests for Internal Society (Minsky's Society of Mind)."""

import numpy as np

from fpi.agent.internal_society import (
    ActionContext, ActionProposal, Explorer, Exploiter, Guardian,
    InternalSociety, Specialist,
)
from fpi.primitives.vitality import Vitality
from fpi.primitives.valence import Valence
from fpi.world_model.model import WorldModel


def _make_context(
    urgency: float = 0.0, seed: int = 42, with_pattern: bool = False,
) -> ActionContext:
    wm = WorldModel(similarity_threshold=0.7)
    vit = Vitality()
    if urgency > 0:
        # Spend energy to create urgency
        vit.spend(urgency * vit.max_energy)
    pattern = None
    if with_pattern:
        from fpi.primitives.signal import Signal
        sig = Signal(data=np.array([1.0, 0.0, 0.5, 0.3]))
        wm.observe(sig)
        pattern = wm.current_pattern
    return ActionContext(
        current_pattern=pattern,
        vitality=vit,
        valence=Valence(),
        world_model=wm,
        action_space=[0, 1, 2],
        rng=np.random.default_rng(seed),
    )


class TestExplorer:
    def test_proposes_action(self):
        ctx = _make_context()
        exp = Explorer()
        proposal = exp.propose(ctx)
        assert proposal.action in [0, 1, 2]
        assert proposal.source == "explorer"

    def test_priority_decreases_with_urgency(self):
        exp = Explorer(base_priority=1.0)
        ctx_safe = _make_context(urgency=0.0, with_pattern=True)
        ctx_urgent = _make_context(urgency=0.8, with_pattern=True)
        p_safe = exp.propose(ctx_safe)
        p_urgent = exp.propose(ctx_urgent)
        assert p_safe.priority > p_urgent.priority


class TestExploiter:
    def test_proposes_action(self):
        ctx = _make_context()
        exp = Exploiter()
        proposal = exp.propose(ctx)
        assert proposal.action in [0, 1, 2]
        assert proposal.source == "exploiter"

    def test_priority_increases_with_urgency(self):
        exp = Exploiter(base_priority=1.0)
        ctx_safe = _make_context(urgency=0.0, with_pattern=True)
        ctx_urgent = _make_context(urgency=0.8, with_pattern=True)
        p_safe = exp.propose(ctx_safe)
        p_urgent = exp.propose(ctx_urgent)
        assert p_urgent.priority > p_safe.priority


class TestGuardian:
    def test_proposes_action(self):
        ctx = _make_context()
        guard = Guardian()
        proposal = guard.propose(ctx)
        assert proposal.action in [0, 1, 2]
        assert proposal.source == "guardian"

    def test_priority_increases_with_urgency(self):
        guard = Guardian(base_priority=1.0)
        ctx_safe = _make_context(urgency=0.0, with_pattern=True)
        ctx_urgent = _make_context(urgency=0.8, with_pattern=True)
        p_safe = guard.propose(ctx_safe)
        p_urgent = guard.propose(ctx_urgent)
        assert p_urgent.priority > p_safe.priority


class TestInternalSociety:
    def test_default_specialists(self):
        soc = InternalSociety()
        names = {s.name for s in soc.specialists}
        assert names == {"explorer", "exploiter", "guardian", "planner", "social_modeler"}

    def test_select_action_returns_valid(self):
        soc = InternalSociety(seed=42)
        ctx = _make_context()
        action = soc.select_action(ctx)
        assert action in [0, 1, 2]

    def test_tracks_proposals(self):
        soc = InternalSociety(seed=42)
        ctx = _make_context()
        soc.select_action(ctx)
        assert len(soc.last_proposals) == 5  # 5 default specialists

    def test_custom_specialists(self):
        soc = InternalSociety(specialists=[Explorer(), Exploiter()])
        assert len(soc.specialists) == 2

    def test_deterministic_with_seed(self):
        ctx1 = _make_context(seed=1)
        ctx2 = _make_context(seed=1)
        soc1 = InternalSociety(seed=42)
        soc2 = InternalSociety(seed=42)
        a1 = soc1.select_action(ctx1)
        a2 = soc2.select_action(ctx2)
        assert a1 == a2

    def test_stochastic_selection(self):
        """Over many runs, different specialists should win sometimes."""
        actions_seen = set()
        for seed in range(100):
            ctx = _make_context(urgency=0.3, seed=seed)
            soc = InternalSociety(seed=seed)
            action = soc.select_action(ctx)
            actions_seen.add(action)
        # With stochastic selection, should see multiple different actions
        assert len(actions_seen) >= 2
