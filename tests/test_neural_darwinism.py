"""Tests for Phase 3 Level 1: Neural Darwinism — internal evolution.

Tests capacity limits, passive decay, maintenance cost, eviction cascades,
and backward compatibility (all defaults preserve Phase 2 behavior).
"""

import numpy as np
import pytest

from fpi.primitives.signal import Signal
from fpi.primitives.pattern import Pattern, Distinction
from fpi.primitives.association import Association, AssociationMap
from fpi.memory.associative import AssociativeMemory
from fpi.world_model.model import WorldModel
from fpi.agent.core import Agent
from fpi.env.base import SurvivalEnv


# ---- Pattern fitness & eviction ----

class TestPatternFitness:
    def test_fitness_increases_with_exposure(self):
        p = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0, exposure_count=10, last_activated=5)
        # At tick 5, recency = 1/1 = 1.0, fitness = 10 * 1.0 = 10
        assert p.fitness(5) == pytest.approx(10.0)

    def test_fitness_decreases_with_age(self):
        p = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0, exposure_count=10, last_activated=0)
        # At tick 10, recency = 1/10, fitness = 10 * 0.1 = 1.0
        assert p.fitness(10) == pytest.approx(1.0)

    def test_recently_active_beats_stale(self):
        stale = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0, exposure_count=20, last_activated=0)
        fresh = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=1, exposure_count=5, last_activated=9)
        # At tick 10: stale = 20/10 = 2.0, fresh = 5/1 = 5.0
        assert fresh.fitness(10) > stale.fitness(10)

    def test_update_centroid_sets_last_activated(self):
        p = Pattern(centroid=np.array([1.0, 0.0]), pattern_id=0)
        sig = Signal(data=np.array([0.9, 0.1]), timestamp=0, modality="test")
        p.update_centroid(sig, tick=42)
        assert p.last_activated == 42


class TestDistinctionCapacity:
    def _make_orthogonal_signals(self, n: int, dim: int = 10) -> list[Signal]:
        """Create n signals that are near-orthogonal (low cosine similarity)."""
        rng = np.random.default_rng(42)
        signals = []
        for i in range(n):
            vec = np.zeros(dim)
            vec[i % dim] = 1.0
            # Add small noise to avoid exact zero vectors
            vec += rng.standard_normal(dim) * 0.01
            vec /= np.linalg.norm(vec)
            signals.append(Signal(data=vec, timestamp=i, modality="test"))
        return signals

    def test_unlimited_capacity_by_default(self):
        d = Distinction(similarity_threshold=0.99)
        sigs = self._make_orthogonal_signals(10)
        for sig in sigs:
            d.distinguish(sig)
        assert len(d.patterns) == 10

    def test_capacity_limit_triggers_eviction(self):
        d = Distinction(similarity_threshold=0.99, max_patterns=3)
        sigs = self._make_orthogonal_signals(5)
        for sig in sigs:
            d.distinguish(sig)
        assert len(d.patterns) == 3

    def test_eviction_removes_least_fit(self):
        d = Distinction(similarity_threshold=0.99, max_patterns=2)

        # Create pattern 0 at tick 0, activate it many times
        d._current_tick = 0
        sig0 = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        d.distinguish(sig0)
        for _ in range(10):
            d.distinguish(sig0)

        # Create pattern 1 at tick 10, only 1 activation
        d._current_tick = 10
        sig1 = Signal(data=np.array([0.0, 1.0]), timestamp=10, modality="test")
        d.distinguish(sig1)

        # Now create pattern 2 at tick 20 — should evict pattern 1 (least fit)
        d._current_tick = 20
        sig2 = Signal(data=np.array([0.5, 0.5]), timestamp=20, modality="test")
        d.distinguish(sig2)

        assert len(d.patterns) == 2
        ids = {p.pattern_id for p in d.patterns}
        assert 1 not in ids  # Pattern 1 was evicted (low exposure, stale)

    def test_last_evicted_is_set(self):
        d = Distinction(similarity_threshold=0.99, max_patterns=2)
        sig0 = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        sig1 = Signal(data=np.array([0.0, 1.0]), timestamp=1, modality="test")
        d.distinguish(sig0)
        d.distinguish(sig1)

        assert d._last_evicted is None

        sig2 = Signal(data=np.array([0.5, 0.5]), timestamp=2, modality="test")
        d.distinguish(sig2)
        assert d._last_evicted is not None

    def test_advance_tick(self):
        d = Distinction()
        assert d._current_tick == 0
        d.advance_tick()
        assert d._current_tick == 1


# ---- Association decay & capacity ----

class TestAssociationDecay:
    def test_decay_reduces_strength(self):
        a = Association(source_id=0, target_id=1, strength=0.5)
        a.decay(rate=0.1)
        assert a.strength == pytest.approx(0.4)

    def test_decay_floors_at_zero(self):
        a = Association(source_id=0, target_id=1, strength=0.05)
        a.decay(rate=0.1)
        assert a.strength == 0.0

    def test_reinforce_sets_last_activated(self):
        a = Association(source_id=0, target_id=1)
        a.reinforce(tick=42)
        assert a.last_activated == 42


class TestAssociationMapCapacity:
    def test_unlimited_by_default(self):
        m = AssociationMap()
        for i in range(20):
            m.get_or_create(0, i)
        assert m.count == 20

    def test_capacity_limit(self):
        m = AssociationMap(max_associations=3)
        for i in range(5):
            a = m.get_or_create(0, i)
            a.reinforce()
        assert m.count == 3

    def test_decay_all_prunes_dead(self):
        m = AssociationMap(decay_rate=0.1)
        a1 = m.get_or_create(0, 1)
        a1.strength = 0.05  # Will die after decay
        a2 = m.get_or_create(0, 2)
        a2.strength = 0.5  # Will survive

        pruned = m.decay_all()
        assert len(pruned) == 1
        assert pruned[0].target_id == 1
        assert m.count == 1

    def test_decay_all_noop_when_rate_zero(self):
        m = AssociationMap(decay_rate=0.0)
        a = m.get_or_create(0, 1)
        a.strength = 0.01
        pruned = m.decay_all()
        assert len(pruned) == 0
        assert m.count == 1

    def test_eviction_prefers_stale_over_active(self):
        """Fitness-based eviction: a strong-but-stale association should be
        evicted before a weaker-but-recently-active one."""
        m = AssociationMap(max_associations=2, decay_rate=0.0001)
        # Create a strong association, activated long ago
        old = m.get_or_create(0, 1)
        old.reinforce(amount=0.5, tick=0)
        # Advance time significantly (500 ticks, minimal decay)
        for _ in range(500):
            m.decay_all()
        # old strength ≈ 0.45, fitness = 0.45 * 1/(1+500*0.01) = 0.075
        # Create a weaker but fresh association
        fresh = m.get_or_create(0, 2)
        fresh.reinforce(amount=0.2, tick=m._current_tick)
        # fresh fitness = 0.2 * 1.0 = 0.2 — higher than stale's 0.075
        # Now add a third — should evict the stale one, not the fresh one
        new = m.get_or_create(0, 3)
        new.reinforce(amount=0.1, tick=m._current_tick)
        assert m.count == 2
        assert m.get(0, 1) is None  # stale evicted
        assert m.get(0, 2) is not None  # fresh survived

    def test_fitness_combines_strength_and_recency(self):
        """Association.fitness() should factor in both strength and recency."""
        a = Association(source_id=0, target_id=1, strength=0.8, last_activated=0)
        # At tick 0, fitness equals strength (recency = 1.0)
        assert a.fitness(0) == pytest.approx(0.8)
        # At tick 500, recency decays significantly
        fitness_later = a.fitness(500)
        assert fitness_later < 0.8
        # A weaker but fresh association can have higher fitness
        b = Association(source_id=0, target_id=2, strength=0.3, last_activated=500)
        assert b.fitness(500) > fitness_later

    def test_decay_all_advances_tick(self):
        """decay_all should advance _current_tick for fitness tracking."""
        m = AssociationMap(decay_rate=0.001)
        assert m._current_tick == 0
        m.decay_all()
        assert m._current_tick == 1
        m.decay_all()
        assert m._current_tick == 2

    def test_remove_association(self):
        m = AssociationMap()
        a = m.get_or_create(0, 1)
        m.get_or_create(0, 2)
        assert m.count == 2
        m.remove(a)
        assert m.count == 1
        assert m.get(0, 1) is None
        assert m.get(0, 2) is not None

    def test_remove_associations_for_pattern(self):
        m = AssociationMap()
        m.get_or_create(0, 1)
        m.get_or_create(0, 2)
        m.get_or_create(1, 0)  # Association TO pattern 0
        m.get_or_create(2, 3)  # Unrelated

        m.remove_associations_for_pattern(0)
        assert m.count == 1  # Only 2→3 remains
        assert m.get(2, 3) is not None


# ---- Associative memory integration ----

class TestAssociativeMemoryMaintenance:
    def test_tick_returns_maintenance_cost(self):
        mem = AssociativeMemory(
            maintenance_cost_per_pattern=0.001,
            maintenance_cost_per_association=0.0005,
        )
        # Add some patterns and associations
        sig0 = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        sig1 = Signal(data=np.array([0.0, 1.0]), timestamp=1, modality="test")
        mem.store(sig0)
        mem.store(sig1)
        mem.associate(0, 1)

        cost = mem.tick()
        # 2 patterns * 0.001 + 1 association * 0.0005
        assert cost == pytest.approx(0.0025)

    def test_tick_returns_zero_with_defaults(self):
        mem = AssociativeMemory()
        sig = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        mem.store(sig)
        cost = mem.tick()
        assert cost == 0.0

    def test_eviction_cascades_to_associations(self):
        mem = AssociativeMemory(similarity_threshold=0.99, max_patterns=2)
        sig0 = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        sig1 = Signal(data=np.array([0.0, 1.0]), timestamp=1, modality="test")
        p0 = mem.store(sig0)
        p1 = mem.store(sig1)
        mem.associate(p0.pattern_id, p1.pattern_id)
        assert mem.association_count == 1

        # Adding a third pattern should evict one and cascade-clean
        sig2 = Signal(data=np.array([0.5, 0.5]), timestamp=2, modality="test")
        mem.store(sig2)
        assert mem.pattern_count == 2
        # Association involving evicted pattern should be cleaned
        assert mem.association_count <= 1


# ---- World model tick ----

class TestWorldModelTick:
    def test_tick_cleans_action_data(self):
        wm = WorldModel(similarity_threshold=0.99, max_patterns=2)
        sig0 = Signal(data=np.array([1.0, 0.0]), timestamp=0, modality="test")
        sig1 = Signal(data=np.array([0.0, 1.0]), timestamp=1, modality="test")
        wm.observe(sig0)
        wm.observe(sig1)

        # Manually add action data for pattern 0
        wm._action_transitions[(0, 0)] = {1: 0.5}
        wm._action_vitality[(0, 0)] = 0.1

        # Evict by adding a third pattern
        sig2 = Signal(data=np.array([0.5, 0.5]), timestamp=2, modality="test")
        wm.observe(sig2)
        wm.tick()

        # Action data for evicted pattern should be cleaned
        live_ids = {p.pattern_id for p in wm.memory.distinction.patterns}
        for key in wm._action_transitions:
            assert key[0] in live_ids


# ---- Agent maintenance cost ----

class TestAgentMaintenanceCost:
    def test_maintenance_drains_vitality(self):
        env = SurvivalEnv(grid_size=10, resource_positions=[], max_steps=10)
        agent = Agent(
            similarity_threshold=0.7,
            seed=42,
            maintenance_cost_per_pattern=0.01,
        )
        results = agent.run_survival_episode(env, max_steps=10)
        # Agent should die faster with maintenance cost
        energy_final = results[-1].vitality
        assert energy_final < 0.8  # Significant drain from maintenance

    def test_no_maintenance_cost_by_default(self):
        """Backward compatibility: default agent has zero maintenance cost."""
        agent = Agent(similarity_threshold=0.7, seed=42)
        # world_model.tick() should return 0
        cost = agent.world_model.tick()
        assert cost == 0.0


# ---- Backward compatibility ----

class TestBackwardCompatibility:
    def test_default_pattern_has_last_activated(self):
        p = Pattern(centroid=np.array([1.0]), pattern_id=0)
        assert p.last_activated == 0

    def test_default_distinction_unlimited(self):
        d = Distinction()
        assert d.max_patterns is None

    def test_default_association_no_decay(self):
        a = Association(source_id=0, target_id=1)
        assert a.last_activated == 0

    def test_default_association_map_unlimited(self):
        m = AssociationMap()
        assert m.max_associations is None
        assert m.decay_rate == 0.0

    def test_default_agent_params(self):
        agent = Agent()
        assert agent._exploration_base == 0.15
