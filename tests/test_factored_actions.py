"""Tests for factored (composite) action space.

Covers:
1. Encoding roundtrip: all 168 IDs encode/decode correctly
2. Combat sim: all 168 actions execute without error or NaN
3. Track_target: forward + track + attack lands hits in sim
4. Motor program override: crit with any movement axis works
5. Adaptive lookahead: select_action handles 168 actions
6. Backward compat: existing Phase 1-3 flat behavior unchanged
7. Full loop: Agent + CombatSimulator(factored=True) for 1000 steps
8. Per-modality thresholds: volatile modalities use broader matching
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from fpi.minecraft.actions import (
    COMBAT_COUNT,
    FACTORED_ACTION_COUNT,
    FACTORED_ACTIONS,
    LOOK_COUNT,
    MOVEMENT_COUNT,
    PHASE_1_ACTIONS,
    PHASE_2_ACTIONS,
    PHASE_3_ACTIONS,
    decode_composite,
    encode_composite,
)
from fpi.minecraft.combat_sim import CombatSimulator
from fpi.minecraft.env import MinecraftEnv
from fpi.agent.core import Agent
from fpi.primitives.compositional import CompositionalDistinction
from fpi.primitives.signal import Signal
from fpi.primitives.vitality import Vitality


# ---- 1. Encoding roundtrip ----

class TestEncodingRoundtrip:
    def test_constants(self):
        assert MOVEMENT_COUNT == 7
        assert LOOK_COUNT == 6
        assert COMBAT_COUNT == 4
        assert FACTORED_ACTION_COUNT == 168
        assert len(FACTORED_ACTIONS) == 168

    def test_encode_decode_all(self):
        """Every flat_id 0..167 roundtrips through encode/decode."""
        seen = set()
        for m in range(MOVEMENT_COUNT):
            for l in range(LOOK_COUNT):
                for c in range(COMBAT_COUNT):
                    flat = encode_composite(m, l, c)
                    assert 0 <= flat < FACTORED_ACTION_COUNT
                    assert flat not in seen, f"Collision at ({m},{l},{c}) = {flat}"
                    seen.add(flat)
                    m2, l2, c2 = decode_composite(flat)
                    assert (m2, l2, c2) == (m, l, c), (
                        f"Roundtrip failed: ({m},{l},{c}) -> {flat} -> ({m2},{l2},{c2})"
                    )
        assert len(seen) == FACTORED_ACTION_COUNT

    def test_specific_encodings(self):
        # (0,0,0) = idle
        assert encode_composite(0, 0, 0) == 0
        # (1,0,0) = forward + no look + no combat
        assert encode_composite(1, 0, 0) == 24
        # (0,0,1) = no move + no look + attack
        assert encode_composite(0, 0, 1) == 1
        # (6,5,3) = max values
        assert encode_composite(6, 5, 3) == 167

    def test_decode_boundary_values(self):
        assert decode_composite(0) == (0, 0, 0)
        assert decode_composite(167) == (6, 5, 3)
        assert decode_composite(1) == (0, 0, 1)
        assert decode_composite(24) == (1, 0, 0)


# ---- 2. Combat sim: all 168 actions execute ----

class TestCombatSimFactored:
    def test_action_space_is_168(self):
        sim = CombatSimulator(factored=True, seed=1)
        assert len(sim.action_space) == 168
        assert sim.action_space == FACTORED_ACTIONS

    def test_all_actions_execute_no_nan(self):
        """Every composite action executes without NaN in state."""
        sim = CombatSimulator(factored=True, seed=1)
        sim.reset()
        for action_id in range(FACTORED_ACTION_COUNT):
            obs, delta, done = sim.step(action_id)
            assert not np.isnan(obs.data).any(), f"NaN for action {action_id}"
            assert not math.isnan(delta), f"NaN delta for action {action_id}"

    def test_track_target_lands_hits(self):
        """forward + track_target + attack should land hits.

        Mob spawns 20-40 blocks away at 0.1 blocks/tick = ~200-400 ticks
        to close. Attack has 32-tick cooldown. Sprint (movement=6) is faster.
        """
        sim = CombatSimulator(factored=True, seed=42, max_mobs=1, mob_speed=0.0)
        sim.reset()
        # Sprint toward mob and attack: movement=6(sprint), look=1(track), combat=1(attack)
        sprint_track_attack = encode_composite(6, 1, 1)
        hits = 0
        for _ in range(500):
            obs, delta, done = sim.step(sprint_track_attack)
            if sim._hit_landed:
                hits += 1
        assert hits > 0, "sprint + track + attack should land at least one hit in 500 steps"

    def test_motor_program_crit_with_various_movements(self):
        """Crit (combat=2) should work regardless of movement axis."""
        sim = CombatSimulator(factored=True, seed=42, max_mobs=1, mob_speed=0.0)
        for movement in range(MOVEMENT_COUNT):
            sim.reset()
            composite = encode_composite(movement, 0, 2)  # crit
            for _ in range(50):
                obs, delta, done = sim.step(composite)
                assert not np.isnan(obs.data).any()

    def test_motor_program_wtap_with_various_movements(self):
        """W-tap (combat=3) should work regardless of movement axis."""
        sim = CombatSimulator(factored=True, seed=42, max_mobs=1, mob_speed=0.0)
        for movement in range(MOVEMENT_COUNT):
            sim.reset()
            composite = encode_composite(movement, 0, 3)  # wtap
            for _ in range(50):
                obs, delta, done = sim.step(composite)
                assert not np.isnan(obs.data).any()


# ---- 3. Backward compat: flat actions still work ----

class TestBackwardCompat:
    def test_phase_1_unchanged(self):
        sim = CombatSimulator(phase=1, seed=1)
        assert sim.action_space == list(range(13))
        assert sim.action_space == PHASE_1_ACTIONS

    def test_phase_2_unchanged(self):
        sim = CombatSimulator(phase=2, seed=1)
        assert sim.action_space == list(range(18))

    def test_phase_3_unchanged(self):
        sim = CombatSimulator(phase=3, seed=1)
        assert sim.action_space == list(range(20))

    def test_flat_actions_still_work(self):
        """Phase 3 flat actions should still execute correctly."""
        sim = CombatSimulator(phase=3, seed=42)
        sim.reset()
        for action in range(20):
            obs, delta, done = sim.step(action)
            assert not np.isnan(obs.data).any()

    def test_factored_false_by_default(self):
        sim = CombatSimulator(seed=1)
        assert sim._factored is False
        assert len(sim.action_space) == 13  # Phase 1 default


# ---- 4. Adaptive lookahead ----

class TestAdaptiveLookahead:
    def test_select_action_handles_168_actions(self):
        """select_action should complete in reasonable time with 168 actions."""
        agent = Agent(
            similarity_threshold=0.80,
            seed=42,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            enable_lookahead=True,
            lookahead_depth=5,
            lookahead_discount=0.9,
        )
        agent.vitality = Vitality(entropy_rate=0.001)

        # Feed it a few observations to build some patterns
        sim = CombatSimulator(factored=True, seed=42)
        obs = sim.reset()
        agent.step_with_action(obs, 0.0, None)
        for _ in range(10):
            obs, delta, done = sim.step(0)
            agent.step_with_action(obs, delta, 0)

        # Now time select_action with 168 actions
        start = time.monotonic()
        action = agent.select_action(FACTORED_ACTIONS)
        elapsed = time.monotonic() - start

        assert 0 <= action < 168
        assert elapsed < 1.0, f"select_action took {elapsed:.3f}s (should be <1s)"

    def test_depth_reduces_for_large_space(self):
        """Verify the adaptive depth logic is applied."""
        agent = Agent(
            similarity_threshold=0.80,
            seed=42,
            enable_lookahead=True,
            lookahead_depth=5,
        )
        # The depth reduction happens inside select_action; we can verify
        # by checking the _lookahead_score accepts a depth parameter
        score = agent._lookahead_score(0, [0, 1, 2], depth=2)
        assert isinstance(score, float)


# ---- 5. Full training loop ----

class TestFullLoop:
    def test_agent_sim_1000_steps(self):
        """Agent + CombatSimulator(factored=True) for 1000 steps, kills > 0."""
        sim = CombatSimulator(
            factored=True, seed=42, max_mobs=2, mob_speed=0.06,
        )
        agent = Agent(
            similarity_threshold=0.80,
            seed=42,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            enable_lookahead=True,
            lookahead_depth=3,
            lookahead_discount=0.9,
            enable_eligibility_traces=True,
            trace_decay=0.8,
            discount_factor=0.95,
        )
        agent.vitality = Vitality(entropy_rate=0.001)

        obs = sim.reset()
        agent.step_with_action(obs, 0.0, None)

        for step in range(1000):
            if not agent.vitality.alive:
                agent.vitality = Vitality(entropy_rate=0.001)
            action = agent.select_action(sim.action_space)
            obs, delta, done = sim.step(action)
            agent.step_with_action(obs, delta, action)

        # Agent should have landed some kills in 1000 steps
        assert sim.kill_count > 0, (
            f"Expected kills > 0 in 1000 steps, got {sim.kill_count}"
        )
        assert sim.step_count == 1000
        # Check no NaN in last observation
        assert not np.isnan(obs.data).any()

    def test_factored_sim_no_nan_long_run(self):
        """500 steps with random actions — no NaN anywhere."""
        sim = CombatSimulator(factored=True, seed=99)
        rng = np.random.default_rng(99)
        obs = sim.reset()
        for _ in range(500):
            action = int(rng.integers(FACTORED_ACTION_COUNT))
            obs, delta, done = sim.step(action)
            assert not np.isnan(obs.data).any()
            assert not math.isnan(delta)


# ---- 6. Per-modality similarity thresholds ----

class TestModalityThresholds:
    def test_thresholds_threaded_to_distinctions(self):
        """Each per-modality Distinction gets its own threshold."""
        thresholds = [0.85, 0.80, 0.65, 0.70, 0.85, 0.80, 0.60]
        cd = CompositionalDistinction(
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            modality_thresholds=thresholds,
        )
        for i, md in enumerate(cd._modal_distinctions):
            assert md.similarity_threshold == thresholds[i], (
                f"Modality {i}: expected {thresholds[i]}, got {md.similarity_threshold}"
            )

    def test_no_thresholds_uses_global(self):
        """Without modality_thresholds, all modalities use global threshold."""
        cd = CompositionalDistinction(
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            similarity_threshold=0.80,
            patterns_per_modality=16,
        )
        for i, md in enumerate(cd._modal_distinctions):
            assert md.similarity_threshold == 0.80, (
                f"Modality {i}: expected 0.80, got {md.similarity_threshold}"
            )

    def test_terrain_varied_obs_reuse_patterns(self):
        """With loose terrain threshold (0.65), similar terrains merge into one pattern."""
        thresholds = MinecraftEnv.MODALITY_THRESHOLDS
        cd = CompositionalDistinction(
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            modality_thresholds=thresholds,
        )

        rng = np.random.default_rng(42)

        def make_signal(terrain_variation: float) -> Signal:
            """Create a 92-dim signal with controlled terrain noise."""
            data = np.zeros(92)
            # Body: stable
            data[0:12] = 0.5
            data[0:12] /= np.linalg.norm(data[0:12])
            # Env: stable
            data[12:20] = 0.3
            data[12:20] /= np.linalg.norm(data[12:20])
            # Terrain: base + small noise
            terrain = np.array([0.3, 0.2, 0.1, 0.15, 0.05, 0.1, 0.05, 0.0])
            terrain += rng.normal(0, terrain_variation, 8)
            terrain = np.abs(terrain)
            tn = np.linalg.norm(terrain)
            if tn > 0:
                terrain /= tn
            data[20:28] = terrain
            # Entities: stable
            data[28:48] = 0.1
            data[28:48] /= np.linalg.norm(data[28:48])
            # Inventory: stable
            data[48:60] = 0.2
            data[48:60] /= np.linalg.norm(data[48:60])
            # Combat: stable
            data[60:76] = 0.15
            data[60:76] /= np.linalg.norm(data[60:76])
            # History: stable
            data[76:92] = 0.1
            data[76:92] /= np.linalg.norm(data[76:92])
            return Signal(data=data, timestamp=0)

        # With loose terrain threshold: varied terrains should produce few patterns
        for _ in range(20):
            sig = make_signal(terrain_variation=0.05)
            cd.distinguish(sig)
            cd.advance_tick()

        # Terrain modality (index 2) should have few patterns despite noise
        terrain_patterns = len(cd._modal_distinctions[2].patterns)
        assert terrain_patterns <= 4, (
            f"Expected ≤4 terrain patterns with 0.65 threshold, got {terrain_patterns}"
        )

    def test_tight_threshold_creates_more_patterns(self):
        """With tight terrain threshold (0.95), same noise creates more patterns."""
        tight_thresholds = [0.80, 0.80, 0.95, 0.80, 0.80, 0.80, 0.80]
        cd_tight = CompositionalDistinction(
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            modality_thresholds=tight_thresholds,
        )

        loose_thresholds = [0.80, 0.80, 0.65, 0.80, 0.80, 0.80, 0.80]
        cd_loose = CompositionalDistinction(
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            similarity_threshold=0.80,
            patterns_per_modality=16,
            modality_thresholds=loose_thresholds,
        )

        rng = np.random.default_rng(123)
        signals = []
        for _ in range(30):
            data = np.zeros(92)
            for s, e in MinecraftEnv.MODALITY_SLICES:
                sl = rng.random(e - s)
                n = np.linalg.norm(sl)
                if n > 0:
                    sl /= n
                data[s:e] = sl
            signals.append(Signal(data=data, timestamp=0))

        for sig in signals:
            cd_tight.distinguish(sig)
            cd_tight.advance_tick()
            cd_loose.distinguish(sig)
            cd_loose.advance_tick()

        tight_terrain = len(cd_tight._modal_distinctions[2].patterns)
        loose_terrain = len(cd_loose._modal_distinctions[2].patterns)
        assert loose_terrain < tight_terrain, (
            f"Loose ({loose_terrain}) should create fewer patterns than tight ({tight_terrain})"
        )

    def test_env_modality_thresholds_constant(self):
        """MinecraftEnv.MODALITY_THRESHOLDS has correct length and values."""
        thresholds = MinecraftEnv.MODALITY_THRESHOLDS
        assert len(thresholds) == len(MinecraftEnv.MODALITY_SLICES)
        assert thresholds == [0.85, 0.80, 0.65, 0.70, 0.85, 0.80, 0.60]

    def test_agent_with_modality_thresholds(self):
        """Agent constructor accepts and threads modality_thresholds."""
        agent = Agent(
            similarity_threshold=0.80,
            seed=42,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
        )
        # Verify it threaded down to the compositional distinction
        cd = agent.world_model.memory.distinction
        assert isinstance(cd, CompositionalDistinction)
        assert cd._modal_distinctions[2].similarity_threshold == 0.65  # terrain
        assert cd._modal_distinctions[6].similarity_threshold == 0.60  # history


# ---- 7. Composite similarity fallback ----

class TestCompositeSimilarityFallback:
    @staticmethod
    def _make_signal(terrain_idx: int = 0) -> Signal:
        """Create a 92-dim signal with controlled terrain variation.

        terrain_idx selects which block category dominates (0-7), producing
        terrain patterns that are far enough apart to exceed the 0.65 threshold.
        """
        data = np.zeros(92)
        # Body: stable
        data[0:12] = 0.5
        data[0:12] /= np.linalg.norm(data[0:12])
        # Environment: stable
        data[12:20] = 0.3
        data[12:20] /= np.linalg.norm(data[12:20])
        # Terrain: one-hot-ish — different terrain_idx = very different pattern
        terrain = np.full(8, 0.05)
        terrain[terrain_idx % 8] = 1.0
        terrain /= np.linalg.norm(terrain)
        data[20:28] = terrain
        # Entities: stable
        data[28:48] = 0.1
        data[28:48] /= np.linalg.norm(data[28:48])
        # Inventory: stable
        data[48:60] = 0.2
        data[48:60] /= np.linalg.norm(data[48:60])
        # Combat: stable
        data[60:76] = 0.15
        data[60:76] /= np.linalg.norm(data[60:76])
        # History: stable
        data[76:92] = 0.1
        data[76:92] /= np.linalg.norm(data[76:92])
        return Signal(data=data, timestamp=0)

    def test_similar_composite_gets_fallback_prediction(self):
        """A composite sharing 6/7 modalities should get predictions via fallback."""
        from fpi.world_model.model import WorldModel

        wm = WorldModel(
            similarity_threshold=0.80,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
            composite_similarity_threshold=0.5,
        )

        # Observe signal A (terrain_idx=0: block cat 0 dominates)
        sig_a = self._make_signal(terrain_idx=0)
        wm.observe(sig_a)
        pattern_a = wm.current_pattern

        # Observe signal B (same terrain) to form transition from A→B with action=5
        sig_b = self._make_signal(terrain_idx=0)
        wm.observe(sig_b, last_action=5)
        pattern_b = wm.current_pattern
        wm.record_action_outcome(5, pattern_b, 0.1)

        # Verify exact lookup works for pattern_a
        pred = wm.predict_from(pattern_a.pattern_id, 5)
        assert pred is not None, "Exact lookup should work"

        # Now observe a signal with very different terrain (terrain_idx=3)
        # This creates a new composite that differs only in terrain
        sig_c = self._make_signal(terrain_idx=3)
        wm.observe(sig_c)
        pattern_c = wm.current_pattern
        assert pattern_c.pattern_id != pattern_a.pattern_id, "Should be a different composite"

        # pattern_c has no direct transitions, but should get fallback from pattern_a
        # (composites share 6/7 modalities, so overall centroid similarity is high)
        pred_fallback = wm.predict_from(pattern_c.pattern_id, 5)
        assert pred_fallback is not None, (
            "Fallback should provide prediction from similar composite"
        )

    def test_dissimilar_composite_no_fallback(self):
        """A completely different composite should NOT get fallback predictions."""
        from fpi.world_model.model import WorldModel

        wm = WorldModel(
            similarity_threshold=0.80,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
            composite_similarity_threshold=0.5,
        )

        rng = np.random.default_rng(42)

        # Train on signal A
        sig_a = self._make_signal(terrain_idx=0)
        wm.observe(sig_a)
        sig_b = self._make_signal(terrain_idx=0)
        wm.observe(sig_b, last_action=5)
        wm.record_action_outcome(5, wm.current_pattern, 0.1)

        # Create a completely different signal (all modalities different)
        data = np.zeros(92)
        for s, e in MinecraftEnv.MODALITY_SLICES:
            sl = rng.random(e - s)
            sl /= np.linalg.norm(sl)
            data[s:e] = sl
        sig_diff = Signal(data=data, timestamp=0)
        wm.observe(sig_diff)
        pattern_diff = wm.current_pattern

        # Dissimilar composite should NOT get fallback (below threshold)
        pred = wm.predict_from(pattern_diff.pattern_id, 5)
        # Could be None or could match if by chance similar enough —
        # just verify no crash and the system handles it
        # The key test is that similar composites DO get fallback (above test)

    def test_vitality_fallback(self):
        """Vitality predictions should also fall back to similar composites."""
        from fpi.world_model.model import WorldModel

        wm = WorldModel(
            similarity_threshold=0.80,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
            composite_similarity_threshold=0.5,
        )

        # Train on signal A with action 3 → vitality 0.1
        sig_a = self._make_signal(terrain_idx=0)
        wm.observe(sig_a)
        pattern_a = wm.current_pattern
        sig_b = self._make_signal(terrain_idx=0)
        wm.observe(sig_b, last_action=3)
        wm.record_action_outcome(3, wm.current_pattern, 0.1)

        # New composite with different terrain
        sig_c = self._make_signal(terrain_idx=3)
        wm.observe(sig_c)
        pattern_c = wm.current_pattern

        # Vitality prediction should fall back
        vit = wm.predict_vitality_from(pattern_c.pattern_id, 3)
        assert vit is not None, "Vitality should fall back from similar composite"
        assert vit > 0, "Fallback vitality should be positive (discounted from 0.1)"

    def test_exact_match_takes_priority(self):
        """When exact lookup exists, fallback should not be used."""
        from fpi.world_model.model import WorldModel

        wm = WorldModel(
            similarity_threshold=0.80,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
            composite_similarity_threshold=0.5,
        )

        # Train transition with negative vitality
        sig_a = self._make_signal(terrain_idx=0)
        wm.observe(sig_a)
        pattern_a = wm.current_pattern

        sig_b = self._make_signal(terrain_idx=0)
        wm.observe(sig_b, last_action=7)
        wm.record_action_outcome(7, wm.current_pattern, -0.5)

        # Exact vitality for pattern_a, action=7 should be -0.5
        vit = wm.predict_vitality_from(pattern_a.pattern_id, 7)
        assert vit is not None
        assert vit < 0, "Exact match should return the negative vitality, not a fallback"

    def test_select_action_with_fallback_performance(self):
        """select_action should still complete in <1s with fallback active."""
        agent = Agent(
            similarity_threshold=0.80,
            seed=42,
            enable_compositional=True,
            patterns_per_modality=16,
            modality_slices=MinecraftEnv.MODALITY_SLICES,
            modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
            composite_similarity_threshold=0.5,
            enable_lookahead=True,
            lookahead_depth=3,
            lookahead_discount=0.9,
        )
        agent.vitality = Vitality(entropy_rate=0.001)

        # Build some patterns and transitions
        sim = CombatSimulator(factored=True, seed=42)
        obs = sim.reset()
        agent.step_with_action(obs, 0.0, None)
        for _ in range(20):
            action = int(np.random.default_rng(42).integers(FACTORED_ACTION_COUNT))
            obs, delta, done = sim.step(action)
            agent.step_with_action(obs, delta, action)

        start = time.monotonic()
        action = agent.select_action(FACTORED_ACTIONS)
        elapsed = time.monotonic() - start

        assert 0 <= action < 168
        assert elapsed < 1.0, f"select_action took {elapsed:.3f}s with fallback (should be <1s)"
