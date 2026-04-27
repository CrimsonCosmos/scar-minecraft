"""Data pipeline contract tests — validates golden_state.json through
the full Python encoding pipeline: env.py validation → encoder.py → 428-dim signal.

Run: python -m pytest tests/test_data_pipeline.py -x -q
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fpi.minecraft.encoder import (
    CROWD_DIM,
    HOSTILE_SLOT_DIM,
    MAX_HOSTILES,
    MAX_PASSIVES,
    MAX_PLAYERS,
    PASSIVE_SLOT_DIM,
    PLAYER_SLOT_DIM,
    SPATIAL_DIM,
    MinecraftStateEncoder,
)
from fpi.minecraft.env import (
    KNOWN_STATE_KEYS,
    OPTIONAL_STATE_KEYS,
    REQUIRED_STATE_KEYS,
    MinecraftEnv,
    compute_energy_delta,
)

GOLDEN_PATH = Path(__file__).parent / "golden_state.json"


@pytest.fixture
def golden_state():
    with open(GOLDEN_PATH) as f:
        state = json.load(f)
    # Remove non-state metadata
    state.pop("_comment", None)
    return state


@pytest.fixture
def encoder():
    return MinecraftStateEncoder()


# ── Schema validation ───────────────────────────────────────────────


class TestSchemaValidation:
    """Golden state must pass env.py schema checks."""

    def test_all_required_keys_present(self, golden_state):
        state_keys = frozenset(golden_state.keys())
        missing = REQUIRED_STATE_KEYS - state_keys
        assert not missing, f"Missing required keys: {sorted(missing)}"

    def test_no_unknown_keys(self, golden_state):
        state_keys = frozenset(golden_state.keys())
        unknown = {k for k in state_keys - KNOWN_STATE_KEYS if not k.startswith("_")}
        assert not unknown, f"Unknown keys: {sorted(unknown)}"

    def test_optional_keys_are_recognized(self, golden_state):
        state_keys = frozenset(golden_state.keys())
        optional_present = state_keys & OPTIONAL_STATE_KEYS
        # At minimum, position and crowd should be present
        assert "position" in optional_present
        assert "crowd" in optional_present

    def test_required_and_optional_are_disjoint(self):
        overlap = REQUIRED_STATE_KEYS & OPTIONAL_STATE_KEYS
        assert not overlap, f"Keys in both required and optional: {sorted(overlap)}"


# ── Encoder output ──────────────────────────────────────────────────


class TestEncoderOutput:
    """Golden state produces correct 396-dim encoded signal."""

    def test_output_dimension(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        assert signal.dim == 396

    def test_no_nans(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        assert not np.any(np.isnan(signal.data))

    def test_no_infs(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        assert not np.any(np.isinf(signal.data))

    def test_data_is_float64(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        assert signal.data.dtype == np.float64


# ── Modality slices ─────────────────────────────────────────────────


class TestModalitySlices:
    """Verify modality slice dimensions add up correctly."""

    def test_base_slices_cover_396(self, encoder):
        covered = set()
        for start, end in encoder.MODALITY_SLICES:
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(396))

    def test_env_slices_cover_428(self):
        covered = set()
        for start, end in MinecraftEnv.MODALITY_SLICES:
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(428))

    def test_body_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[0] == (0, 12)

    def test_environment_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[1] == (12, 20)

    def test_terrain_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[2] == (20, 44)

    def test_entity_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[3] == (44, 332)

    def test_inventory_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[4] == (332, 352)

    def test_combat_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[5] == (352, 364)

    def test_self_awareness_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[6] == (364, 380)

    def test_threat_dynamics_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[7] == (380, 396)

    def test_vision_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[8] == (396, 412)

    def test_history_slice(self):
        assert MinecraftEnv.MODALITY_SLICES[9] == (412, 428)


# ── Entity dimension math ──────────────────────────────────────────


class TestEntityDimensions:
    """Verify entity slot dimensions add up to the entity modality."""

    def test_hostile_total(self):
        assert MAX_HOSTILES * HOSTILE_SLOT_DIM == 160

    def test_passive_total(self):
        assert MAX_PASSIVES * PASSIVE_SLOT_DIM == 56

    def test_player_total(self):
        assert MAX_PLAYERS * PLAYER_SLOT_DIM == 56

    def test_entity_modality_size(self):
        # 160 hostile + 56 passive + 56 player + 16 crowd = 288
        entity_total = (
            MAX_HOSTILES * HOSTILE_SLOT_DIM
            + MAX_PASSIVES * PASSIVE_SLOT_DIM
            + MAX_PLAYERS * PLAYER_SLOT_DIM
            + CROWD_DIM  # 16
        )
        entity_slice = MinecraftEnv.MODALITY_SLICES[3]
        assert entity_total == entity_slice[1] - entity_slice[0]

    def test_spatial_dim_matches_slice(self):
        terrain_slice = MinecraftEnv.MODALITY_SLICES[2]
        assert SPATIAL_DIM == terrain_slice[1] - terrain_slice[0]


# ── Per-modality normalization ──────────────────────────────────────


class TestModalityNormalization:
    """Each non-zero modality slice should be L2-normalized."""

    def test_body_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[0:12]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6

    def test_environment_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[12:20]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6

    def test_terrain_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[20:44]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6

    def test_entity_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[44:332]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6

    def test_inventory_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[332:352]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6

    def test_combat_normalized(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        slc = signal.data[352:364]
        if np.linalg.norm(slc) > 0:
            assert abs(np.linalg.norm(slc) - 1.0) < 1e-6


# ── Reward computation ──────────────────────────────────────────────


class TestRewardComputation:
    """compute_energy_delta works with golden state."""

    def test_same_state_near_zero(self, golden_state):
        delta = compute_energy_delta(golden_state, golden_state)
        # Same state → only idle penalty, very small
        assert abs(delta) < 0.05

    def test_death_returns_minus_one(self, golden_state):
        dead = {**golden_state, "alive": False}
        delta = compute_energy_delta(golden_state, dead)
        assert delta == -1.0

    def test_health_loss_is_negative(self, golden_state):
        damaged = {**golden_state, "health": 10.0}
        delta = compute_energy_delta(golden_state, damaged)
        assert delta < 0

    def test_hit_landed_is_positive(self, golden_state):
        hit = {**golden_state, "hit_landed": True}
        delta = compute_energy_delta(golden_state, hit)
        # Hit reward (0.1) should dominate idle penalty
        assert delta > 0

    def test_returns_float(self, golden_state):
        delta = compute_energy_delta(golden_state, golden_state)
        assert isinstance(delta, float)


# ── Golden state entity encoding ────────────────────────────────────


class TestGoldenEntityEncoding:
    """Golden state entities produce non-zero encodings in the right slots."""

    def test_hostile_slots_populated(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        # 3 hostiles in golden state → first 3 slots non-zero, remaining zero
        for i in range(3):
            start = 44 + i * HOSTILE_SLOT_DIM
            end = start + HOSTILE_SLOT_DIM
            assert np.any(signal.data[start:end] != 0.0), f"Hostile slot {i} empty"
        for i in range(3, 8):
            start = 44 + i * HOSTILE_SLOT_DIM
            end = start + HOSTILE_SLOT_DIM
            assert np.all(signal.data[start:end] == 0.0), f"Hostile slot {i} should be empty"

    def test_passive_slots_populated(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        # 1 passive → slot 0 non-zero, slots 1-3 zero
        start0 = 204
        assert np.any(signal.data[start0:start0 + PASSIVE_SLOT_DIM] != 0.0)
        for i in range(1, 4):
            start = 204 + i * PASSIVE_SLOT_DIM
            assert np.all(signal.data[start:start + PASSIVE_SLOT_DIM] == 0.0)

    def test_player_slots_populated(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        # 1 player → slot 0 non-zero, slots 1-3 zero
        start0 = 260
        assert np.any(signal.data[start0:start0 + PLAYER_SLOT_DIM] != 0.0)
        for i in range(1, 4):
            start = 260 + i * PLAYER_SLOT_DIM
            assert np.all(signal.data[start:start + PLAYER_SLOT_DIM] == 0.0)

    def test_crowd_encoded(self, golden_state, encoder):
        signal = encoder.encode(golden_state)
        # Crowd at [316:332] should be non-zero (3 hostiles, 1 passive, 1 player)
        assert np.any(signal.data[316:332] != 0.0)
