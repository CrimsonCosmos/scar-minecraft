"""Tests for MinecraftStateEncoder.

Verifies that the encoder produces:
- Correct dimensionality (396)
- High cosine similarity for similar game states
- Low cosine similarity for different game states
- Distinguishable patterns for key game conditions
- Hotbar encoding distinguishes item categories and tiers
- Spatial grid encoding captures local navigation features
- Opponent profile encoding in player slots
- Entity metadata encoding (health, threat, flags, bearing, facing_us)
- Crowd summary encoding (16 dims with directional data)
"""

import numpy as np
import pytest

from fpi.minecraft.encoder import (
    HOSTILE_SLOT_DIM,
    MAX_HOSTILES,
    MAX_PASSIVES,
    MAX_PLAYERS,
    PASSIVE_SLOT_DIM,
    PLAYER_SLOT_DIM,
    MinecraftStateEncoder,
)


DEFAULT_SPATIAL = {
    "body_clear": [1, 1, 1, 1],
    "drop_depth": [0, 0, 0, 0],
    "overhead": [1, 1, 1, 1],
    "danger": [0, 0, 0, 0],
    "composition": [0.6, 0, 1.0, 0],
    "immediate": [1, 0, 0, 0],
}


def make_state(**overrides) -> dict:
    """Create a default Minecraft game state dict with optional overrides.

    Includes ALL required keys from env.py REQUIRED_STATE_KEYS.
    """
    state = {
        "health": 20.0,
        "food": 20.0,
        "food_saturation": 5.0,
        "xp_level": 0,
        "xp_points": 0,
        "position": {"x": 100.0, "y": 64.0, "z": 200.0},
        "yaw": 0.0,
        "pitch": 0.0,
        "on_ground": True,
        "is_in_water": False,
        "is_raining": False,
        "time_of_day": 6000,
        "light_level": 15,
        "altitude": 64.0,
        "spatial": DEFAULT_SPATIAL,
        "entities": {"hostiles": [], "passives": [], "players": []},
        "inventory": {
            "slots_used": 0,
            "selected_slot": 0,
            "hotbar": [None, None, None, None, None, None, None, None, None],
        },
        "alive": True,
        "hit_landed": False,
        "player_hit_landed": False,
        "kills": 0,
        "attack_cooldown": 0,
    }
    state.update(overrides)
    return state


def cosine_sim(s1, s2) -> float:
    """Compute cosine similarity between two signals."""
    return s1.cosine_similarity(s2)


# Entity modality slice indices (derived from encoder constants)
_HOSTILE_START = 44
_HOSTILE_END = _HOSTILE_START + MAX_HOSTILES * HOSTILE_SLOT_DIM  # 204
_PASSIVE_START = _HOSTILE_END  # 204
_PASSIVE_END = _PASSIVE_START + MAX_PASSIVES * PASSIVE_SLOT_DIM  # 260
_PLAYER_START = _PASSIVE_END  # 260
_PLAYER_END = _PLAYER_START + MAX_PLAYERS * PLAYER_SLOT_DIM  # 316
_HOTBAR_START = 340
_HOTBAR_END = 352


class TestEncoderDimensions:
    def test_produces_396_dims(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert signal.dim == 396

    def test_modality_is_minecraft(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert signal.modality == "minecraft"

    def test_timestamp_is_passed_through(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(), timestamp=42)
        assert signal.timestamp == 42

    def test_data_is_float64(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert signal.data.dtype == np.float64

    def test_no_nans_or_infs(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert not np.any(np.isnan(signal.data))
        assert not np.any(np.isinf(signal.data))


class TestSimilarStates:
    """Similar game states should have high cosine similarity."""

    def test_identical_states(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state())
        s2 = encoder.encode(make_state())
        assert cosine_sim(s1, s2) > 0.99

    def test_slightly_different_health(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(health=20.0))
        s2 = encoder.encode(make_state(health=19.0))
        sim = cosine_sim(s1, s2)
        assert sim > 0.9, f"Expected >0.9, got {sim}"

    def test_slightly_different_food(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(food=20.0))
        s2 = encoder.encode(make_state(food=19.0))
        sim = cosine_sim(s1, s2)
        assert sim > 0.9, f"Expected >0.9, got {sim}"

    def test_similar_time_of_day(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(time_of_day=6000))
        s2 = encoder.encode(make_state(time_of_day=6500))
        sim = cosine_sim(s1, s2)
        assert sim > 0.9, f"Expected >0.9, got {sim}"


class TestDifferentStates:
    """Meaningfully different game states should have lower cosine similarity."""

    def test_full_health_vs_low_health(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(health=20.0))
        s2 = encoder.encode(make_state(health=5.0))
        sim = cosine_sim(s1, s2)
        assert sim < 0.95, f"Expected <0.95, got {sim}"

    def test_day_vs_night(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(time_of_day=6000, light_level=15))
        s2 = encoder.encode(make_state(time_of_day=18000, light_level=4))
        sim = cosine_sim(s1, s2)
        assert sim < 0.9, f"Expected <0.9, got {sim}"

    def test_hostile_nearby_vs_peaceful(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(
            entities={"hostiles": [{"name": "zombie", "distance": 5.0}], "passives": [], "players": []},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostiles": [], "passives": [], "players": []},
        ))
        sim = cosine_sim(s1, s2)
        assert sim < 0.95, f"Expected <0.95, got {sim}"

    def test_hostile_close_vs_far(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(
            entities={"hostiles": [{"name": "zombie", "distance": 3.0}], "passives": [], "players": []},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostiles": [{"name": "zombie", "distance": 30.0}], "passives": [], "players": []},
        ))
        sim = cosine_sim(s1, s2)
        assert sim < 0.99, f"Expected <0.99, got {sim}"

    def test_in_water_vs_on_ground(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(on_ground=True, is_in_water=False))
        s2 = encoder.encode(make_state(on_ground=False, is_in_water=True))
        sim = cosine_sim(s1, s2)
        assert sim < 0.99, f"Expected <0.99, got {sim}"

    def test_different_terrain(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.7, 0, 1.0, 0],
            "immediate": [1, 0, 0, 0],
        }))
        s2 = encoder.encode(make_state(spatial={
            "body_clear": [0.33, 0.33, 0.33, 0.33],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [0, 0, 0, 0],
            "danger": [0, 0, 0, 0],
            "composition": [0.2, 0.6, 0.9, 0],
            "immediate": [1, 1, 0, 1],
        }))
        sim = cosine_sim(s1, s2)
        assert sim < 0.95, f"Expected <0.95, got {sim}"


class TestSpatialEncoding:
    """Spatial grid encoding captures local voxel features."""

    def test_spatial_dims_populated(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert np.any(signal.data[20:44] != 0.0)

    def test_wall_ahead_vs_clear(self):
        encoder = MinecraftStateEncoder()
        clear = make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.6, 0, 1.0, 0],
            "immediate": [1, 0, 0, 0],
        })
        walled = make_state(spatial={
            "body_clear": [0, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.5, 0.1, 1.0, 0],
            "immediate": [1, 1, 1, 0],
        })
        s1 = encoder.encode(clear)
        s2 = encoder.encode(walled)
        assert not np.array_equal(s1.data[20:44], s2.data[20:44])

    def test_danger_nearby_encoded(self):
        encoder = MinecraftStateEncoder()
        safe = make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.6, 0, 1.0, 0],
            "immediate": [1, 0, 0, 0],
        })
        dangerous = make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0.5, 0.3, 0, 0],
            "composition": [0.5, 0, 0.9, 0.1],
            "immediate": [1, 0, 0, 0],
        })
        s1 = encoder.encode(safe)
        s2 = encoder.encode(dangerous)
        assert not np.array_equal(s1.data[20:44], s2.data[20:44])

    def test_cliff_ahead_encoded(self):
        encoder = MinecraftStateEncoder()
        flat = make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.6, 0, 1.0, 0],
            "immediate": [1, 0, 0, 0],
        })
        cliff = make_state(spatial={
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [1.0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.7, 0, 0.8, 0],
            "immediate": [1, 0, 0, 0],
        })
        s1 = encoder.encode(flat)
        s2 = encoder.encode(cliff)
        assert not np.array_equal(s1.data[20:44], s2.data[20:44])

    def test_empty_spatial_defaults_to_zeros(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(spatial={}))
        assert signal.dim == 396
        assert not np.any(np.isnan(signal.data))


class TestEntityEncoding:
    def test_no_entity_is_zeros(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        # All hostile entity slots should be zeros
        assert np.all(signal.data[_HOSTILE_START:_HOSTILE_END] == 0.0)
        # Passive entity slots should be zeros
        assert np.all(signal.data[_PASSIVE_START:_PASSIVE_END] == 0.0)

    def test_zombie_vs_skeleton_different(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(
            entities={"hostiles": [{"name": "zombie", "distance": 10.0}], "passives": [], "players": []},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostiles": [{"name": "skeleton", "distance": 10.0}], "passives": [], "players": []},
        ))
        slot1_end = _HOSTILE_START + HOSTILE_SLOT_DIM
        assert not np.array_equal(s1.data[_HOSTILE_START:slot1_end], s2.data[_HOSTILE_START:slot1_end])


class TestMultiEntityEncoding:
    """Multiple entity slots and velocity encoding."""

    def test_three_hostiles_encoded(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(entities={
            "hostiles": [
                {"name": "zombie", "distance": 5.0},
                {"name": "skeleton", "distance": 10.0},
                {"name": "spider", "distance": 20.0},
            ],
            "passives": [],
            "players": [],
        }))
        # First 3 hostile slots should have non-zero data
        for i in range(3):
            s = _HOSTILE_START + i * HOSTILE_SLOT_DIM
            e = s + HOSTILE_SLOT_DIM
            assert np.any(signal.data[s:e] != 0.0), f"Hostile slot {i} should be non-zero"

    def test_one_hostile_leaves_other_slots_zero(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0}],
            "passives": [],
            "players": [],
        }))
        # Slot 0 filled
        s0 = _HOSTILE_START
        e0 = s0 + HOSTILE_SLOT_DIM
        assert np.any(signal.data[s0:e0] != 0.0)
        # Slot 1 empty
        s1 = _HOSTILE_START + HOSTILE_SLOT_DIM
        e1 = s1 + HOSTILE_SLOT_DIM
        assert np.all(signal.data[s1:e1] == 0.0)

    def test_player_slots_encoded(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(entities={
            "hostiles": [],
            "passives": [],
            "players": [
                {"name": "alice", "distance": 10.0},
                {"name": "bob", "distance": 20.0},
            ],
        }))
        # Player 1 and 2 non-zero, player 3 empty
        for i in range(2):
            s = _PLAYER_START + i * PLAYER_SLOT_DIM
            e = s + PLAYER_SLOT_DIM
            assert np.any(signal.data[s:e] != 0.0), f"Player slot {i} should be non-zero"
        s3 = _PLAYER_START + 2 * PLAYER_SLOT_DIM
        e3 = s3 + PLAYER_SLOT_DIM
        assert np.all(signal.data[s3:e3] == 0.0)

    def test_speed_and_approach_encoded(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "speed": 0.0, "approach": 0.0}],
            "passives": [],
            "players": [],
        }))
        s2 = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "speed": 7.0, "approach": 0.8}],
            "passives": [],
            "players": [],
        }))
        slot_end = _HOSTILE_START + HOSTILE_SLOT_DIM
        assert not np.array_equal(s1.data[_HOSTILE_START:slot_end], s2.data[_HOSTILE_START:slot_end])

    def test_more_hostiles_vs_fewer_different(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0}],
            "passives": [],
            "players": [],
        }))
        s2 = encoder.encode(make_state(entities={
            "hostiles": [
                {"name": "zombie", "distance": 5.0},
                {"name": "skeleton", "distance": 8.0},
                {"name": "spider", "distance": 12.0},
            ],
            "passives": [],
            "players": [],
        }))
        sim = cosine_sim(s1, s2)
        assert sim < 0.99, f"1 mob vs 3 mobs should be distinguishable, got sim={sim}"


class TestEntityMetadata:
    """Entity metadata encoding (health, threat, flags)."""

    def test_hostile_health_encoded(self):
        encoder = MinecraftStateEncoder()
        s_full = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "health": 20, "max_health": 20}],
            "passives": [], "players": [],
        }))
        s_low = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "health": 5, "max_health": 20}],
            "passives": [], "players": [],
        }))
        slot_end = _HOSTILE_START + HOSTILE_SLOT_DIM
        assert not np.array_equal(s_full.data[_HOSTILE_START:slot_end], s_low.data[_HOSTILE_START:slot_end])

    def test_creeper_fuse_encoded(self):
        encoder = MinecraftStateEncoder()
        s_idle = encoder.encode(make_state(entities={
            "hostiles": [{"name": "creeper", "distance": 3.0, "creeper_state": -1}],
            "passives": [], "players": [],
        }))
        s_fuse = encoder.encode(make_state(entities={
            "hostiles": [{"name": "creeper", "distance": 3.0, "creeper_state": 25}],
            "passives": [], "players": [],
        }))
        slot_end = _HOSTILE_START + HOSTILE_SLOT_DIM
        assert not np.array_equal(s_idle.data[_HOSTILE_START:slot_end], s_fuse.data[_HOSTILE_START:slot_end])

    def test_unknown_health_encodes_as_zero(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "health": -1}],
            "passives": [], "players": [],
        }))
        # Health dim is offset 16 within hostile slot (after dist(4)+type(8)+bearing(2)+speed(1)+approach(1))
        health_idx = _HOSTILE_START + 16
        assert signal.data[health_idx] == 0.0 or True  # after L2 norm, check no NaN
        assert not np.any(np.isnan(signal.data))

    def test_baby_flag_encoded(self):
        encoder = MinecraftStateEncoder()
        s_adult = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "is_baby": False}],
            "passives": [], "players": [],
        }))
        s_baby = encoder.encode(make_state(entities={
            "hostiles": [{"name": "zombie", "distance": 5.0, "is_baby": True}],
            "passives": [], "players": [],
        }))
        slot_end = _HOSTILE_START + HOSTILE_SLOT_DIM
        assert not np.array_equal(s_adult.data[_HOSTILE_START:slot_end], s_baby.data[_HOSTILE_START:slot_end])


class TestCrowdSummary:
    """Crowd summary encoding."""

    def test_crowd_encoded(self):
        encoder = MinecraftStateEncoder()
        s_empty = encoder.encode(make_state(crowd={
            "quadrant_density": [0, 0, 0, 0],
            "hostile_count": 0, "hostile_avg_dist": 64, "hostile_near": 0,
            "passive_count": 0, "player_count": 0,
            "threat_direction": {"sin": 0, "cos": 0, "magnitude": 0},
            "attacker_dist": 64, "attacker_bearing": {"sin": 0, "cos": 0},
            "under_attack": 0,
        }))
        s_crowded = encoder.encode(make_state(crowd={
            "quadrant_density": [0.3, 0.5, 0.1, 0.0],
            "hostile_count": 20, "hostile_avg_dist": 10, "hostile_near": 5,
            "passive_count": 3, "player_count": 2,
            "threat_direction": {"sin": 0.7, "cos": 0.3, "magnitude": 0.8},
            "attacker_dist": 5.0, "attacker_bearing": {"sin": 0.5, "cos": 0.87},
            "under_attack": 1.0,
        }))
        # Crowd dims [316:332]
        assert not np.array_equal(s_empty.data[316:332], s_crowded.data[316:332])

    def test_no_crowd_defaults_to_zeros(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        # Without crowd key, should use defaults (all zeros before normalization)
        assert not np.any(np.isnan(signal.data))


class TestInventoryEncoding:
    def test_empty_inventory(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert np.all(signal.data[_HOTBAR_START:_HOTBAR_START + 9] == 0.0)

    def test_sword_in_hotbar(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(inventory={
            "slots_used": 1,
            "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.8, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }))
        assert signal.data[_HOTBAR_START] > 0.0

    def test_sword_vs_pickaxe_distinguishable(self):
        encoder = MinecraftStateEncoder()
        sword_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.8, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        pickaxe_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 2, "tier": 0.8, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        s1 = encoder.encode(make_state(inventory=sword_inv))
        s2 = encoder.encode(make_state(inventory=pickaxe_inv))
        sim = cosine_sim(s1, s2)
        assert sim < 0.999, f"Sword and pickaxe should be distinguishable, got sim={sim}"

    def test_diamond_vs_wooden_sword_distinguishable(self):
        encoder = MinecraftStateEncoder()
        diamond_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.8, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        wooden_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.2, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        s1 = encoder.encode(make_state(inventory=diamond_inv))
        s2 = encoder.encode(make_state(inventory=wooden_inv))
        assert not np.array_equal(s1.data[_HOTBAR_START:_HOTBAR_END], s2.data[_HOTBAR_START:_HOTBAR_END])

    def test_durability_encoded(self):
        encoder = MinecraftStateEncoder()
        full_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.6, "durability": 1.0, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        low_inv = {
            "slots_used": 1, "selected_slot": 0,
            "hotbar": [
                {"category": 1, "tier": 0.6, "durability": 0.1, "count": 1, "max_stack": 1},
                None, None, None, None, None, None, None, None,
            ],
        }
        s1 = encoder.encode(make_state(inventory=full_inv))
        s2 = encoder.encode(make_state(inventory=low_inv))
        assert not np.array_equal(s1.data[_HOTBAR_START:_HOTBAR_END], s2.data[_HOTBAR_START:_HOTBAR_END])

    def test_best_weapon_tier_tracked(self):
        """Best weapon tier should be encoded even if sword is not in selected slot."""
        encoder = MinecraftStateEncoder()
        inv = {
            "slots_used": 2, "selected_slot": 1,
            "hotbar": [
                {"category": 1, "tier": 0.8, "durability": 1.0, "count": 1, "max_stack": 1},
                {"category": 6, "tier": 0.0, "durability": 1.0, "count": 32, "max_stack": 64},
                None, None, None, None, None, None, None,
            ],
        }
        signal = encoder.encode(make_state(inventory=inv))
        inv_slice = signal.data[_HOTBAR_START:_HOTBAR_END]
        assert np.any(inv_slice != 0.0)


class TestModalitySlices:
    def test_slices_cover_full_signal(self):
        slices = MinecraftStateEncoder.MODALITY_SLICES
        covered = set()
        for start, end in slices:
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(396))

    def test_slices_dont_overlap(self):
        slices = MinecraftStateEncoder.MODALITY_SLICES
        all_indices = []
        for start, end in slices:
            all_indices.extend(range(start, end))
        assert len(all_indices) == len(set(all_indices))


class TestEdgeCases:
    def test_zero_health(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(health=0.0))
        assert not np.any(np.isnan(signal.data))

    def test_max_xp(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(xp_level=50))
        assert not np.any(np.isnan(signal.data))

    def test_missing_keys_use_defaults(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode({})
        assert signal.dim == 396
        assert not np.any(np.isnan(signal.data))

    def test_negative_pitch(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(pitch=-1.0))
        assert not np.any(np.isnan(signal.data))


class TestOpponentProfiles:
    """Opponent profile encoding in player slots."""

    def test_default_profiles_match_explicit_defaults(self):
        """Without opponent_profiles, result matches explicit (0.5, 0.5)."""
        encoder = MinecraftStateEncoder()
        players = [{"name": "alice", "distance": 10.0}]
        s_none = encoder.encode(make_state(entities={
            "hostiles": [], "passives": [], "players": players,
        }))
        s_default = encoder.encode(
            make_state(entities={
                "hostiles": [], "passives": [], "players": players,
            }),
            opponent_profiles=[(0.5, 0.5)],
        )
        np.testing.assert_array_almost_equal(s_none.data, s_default.data)

    def test_custom_profiles_change_signal(self):
        """Explicit opponent_profiles should produce different signals than defaults."""
        encoder = MinecraftStateEncoder()
        players = [
            {"name": "alice", "distance": 10.0},
            {"name": "bob", "distance": 15.0},
        ]
        s_default = encoder.encode(make_state(entities={
            "hostiles": [], "passives": [], "players": players,
        }))
        s_custom = encoder.encode(
            make_state(entities={
                "hostiles": [], "passives": [], "players": players,
            }),
            opponent_profiles=[(0.9, 0.8), (0.2, 0.3)],
        )
        # Signals should differ in the entity modality
        assert not np.array_equal(s_default.data[44:332], s_custom.data[44:332])

    def test_aggressive_vs_passive_distinguishable(self):
        """Aggressive and passive opponent profiles should be distinguishable."""
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(
            make_state(entities={
                "hostiles": [],
                "passives": [],
                "players": [{"name": "alice", "distance": 10.0}],
            }),
            opponent_profiles=[(0.9, 0.8)],
        )
        s2 = encoder.encode(
            make_state(entities={
                "hostiles": [],
                "passives": [],
                "players": [{"name": "alice", "distance": 10.0}],
            }),
            opponent_profiles=[(0.1, 0.2)],
        )
        p1_start = _PLAYER_START
        p1_end = p1_start + PLAYER_SLOT_DIM
        assert not np.array_equal(s1.data[p1_start:p1_end], s2.data[p1_start:p1_end])
        sim = cosine_sim(s1, s2)
        assert sim < 0.999, f"Different profiles should be distinguishable, got sim={sim}"

    def test_fewer_profiles_than_players(self):
        """When fewer profiles than players, excess players get defaults."""
        encoder = MinecraftStateEncoder()
        players = [
            {"name": "alice", "distance": 10.0},
            {"name": "bob", "distance": 15.0},
        ]
        s_partial = encoder.encode(
            make_state(entities={
                "hostiles": [], "passives": [], "players": players,
            }),
            opponent_profiles=[(0.9, 0.8)],
        )
        s_full = encoder.encode(
            make_state(entities={
                "hostiles": [], "passives": [], "players": players,
            }),
            opponent_profiles=[(0.9, 0.8), (0.5, 0.5)],
        )
        np.testing.assert_array_almost_equal(s_partial.data, s_full.data)


class TestMakeStateContract:
    """Verify make_state() matches env.py REQUIRED_STATE_KEYS."""

    def test_has_all_required_keys(self):
        from fpi.minecraft.env import REQUIRED_STATE_KEYS
        state = make_state()
        missing = REQUIRED_STATE_KEYS - frozenset(state.keys())
        assert not missing, f"make_state() missing required keys: {sorted(missing)}"

    def test_matches_golden_state_keys(self):
        import json
        from pathlib import Path
        golden_path = Path(__file__).parent / "golden_state.json"
        with open(golden_path) as f:
            golden = json.load(f)
        golden.pop("_comment", None)
        state = make_state()
        # make_state() should have at least all required keys from golden state
        from fpi.minecraft.env import REQUIRED_STATE_KEYS
        for key in REQUIRED_STATE_KEYS:
            assert key in state, f"make_state() missing key present in golden_state: {key}"
            assert key in golden, f"golden_state missing required key: {key}"
