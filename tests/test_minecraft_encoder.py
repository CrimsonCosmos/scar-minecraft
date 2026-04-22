"""Tests for MinecraftStateEncoder.

Verifies that the encoder produces:
- Correct dimensionality (76)
- High cosine similarity for similar game states
- Low cosine similarity for different game states
- Distinguishable patterns for key game conditions
"""

import numpy as np
import pytest

from fpi.minecraft.encoder import MinecraftStateEncoder


def make_state(**overrides) -> dict:
    """Create a default Minecraft game state dict with optional overrides."""
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
        "block_composition": {
            "air": 0.4,
            "stone": 0.1,
            "dirt": 0.3,
            "wood": 0.05,
            "water": 0.0,
            "ore": 0.0,
            "danger": 0.0,
            "other": 0.15,
        },
        "entities": {"hostile": None, "passive": None, "player": None},
        "inventory": {
            "slots_used": 0,
            "has_weapon": False,
            "has_food": False,
            "has_wood": False,
            "has_tool": False,
        },
        "alive": True,
    }
    state.update(overrides)
    return state


def cosine_sim(s1, s2) -> float:
    """Compute cosine similarity between two signals."""
    return s1.cosine_similarity(s2)


class TestEncoderDimensions:
    def test_produces_76_dims(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        assert signal.dim == 76

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
            entities={"hostile": {"name": "zombie", "distance": 5.0}, "passive": None, "player": None},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostile": None, "passive": None, "player": None},
        ))
        sim = cosine_sim(s1, s2)
        assert sim < 0.95, f"Expected <0.95, got {sim}"

    def test_hostile_close_vs_far(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(
            entities={"hostile": {"name": "zombie", "distance": 3.0}, "passive": None, "player": None},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostile": {"name": "zombie", "distance": 30.0}, "passive": None, "player": None},
        ))
        sim = cosine_sim(s1, s2)
        assert sim < 0.97, f"Expected <0.97, got {sim}"

    def test_in_water_vs_on_ground(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(on_ground=True, is_in_water=False))
        s2 = encoder.encode(make_state(on_ground=False, is_in_water=True))
        sim = cosine_sim(s1, s2)
        assert sim < 0.99, f"Expected <0.99, got {sim}"

    def test_different_terrain(self):
        encoder = MinecraftStateEncoder()
        # Forest terrain
        s1 = encoder.encode(make_state(block_composition={
            "air": 0.3, "stone": 0.0, "dirt": 0.2, "wood": 0.4,
            "water": 0.0, "ore": 0.0, "danger": 0.0, "other": 0.1,
        }))
        # Cave terrain
        s2 = encoder.encode(make_state(block_composition={
            "air": 0.2, "stone": 0.6, "dirt": 0.0, "wood": 0.0,
            "water": 0.0, "ore": 0.1, "danger": 0.0, "other": 0.1,
        }))
        sim = cosine_sim(s1, s2)
        assert sim < 0.95, f"Expected <0.95, got {sim}"


class TestEntityEncoding:
    def test_no_entity_is_zeros(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        # Hostile entity slice [28:36] should be all zeros
        assert np.all(signal.data[28:36] == 0.0)
        # Passive entity slice [36:44] should be all zeros
        assert np.all(signal.data[36:44] == 0.0)

    def test_zombie_vs_skeleton_different(self):
        encoder = MinecraftStateEncoder()
        s1 = encoder.encode(make_state(
            entities={"hostile": {"name": "zombie", "distance": 10.0}, "passive": None, "player": None},
        ))
        s2 = encoder.encode(make_state(
            entities={"hostile": {"name": "skeleton", "distance": 10.0}, "passive": None, "player": None},
        ))
        # Different entity type should create different signals
        assert not np.array_equal(s1.data[28:36], s2.data[28:36])


class TestInventoryEncoding:
    def test_empty_inventory(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state())
        # Inventory flags [56:60] should all be 0
        assert signal.data[56] == 0.0  # no weapon
        assert signal.data[57] == 0.0  # no food
        assert signal.data[58] == 0.0  # no tool
        assert signal.data[59] == 0.0  # no wood

    def test_equipped_inventory(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(inventory={
            "slots_used": 10,
            "has_weapon": True,
            "has_food": True,
            "has_wood": False,
            "has_tool": True,
        }))
        # After per-modality L2 normalization, flags are no longer exactly 1.0
        # but the relative pattern should be: weapon > 0, food > 0, tool > 0, wood == 0
        assert signal.data[56] > 0.0  # weapon
        assert signal.data[57] > 0.0  # food
        assert signal.data[58] > 0.0  # tool
        assert signal.data[59] == 0.0  # no wood


class TestModalitySlices:
    def test_slices_cover_full_signal(self):
        slices = MinecraftStateEncoder.MODALITY_SLICES
        # Should cover [0, 76) with no gaps
        covered = set()
        for start, end in slices:
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(76))

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
        signal = encoder.encode({})  # Empty state
        assert signal.dim == 76
        assert not np.any(np.isnan(signal.data))

    def test_negative_pitch(self):
        encoder = MinecraftStateEncoder()
        signal = encoder.encode(make_state(pitch=-1.0))
        assert not np.any(np.isnan(signal.data))
