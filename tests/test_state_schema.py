"""Tests for state dict schema validation."""

import logging

import pytest

from fpi.minecraft.env import (
    KNOWN_STATE_KEYS,
    OPTIONAL_STATE_KEYS,
    REQUIRED_STATE_KEYS,
    _validate_state_schema,
)
from fpi.minecraft.fast_train import CombatArena


def _make_complete_state() -> dict:
    """Minimal complete state dict with all required keys."""
    return {
        "health": 20.0,
        "food": 20.0,
        "alive": True,
        "time_of_day": 6000,
        "light_level": 15,
        "yaw": 0.0,
        "pitch": 0.0,
        "on_ground": True,
        "is_in_water": False,
        "is_raining": False,
        "altitude": 64.0,
        "spatial": {
            "body_clear": [1, 1, 1, 1],
            "drop_depth": [0, 0, 0, 0],
            "overhead": [1, 1, 1, 1],
            "danger": [0, 0, 0, 0],
            "composition": [0.6, 0, 1.0, 0],
            "immediate": [1, 0, 0, 0],
        },
        "entities": {"hostiles": [], "passives": [], "players": []},
        "inventory": {"slots_used": 0, "selected_slot": 0, "hotbar": [None] * 9},
        "xp_level": 0,
        "xp_points": 0,
        "attack_cooldown": 0,
        "hit_landed": False,
        "player_hit_landed": False,
        "kills": 0,
    }


class TestSchemaConstants:
    def test_required_and_optional_disjoint(self):
        overlap = REQUIRED_STATE_KEYS & OPTIONAL_STATE_KEYS
        assert not overlap, f"Keys in both required and optional: {overlap}"

    def test_known_is_union(self):
        assert KNOWN_STATE_KEYS == REQUIRED_STATE_KEYS | OPTIONAL_STATE_KEYS

    def test_required_keys_count(self):
        assert len(REQUIRED_STATE_KEYS) == 20


class TestValidation:
    def test_complete_state_no_warnings(self, caplog):
        state = _make_complete_state()
        with caplog.at_level(logging.WARNING, logger="fpi.minecraft.env"):
            _validate_state_schema(state)
        assert not caplog.records

    def test_missing_required_key_warns(self, caplog):
        state = _make_complete_state()
        del state["health"]
        with caplog.at_level(logging.WARNING, logger="fpi.minecraft.env"):
            _validate_state_schema(state)
        assert any("missing" in r.message.lower() for r in caplog.records)
        assert any("health" in r.message for r in caplog.records)

    def test_unknown_key_warns(self, caplog):
        state = _make_complete_state()
        state["brand_new_js_key"] = 42
        with caplog.at_level(logging.WARNING, logger="fpi.minecraft.env"):
            _validate_state_schema(state)
        assert any("unknown" in r.message.lower() for r in caplog.records)
        assert any("brand_new_js_key" in r.message for r in caplog.records)

    def test_internal_keys_ignored(self, caplog):
        """Keys starting with _ are internal (e.g., _idle_ticks) and not flagged."""
        state = _make_complete_state()
        state["_idle_ticks"] = 0
        state["_internal_thing"] = True
        with caplog.at_level(logging.WARNING, logger="fpi.minecraft.env"):
            _validate_state_schema(state)
        assert not caplog.records

    def test_optional_keys_not_flagged(self, caplog):
        state = _make_complete_state()
        state["food_saturation"] = 5.0
        state["type"] = "state"
        with caplog.at_level(logging.WARNING, logger="fpi.minecraft.env"):
            _validate_state_schema(state)
        assert not caplog.records


class TestSimStateCompleteness:
    def test_fast_train_state_has_required_keys(self):
        """The 2D sim's _build_state() should produce all required keys."""
        arena = CombatArena(num_zombies=2, seed=42)
        state = arena._build_state()
        state_keys = frozenset(state.keys())
        missing = REQUIRED_STATE_KEYS - state_keys
        assert not missing, f"fast_train state missing required keys: {missing}"

    def test_fast_train_spatial_has_variation(self):
        """Spatial encoding should vary across different terrain types."""
        arena = CombatArena(num_zombies=2, seed=42)
        seen_types = set()
        spatial_norms = []
        for _ in range(50):
            arena.reset()
            seen_types.add(arena._terrain_type)
            state = arena._build_state()
            spatial = state["spatial"]
            # Flatten spatial to a vector and check it's not always the same
            vec = (
                spatial["body_clear"]
                + spatial["drop_depth"]
                + spatial["overhead"]
                + spatial["composition"]
                + spatial["immediate"]
            )
            spatial_norms.append(tuple(round(v, 2) for v in vec))

        # Should see multiple terrain types across 50 resets
        assert len(seen_types) >= 2, f"Only saw terrain types: {seen_types}"
        # Should see spatial variation
        unique_spatials = len(set(spatial_norms))
        assert unique_spatials >= 3, f"Only {unique_spatials} unique spatial vectors in 50 resets"
