"""Tests for OpponentTracker — per-player behavioral profiling.

Covers:
1. Profile creation and defaults
2. Aggression EMA from approach velocity
3. Skill EMA from damage attribution
4. Encounter tracking
5. Kill/death attribution
6. get_profiles_for_state ordering
"""

from __future__ import annotations

import pytest

from fpi.minecraft.opponent import OpponentProfile, OpponentTracker


def make_state(
    health: float = 20.0,
    alive: bool = True,
    players: list[dict] | None = None,
    player_hit_landed: bool = False,
) -> dict:
    """Create a minimal state dict for opponent tracking tests."""
    return {
        "health": health,
        "alive": alive,
        "player_hit_landed": player_hit_landed,
        "entities": {
            "hostiles": [],
            "passives": [],
            "players": players or [],
        },
    }


class TestProfileDefaults:
    def test_new_profile_defaults(self):
        p = OpponentProfile(username="alice")
        assert p.aggression == 0.5
        assert p.skill == 0.5
        assert p.encounters == 0
        assert p.kills_on_us == 0
        assert p.deaths_to_us == 0

    def test_unknown_player_returns_none(self):
        tracker = OpponentTracker()
        assert tracker.get_profile("unknown") is None

    def test_profiles_for_empty_state(self):
        tracker = OpponentTracker()
        result = tracker.get_profiles_for_state(make_state(players=[]))
        assert result == []


class TestAggressionTracking:
    def test_approaching_player_increases_aggression(self):
        tracker = OpponentTracker()
        # Player approaches from 15 to 5 blocks over 10 ticks
        prev = make_state(players=[{"name": "alice", "distance": 15.0}])
        for step in range(10):
            dist = 15.0 - step * 1.0
            curr = make_state(players=[{"name": "alice", "distance": dist}])
            tracker.update(curr, prev, step)
            prev = curr

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert profile.aggression > 0.5, (
            f"Approaching player should have aggression > 0.5, got {profile.aggression}"
        )

    def test_fleeing_player_decreases_aggression(self):
        tracker = OpponentTracker()
        # Player retreats from 5 to 20 blocks over 10 ticks
        prev = make_state(players=[{"name": "alice", "distance": 5.0}])
        for step in range(10):
            dist = 5.0 + step * 1.5
            curr = make_state(players=[{"name": "alice", "distance": dist}])
            tracker.update(curr, prev, step)
            prev = curr

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert profile.aggression < 0.5, (
            f"Fleeing player should have aggression < 0.5, got {profile.aggression}"
        )

    def test_stationary_player_stays_neutral(self):
        tracker = OpponentTracker()
        prev = make_state(players=[{"name": "alice", "distance": 10.0}])
        for step in range(10):
            curr = make_state(players=[{"name": "alice", "distance": 10.0}])
            tracker.update(curr, prev, step)
            prev = curr

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert 0.45 <= profile.aggression <= 0.55, (
            f"Stationary player should have ~0.5 aggression, got {profile.aggression}"
        )


class TestSkillTracking:
    def test_damage_increases_skill(self):
        tracker = OpponentTracker()
        # Repeated high damage pushes skill above 0.5 over several ticks
        prev = make_state(
            health=20.0,
            players=[{"name": "alice", "distance": 3.0}],
        )
        for step in range(5):
            # Player hits us for 10 damage each tick (high damage)
            curr = make_state(
                health=10.0,
                players=[{"name": "alice", "distance": 3.0}],
            )
            tracker.update(curr, prev, step=step)
            prev = make_state(
                health=20.0,  # heal back up for next tick
                players=[{"name": "alice", "distance": 3.0}],
            )

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert profile.skill > 0.5, (
            f"Player dealing repeated damage should have skill > 0.5, got {profile.skill}"
        )

    def test_no_damage_keeps_skill_default(self):
        tracker = OpponentTracker()
        prev = make_state(players=[{"name": "alice", "distance": 3.0}])
        curr = make_state(players=[{"name": "alice", "distance": 3.0}])
        tracker.update(curr, prev, step=1)

        profile = tracker.get_profile("alice")
        assert profile is not None
        # No damage dealt = skill stays at default
        assert profile.skill == pytest.approx(0.5)


class TestEncounterTracking:
    def test_entering_combat_range_counts_encounter(self):
        tracker = OpponentTracker()
        # Player enters combat range (<10 blocks)
        prev = make_state(players=[{"name": "alice", "distance": 15.0}])
        curr = make_state(players=[{"name": "alice", "distance": 8.0}])
        tracker.update(curr, prev, step=1)

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert profile.encounters == 1

    def test_staying_in_range_no_new_encounter(self):
        tracker = OpponentTracker()
        prev = make_state(players=[{"name": "alice", "distance": 8.0}])
        curr = make_state(players=[{"name": "alice", "distance": 8.0}])
        # First tick establishes _in_combat
        tracker.update(curr, prev, step=1)
        # Same range — no new encounter
        tracker.update(curr, curr, step=2)

        profile = tracker.get_profile("alice")
        assert profile.encounters == 1  # not 2

    def test_leave_and_reenter_counts_new_encounter(self):
        tracker = OpponentTracker()
        prev = make_state(players=[{"name": "alice", "distance": 15.0}])
        # Enter
        curr = make_state(players=[{"name": "alice", "distance": 8.0}])
        tracker.update(curr, prev, step=1)
        # Leave
        prev2 = curr
        curr2 = make_state(players=[{"name": "alice", "distance": 15.0}])
        tracker.update(curr2, prev2, step=2)
        # Re-enter
        prev3 = curr2
        curr3 = make_state(players=[{"name": "alice", "distance": 8.0}])
        tracker.update(curr3, prev3, step=3)

        profile = tracker.get_profile("alice")
        assert profile.encounters == 2


class TestKillDeathAttribution:
    def test_death_attributed_to_nearest_player(self):
        tracker = OpponentTracker()
        prev = make_state(
            health=5.0,
            alive=True,
            players=[
                {"name": "alice", "distance": 3.0},
                {"name": "bob", "distance": 20.0},
            ],
        )
        curr = make_state(
            health=0.0,
            alive=False,
            players=[
                {"name": "alice", "distance": 3.0},
                {"name": "bob", "distance": 20.0},
            ],
        )
        tracker.update(curr, prev, step=1)

        assert tracker.get_profile("alice").kills_on_us == 1
        assert tracker.get_profile("bob").kills_on_us == 0

    def test_kill_attributed_when_player_disappears_after_hit(self):
        tracker = OpponentTracker()
        prev = make_state(
            players=[{"name": "alice", "distance": 3.0}],
            player_hit_landed=False,
        )
        # We hit alice and she disappears (killed)
        curr = make_state(
            players=[],
            player_hit_landed=True,
        )
        # Need to have a profile first
        tracker.update(prev, make_state(players=[]), step=0)
        tracker.update(curr, prev, step=1)

        profile = tracker.get_profile("alice")
        assert profile is not None
        assert profile.deaths_to_us == 1


class TestProfilesForState:
    def test_order_matches_state_players(self):
        tracker = OpponentTracker()
        # Create profiles with different stats
        prev = make_state(players=[
            {"name": "alice", "distance": 3.0},
            {"name": "bob", "distance": 15.0},
        ])
        curr = make_state(
            health=15.0,  # took damage
            players=[
                {"name": "alice", "distance": 3.0},
                {"name": "bob", "distance": 15.0},
            ],
        )
        tracker.update(curr, prev, step=1)

        profiles = tracker.get_profiles_for_state(curr)
        assert len(profiles) == 2
        # Each should be (aggression, skill) tuples
        assert len(profiles[0]) == 2
        assert len(profiles[1]) == 2

    def test_unknown_player_gets_defaults(self):
        tracker = OpponentTracker()
        state = make_state(players=[{"name": "unknown", "distance": 10.0}])
        profiles = tracker.get_profiles_for_state(state)
        assert profiles == [(0.5, 0.5)]

    def test_reset_clears_profiles(self):
        tracker = OpponentTracker()
        prev = make_state(players=[{"name": "alice", "distance": 10.0}])
        tracker.update(prev, make_state(players=[]), step=0)
        assert tracker.get_profile("alice") is not None

        tracker.reset()
        assert tracker.get_profile("alice") is None
        assert tracker.profiles == {}
