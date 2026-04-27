"""Opponent modeling — per-player behavioral profiling.

Tracks per-player behavioral statistics from the state dict stream and
encodes them as (aggression, skill) pairs that feed into the FPI signal.

Rather than a separate per-opponent world model (too sparse — you might
fight each player only a few times), we profile opponents into behavioral
features. The existing FPI compositional pattern matching naturally creates
patterns for "facing aggressive opponent" vs "facing defensive opponent"
without any new learning machinery.

This is biologically plausible: humans don't learn separate brain circuits
per opponent — they categorize opponents into archetypes and adapt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# EMA learning rate for aggression and skill updates
_EMA_ALPHA = 0.3
# Maximum approach speed in blocks/second for normalization
_MAX_APPROACH_SPEED = 6.0  # sprinting speed ~5.6 b/s
# Distance threshold for "in combat range" (triggers encounter tracking)
_COMBAT_RANGE = 10.0
# Max damage a player can deal in one encounter tick (diamond sword crit)
_MAX_TICK_DAMAGE = 15.0
# Rolling window size for approach/damage samples
_SAMPLE_WINDOW = 50


@dataclass(slots=True)
class OpponentProfile:
    """Behavioral profile for a single player opponent."""

    username: str
    aggression: float = 0.5      # 0=passive/fleeing, 0.5=neutral, 1=rushing
    skill: float = 0.5           # 0=harmless, 1=high damage per encounter
    encounters: int = 0          # combat engagements (entered <10 blocks)
    kills_on_us: int = 0         # times they killed us
    deaths_to_us: int = 0        # times we killed them
    last_seen_step: int = 0
    _in_combat: bool = False     # currently within combat range
    _approach_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=_SAMPLE_WINDOW),
    )
    _damage_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=_SAMPLE_WINDOW),
    )


class OpponentTracker:
    """Accumulates per-player behavioral stats from the state dict stream.

    Call update() every tick with current and previous state dicts.
    Call get_profiles_for_state() to get (aggression, skill) tuples
    matching the player order in the current state dict.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, OpponentProfile] = {}

    @property
    def profiles(self) -> dict[str, OpponentProfile]:
        return self._profiles

    def update(self, state: dict, prev_state: dict, step: int) -> None:
        """Process one tick of state to update opponent profiles.

        Args:
            state: Current game state dict.
            prev_state: Previous game state dict.
            step: Current step number (for recency tracking).
        """
        players = state.get("entities", {}).get("players", [])
        prev_players = prev_state.get("entities", {}).get("players", [])

        # Build lookup for previous player distances
        prev_by_name: dict[str, float] = {}
        for p in prev_players:
            name = p.get("name", "")
            if name:
                prev_by_name[name] = float(p.get("distance", 64.0))

        # Health change detection (did we take damage this tick?)
        curr_health = float(state.get("health") or 20.0)
        prev_health = float(prev_state.get("health") or 20.0)
        health_loss = max(0.0, prev_health - curr_health)

        for player in players:
            name = player.get("name", "")
            if not name:
                continue

            dist = float(player.get("distance", 64.0))
            profile = self._profiles.get(name)
            if profile is None:
                profile = OpponentProfile(username=name)
                self._profiles[name] = profile

            profile.last_seen_step = step

            # Approach velocity: positive = approaching us
            prev_dist = prev_by_name.get(name)
            if prev_dist is not None:
                approach = prev_dist - dist  # positive = got closer
                # Normalize to [0, 1]: 0 = fleeing at max speed, 1 = charging
                norm_approach = (approach / _MAX_APPROACH_SPEED + 1.0) / 2.0
                norm_approach = max(0.0, min(1.0, norm_approach))
                profile._approach_samples.append(norm_approach)

            # Encounter tracking
            if dist < _COMBAT_RANGE:
                if not profile._in_combat:
                    profile._in_combat = True
                    profile.encounters += 1

                # Damage attribution: if we lost health and this player
                # is the nearest player within attack range (~3.5 blocks)
                if health_loss > 0 and dist < 4.0:
                    norm_damage = min(1.0, health_loss / _MAX_TICK_DAMAGE)
                    profile._damage_samples.append(norm_damage)
            else:
                profile._in_combat = False

            # Update EMA for aggression (from approach samples)
            if profile._approach_samples:
                recent_aggression = (
                    sum(profile._approach_samples) / len(profile._approach_samples)
                )
                profile.aggression = (
                    (1.0 - _EMA_ALPHA) * profile.aggression
                    + _EMA_ALPHA * recent_aggression
                )

            # Update EMA for skill (from damage samples)
            if profile._damage_samples:
                recent_skill = (
                    sum(profile._damage_samples) / len(profile._damage_samples)
                )
                profile.skill = (
                    (1.0 - _EMA_ALPHA) * profile.skill
                    + _EMA_ALPHA * recent_skill
                )

        # Track kills: if player_hit_landed and a player disappeared
        if state.get("player_hit_landed", False):
            prev_names = {p.get("name") for p in prev_players if p.get("name")}
            curr_names = {p.get("name") for p in players if p.get("name")}
            disappeared = prev_names - curr_names
            for name in disappeared:
                if name in self._profiles:
                    self._profiles[name].deaths_to_us += 1

        # Track deaths on us: if we died and a player was nearby
        if not state.get("alive", True) and prev_state.get("alive", True):
            nearest_player = None
            nearest_dist = float("inf")
            for p in prev_players:
                d = float(p.get("distance", 64.0))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_player = p.get("name")
            if nearest_player and nearest_dist < 5.0:
                if nearest_player in self._profiles:
                    self._profiles[nearest_player].kills_on_us += 1

    def get_profile(self, username: str) -> OpponentProfile | None:
        """Get profile for a specific player."""
        return self._profiles.get(username)

    def get_profiles_for_state(
        self, state: dict,
    ) -> list[tuple[float, float]]:
        """Return (aggression, skill) for each player in the state dict.

        Matches by username. Unknown players get (0.5, 0.5) defaults.
        Order matches state["entities"]["players"] order.
        """
        players = state.get("entities", {}).get("players", [])
        result: list[tuple[float, float]] = []
        for player in players:
            name = player.get("name", "")
            profile = self._profiles.get(name)
            if profile is not None:
                result.append((profile.aggression, profile.skill))
            else:
                result.append((0.5, 0.5))
        return result

    def reset(self) -> None:
        """Clear all profiles (e.g., on session restart)."""
        self._profiles.clear()
