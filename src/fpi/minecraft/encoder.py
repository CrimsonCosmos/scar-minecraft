"""Minecraft state encoder — game state dict to 76-dim Signal.

Encodes Minecraft game state using Gaussian basis functions, the same
technique used by SurvivalEnv (position encoding) and SocialGridEnv
(vitality/surprise/direction encoding). This ensures cosine similarity
produces meaningful distances: similar game states have high similarity,
different states have low similarity.

Signal layout (76 dimensions):

  Slice    Dims  What                    Encoding
  [0:4]      4   Health                  Gaussian bases over [0, 20]
  [4:8]      4   Food / hunger           Gaussian bases over [0, 20]
  [8:12]     4   Combined urgency        1-(health+food)/40, Gaussian over [0,1]
  [12:16]    4   Time of day             Gaussian bases over [0, 24000]
  [16:20]    4   Light level             Gaussian bases over [0, 15]
  [20:28]    8   Block composition       8 category ratios, L2-normalized
  [28:36]    8   Nearest hostile entity   4 distance + 4 type bases
  [36:44]    8   Nearest passive entity   4 distance + 4 type bases
  [44:48]    4   Nearest player          4 distance Gaussian bases
  [48:52]    4   Situational flags       on_ground, in_water, raining, altitude
  [52:56]    4   Inventory fullness      Gaussian bases over [0, 36]
  [56:60]    4   Inventory flags         has_weapon, has_food, has_tool, has_wood
  [60:64]    4   XP level                Gaussian bases over [0, 30]
  [64:68]    4   Movement / orientation  yaw quadrant + pitch sign
  [68:72]    4   Attack cooldown         Gaussian bases over [0, 32] ticks
  [72:74]    2   Hostile facing          facing_toward_us + angle_diff
  [74:76]    2   Player facing           facing_toward_us + angle_diff
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal


# Block category order matching bot.js scanNearbyBlocks output
BLOCK_CATEGORIES = ("air", "stone", "dirt", "wood", "water", "ore", "danger", "other")

# Hostile entity type indices (matching bot.js HOSTILE_TYPE_MAP)
HOSTILE_TYPES = {"zombie": 0, "skeleton": 1, "spider": 2, "creeper": 3}

# Passive entity type indices (matching bot.js PASSIVE_TYPE_MAP)
PASSIVE_TYPES = {"cow": 0, "pig": 1, "sheep": 2, "chicken": 3}


class MinecraftStateEncoder:
    """Encodes Minecraft game state into a 76-dimensional Signal.

    Uses Gaussian basis encoding throughout, following the pattern
    established by SurvivalEnv and SocialGridEnv.
    """

    SIGNAL_DIM = 76

    # Modality slices for attention gating and salience learning.
    MODALITY_SLICES: list[tuple[int, int]] = [
        (0, 12),   # body: health + food + urgency
        (12, 20),  # environment: time + light
        (20, 28),  # terrain: block composition
        (28, 48),  # entities: hostile + passive + player
        (48, 60),  # inventory + situation
        (60, 76),  # combat + movement: xp + orientation + cooldown + facing
    ]

    def __init__(self) -> None:
        # Health: 4 bases over [0, 20]
        self._health_centers = np.linspace(0.0, 20.0, 4)
        self._health_sigma = 3.0

        # Food: 4 bases over [0, 20]
        self._food_centers = np.linspace(0.0, 20.0, 4)
        self._food_sigma = 3.0

        # Urgency: 4 bases over [0, 1]
        self._urgency_centers = np.linspace(0.0, 1.0, 4)
        self._urgency_sigma = 0.2

        # Time of day: 4 bases over [0, 24000]
        self._time_centers = np.linspace(0.0, 24000.0, 4)
        self._time_sigma = 4000.0

        # Light level: 4 bases over [0, 15]
        self._light_centers = np.linspace(0.0, 15.0, 4)
        self._light_sigma = 2.5

        # Entity distance: 4 bases over [0, 64]
        # Broad sigma (16.0) ensures small mob movements don't create new patterns.
        # A mob must move ~25+ blocks to change the encoding significantly.
        # 64-block range lets the agent "see" mobs from far away and plan approach.
        self._entity_dist_centers = np.linspace(0.0, 64.0, 4)
        self._entity_dist_sigma = 16.0

        # Inventory fullness: 4 bases over [0, 36]
        self._inv_centers = np.linspace(0.0, 36.0, 4)
        self._inv_sigma = 6.0

        # XP level: 4 bases over [0, 30]
        self._xp_centers = np.linspace(0.0, 30.0, 4)
        self._xp_sigma = 5.0

        # Yaw quadrant: 4 bases over [0, 3] (N=0, E=1, S=2, W=3)
        self._yaw_centers = np.linspace(0.0, 3.0, 4)
        self._yaw_sigma = 0.5

        # Attack cooldown: 4 bases over [0, 32] ticks
        # Full cooldown at 1.6s = 32 ticks (sword), 0 = ready to attack
        self._cooldown_centers = np.linspace(0.0, 32.0, 4)
        self._cooldown_sigma = 6.0

    def encode(self, state: dict, timestamp: int = 0) -> Signal:
        """Encode a Minecraft game state dict into a 76-dim Signal.

        Args:
            state: Game state dict from the Mineflayer bridge. Expected keys:
                health, food, time_of_day, light_level, block_composition,
                entities, inventory, on_ground, is_in_water, is_raining,
                altitude, xp_level, yaw, pitch, alive, attack_cooldown,
                hostile_facing, player_facing.
            timestamp: Discrete tick number.

        Returns:
            Signal with 76-dim float64 data array, modality "minecraft".
        """
        data = np.zeros(self.SIGNAL_DIM, dtype=np.float64)

        # [0:4] Health
        health = float(state.get("health") or 20.0)
        data[0:4] = self._gaussian(health, self._health_centers, self._health_sigma)

        # [4:8] Food
        food = float(state.get("food") or 20.0)
        data[4:8] = self._gaussian(food, self._food_centers, self._food_sigma)

        # [8:12] Combined urgency: 1 - (health + food) / 40
        urgency = 1.0 - (health + food) / 40.0
        urgency = max(0.0, min(1.0, urgency))
        data[8:12] = self._gaussian(urgency, self._urgency_centers, self._urgency_sigma)

        # [12:16] Time of day
        time_of_day = float(state.get("time_of_day") or 6000)
        data[12:16] = self._gaussian(time_of_day, self._time_centers, self._time_sigma)

        # [16:20] Light level
        light = float(state.get("light_level") if state.get("light_level") is not None else 15)
        data[16:20] = self._gaussian(light, self._light_centers, self._light_sigma)

        # [20:28] Block composition (8 category ratios, L2-normalized)
        block_comp = state.get("block_composition", {})
        block_vec = np.array(
            [float(block_comp.get(cat, 0.0)) for cat in BLOCK_CATEGORIES],
            dtype=np.float64,
        )
        norm = np.linalg.norm(block_vec)
        if norm > 0:
            block_vec /= norm
        data[20:28] = block_vec

        # [28:36] Nearest hostile entity
        entities = state.get("entities", {})
        hostile = entities.get("hostile")
        data[28:36] = self._encode_entity(hostile, HOSTILE_TYPES)

        # [36:44] Nearest passive entity
        passive = entities.get("passive")
        data[36:44] = self._encode_entity(passive, PASSIVE_TYPES)

        # [44:48] Nearest player (4 distance Gaussian bases)
        player = entities.get("player")
        data[44:48] = self._encode_player(player)

        # [48:52] Situational flags
        data[48] = 1.0 if state.get("on_ground", True) else 0.0
        data[49] = 1.0 if state.get("is_in_water", False) else 0.0
        data[50] = 1.0 if state.get("is_raining", False) else 0.0
        # Altitude band: normalize to [0, 1] range (y=0 to y=320)
        altitude = float(state.get("altitude") or 64.0)
        data[51] = max(0.0, min(1.0, altitude / 320.0))

        # [52:56] Inventory fullness
        inv = state.get("inventory", {})
        slots_used = float(inv.get("slots_used", 0))
        data[52:56] = self._gaussian(slots_used, self._inv_centers, self._inv_sigma)

        # [56:60] Inventory flags
        data[56] = 1.0 if inv.get("has_weapon", False) else 0.0
        data[57] = 1.0 if inv.get("has_food", False) else 0.0
        data[58] = 1.0 if inv.get("has_tool", False) else 0.0
        data[59] = 1.0 if inv.get("has_wood", False) else 0.0

        # [60:64] XP level
        xp_level = float(state.get("xp_level") or 0)
        data[60:64] = self._gaussian(xp_level, self._xp_centers, self._xp_sigma)

        # [64:68] Movement / orientation
        yaw = float(state.get("yaw") or 0.0)
        pitch = float(state.get("pitch") or 0.0)
        data[64:68] = self._encode_orientation(yaw, pitch)

        # [68:72] Attack cooldown (0 = ready, up to 32 ticks for sword)
        cooldown = float(state.get("attack_cooldown") or 0)
        data[68:72] = self._gaussian(cooldown, self._cooldown_centers, self._cooldown_sigma)

        # [72:74] Hostile facing: how much the nearest hostile is facing toward us
        data[72:74] = self._encode_facing(state.get("hostile_facing"))

        # [74:76] Player facing: how much the nearest player is facing toward us
        data[74:76] = self._encode_facing(state.get("player_facing"))

        # L2-normalize each modality slice independently.
        # This ensures no single modality dominates cosine similarity.
        # Without this, constant dimensions (health=20, food=20, empty
        # inventory) produce identical high-magnitude activations that
        # overwhelm the few varying dimensions (yaw, block composition).
        for start, end in self.MODALITY_SLICES:
            slc = data[start:end]
            norm = np.linalg.norm(slc)
            if norm > 0:
                data[start:end] = slc / norm

        return Signal(data=data, timestamp=timestamp, modality="minecraft")

    def _gaussian(
        self,
        value: float,
        centers: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64]:
        """Gaussian basis encoding of a scalar value.

        Same function used in SurvivalEnv._make_observation and
        SocialGridEnv._encode_gaussian.
        """
        return np.exp(-((value - centers) ** 2) / (2 * sigma**2))

    def _encode_entity(
        self,
        entity: dict | None,
        type_map: dict[str, int],
    ) -> NDArray[np.float64]:
        """Encode nearest entity as 8 dims: 4 distance bases + 4 type bases.

        If no entity is present, returns zeros (distinct "alone" pattern).
        """
        result = np.zeros(8, dtype=np.float64)
        if entity is None:
            return result

        # Distance: 4 Gaussian bases over [0, 32]
        dist = float(entity.get("distance", 32.0))
        result[0:4] = self._gaussian(dist, self._entity_dist_centers, self._entity_dist_sigma)

        # Type: one-hot-ish encoding over 4 types
        name = entity.get("name", "")
        type_idx = type_map.get(name, -1)
        if type_idx < 0:
            # Unknown type: hash to index
            type_idx = abs(hash(name)) % 4
        result[4 + type_idx] = 1.0

        return result

    def _encode_player(
        self,
        player: dict | None,
    ) -> NDArray[np.float64]:
        """Encode nearest player as 4 dims: distance Gaussian bases.

        If no player is present, returns zeros (distinct "alone" pattern).
        """
        result = np.zeros(4, dtype=np.float64)
        if player is None:
            return result
        dist = float(player.get("distance", 64.0))
        result[0:4] = self._gaussian(dist, self._entity_dist_centers, self._entity_dist_sigma)
        return result

    def _encode_facing(
        self,
        facing: dict | None,
    ) -> NDArray[np.float64]:
        """Encode entity facing direction relative to us as 2 dims.

        facing dict has:
          - facing_us: float in [0, 1] — 1.0 = looking directly at us
          - angle_diff: float in [0, pi] — absolute angle between their
            facing vector and the vector toward us

        Returns 2 dims: [facing_us_score, angle_diff_normalized]
        """
        result = np.zeros(2, dtype=np.float64)
        if facing is None:
            return result
        # Facing score: 1.0 = looking at us, 0.0 = looking away
        result[0] = float(facing.get("facing_us", 0.0))
        # Angle diff: normalized to [0, 1] range
        angle_diff = float(facing.get("angle_diff", math.pi))
        result[1] = 1.0 - (angle_diff / math.pi)  # 1.0 = facing us, 0.0 = away
        return result

    def _encode_orientation(self, yaw: float, pitch: float) -> NDArray[np.float64]:
        """Encode yaw/pitch as 4 dims.

        Yaw is converted to a quadrant index (N=0, E=1, S=2, W=3) and
        encoded with Gaussian bases. Pitch sign indicates up/level/down
        as a continuous value.
        """
        result = np.zeros(4, dtype=np.float64)

        # Yaw to quadrant: normalize to [0, 2*pi), then to [0, 4)
        yaw_norm = yaw % (2 * math.pi)
        quadrant = (yaw_norm / (2 * math.pi)) * 4.0
        result[0:3] = self._gaussian(quadrant, self._yaw_centers[:3], self._yaw_sigma)

        # Pitch: normalize from [-pi/2, pi/2] to [0, 1]
        # 0 = looking straight up, 0.5 = level, 1.0 = looking down
        pitch_norm = (pitch + math.pi / 2) / math.pi
        pitch_norm = max(0.0, min(1.0, pitch_norm))
        result[3] = pitch_norm

        return result


class HistoryTrace:
    """Exponentially decaying trace of recent observations.

    Compresses the full observation signal into a fixed-size history
    vector using a random projection, then maintains an exponential
    moving average. This gives the agent temporal context: "where I
    came from" encoded as a compact vector.

    The output is L2-normalized so it can be used as a modality slice
    alongside the existing signal encoding.

    Args:
        decay: Exponential decay factor (0-1). Higher = longer memory.
        output_dims: Dimensionality of the history vector.
        input_dims: Dimensionality of the input signal (must match encoder SIGNAL_DIM).
        seed: Random seed for the projection matrix.
    """

    def __init__(
        self,
        decay: float = 0.6,
        output_dims: int = 16,
        input_dims: int = 68,
        seed: int = 42,
    ) -> None:
        self._trace = np.zeros(output_dims, dtype=np.float64)
        self._decay = decay
        self._output_dims = output_dims
        # Fixed random projection: compress full signal to output_dims
        rng = np.random.default_rng(seed)
        self._projection = rng.standard_normal((output_dims, input_dims))
        # Normalize each row for unit-variance projections
        row_norms = np.linalg.norm(self._projection, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        self._projection /= row_norms

    def update(self, signal_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update trace with new observation, return current history encoding.

        Args:
            signal_data: The raw encoded signal (before history appending).

        Returns:
            L2-normalized history vector of shape (output_dims,).
        """
        compressed = self._projection @ signal_data
        self._trace = self._decay * self._trace + (1.0 - self._decay) * compressed
        norm = np.linalg.norm(self._trace)
        if norm > 0:
            return self._trace / norm
        return self._trace.copy()

    def reset(self) -> None:
        """Clear the trace (e.g., on death/respawn)."""
        self._trace[:] = 0.0
