"""Minecraft state encoder — game state dict to 396-dim Signal.

Encodes Minecraft game state using Gaussian basis functions, the same
technique used by SurvivalEnv (position encoding) and SocialGridEnv
(vitality/surprise/direction encoding). This ensures cosine similarity
produces meaningful distances: similar game states have high similarity,
different states have low similarity.

Signal layout (396 dimensions):

  Slice        Dims  What                    Encoding
  [0:4]          4   Health                  Gaussian bases over [0, 20]
  [4:8]          4   Food / hunger           Gaussian bases over [0, 20]
  [8:12]         4   Combined urgency        1-(health+food)/40, Gaussian over [0,1]
  [12:16]        4   Time of day             Gaussian bases over [0, 24000]
  [16:20]        4   Light level             Gaussian bases over [0, 15]
  [20:44]       24   Spatial grid            Player-relative 7x7x5 voxel features
  [44:204]     160   Hostile entities        8 x 20 dims
  [204:260]     56   Passive entities        4 x 14 dims
  [260:316]     56   Player entities         4 x 14 dims
  [316:332]     16   Crowd summary           directional + attacker tracking
  [332:336]      4   Situational flags       on_ground, in_water, raining, altitude
  [336:340]      4   Inventory fullness      Gaussian bases over [0, 36]
  [340:352]     12   Hotbar slot encoding    per-slot category+tier scalar (slots 0-8)
  [352:356]      4   XP level                Gaussian bases over [0, 30]
  [356:360]      4   Movement / orientation  yaw quadrant + pitch sign
  [360:364]      4   Attack cooldown         Gaussian bases over [0, 32] ticks
  [364:380]     16   Self-awareness          velocity, deltas, sprint, effects, armor
  [380:396]     16   Threat dynamics         projectiles, timing, momentum, height

Per-hostile slot (20 dims):

  [0:4]   distance       4 Gaussian bases over [0, 64]
  [4:12]  type           8 one-hot: zombie, skeleton, spider, creeper, slime, enderman, phantom, witch
  [12:14] bearing        2 dims: sin(rel_angle), cos(rel_angle) from player facing
  [14]    speed          1 dim: horizontal speed / 20 (blocks/sec)
  [15]    approach       1 dim: velocity dot product mapped [0,1]
  [16]    health         1 dim: health / max_health
  [17]    threat         1 dim: mob-specific (creeper fuse, skeleton bow, melee proximity)
  [18]    facing_us      1 dim: headYaw-based facing score [0,1]
  [19]    flags          1 dim: packed on_fire + baby + sprinting + using_item (0.25 each)

Per-passive slot (14 dims):

  [0:4]   distance       4 Gaussian bases
  [4:12]  type           8 one-hot: cow, pig, sheep, chicken, horse, villager, wolf, (other=7)
  [12]    speed          1 dim: horizontal speed
  [13]    facing_us      1 dim: headYaw-based facing score

Per-player slot (14 dims):

  [0:4]   distance       4 Gaussian bases
  [4:6]   bearing        2 dims: sin/cos relative angle
  [6]     speed          1 dim: horizontal speed
  [7]     approach       1 dim: velocity dot product
  [8:10]  profile        2 dims: aggression + skill from OpponentTracker
  [10]    health         1 dim: health / 20
  [11]    facing_us      1 dim: headYaw-based facing score
  [12]    hand_flags     1 dim: using_item + offhand (0.5 each)
  [13]    equipment      1 dim: weapon tier (0=none, 0.2-1.0 = wood-netherite)

Crowd summary (16 dims):

  [0:4]   quadrant_density  FL/FR/BL/BR hostile density, proximity-weighted
  [4]     hostile_count     / 25
  [5]     hostile_avg_dist  / 64
  [6]     hostile_near      within 8 blocks / 10
  [7]     passive_count     / 10
  [8]     player_count      / 5
  [9:12]  threat_direction  sin(angle), cos(angle), magnitude
  [12]    attacker_dist     / 64
  [13:15] attacker_bearing  sin, cos
  [15]    under_attack      1.0 if took damage recently

Self-awareness [364:380] (16 dims):

  [364]  self_speed           horiz speed / 10
  [365]  self_approach        approach to nearest entity [-1,1] mapped [0,1]
  [366]  vertical_velocity    vy / 20 clamped [-1,1]
  [367]  food_saturation      saturation / 20
  [368]  health_delta         Δhealth / 10 clamped [-1,1]
  [369]  food_delta           Δfood / 10 clamped [-1,1]
  [370]  sprint_ability       1.0/0.5/0.0 from food+saturation thresholds
  [371]  effect_speed         amplifier / 3
  [372]  effect_strength      amplifier / 3
  [373]  effect_resistance    amplifier / 5
  [374]  effect_regeneration  amplifier / 3
  [375]  self_armor_tier      avg armor material [0,1]
  [376]  weather              0=clear, 0.5=rain, 1.0=thunder
  [377]  is_falling           1.0 if vy < -3
  [378]  time_airborne        ticks / 20
  [379]  movement_efficiency  actual_speed / 5.612

Threat dynamics [380:396] (16 dims):

  [380]  proj_urgency         exp(-dist²/128)
  [381]  proj_bearing_sin     sin(relative angle)
  [382]  proj_bearing_cos     cos(relative angle)
  [383]  proj_type            0=none, 0.5=arrow, 1.0=fireball
  [384]  time_to_hostile      dist/approach/20 clamped [0,1]
  [385]  time_to_projectile   proj_dist/proj_speed/20 clamped [0,1]
  [386]  hostile_accel        Δspeed clamped [-1,1]
  [387]  player_armor         avg armor tier of nearest player
  [388]  height_vs_hostile    Δy/10 clamped [-1,1]
  [389]  height_vs_player     Δy/10 clamped [-1,1]
  [390]  damage_rate_5s       total dmg last 5s / 20
  [391]  hits_landed_5s       hits last 5s / 10
  [392]  time_since_hit       ticks since last hit / 40 (capped 1.0)
  [393]  kill_streak          kills last 30s / 5
  [394]  combat_advantage     (my_hits - dmg_taken) / 10 clamped [-1,1]
  [395]  strafing_direction   lateral velocity / 5.612 clamped [-1,1]

Spatial grid sub-layout [20:44] (24 dims, player-relative):

  [20:24]  4  Body clearance   distance to solid at foot+head in F/B/L/R
  [24:28]  4  Drop depth       air blocks below feet in F/B/L/R
  [28:32]  4  Overhead         headroom in F/B/L/R
  [32:36]  4  Danger map       danger density in FL/FR/BL/BR quadrants
  [36:40]  4  Composition      air_ratio, wall_density, ground_coverage, danger_ratio
  [40:44]  4  Immediate        solid flags: below, front_foot, front_head, above
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..primitives.signal import Signal


# Hostile entity type indices — 8 types (matching categories.js HOSTILE_TYPE_MAP)
HOSTILE_TYPES = {
    "zombie": 0, "skeleton": 1, "spider": 2, "creeper": 3,
    "slime": 4, "enderman": 5, "phantom": 6, "witch": 7,
}

# Passive entity type indices — 8 types (matching categories.js PASSIVE_TYPE_MAP)
PASSIVE_TYPES = {
    "cow": 0, "pig": 1, "sheep": 2, "chicken": 3,
    "horse": 4, "villager": 5, "wolf": 6,
}

# Spatial grid feature group names and sizes
SPATIAL_GROUPS = ("body_clear", "drop_depth", "overhead", "danger", "composition", "immediate")
SPATIAL_DIM = 24  # 6 groups x 4 dims each

# How many entities to track per category
MAX_HOSTILES = 8
MAX_PASSIVES = 4
MAX_PLAYERS = 4

# Per-entity dims
HOSTILE_SLOT_DIM = 20   # 4 dist + 8 type + 2 bearing + 1 speed + 1 approach + 1 health + 1 threat + 1 facing_us + 1 flags
PASSIVE_SLOT_DIM = 14   # 4 dist + 8 type + 1 speed + 1 facing_us
PLAYER_SLOT_DIM = 14    # 4 dist + 2 bearing + 1 speed + 1 approach + 2 profile + 1 health + 1 facing_us + 1 hand_flags + 1 equipment
CROWD_DIM = 16


class MinecraftStateEncoder:
    """Encodes Minecraft game state into a 396-dimensional Signal.

    Uses Gaussian basis encoding throughout, following the pattern
    established by SurvivalEnv and SocialGridEnv.
    """

    SIGNAL_DIM = 396

    # Modality slices for attention gating and salience learning.
    MODALITY_SLICES: list[tuple[int, int]] = [
        (0, 12),     # body: health + food + urgency
        (12, 20),    # environment: time + light
        (20, 44),    # terrain: spatial grid (24 dims)
        (44, 332),   # entities: 8 hostile + 4 passive + 4 player + crowd
        (332, 352),  # inventory + situation
        (352, 364),  # combat + movement: xp + orientation + cooldown
        (364, 380),  # self-awareness: velocity, deltas, effects, armor
        (380, 396),  # threat dynamics: projectiles, timing, momentum
    ]

    NUM_ITEM_CATEGORIES = 8  # SWORD=1 through OTHER=8 (0=empty)

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

    def encode(
        self,
        state: dict,
        timestamp: int = 0,
        opponent_profiles: list[tuple[float, float]] | None = None,
    ) -> Signal:
        """Encode a Minecraft game state dict into a 396-dim Signal.

        Args:
            state: Game state dict from the relay bridge.
            timestamp: Discrete tick number.
            opponent_profiles: Optional list of (aggression, skill) tuples,
                one per player in state["entities"]["players"]. When None,
                defaults to (0.5, 0.5) for all players.

        Returns:
            Signal with SIGNAL_DIM float64 data array, modality "minecraft".
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

        # [20:44] Spatial grid (24 dims from 7x7x5 voxel scan)
        spatial = state.get("spatial", {})
        offset = 20
        for group in SPATIAL_GROUPS:
            vals = spatial.get(group, [0, 0, 0, 0])
            for i, v in enumerate(vals[:4]):
                data[offset + i] = float(v)
            offset += 4

        # --- Entities [44:332] ---
        entities = state.get("entities", {})
        hostiles = entities.get("hostiles", [])
        passives = entities.get("passives", [])
        players = entities.get("players", [])

        # Hostile entities [44:204]: 8 slots x 20 dims
        for i in range(MAX_HOSTILES):
            ent = hostiles[i] if i < len(hostiles) else None
            start = 44 + i * HOSTILE_SLOT_DIM
            data[start : start + HOSTILE_SLOT_DIM] = self._encode_hostile_slot(ent)

        # Passive entities [204:260]: 4 slots x 14 dims
        for i in range(MAX_PASSIVES):
            pas = passives[i] if i < len(passives) else None
            start = 204 + i * PASSIVE_SLOT_DIM
            data[start : start + PASSIVE_SLOT_DIM] = self._encode_passive_slot(pas)

        # Player entities [260:316]: 4 slots x 14 dims
        for i in range(MAX_PLAYERS):
            plr = players[i] if i < len(players) else None
            profile = (
                opponent_profiles[i]
                if opponent_profiles is not None and i < len(opponent_profiles)
                else None
            )
            start = 260 + i * PLAYER_SLOT_DIM
            data[start : start + PLAYER_SLOT_DIM] = self._encode_player_slot(plr, profile)

        # [316:332] Crowd summary (16 dims)
        data[316:332] = self._encode_crowd(state.get("crowd", {}))

        # --- Inventory + situation [332:352] ---

        # [332:336] Situational flags
        data[332] = 1.0 if state.get("on_ground", True) else 0.0
        data[333] = 1.0 if state.get("is_in_water", False) else 0.0
        data[334] = 1.0 if state.get("is_raining", False) else 0.0
        altitude = float(state.get("altitude") or 64.0)
        data[335] = max(0.0, min(1.0, altitude / 320.0))

        # [336:340] Inventory fullness
        inv = state.get("inventory", {})
        slots_used = float(inv.get("slots_used", 0))
        data[336:340] = self._gaussian(slots_used, self._inv_centers, self._inv_sigma)

        # [340:352] Hotbar encoding (12 dims)
        data[340:352] = self._encode_hotbar(inv)

        # --- Combat + movement [352:364] ---

        # [352:356] XP level
        xp_level = float(state.get("xp_level") or 0)
        data[352:356] = self._gaussian(xp_level, self._xp_centers, self._xp_sigma)

        # [356:360] Movement / orientation
        yaw = float(state.get("yaw") or 0.0)
        pitch = float(state.get("pitch") or 0.0)
        data[356:360] = self._encode_orientation(yaw, pitch)

        # [360:364] Attack cooldown (0 = ready, up to 32 ticks for sword)
        cooldown = float(state.get("attack_cooldown") or 0)
        data[360:364] = self._gaussian(cooldown, self._cooldown_centers, self._cooldown_sigma)

        # [364:380] Self-awareness (16 dims)
        data[364:380] = self._encode_self_awareness(state)

        # [380:396] Threat dynamics (16 dims)
        data[380:396] = self._encode_threat_dynamics(state)

        # L2-normalize each modality slice independently.
        for start, end in self.MODALITY_SLICES:
            slc = data[start:end]
            norm = np.linalg.norm(slc)
            if norm > 0:
                data[start:end] = slc / norm

        return Signal(data=data, timestamp=timestamp, modality="minecraft")

    def _encode_hotbar(self, inv: dict) -> NDArray[np.float64]:
        """Encode hotbar contents as 12 dims.

        [0:9]  Per-slot category+tier scalar (slots 0-8).
               Value = category / NUM_ITEM_CATEGORIES + tier * 0.01
               Empty slots = 0.0
        [9]    Held item durability fraction (0-1, 1.0 if N/A).
        [10]   Held item stack fraction (count / max_stack).
        [11]   Best weapon tier available in hotbar (0-1).
        """
        result = np.zeros(12, dtype=np.float64)
        hotbar = inv.get("hotbar", [])
        selected = int(inv.get("selected_slot", 0))
        best_weapon_tier = 0.0

        for i in range(min(9, len(hotbar))):
            slot = hotbar[i] if i < len(hotbar) else None
            if slot is None:
                continue
            cat = float(slot.get("category", 0))
            tier = float(slot.get("tier", 0))
            result[i] = cat / self.NUM_ITEM_CATEGORIES + tier * 0.01

            # Track best weapon tier (category 1 = sword)
            if cat == 1 and tier > best_weapon_tier:
                best_weapon_tier = tier

        # Held item details
        if 0 <= selected < len(hotbar) and hotbar[selected] is not None:
            held = hotbar[selected]
            result[9] = float(held.get("durability", 1.0))
            max_stack = float(held.get("max_stack", 64))
            count = float(held.get("count", 1))
            result[10] = count / max_stack if max_stack > 0 else 0.0
        else:
            result[9] = 1.0  # No item = full durability (N/A)
            result[10] = 0.0  # No item = empty stack

        result[11] = best_weapon_tier

        return result

    def _gaussian(
        self,
        value: float,
        centers: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64]:
        """Gaussian basis encoding of a scalar value."""
        return np.exp(-((value - centers) ** 2) / (2 * sigma**2))

    def _encode_hostile_slot(
        self,
        entity: dict | None,
    ) -> NDArray[np.float64]:
        """Encode one hostile entity as 20 dims.

        [0:4]   distance      Gaussian bases over [0, 64]
        [4:12]  type          8 one-hot
        [12:14] bearing       sin(rel_angle), cos(rel_angle)
        [14]    speed         horizontal speed / 20
        [15]    approach      velocity dot product mapped [0,1]
        [16]    health        mob health / max_health
        [17]    threat        mob-specific danger
        [18]    facing_us     headYaw-based facing score [0,1]
        [19]    flags         packed: on_fire + baby + sprinting + using_item

        If no entity is present, returns zeros (distinct "empty slot" pattern).
        """
        result = np.zeros(HOSTILE_SLOT_DIM, dtype=np.float64)
        if entity is None:
            return result

        # Distance: 4 Gaussian bases over [0, 64]
        dist = float(entity.get("distance", 32.0))
        result[0:4] = self._gaussian(dist, self._entity_dist_centers, self._entity_dist_sigma)

        # Type: 8 one-hot
        name = entity.get("name", "")
        type_idx = HOSTILE_TYPES.get(name, -1)
        if type_idx < 0:
            type_idx = abs(hash(name)) % 8
        result[4 + type_idx] = 1.0

        # Bearing: sin/cos of relative angle from player facing
        bearing = entity.get("bearing")
        if bearing is not None:
            result[12] = float(bearing.get("sin", 0.0))
            result[13] = float(bearing.get("cos", 0.0))

        # Speed: horizontal speed in blocks/sec, normalized by /20
        result[14] = min(1.0, float(entity.get("speed", 0.0)) / 20.0)

        # Approach: pre-computed dot product [-1,1] mapped to [0,1]
        approach = float(entity.get("approach", 0.0))
        result[15] = (approach + 1.0) / 2.0

        # Health fraction (0 = unknown/not received yet)
        health = entity.get("health", -1)
        if isinstance(health, (int, float)) and health >= 0:
            max_health = float(entity.get("max_health", 20))
            result[16] = min(1.0, float(health) / max(1.0, max_health))

        # Threat level: mob-specific danger signal
        result[17] = self._compute_threat_level(entity, name, dist)

        # Facing us: per-entity facing score from headYaw
        result[18] = float(entity.get("facing_us", 0.0))

        # Entity flags: packed scalar
        flags = int(entity.get("flags", 0))
        on_fire = 0.25 if (flags & 0x01) else 0.0
        sprinting = 0.25 if (flags & 0x08) else 0.0
        is_baby = 0.25 if entity.get("is_baby", False) else 0.0
        hand_state = int(entity.get("hand_state", 0))
        using_item = 0.25 if (hand_state & 0x01) else 0.0
        result[19] = on_fire + is_baby + sprinting + using_item

        return result

    def _compute_threat_level(
        self, entity: dict, name: str, dist: float
    ) -> float:
        """Compute mob-specific threat level in [0, 1].

        - Creeper: fuse progress (0 = idle, 1 = about to explode)
        - Skeleton: 1.0 if bow drawn (hand_active), else range-based
        - Default melee: 0.5 when in range, lower when far
        """
        creeper_state = entity.get("creeper_state", -1)
        if name == "creeper" and isinstance(creeper_state, (int, float)):
            if creeper_state >= 0:
                return min(1.0, float(creeper_state) / 30.0)
            if entity.get("creeper_charged", False):
                return 0.7
            return 0.3 if dist < 5.0 else 0.1

        if name == "skeleton":
            hand_state = int(entity.get("hand_state", 0))
            if hand_state & 0x01:  # bow drawn
                return 1.0
            return 0.4 if dist < 16.0 else 0.2

        # Default melee mob: threat based on proximity
        if dist < 3.0:
            return 0.8
        if dist < 8.0:
            return 0.5
        return 0.2

    def _encode_passive_slot(
        self,
        entity: dict | None,
    ) -> NDArray[np.float64]:
        """Encode one passive entity as 14 dims.

        [0:4]   distance       Gaussian bases over [0, 64]
        [4:12]  type           8 one-hot
        [12]    speed          horizontal speed / 20
        [13]    facing_us      headYaw-based facing score

        If no entity is present, returns zeros.
        """
        result = np.zeros(PASSIVE_SLOT_DIM, dtype=np.float64)
        if entity is None:
            return result

        # Distance: 4 Gaussian bases over [0, 64]
        dist = float(entity.get("distance", 32.0))
        result[0:4] = self._gaussian(dist, self._entity_dist_centers, self._entity_dist_sigma)

        # Type: 8 one-hot (7 named + hash fallback into slot 7)
        name = entity.get("name", "")
        type_idx = PASSIVE_TYPES.get(name, -1)
        if type_idx < 0:
            type_idx = 7  # "other" slot
        result[4 + type_idx] = 1.0

        # Speed
        result[12] = min(1.0, float(entity.get("speed", 0.0)) / 20.0)

        # Facing us
        result[13] = float(entity.get("facing_us", 0.0))

        return result

    def _encode_player_slot(
        self,
        player: dict | None,
        profile: tuple[float, float] | None = None,
    ) -> NDArray[np.float64]:
        """Encode one player as 14 dims.

        [0:4]   distance       Gaussian bases
        [4:6]   bearing        sin/cos relative angle
        [6]     speed          horizontal speed / 20
        [7]     approach       velocity dot product mapped [0,1]
        [8:10]  profile        aggression + skill
        [10]    health         player health / 20
        [11]    facing_us      headYaw-based facing score
        [12]    hand_flags     using_item + offhand (0.5 each)
        [13]    equipment      weapon tier (0-1)

        If no player is present, returns zeros.
        """
        result = np.zeros(PLAYER_SLOT_DIM, dtype=np.float64)
        if player is None:
            return result

        dist = float(player.get("distance", 64.0))
        result[0:4] = self._gaussian(dist, self._entity_dist_centers, self._entity_dist_sigma)

        # Bearing: sin/cos of relative angle from player facing
        bearing = player.get("bearing")
        if bearing is not None:
            result[4] = float(bearing.get("sin", 0.0))
            result[5] = float(bearing.get("cos", 0.0))

        # Speed
        result[6] = min(1.0, float(player.get("speed", 0.0)) / 20.0)

        # Approach: pre-computed dot product [-1,1] mapped to [0,1]
        approach = float(player.get("approach", 0.0))
        result[7] = (approach + 1.0) / 2.0

        # Opponent profile: aggression + skill
        if profile is not None:
            result[8] = float(profile[0])
            result[9] = float(profile[1])
        else:
            result[8] = 0.5  # neutral aggression
            result[9] = 0.5  # neutral skill

        # Player health (0 = unknown)
        health = player.get("health", -1)
        if isinstance(health, (int, float)) and health >= 0:
            result[10] = min(1.0, float(health) / 20.0)

        # Facing us
        result[11] = float(player.get("facing_us", 0.0))

        # Hand flags
        hand_state = int(player.get("hand_state", 0))
        using_item = 0.5 if (hand_state & 0x01) else 0.0
        offhand = 0.5 if (hand_state & 0x02) else 0.0
        result[12] = using_item + offhand

        # Equipment tier
        result[13] = float(player.get("equipment_tier", 0.0))

        return result

    def _encode_crowd(self, crowd: dict) -> NDArray[np.float64]:
        """Encode crowd summary as 16 dims.

        [0:4]   quadrant_density  FL/FR/BL/BR hostile density
        [4]     hostile_count     / 25
        [5]     hostile_avg_dist  / 64
        [6]     hostile_near      / 10
        [7]     passive_count     / 10
        [8]     player_count      / 5
        [9:12]  threat_direction  sin, cos, magnitude
        [12]    attacker_dist     / 64
        [13:15] attacker_bearing  sin, cos
        [15]    under_attack      1.0 if took damage recently
        """
        result = np.zeros(CROWD_DIM, dtype=np.float64)

        # Quadrant density: FL/FR/BL/BR
        qd = crowd.get("quadrant_density", [0, 0, 0, 0])
        for i in range(min(4, len(qd))):
            result[i] = min(1.0, float(qd[i]))

        # Counts and distances
        result[4] = min(1.0, float(crowd.get("hostile_count", 0)) / 25.0)
        result[5] = min(1.0, float(crowd.get("hostile_avg_dist", 64)) / 64.0)
        result[6] = min(1.0, float(crowd.get("hostile_near", 0)) / 10.0)
        result[7] = min(1.0, float(crowd.get("passive_count", 0)) / 10.0)
        result[8] = min(1.0, float(crowd.get("player_count", 0)) / 5.0)

        # Threat direction
        td = crowd.get("threat_direction", {})
        result[9] = float(td.get("sin", 0.0))
        result[10] = float(td.get("cos", 0.0))
        result[11] = min(1.0, float(td.get("magnitude", 0.0)))

        # Attacker info
        result[12] = min(1.0, float(crowd.get("attacker_dist", 64)) / 64.0)
        ab = crowd.get("attacker_bearing", {})
        result[13] = float(ab.get("sin", 0.0))
        result[14] = float(ab.get("cos", 0.0))
        result[15] = float(crowd.get("under_attack", 0.0))

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
        pitch_norm = (pitch + math.pi / 2) / math.pi
        pitch_norm = max(0.0, min(1.0, pitch_norm))
        result[3] = pitch_norm

        return result

    def _encode_self_awareness(self, state: dict) -> NDArray[np.float64]:
        """Encode self-awareness modality as 16 dims.

        [0]  self_speed           horiz speed / 10
        [1]  self_approach        approach to nearest [-1,1] → [0,1]
        [2]  vertical_velocity    vy / 20 clamped [-1,1]
        [3]  food_saturation      saturation / 20
        [4]  health_delta         Δhealth / 10 clamped [-1,1]
        [5]  food_delta           Δfood / 10 clamped [-1,1]
        [6]  sprint_ability       1.0/0.5/0.0
        [7]  effect_speed         amplifier / 3
        [8]  effect_strength      amplifier / 3
        [9]  effect_resistance    amplifier / 5
        [10] effect_regeneration  amplifier / 3
        [11] self_armor_tier      [0,1]
        [12] weather              0/0.5/1.0
        [13] is_falling           0 or 1
        [14] time_airborne        ticks / 20 capped at 1.0
        [15] movement_efficiency  actual_speed / 5.612 capped at 1.0
        """
        result = np.zeros(16, dtype=np.float64)

        # Self velocity
        vel = state.get("self_velocity", {})
        vx = float(vel.get("x", 0.0))
        vy = float(vel.get("y", 0.0))
        vz = float(vel.get("z", 0.0))
        horiz_speed = math.sqrt(vx * vx + vz * vz)

        result[0] = min(1.0, horiz_speed / 10.0)

        # Approach to nearest entity (from entities)
        entities = state.get("entities", {})
        hostiles = entities.get("hostiles", [])
        players = entities.get("players", [])
        nearest_approach = 0.0
        if hostiles:
            nearest_approach = float(hostiles[0].get("approach", 0.0))
        elif players:
            nearest_approach = float(players[0].get("approach", 0.0))
        result[1] = (nearest_approach + 1.0) / 2.0  # map [-1,1] to [0,1]

        # Vertical velocity
        result[2] = max(-1.0, min(1.0, vy / 20.0))

        # Food saturation
        saturation = float(state.get("food_saturation", 0.0))
        result[3] = min(1.0, saturation / 20.0)

        # Health delta
        health_delta = float(state.get("health_delta", 0.0))
        result[4] = max(-1.0, min(1.0, health_delta / 10.0))

        # Food delta
        food_delta = float(state.get("food_delta", 0.0))
        result[5] = max(-1.0, min(1.0, food_delta / 10.0))

        # Sprint ability: computed from food + saturation
        food = float(state.get("food", 20.0))
        if food > 6:
            result[6] = 1.0
        elif food > 0 and saturation > 0:
            result[6] = 0.5
        else:
            result[6] = 0.0

        # Status effects
        effects = state.get("self_effects", {})
        result[7] = min(1.0, float(effects.get("speed", 0)) / 3.0)
        result[8] = min(1.0, float(effects.get("strength", 0)) / 3.0)
        result[9] = min(1.0, float(effects.get("resistance", 0)) / 5.0)
        result[10] = min(1.0, float(effects.get("regeneration", 0)) / 3.0)

        # Self armor tier
        result[11] = float(state.get("self_armor_tier", 0.0))

        # Weather: 0=clear, 0.5=rain, 1.0=thunder
        is_raining = state.get("is_raining", False)
        is_thundering = state.get("is_thundering", False)
        if is_thundering:
            result[12] = 1.0
        elif is_raining:
            result[12] = 0.5

        # Is falling (vy < -3 blocks/s)
        result[13] = 1.0 if vy < -3.0 else 0.0

        # Time airborne
        ticks_airborne = float(state.get("ticks_airborne", 0))
        result[14] = min(1.0, ticks_airborne / 20.0)

        # Movement efficiency: actual speed / sprint speed (5.612 blocks/s)
        result[15] = min(1.0, horiz_speed / 5.612)

        return result

    def _encode_threat_dynamics(self, state: dict) -> NDArray[np.float64]:
        """Encode threat dynamics modality as 16 dims.

        [0]  proj_urgency         exp(-dist²/128)
        [1]  proj_bearing_sin     sin of relative angle
        [2]  proj_bearing_cos     cos of relative angle
        [3]  proj_type            0=none, 0.5=arrow, 1.0=fireball
        [4]  time_to_hostile      dist / approach / 20 clamped [0,1]
        [5]  time_to_projectile   proj_dist / proj_speed / 20 clamped [0,1]
        [6]  hostile_accel        Δspeed clamped [-1,1]
        [7]  player_armor         avg armor tier of nearest player
        [8]  height_vs_hostile    Δy / 10 clamped [-1,1]
        [9]  height_vs_player     Δy / 10 clamped [-1,1]
        [10] damage_rate_5s       total dmg / 20
        [11] hits_landed_5s       hits / 10
        [12] time_since_hit       seconds / 10 capped 1.0
        [13] kill_streak          kills / 5 capped 1.0
        [14] combat_advantage     (hits - dmg) / 10 clamped [-1,1]
        [15] strafing_direction   lateral velocity / 5.612 clamped [-1,1]
        """
        result = np.zeros(16, dtype=np.float64)

        # Incoming projectile
        proj = state.get("incoming_projectile")
        if proj is not None:
            proj_dist = float(proj.get("distance", 32.0))
            result[0] = math.exp(-(proj_dist * proj_dist) / 128.0)
            bearing = proj.get("bearing", {})
            result[1] = float(bearing.get("sin", 0.0))
            result[2] = float(bearing.get("cos", 0.0))
            proj_name = proj.get("name", "")
            if "fireball" in proj_name or "wither_skull" in proj_name:
                result[3] = 1.0
            elif proj_name:
                result[3] = 0.5  # arrow, trident, etc.
            # Time to projectile impact
            proj_speed = float(proj.get("speed", 1.0))
            if proj_speed > 0.1:
                result[5] = min(1.0, (proj_dist / proj_speed) / 20.0)

        # Time to nearest hostile
        entities = state.get("entities", {})
        hostiles = entities.get("hostiles", [])
        if hostiles:
            h_dist = float(hostiles[0].get("distance", 64.0))
            h_approach = float(hostiles[0].get("approach", 0.0))
            if h_approach > 0.1:
                result[4] = min(1.0, (h_dist / h_approach) / 20.0)
            else:
                result[4] = 1.0  # not approaching = max time

        # Hostile acceleration
        accel = float(state.get("nearest_hostile_accel", 0.0))
        result[6] = max(-1.0, min(1.0, accel))

        # Nearest player armor
        result[7] = float(state.get("nearest_player_armor", 0.0))

        # Height advantage
        result[8] = max(-1.0, min(1.0, float(state.get("height_vs_hostile", 0.0)) / 10.0))
        result[9] = max(-1.0, min(1.0, float(state.get("height_vs_player", 0.0)) / 10.0))

        # Combat momentum
        result[10] = min(1.0, float(state.get("combat_damage_5s", 0.0)) / 20.0)
        result[11] = min(1.0, float(state.get("combat_hits_5s", 0)) / 10.0)
        time_since = float(state.get("time_since_hit", 10.0))
        result[12] = min(1.0, time_since / 10.0)
        result[13] = min(1.0, float(state.get("kill_streak", 0)) / 5.0)

        # Combat advantage: hits landed - damage taken (relative)
        hits = float(state.get("combat_hits_5s", 0))
        damage = float(state.get("combat_damage_5s", 0.0))
        result[14] = max(-1.0, min(1.0, (hits - damage) / 10.0))

        # Strafing direction: lateral velocity / sprint speed
        strafing = float(state.get("strafing", 0.0))
        result[15] = max(-1.0, min(1.0, strafing / 5.612))

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
        input_dims: int = 396,
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
