"""fast_train — Offline 3D combat simulator for FPI pre-training.

Trains an FPI agent in a physics-accurate 3D arena against zombies and
skeletons, producing a pretrained.pkl that transfers to real Minecraft via:

    python -m fpi.minecraft.runner --transfer pretrained.pkl

The sim models real Bedrock Edition mechanics:
  - Velocity-based movement with friction (0.91 ground, 0.98 air vertical)
  - Gravity + jump physics (0.42 blocks/tick jump, 0.08 gravity)
  - Facing requirement for attacks (±30° cone)
  - Real crit detection (must be falling: vy < 0 and not on_ground)
  - Player knockback when hit by mobs
  - Sprint state machine with food/saturation exhaustion
  - CPS-limited attacks (10 CPS = 2 tick minimum interval)
  - W-tap: stop movement → attack (full KB, no forward reduction) → re-sprint
  - Skeleton AI with arrow projectiles (gravity, drag, hit detection)
  - Baby zombies (1.5x speed)

What transfers:
  - "Jump → wait for descent → swing = crit" (timing)
  - "Face target before attacking" (tracking)
  - "Stop moving before hit = more KB" (w-tap)
  - "Dodge arrows by strafing" (projectile avoidance)
  - "Manage food for sprint ability" (resource management)
  - "Space after being hit to avoid follow-up" (KB recovery)

What doesn't transfer (learned online in MC):
  - Complex 3D terrain (stairs, water, lava)
  - Inventory/crafting decisions
  - Block-based pathfinding and obstacle avoidance

Usage:
    python -m fpi.minecraft.fast_train --steps 500000 --save ~/Desktop/pretrained.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from ..agent.core import Agent
from ..primitives.signal import Signal
from ..primitives.vitality import Vitality
from .actions import (
    FACTORED_ACTIONS, PHASE_3_ACTIONS, PHASE_4_ACTIONS,
    decode_composite,
)
from .encoder import HistoryTrace, MinecraftStateEncoder
from .env import MinecraftEnv, compute_energy_delta


# ─── Physics Constants (Bedrock Edition) ─────────────────────────────

# Movement
GROUND_ACCEL = 0.098          # blocks/tick² (walking input)
SPRINT_ACCEL = 0.1274         # blocks/tick² (sprint input, ~1.3x walk)
GROUND_FRICTION = 0.91        # velocity multiplier per tick on ground
AIR_ACCEL_MULT = 0.20         # air control = 20% of ground acceleration
AIR_FRICTION_VERT = 0.98      # vertical drag multiplier

# Jump / Gravity
JUMP_VELOCITY = 0.42          # blocks/tick upward
GRAVITY = 0.08                # blocks/tick² downward
SPRINT_JUMP_BOOST = 0.2       # extra forward accel on sprint-jump tick

# Combat
MELEE_REACH = 3.0             # blocks
FACING_CONE_RAD = math.radians(30)  # ±30° half-angle
CRIT_MULT = 1.5               # damage multiplier when falling
SWORD_DAMAGE = 7.0            # iron sword base damage
BASE_KB = 0.4                 # base knockback velocity (blocks/tick)
KB_UPWARD = 0.36              # vertical KB component
SPRINT_KB_BONUS = 0.4         # additional KB when sprinting
FORWARD_KB_REDUCTION = 0.6    # KB × this if attacker moving forward
MIN_ATTACK_INTERVAL = 2       # ticks between swings (10 CPS)

# Sprint / Food
MIN_FOOD_TO_SPRINT = 7        # need food >= 7 to sprint
EXHAUSTION_PER_SPRINT_TICK = 0.005
EXHAUSTION_PER_JUMP = 0.2     # per jump while sprinting
EXHAUSTION_PER_ATTACK = 0.1
EXHAUSTION_THRESHOLD = 4.0    # drain 1 saturation/food at this

# Arena
ARENA_SIZE = 24.0
ARENA_INT = 24                # integer size for heightmap indexing
GROUND_Y = 4.0                # ground level in world coords
FALL_DAMAGE_THRESHOLD = 3.0   # blocks of fall before taking damage
STEP_UP_HEIGHT = 0.6          # max auto-step height
JUMP_CLEAR_HEIGHT = 1.25      # max height clearable with a jump
MOB_STEP_HEIGHT = 1.0         # mobs can auto-step up to this

# Zombie
ZOMBIE_SPEED = 0.12           # blocks/tick
ZOMBIE_DAMAGE = 3.0           # normal difficulty
ZOMBIE_ATTACK_RANGE = 1.5
ZOMBIE_ATTACK_CD = 20         # ticks (1s)
ZOMBIE_INVULN = 10            # invulnerability ticks after hit
ZOMBIE_HEALTH = 20.0
ZOMBIE_BABY_SPEED_MULT = 1.5
ZOMBIE_BABY_CHANCE = 0.05     # 5% of zombie spawns are babies
ZOMBIE_WOBBLE_RAD = math.radians(5)  # ±5° path noise
ZOMBIE_KB_FRICTION = 0.85     # zombie decelerates after KB
ZOMBIE_COUNT = 2
ZOMBIE_SPAWN_MIN = 5.0
ZOMBIE_SPAWN_MAX = 12.0

# Skeleton
SKELETON_RANGE = 16.0
SKELETON_HEALTH = 20.0
SKELETON_FIRE_CD_MIN = 30
SKELETON_FIRE_CD_MAX = 50
SKELETON_RETREAT_DIST = 4.0
SKELETON_SPEED = 0.10         # slightly slower than zombie
SKELETON_ARROW_SPEED = 1.6    # blocks/tick
SKELETON_ARROW_GRAVITY = 0.05
SKELETON_ARROW_DRAG = 0.99
SKELETON_ARROW_DAMAGE = 4.0
SKELETON_ACCURACY_ERROR = math.radians(6)  # ±6°

LOOK_DELTA = math.radians(30)

# Weapon / Armor / Enchantments
WEAPON_TIERS = {
    "wood":          {"tier": 0.2, "damage": 5.0, "is_axe": False},
    "stone":         {"tier": 0.4, "damage": 6.0, "is_axe": False},
    "iron":          {"tier": 0.6, "damage": 7.0, "is_axe": False},
    "diamond":       {"tier": 0.8, "damage": 8.0, "is_axe": False},
    "netherite":     {"tier": 1.0, "damage": 9.0, "is_axe": False},
    "iron_axe":      {"tier": 0.6, "damage": 9.0, "is_axe": True},
    "diamond_axe":   {"tier": 0.8, "damage": 9.0, "is_axe": True},
    "netherite_axe": {"tier": 1.0, "damage": 10.0, "is_axe": True},
}
ARMOR_TIERS = {
    "none":      {"tier": 0.0, "reduction": 0.00},
    "leather":   {"tier": 0.2, "reduction": 0.28},
    "gold":      {"tier": 0.3, "reduction": 0.44},
    "chain":     {"tier": 0.5, "reduction": 0.48},
    "iron":      {"tier": 0.6, "reduction": 0.60},
    "diamond":   {"tier": 0.8, "reduction": 0.80},
    "netherite": {"tier": 1.0, "reduction": 0.80},
}
SHARPNESS_BONUS_PER_LEVEL = 1.25   # +damage per sharpness level (Bedrock)
KB_BONUS_PER_LEVEL = 0.3           # +KB per knockback enchant level
PROTECTION_PER_LEVEL = 0.04 * 4    # 4% per level × 4 armor pieces = 16%/level
MAX_DAMAGE_REDUCTION = 0.96        # cap total reduction at 96%
REGEN_INTERVAL = 80                # ticks between natural regen (4 seconds)
REGEN_MIN_FOOD = 18                # need food >= 18 for natural regen
REGEN_EXHAUSTION = 6.0             # exhaustion per regen tick
FIRE_ASPECT_DURATION = 80          # ticks per fire aspect level (4 seconds)
FIRE_ASPECT_DPS = 1.0              # damage per second from fire (0.05/tick)

# Creeper
CREEPER_SPEED = 0.10
CREEPER_HEALTH = 20.0
CREEPER_FUSE = 30                  # ticks (1.5s)
CREEPER_BLAST_RADIUS = 3.0
CREEPER_DAMAGE = 24.5              # normal difficulty
CREEPER_CHARGED_MULT = 2.0
CREEPER_CHARGED_CHANCE = 0.05
CREEPER_KB_FRICTION = 0.85

# Water
WATER_FRICTION = 0.80              # slower than ground (0.91)

# Eating
EAT_DURATION = 32                  # ticks (1.6s)
EAT_HEAL = 6.0                    # HP restored
PEARL_DAMAGE = 5.0                 # damage on landing

# Loadout profiles: (weight, weapon_choices, armor_choices, sharp_range, kb_range, prot_range, fire_range, axe_chance)
LOADOUT_PROFILES = [
    # early_game: 20%
    (0.20, ["wood", "stone"], ["none", "leather"], (0, 0), (0, 0), (0, 0), (0, 0), 0.0),
    # mid_game: 30%
    (0.30, ["iron"], ["chain", "iron"], (1, 2), (0, 0), (1, 2), (0, 1), 0.10),
    # late_game: 25%
    (0.25, ["diamond"], ["diamond"], (3, 4), (0, 1), (3, 4), (1, 2), 0.15),
    # endgame: 15%
    (0.15, ["netherite"], ["netherite"], (5, 5), (1, 2), (4, 4), (2, 2), 0.15),
    # random: 10%
    (0.10, ["wood", "stone", "iron", "diamond", "netherite"],
     ["none", "leather", "gold", "chain", "iron", "diamond", "netherite"],
     (0, 5), (0, 2), (0, 4), (0, 2), 0.10),
]

# Shield disable duration (ticks) when hit by axe
SHIELD_DISABLE_DURATION = 100


# ─── Arrow Projectile ────────────────────────────────────────────────

@dataclass
class Arrow:
    """Projectile with gravity and drag."""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    lifetime: int = 0
    is_player: bool = False   # True for player-shot arrows
    damage: float = 0.0       # damage on hit (player arrows only)
    is_pearl: bool = False    # True for enderpearl projectile

    def tick(self) -> str:
        """Advance arrow one tick. Returns 'flying' or 'despawn'."""
        self.vy -= SKELETON_ARROW_GRAVITY
        self.vx *= SKELETON_ARROW_DRAG
        self.vy *= SKELETON_ARROW_DRAG
        self.vz *= SKELETON_ARROW_DRAG
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        self.lifetime += 1
        # Terrain collision is checked in _tick_arrows(); only timeout here
        if self.lifetime > 200:
            return "despawn"
        return "flying"


# ─── Combat Arena ────────────────────────────────────────────────────

class CombatArena:
    """3D physics-accurate combat arena: player vs zombies/skeletons."""

    def __init__(self, num_zombies: int = ZOMBIE_COUNT, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._num_zombies = num_zombies

        # Player state (initialized in reset)
        self.px = 0.0
        self.pz = 0.0
        self.py = 0.0       # vertical (height above ground)
        self.vx = 0.0
        self.vz = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.on_ground = True
        self.is_sprinting = False
        self.sprint_ticks = 0
        self.ticks_airborne = 0
        self.health = 20.0
        self.food = 20
        self.saturation = 5.0
        self.exhaustion = 0.0
        self.last_attack_tick = -999
        self.alive = True

        # Zombie state (parallel arrays)
        self.zx: list[float] = []
        self.zz: list[float] = []
        self.zy: list[float] = []   # zombie vertical (always 0 for ground mobs)
        self.zvx: list[float] = []
        self.zvz: list[float] = []
        self.zh: list[float] = []
        self.z_invuln: list[int] = []
        self.z_atk_cd: list[int] = []
        self.z_is_baby: list[bool] = []

        # Skeleton state (parallel arrays, separate from zombies)
        self.sx: list[float] = []
        self.sz: list[float] = []
        self.sh: list[float] = []
        self.svx: list[float] = []
        self.svz: list[float] = []
        self.s_fire_cd: list[int] = []
        self.s_invuln: list[int] = []

        # Arrows in flight
        self.arrows: list[Arrow] = []

        # Creeper state (parallel arrays)
        self.cx: list[float] = []
        self.cz: list[float] = []
        self.cy: list[float] = []
        self.cvx: list[float] = []
        self.cvz: list[float] = []
        self.ch: list[float] = []
        self.c_fuse: list[int] = []       # -1 = not fusing, 0+ = countdown
        self.c_charged: list[bool] = []
        self.c_invuln: list[int] = []

        # Fire ticks on mobs
        self.z_fire_ticks: list[int] = []
        self.s_fire_ticks: list[int] = []
        self.c_fire_ticks: list[int] = []

        # Per-step outputs
        self.hit_landed = False
        self.kills_this_tick = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.tick = 0

        # Terrain
        self._terrain_type = "flat"
        self._heights = np.zeros((ARENA_INT, ARENA_INT), dtype=np.float32)
        self._water_map = np.zeros((ARENA_INT, ARENA_INT), dtype=bool)
        self._fall_start_height = 0.0

        # Loadout (randomized per episode)
        self._weapon_name = "iron"
        self._weapon_damage = 7.0
        self._weapon_tier = 0.6
        self._is_axe = False
        self._armor_name = "iron"
        self._armor_tier = 0.6
        self._armor_reduction = 0.60
        self._sharpness_level = 0
        self._knockback_level = 0
        self._protection_level = 0
        self._fire_aspect_level = 0

        # Equipment flags
        self._has_shield = False
        self._shield_raised = False
        self._shield_disabled_ticks = 0
        self._has_bow = False
        self._bow_charge_ticks = 0
        self._is_charging_bow = False
        self._has_blocks = False
        self._block_count = 0
        self._has_pearl = False
        self._pearl_count = 0
        self._selected_slot = 0
        self._is_eating = False
        self._eat_ticks = 0

        # Potion effects (per-episode)
        self._effect_speed = 0
        self._effect_strength = 0
        self._effect_resistance = 0

        # Episode type
        self._episode_type = "zombies"  # zombies, mixed, horde, hard, creeper

        # Self-awareness / threat dynamics tracking
        self._prev_health = 20.0
        self._prev_food = 20
        self._recent_hits: list[int] = []
        self._recent_damage: list[float] = []
        self._recent_kills: list[int] = []
        self._prev_zombie_speeds: list[float] = []

        self.reset()

    def reset(self):
        """Reset arena for new episode."""
        self.px = ARENA_SIZE / 2
        self.pz = ARENA_SIZE / 2
        self.py = 0.0
        self.vx = 0.0
        self.vz = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.on_ground = True
        self.is_sprinting = False
        self.sprint_ticks = 0
        self.ticks_airborne = 0
        self.health = 20.0
        self.food = 20
        self.saturation = 5.0
        self.exhaustion = 0.0
        self.last_attack_tick = -999
        self.alive = True
        self.tick = 0
        self.hit_landed = False
        self.kills_this_tick = 0

        # Clear entities
        self.zx.clear(); self.zz.clear(); self.zy.clear()
        self.zvx.clear(); self.zvz.clear()
        self.zh.clear(); self.z_invuln.clear()
        self.z_atk_cd.clear(); self.z_is_baby.clear()
        self.z_fire_ticks.clear()
        self.sx.clear(); self.sz.clear(); self.sh.clear()
        self.svx.clear(); self.svz.clear()
        self.s_fire_cd.clear(); self.s_invuln.clear()
        self.s_fire_ticks.clear()
        self.cx.clear(); self.cz.clear(); self.cy.clear()
        self.cvx.clear(); self.cvz.clear()
        self.ch.clear(); self.c_fuse.clear()
        self.c_charged.clear(); self.c_invuln.clear()
        self.c_fire_ticks.clear()
        self.arrows.clear()
        self._prev_zombie_speeds.clear()

        # Reset equipment state
        self._shield_raised = False
        self._shield_disabled_ticks = 0
        self._bow_charge_ticks = 0
        self._is_charging_bow = False
        self._selected_slot = 0
        self._is_eating = False
        self._eat_ticks = 0

        # Generate terrain
        self._generate_terrain()
        # Set player starting height to terrain at spawn
        spawn_gx = int(ARENA_SIZE / 2)
        spawn_gz = int(ARENA_SIZE / 2)
        self.py = float(self._heights[spawn_gx, spawn_gz])
        self._fall_start_height = self.py

        # Roll loadout profile
        self._roll_loadout()

        # Sample episode type
        ep_roll = self._rng.random()
        if ep_roll < 0.60:
            self._episode_type = "zombies"
        elif ep_roll < 0.75:
            self._episode_type = "mixed"
        elif ep_roll < 0.85:
            self._episode_type = "horde"
        elif ep_roll < 0.95:
            self._episode_type = "hard"
        else:
            self._episode_type = "creeper"

        # Spawn entities based on episode type
        if self._episode_type == "zombies":
            for _ in range(self._num_zombies):
                self._spawn_zombie()
        elif self._episode_type == "mixed":
            self._spawn_zombie()
            self._spawn_skeleton()
        elif self._episode_type == "horde":
            for _ in range(3):
                self._spawn_zombie()
        elif self._episode_type == "hard":
            self._spawn_zombie(force_baby=True)
            self._spawn_skeleton()
        elif self._episode_type == "creeper":
            self._spawn_creeper()
            self._spawn_zombie()

    def _clamp_pos(self, v: float) -> float:
        return max(1.0, min(ARENA_SIZE - 1, v))

    def _roll_loadout(self):
        """Randomize weapon, armor, enchantments, equipment, and effects."""
        roll = float(self._rng.random())
        cumulative = 0.0
        profile = LOADOUT_PROFILES[-1]  # fallback
        for p in LOADOUT_PROFILES:
            cumulative += p[0]
            if roll < cumulative:
                profile = p
                break

        _, weapons, armors, sharp_range, kb_range, prot_range, fire_range, axe_chance = profile

        # Weapon (possibly axe)
        if float(self._rng.random()) < axe_chance:
            # Pick an axe matching the tier range
            axe_names = [n for n, w in WEAPON_TIERS.items() if w["is_axe"]]
            self._weapon_name = str(self._rng.choice(axe_names))
        else:
            self._weapon_name = str(self._rng.choice(weapons))
        w = WEAPON_TIERS[self._weapon_name]
        self._weapon_damage = w["damage"]
        self._weapon_tier = w["tier"]
        self._is_axe = w["is_axe"]

        # Armor
        self._armor_name = str(self._rng.choice(armors))
        a = ARMOR_TIERS[self._armor_name]
        self._armor_tier = a["tier"]
        self._armor_reduction = a["reduction"]

        # Enchantments
        self._sharpness_level = int(self._rng.integers(sharp_range[0], sharp_range[1] + 1))
        self._knockback_level = int(self._rng.integers(kb_range[0], kb_range[1] + 1))
        self._protection_level = int(self._rng.integers(prot_range[0], prot_range[1] + 1))
        self._fire_aspect_level = int(self._rng.integers(fire_range[0], fire_range[1] + 1))

        # Shield (50% when armor >= iron)
        self._has_shield = (self._armor_tier >= 0.6 and float(self._rng.random()) < 0.5)

        # Bow (30% chance)
        self._has_bow = float(self._rng.random()) < 0.30

        # Blocks (40% chance, 16-32 blocks)
        self._has_blocks = float(self._rng.random()) < 0.40
        self._block_count = int(self._rng.integers(16, 33)) if self._has_blocks else 0

        # Enderpearl (20% chance, 1-3 pearls)
        self._has_pearl = float(self._rng.random()) < 0.20
        self._pearl_count = int(self._rng.integers(1, 4)) if self._has_pearl else 0

        # Potion effects
        self._effect_speed = int(self._rng.choice([0, 1, 2])) if float(self._rng.random()) < 0.30 else 0
        self._effect_strength = int(self._rng.choice([0, 1, 2])) if float(self._rng.random()) < 0.20 else 0
        self._effect_resistance = int(self._rng.choice([0, 1])) if float(self._rng.random()) < 0.15 else 0

    def _get_damage_reduction(self) -> float:
        """Total damage reduction from armor + protection enchant."""
        prot_bonus = self._protection_level * PROTECTION_PER_LEVEL
        return min(MAX_DAMAGE_REDUCTION, self._armor_reduction + prot_bonus)

    def _spawn_zombie(self, force_baby: bool = False):
        angle = self._rng.uniform(0, 2 * math.pi)
        dist = self._rng.uniform(ZOMBIE_SPAWN_MIN, ZOMBIE_SPAWN_MAX)
        x = self.px + dist * math.cos(angle)
        z = self.pz + dist * math.sin(angle)
        self.zx.append(x)
        self.zz.append(z)
        self.zy.append(self._terrain_height_at(x, z))
        self.zvx.append(0.0)
        self.zvz.append(0.0)
        # Scale mob health with weapon tier (stronger weapons → tougher mobs)
        hp = ZOMBIE_HEALTH * (1.0 + self._weapon_tier * 0.5)  # 20-30 HP
        self.zh.append(hp)
        self.z_invuln.append(0)
        self.z_atk_cd.append(0)
        is_baby = force_baby or (self._rng.random() < ZOMBIE_BABY_CHANCE)
        self.z_is_baby.append(is_baby)
        self.z_fire_ticks.append(0)

    def _spawn_skeleton(self):
        angle = self._rng.uniform(0, 2 * math.pi)
        dist = self._rng.uniform(10.0, 15.0)
        x = self.px + dist * math.cos(angle)
        z = self.pz + dist * math.sin(angle)
        self.sx.append(x)
        self.sz.append(z)
        hp = SKELETON_HEALTH * (1.0 + self._weapon_tier * 0.5)
        self.sh.append(hp)
        self.svx.append(0.0)
        self.svz.append(0.0)
        self.s_fire_cd.append(int(self._rng.integers(SKELETON_FIRE_CD_MIN, SKELETON_FIRE_CD_MAX)))
        self.s_invuln.append(0)
        self.s_fire_ticks.append(0)

    def _spawn_creeper(self):
        angle = self._rng.uniform(0, 2 * math.pi)
        dist = self._rng.uniform(ZOMBIE_SPAWN_MIN, ZOMBIE_SPAWN_MAX)
        x = self.px + dist * math.cos(angle)
        z = self.pz + dist * math.sin(angle)
        self.cx.append(x)
        self.cz.append(z)
        self.cy.append(self._terrain_height_at(x, z))
        self.cvx.append(0.0)
        self.cvz.append(0.0)
        hp = CREEPER_HEALTH * (1.0 + self._weapon_tier * 0.5)
        self.ch.append(hp)
        self.c_fuse.append(-1)  # not fusing
        charged = float(self._rng.random()) < CREEPER_CHARGED_CHANCE
        self.c_charged.append(charged)
        self.c_invuln.append(0)
        self.c_fire_ticks.append(0)

    # ── Terrain Generation ─────────────────────────────────────────

    def _generate_terrain(self):
        """Generate procedural heightmap for this episode."""
        # Select terrain type
        roll = float(self._rng.random())
        if roll < 0.30:
            self._terrain_type = "flat"
        elif roll < 0.55:
            self._terrain_type = "hills"
        elif roll < 0.70:
            self._terrain_type = "pillars"
        elif roll < 0.85:
            self._terrain_type = "valley"
        elif roll < 0.95:
            self._terrain_type = "ruins"
        else:
            self._terrain_type = "platform"

        heights = np.zeros((ARENA_INT, ARENA_INT), dtype=np.float32)

        if self._terrain_type == "flat":
            pass  # all zeros

        elif self._terrain_type == "hills":
            n_hills = int(self._rng.integers(3, 6))
            for _ in range(n_hills):
                cx = float(self._rng.uniform(4, 20))
                cz = float(self._rng.uniform(4, 20))
                radius = float(self._rng.uniform(3, 6))
                peak = float(self._rng.uniform(1, 3))
                for x in range(ARENA_INT):
                    for z in range(ARENA_INT):
                        d = math.hypot(x - cx, z - cz)
                        if d < radius:
                            h = peak * (1 - d / radius) ** 2
                            heights[x, z] = max(heights[x, z], h)

        elif self._terrain_type == "pillars":
            n_pillars = int(self._rng.integers(4, 7))
            for _ in range(n_pillars):
                px = int(self._rng.integers(3, 21))
                pz = int(self._rng.integers(3, 21))
                h = float(self._rng.uniform(3, 4))
                for dx in range(2):
                    for dz in range(2):
                        if 0 <= px + dx < ARENA_INT and 0 <= pz + dz < ARENA_INT:
                            heights[px + dx, pz + dz] = h

        elif self._terrain_type == "valley":
            for x in range(ARENA_INT):
                for z in range(ARENA_INT):
                    edge_dist = min(x, z, ARENA_INT - 1 - x, ARENA_INT - 1 - z)
                    if edge_dist < 4:
                        heights[x, z] = (4 - edge_dist) * 0.7

        elif self._terrain_type == "ruins":
            # Base noise
            for x in range(ARENA_INT):
                for z in range(ARENA_INT):
                    heights[x, z] = float(self._rng.uniform(0, 0.5))
            # Wall segments
            n_walls = int(self._rng.integers(4, 8))
            for _ in range(n_walls):
                wx = int(self._rng.integers(3, 20))
                wz = int(self._rng.integers(3, 20))
                length = int(self._rng.integers(2, 5))
                horizontal = self._rng.random() < 0.5
                h = float(self._rng.uniform(2, 3))
                for s in range(length):
                    if horizontal:
                        if 0 <= wx + s < ARENA_INT:
                            heights[wx + s, wz] = h
                    else:
                        if 0 <= wz + s < ARENA_INT:
                            heights[wx, wz + s] = h

        elif self._terrain_type == "platform":
            # Central 5x5 raised platform
            for x in range(10, 15):
                for z in range(10, 15):
                    heights[x, z] = 3.0
            # Ramp on one side
            for x in range(7, 10):
                for z in range(10, 15):
                    heights[x, z] = (x - 7) * 1.0

        # Round to nearest 0.5 (MC blocks + half-slabs)
        heights = np.round(heights * 2) / 2
        self._heights = heights

        # Generate water map (water at height=0 cells, certain terrain types)
        self._water_map = np.zeros((ARENA_INT, ARENA_INT), dtype=bool)
        if self._terrain_type in ("flat", "valley"):
            water_chance = 0.12 if self._terrain_type == "flat" else 0.18
            # Place water pools (clusters of 2-4 cells)
            n_pools = int(self._rng.integers(2, 5))
            for _ in range(n_pools):
                wx = int(self._rng.integers(3, ARENA_INT - 3))
                wz = int(self._rng.integers(3, ARENA_INT - 3))
                pool_size = int(self._rng.integers(2, 4))
                for dx in range(-pool_size, pool_size + 1):
                    for dz in range(-pool_size, pool_size + 1):
                        nx, nz = wx + dx, wz + dz
                        if (0 <= nx < ARENA_INT and 0 <= nz < ARENA_INT
                                and heights[nx, nz] == 0.0
                                and float(self._rng.random()) < water_chance * 3):
                            self._water_map[nx, nz] = True

    def _terrain_height_at(self, x: float, z: float) -> float:
        """Get terrain height at continuous position. Returns 0 outside heightmap."""
        gx = int(x)
        gz = int(z)
        if 0 <= gx < ARENA_INT and 0 <= gz < ARENA_INT:
            return float(self._heights[gx, gz])
        return 0.0  # flat ground outside terrain feature area

    def _is_in_water(self, x: float, z: float) -> bool:
        """Check if position is in a water cell."""
        gx = int(x)
        gz = int(z)
        if 0 <= gx < ARENA_INT and 0 <= gz < ARENA_INT:
            return bool(self._water_map[gx, gz])
        return False

    def _can_move_to(self, new_x: float, new_z: float) -> bool:
        """Check if player/entity can move to (new_x, new_z) given terrain."""
        dest_height = self._terrain_height_at(new_x, new_z)
        current_ground = self._terrain_height_at(self.px, self.pz)
        height_diff = dest_height - max(self.py, current_ground)

        # Auto-step: can walk up ≤0.6 blocks without jumping
        if height_diff <= STEP_UP_HEIGHT:
            return True
        # Airborne: can clear up to ~1.25 blocks
        if not self.on_ground and height_diff <= JUMP_CLEAR_HEIGHT:
            return True
        return False

    def _mob_can_move(self, from_x: float, from_z: float, to_x: float, to_z: float) -> bool:
        """Check if a mob can walk between two positions."""
        from_h = self._terrain_height_at(from_x, from_z)
        to_h = self._terrain_height_at(to_x, to_z)
        return to_h - from_h <= MOB_STEP_HEIGHT

    # ── Physics ──────────────────────────────────────────────────────

    def _physics_tick(self):
        """Apply gravity, friction, and update position with terrain collision."""
        in_water = self._is_in_water(self.px, self.pz)

        # Gravity
        if not self.on_ground:
            self.vy -= GRAVITY

        # Friction (applied before acceleration in MC)
        if in_water:
            self.vx *= WATER_FRICTION
            self.vz *= WATER_FRICTION
        else:
            self.vx *= GROUND_FRICTION
            self.vz *= GROUND_FRICTION
        if not self.on_ground:
            self.vy *= AIR_FRICTION_VERT

        # Compute candidate position
        new_x = self.px + self.vx
        new_z = self.pz + self.vz

        # Movement blocking: check if horizontal move is passable
        if not self._can_move_to(new_x, new_z):
            # Try sliding along each axis separately
            if self._can_move_to(new_x, self.pz):
                self.px = new_x
                self.vz = 0.0  # blocked in Z
            elif self._can_move_to(self.px, new_z):
                self.pz = new_z
                self.vx = 0.0  # blocked in X
            else:
                # Fully blocked
                self.vx = 0.0
                self.vz = 0.0
        else:
            self.px = new_x
            self.pz = new_z

        # Vertical movement
        self.py += self.vy

        # Ground collision using heightmap
        ground_h = self._terrain_height_at(self.px, self.pz)
        if self.py <= ground_h:
            # Fall damage check (no fall damage in water)
            fall_dist = self._fall_start_height - ground_h
            if fall_dist > FALL_DAMAGE_THRESHOLD and not in_water:
                damage = fall_dist - FALL_DAMAGE_THRESHOLD
                self.health -= damage
                if self.health <= 0:
                    self.health = 0
                    self.alive = False
                    self.total_deaths += 1

            self.py = ground_h
            self.vy = 0.0
            self.on_ground = True
            self.ticks_airborne = 0
            self._fall_start_height = ground_h
        else:
            self.on_ground = False
            self.ticks_airborne += 1
            # Track peak height for fall damage
            if self.py > self._fall_start_height:
                self._fall_start_height = self.py

        # Auto-step: if on ground and destination is slightly higher, snap up
        if self.on_ground:
            ground_at_pos = self._terrain_height_at(self.px, self.pz)
            if ground_at_pos > self.py and ground_at_pos - self.py <= STEP_UP_HEIGHT:
                self.py = ground_at_pos
                self._fall_start_height = self.py

    def _apply_accel(self, dx: float, dz: float, accel: float):
        """Apply acceleration in direction (dx, dz), respecting air control and effects."""
        # Speed potion effect: +20% per level
        effective_accel = accel * (1.0 + 0.2 * self._effect_speed)
        # Eating slows to walk speed
        if self._is_eating:
            effective_accel = min(effective_accel, GROUND_ACCEL)
        if self.on_ground:
            self.vx += dx * effective_accel
            self.vz += dz * effective_accel
        else:
            self.vx += dx * effective_accel * AIR_ACCEL_MULT
            self.vz += dz * effective_accel * AIR_ACCEL_MULT

    def _jump(self):
        """Initiate jump if on ground."""
        if self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False
            self.ticks_airborne = 1
            # Sprint-jump boost
            if self.is_sprinting:
                cos_y = math.cos(self.yaw)
                sin_y = math.sin(self.yaw)
                self.vx += -sin_y * SPRINT_JUMP_BOOST
                self.vz += cos_y * SPRINT_JUMP_BOOST
            # Exhaustion
            if self.is_sprinting:
                self.exhaustion += EXHAUSTION_PER_JUMP
            else:
                self.exhaustion += 0.05

    # ── Sprint / Food ────────────────────────────────────────────────

    def _update_sprint(self, wants_sprint: bool, wants_forward: bool):
        """Update sprint state with food check. Can't sprint in water or while eating."""
        in_water = self._is_in_water(self.px, self.pz)
        if (wants_sprint and wants_forward and self.food >= MIN_FOOD_TO_SPRINT
                and not in_water and not self._is_eating):
            if not self.is_sprinting:
                self.is_sprinting = True
                self.sprint_ticks = 0
            self.sprint_ticks += 1
        else:
            self.is_sprinting = False
            self.sprint_ticks = 0

    def _drain_exhaustion(self):
        """Drain food from sprint/jump exhaustion."""
        if self.is_sprinting:
            self.exhaustion += EXHAUSTION_PER_SPRINT_TICK
        while self.exhaustion >= EXHAUSTION_THRESHOLD:
            self.exhaustion -= EXHAUSTION_THRESHOLD
            if self.saturation > 0:
                self.saturation = max(0.0, self.saturation - 1.0)
            else:
                self.food = max(0, self.food - 1)

    # ── Combat ───────────────────────────────────────────────────────

    def _in_facing_cone(self, tx: float, tz: float) -> bool:
        """Check if target is within ±30° of player facing."""
        dx = tx - self.px
        dz = tz - self.pz
        angle_to_target = math.atan2(dz, dx)
        # Player yaw uses MC convention: -sin(yaw) = forward X, cos(yaw) = forward Z
        # But in our sim yaw is already the facing angle in (x, z) space
        angle_diff = (angle_to_target - self.yaw + math.pi) % (2 * math.pi) - math.pi
        return abs(angle_diff) <= FACING_CONE_RAD

    def _try_attack(self) -> bool:
        """Attempt melee attack with auto-aim, CPS, and crit checks.

        Auto-aim: snaps yaw to nearest target within range AND forward hemisphere
        (±90°). Simulates clicking on an entity — aim is implicit in the attack
        intent. Agent still needs to be in RANGE and roughly facing forward.
        """
        # CPS limit
        if self.tick - self.last_attack_tick < MIN_ATTACK_INTERVAL:
            return False
        self.last_attack_tick = self.tick
        self.exhaustion += EXHAUSTION_PER_ATTACK

        # Find nearest target in range AND forward hemisphere (±90°)
        best_type = None  # 'z', 's', or 'c'
        best_i = -1
        best_d = float("inf")
        best_angle = 0.0

        half_pi = math.pi / 2  # ±90° = forward hemisphere

        for i in range(len(self.zx)):
            if self.zh[i] <= 0 or self.z_invuln[i] > 0:
                continue
            d = math.sqrt(
                (self.zx[i] - self.px) ** 2
                + (self.zy[i] - self.py) ** 2
                + (self.zz[i] - self.pz) ** 2
            )
            if d > MELEE_REACH:
                continue
            angle_to = math.atan2(self.zz[i] - self.pz, self.zx[i] - self.px)
            angle_diff = (angle_to - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > half_pi:
                continue
            if d < best_d:
                best_type = "z"
                best_i = i
                best_d = d
                best_angle = angle_to

        for i in range(len(self.sx)):
            if self.sh[i] <= 0 or self.s_invuln[i] > 0:
                continue
            d = math.sqrt(
                (self.sx[i] - self.px) ** 2 + (self.sz[i] - self.pz) ** 2
            )
            if d > MELEE_REACH:
                continue
            angle_to = math.atan2(self.sz[i] - self.pz, self.sx[i] - self.px)
            angle_diff = (angle_to - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > half_pi:
                continue
            if d < best_d:
                best_type = "s"
                best_i = i
                best_d = d
                best_angle = angle_to

        for i in range(len(self.cx)):
            if self.ch[i] <= 0 or self.c_invuln[i] > 0:
                continue
            d = math.sqrt(
                (self.cx[i] - self.px) ** 2
                + (self.cy[i] - self.py) ** 2
                + (self.cz[i] - self.pz) ** 2
            )
            if d > MELEE_REACH:
                continue
            angle_to = math.atan2(self.cz[i] - self.pz, self.cx[i] - self.px)
            angle_diff = (angle_to - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > half_pi:
                continue
            if d < best_d:
                best_type = "c"
                best_i = i
                best_d = d
                best_angle = angle_to

        if best_i < 0:
            return False  # whiff — nothing in range or forward hemisphere

        # Auto-aim: snap yaw to target
        self.yaw = best_angle

        # Crit check: falling and not on ground (disabled in water)
        in_water = self._is_in_water(self.px, self.pz)
        is_crit = (not self.on_ground) and (self.vy < 0) and not in_water
        base_damage = self._weapon_damage + self._sharpness_level * SHARPNESS_BONUS_PER_LEVEL
        # Strength potion effect: +3 damage per level
        base_damage += self._effect_strength * 3.0
        damage = base_damage * (CRIT_MULT if is_crit else 1.0)

        # Knockback calculation
        kb_strength = BASE_KB
        if self.is_sprinting:
            kb_strength += SPRINT_KB_BONUS
        kb_strength += self._knockback_level * KB_BONUS_PER_LEVEL

        # Forward movement reduction
        horiz_speed = math.hypot(self.vx, self.vz)
        if horiz_speed > 0.05:
            kb_strength *= FORWARD_KB_REDUCTION

        # Apply to target
        if best_type == "z":
            self._apply_kb_to_zombie(best_i, kb_strength)
            self.zh[best_i] -= damage
            self.z_invuln[best_i] = ZOMBIE_INVULN
            if self._fire_aspect_level > 0:
                self.z_fire_ticks[best_i] = FIRE_ASPECT_DURATION * self._fire_aspect_level
            if self.zh[best_i] <= 0:
                self.kills_this_tick += 1
                self.total_kills += 1
        elif best_type == "s":
            self._apply_kb_to_skeleton(best_i, kb_strength)
            self.sh[best_i] -= damage
            self.s_invuln[best_i] = ZOMBIE_INVULN
            if self._fire_aspect_level > 0:
                self.s_fire_ticks[best_i] = FIRE_ASPECT_DURATION * self._fire_aspect_level
            if self.sh[best_i] <= 0:
                self.kills_this_tick += 1
                self.total_kills += 1
        elif best_type == "c":
            self._apply_kb_to_creeper(best_i, kb_strength)
            self.ch[best_i] -= damage
            self.c_invuln[best_i] = ZOMBIE_INVULN
            if self._fire_aspect_level > 0:
                self.c_fire_ticks[best_i] = FIRE_ASPECT_DURATION * self._fire_aspect_level
            # Hit partially resets creeper fuse
            if self.c_fuse[best_i] >= 0:
                self.c_fuse[best_i] = min(CREEPER_FUSE, self.c_fuse[best_i] + 10)
            if self.ch[best_i] <= 0:
                self.kills_this_tick += 1
                self.total_kills += 1

        return True

    def _apply_kb_to_zombie(self, i: int, strength: float):
        """Apply knockback velocity to zombie."""
        dx = self.zx[i] - self.px
        dz = self.zz[i] - self.pz
        d = math.hypot(dx, dz) or 1.0
        self.zvx[i] = (dx / d) * strength
        self.zvz[i] = (dz / d) * strength

    def _apply_kb_to_skeleton(self, i: int, strength: float):
        """Apply knockback velocity to skeleton."""
        dx = self.sx[i] - self.px
        dz = self.sz[i] - self.pz
        d = math.hypot(dx, dz) or 1.0
        self.svx[i] = (dx / d) * strength
        self.svz[i] = (dz / d) * strength

    def _apply_kb_to_creeper(self, i: int, strength: float):
        """Apply knockback velocity to creeper."""
        dx = self.cx[i] - self.px
        dz = self.cz[i] - self.pz
        d = math.hypot(dx, dz) or 1.0
        self.cvx[i] = (dx / d) * strength
        self.cvz[i] = (dz / d) * strength

    def _apply_player_kb(self, from_x: float, from_z: float):
        """Apply knockback to player when hit by mob."""
        dx = self.px - from_x
        dz = self.pz - from_z
        d = math.hypot(dx, dz) or 1.0
        self.vx = (dx / d) * BASE_KB
        self.vz = (dz / d) * BASE_KB
        self.vy = KB_UPWARD
        self.on_ground = False

    # ── Entity AI ────────────────────────────────────────────────────

    def _zombie_ai(self):
        """Move zombies toward player, handle attacks with terrain navigation."""
        for i in range(len(self.zx)):
            if self.zh[i] <= 0:
                continue

            # Decrement cooldowns
            if self.z_invuln[i] > 0:
                self.z_invuln[i] -= 1
            if self.z_atk_cd[i] > 0:
                self.z_atk_cd[i] -= 1

            # Fire DoT
            if self.z_fire_ticks[i] > 0:
                self.z_fire_ticks[i] -= 1
                self.zh[i] -= FIRE_ASPECT_DPS / 20.0
                if self.zh[i] <= 0:
                    self.kills_this_tick += 1
                    self.total_kills += 1
                    continue

            # If zombie has KB velocity, decelerate
            z_speed = math.hypot(self.zvx[i], self.zvz[i])
            if z_speed > ZOMBIE_SPEED * 1.5:
                # Still in KB, apply friction
                self.zvx[i] *= ZOMBIE_KB_FRICTION
                self.zvz[i] *= ZOMBIE_KB_FRICTION
                self.zx[i] += self.zvx[i]
                self.zz[i] += self.zvz[i]
            else:
                # Normal pathfinding with terrain obstacle avoidance
                dx = self.px - self.zx[i]
                dz = self.pz - self.zz[i]
                d = math.hypot(dx, dz)
                if d > 0.5:
                    angle = math.atan2(dz, dx)
                    wobble = float(self._rng.uniform(-ZOMBIE_WOBBLE_RAD, ZOMBIE_WOBBLE_RAD))
                    angle += wobble
                    speed = ZOMBIE_SPEED * (ZOMBIE_BABY_SPEED_MULT if self.z_is_baby[i] else 1.0)

                    # Try direct path, then alternates if blocked
                    moved = False
                    for angle_offset in (0.0, 0.785, -0.785, 1.57, -1.57):
                        try_angle = angle + angle_offset
                        nx = self.zx[i] + math.cos(try_angle) * speed
                        nz = self.zz[i] + math.sin(try_angle) * speed
                        if self._mob_can_move(self.zx[i], self.zz[i], nx, nz):
                            self.zvx[i] = math.cos(try_angle) * speed
                            self.zvz[i] = math.sin(try_angle) * speed
                            self.zx[i] = nx
                            self.zz[i] = nz
                            moved = True
                            break
                    if not moved:
                        self.zvx[i] = 0.0
                        self.zvz[i] = 0.0
                else:
                    self.zvx[i] = 0.0
                    self.zvz[i] = 0.0

            # Update zombie Y to terrain height
            self.zy[i] = self._terrain_height_at(self.zx[i], self.zz[i])

            # Zombie attack (check 3D distance including height)
            d2 = math.sqrt(
                (self.zx[i] - self.px) ** 2
                + (self.zy[i] - self.py) ** 2
                + (self.zz[i] - self.pz) ** 2
            )
            if d2 <= ZOMBIE_ATTACK_RANGE and self.z_atk_cd[i] <= 0:
                self.z_atk_cd[i] = ZOMBIE_ATTACK_CD
                # Shield check: block frontal damage
                if self._shield_raised and self._shield_disabled_ticks <= 0:
                    angle_to_zombie = math.atan2(
                        self.zz[i] - self.pz, self.zx[i] - self.px
                    )
                    angle_diff = (angle_to_zombie - self.yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(angle_diff) <= math.pi / 2:
                        # Blocked by shield — KB to zombie instead
                        self._apply_kb_to_zombie(i, BASE_KB * 0.5)
                        continue
                # Mob damage scales with armor tier (harder content for geared players)
                raw_damage = ZOMBIE_DAMAGE * (1.0 + self._armor_tier * 1.0)  # 3-6 dmg
                # Resistance effect: -20% per level
                resist_mult = max(0.0, 1.0 - 0.2 * self._effect_resistance)
                actual_damage = raw_damage * (1.0 - self._get_damage_reduction()) * resist_mult
                self.health -= actual_damage
                self._apply_player_kb(self.zx[i], self.zz[i])
                # Cancel eating on hit
                if self._is_eating:
                    self._is_eating = False
                    self._eat_ticks = 0
                if self.health <= 0:
                    self.health = 0
                    self.alive = False
                    self.total_deaths += 1

    def _skeleton_ai(self):
        """Move skeletons, shoot arrows with terrain navigation."""
        for i in range(len(self.sx)):
            if self.sh[i] <= 0:
                continue

            if self.s_invuln[i] > 0:
                self.s_invuln[i] -= 1
            if self.s_fire_cd[i] > 0:
                self.s_fire_cd[i] -= 1

            # Fire DoT
            if self.s_fire_ticks[i] > 0:
                self.s_fire_ticks[i] -= 1
                self.sh[i] -= FIRE_ASPECT_DPS / 20.0
                if self.sh[i] <= 0:
                    self.kills_this_tick += 1
                    self.total_kills += 1
                    continue

            # If skeleton has KB velocity, decelerate
            s_speed = math.hypot(self.svx[i], self.svz[i])
            if s_speed > SKELETON_SPEED * 1.5:
                self.svx[i] *= ZOMBIE_KB_FRICTION
                self.svz[i] *= ZOMBIE_KB_FRICTION
                self.sx[i] += self.svx[i]
                self.sz[i] += self.svz[i]
                continue

            dx = self.px - self.sx[i]
            dz = self.pz - self.sz[i]
            dist = math.hypot(dx, dz)

            if dist < SKELETON_RETREAT_DIST and dist > 0.1:
                angle = math.atan2(-dz, -dx)  # away from player
            elif dist > SKELETON_RANGE:
                angle = math.atan2(dz, dx)  # toward player
            else:
                # In range: stop and shoot
                self.svx[i] = 0.0
                self.svz[i] = 0.0
                angle = None

            if angle is not None:
                # Try movement with obstacle avoidance
                moved = False
                for angle_offset in (0.0, 0.785, -0.785, 1.57, -1.57):
                    try_angle = angle + angle_offset
                    nx = self.sx[i] + math.cos(try_angle) * SKELETON_SPEED
                    nz = self.sz[i] + math.sin(try_angle) * SKELETON_SPEED
                    if self._mob_can_move(self.sx[i], self.sz[i], nx, nz):
                        self.svx[i] = math.cos(try_angle) * SKELETON_SPEED
                        self.svz[i] = math.sin(try_angle) * SKELETON_SPEED
                        self.sx[i] = nx
                        self.sz[i] = nz
                        moved = True
                        break
                if not moved:
                    self.svx[i] = 0.0
                    self.svz[i] = 0.0

            # Shoot arrow
            if dist <= SKELETON_RANGE and self.s_fire_cd[i] <= 0:
                self._shoot_arrow(i)
                self.s_fire_cd[i] = int(self._rng.integers(
                    SKELETON_FIRE_CD_MIN, SKELETON_FIRE_CD_MAX
                ))

    def _creeper_ai(self):
        """Move creepers toward player, handle fuse countdown and explosion."""
        exploded = []
        for i in range(len(self.cx)):
            if self.ch[i] <= 0:
                continue

            if self.c_invuln[i] > 0:
                self.c_invuln[i] -= 1

            # Fire DoT
            if self.c_fire_ticks[i] > 0:
                self.c_fire_ticks[i] -= 1
                self.ch[i] -= FIRE_ASPECT_DPS / 20.0
                if self.ch[i] <= 0:
                    self.kills_this_tick += 1
                    self.total_kills += 1
                    continue

            # KB deceleration
            c_speed = math.hypot(self.cvx[i], self.cvz[i])
            if c_speed > CREEPER_SPEED * 1.5:
                self.cvx[i] *= CREEPER_KB_FRICTION
                self.cvz[i] *= CREEPER_KB_FRICTION
                self.cx[i] += self.cvx[i]
                self.cz[i] += self.cvz[i]
            else:
                # Move toward player
                dx = self.px - self.cx[i]
                dz = self.pz - self.cz[i]
                d = math.hypot(dx, dz)
                if d > 0.5:
                    angle = math.atan2(dz, dx)
                    for angle_offset in (0.0, 0.785, -0.785, 1.57, -1.57):
                        try_angle = angle + angle_offset
                        nx = self.cx[i] + math.cos(try_angle) * CREEPER_SPEED
                        nz = self.cz[i] + math.sin(try_angle) * CREEPER_SPEED
                        if self._mob_can_move(self.cx[i], self.cz[i], nx, nz):
                            self.cvx[i] = math.cos(try_angle) * CREEPER_SPEED
                            self.cvz[i] = math.sin(try_angle) * CREEPER_SPEED
                            self.cx[i] = nx
                            self.cz[i] = nz
                            break
                    else:
                        self.cvx[i] = 0.0
                        self.cvz[i] = 0.0

            # Update Y
            self.cy[i] = self._terrain_height_at(self.cx[i], self.cz[i])

            # Fuse logic
            d_to_player = math.sqrt(
                (self.cx[i] - self.px) ** 2
                + (self.cy[i] - self.py) ** 2
                + (self.cz[i] - self.pz) ** 2
            )
            if d_to_player <= 3.0:
                if self.c_fuse[i] < 0:
                    self.c_fuse[i] = CREEPER_FUSE  # start fuse
                else:
                    self.c_fuse[i] -= 1
                    if self.c_fuse[i] <= 0:
                        # EXPLODE
                        exploded.append(i)
            else:
                # Reset fuse if player escapes
                if self.c_fuse[i] >= 0:
                    self.c_fuse[i] = -1

        # Process explosions (reverse order for safe removal)
        for i in sorted(exploded, reverse=True):
            self._creeper_explode(i)

    def _creeper_explode(self, i: int):
        """Handle creeper explosion: area damage to player and nearby mobs."""
        ex, ez, ey = self.cx[i], self.cz[i], self.cy[i]
        is_charged = self.c_charged[i]
        mult = CREEPER_CHARGED_MULT if is_charged else 1.0

        # Damage to player
        dx = self.px - ex
        dz = self.pz - ez
        dy = self.py - ey
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < CREEPER_BLAST_RADIUS:
            # Shield check
            blocked = False
            if self._shield_raised and self._shield_disabled_ticks <= 0:
                angle_to = math.atan2(ez - self.pz, ex - self.px)
                angle_diff = (angle_to - self.yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(angle_diff) <= math.pi / 2:
                    blocked = True
            if not blocked:
                damage_frac = max(0.0, 1.0 - dist / CREEPER_BLAST_RADIUS)
                resist_mult = max(0.0, 1.0 - 0.2 * self._effect_resistance)
                raw_damage = CREEPER_DAMAGE * damage_frac * mult
                actual = raw_damage * (1.0 - self._get_damage_reduction()) * resist_mult
                self.health -= actual
                self._apply_player_kb(ex, ez)
                if self._is_eating:
                    self._is_eating = False
                    self._eat_ticks = 0
                if self.health <= 0:
                    self.health = 0
                    self.alive = False
                    self.total_deaths += 1

        # Damage to nearby zombies
        for j in range(len(self.zx)):
            if self.zh[j] <= 0:
                continue
            d = math.sqrt((self.zx[j] - ex) ** 2 + (self.zz[j] - ez) ** 2)
            if d < CREEPER_BLAST_RADIUS:
                frac = max(0.0, 1.0 - d / CREEPER_BLAST_RADIUS)
                self.zh[j] -= CREEPER_DAMAGE * frac * mult
                if self.zh[j] <= 0:
                    self.kills_this_tick += 1
                    self.total_kills += 1

        # Damage to nearby skeletons
        for j in range(len(self.sx)):
            if self.sh[j] <= 0:
                continue
            d = math.sqrt((self.sx[j] - ex) ** 2 + (self.sz[j] - ez) ** 2)
            if d < CREEPER_BLAST_RADIUS:
                frac = max(0.0, 1.0 - d / CREEPER_BLAST_RADIUS)
                self.sh[j] -= CREEPER_DAMAGE * frac * mult
                if self.sh[j] <= 0:
                    self.kills_this_tick += 1
                    self.total_kills += 1

        # Creeper dies on explosion
        self.ch[i] = 0

    def _shoot_arrow(self, skeleton_idx: int):
        """Skeleton fires arrow at player with accuracy error."""
        sx = self.sx[skeleton_idx]
        sz = self.sz[skeleton_idx]
        sy = self._terrain_height_at(sx, sz) + 1.5  # arrow from skeleton eye height

        # Direction to player (with lead and accuracy error)
        dx = self.px - sx
        dy = (self.py + 1.0) - sy  # aim at player center
        dz = self.pz - sz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0

        # Normalize direction
        nx = dx / dist
        ny = dy / dist
        nz = dz / dist

        # Add accuracy error (random angular offset)
        err_yaw = float(self._rng.uniform(-SKELETON_ACCURACY_ERROR, SKELETON_ACCURACY_ERROR))
        err_pitch = float(self._rng.uniform(-SKELETON_ACCURACY_ERROR, SKELETON_ACCURACY_ERROR))

        cos_e = math.cos(err_yaw)
        sin_e = math.sin(err_yaw)
        # Rotate around Y axis (yaw error)
        rnx = nx * cos_e - nz * sin_e
        rnz = nx * sin_e + nz * cos_e
        # Rotate pitch (simple approximation)
        rny = ny + math.sin(err_pitch) * 0.3

        # Loft the arrow slightly to account for gravity over distance
        flight_time = dist / SKELETON_ARROW_SPEED
        loft = 0.5 * SKELETON_ARROW_GRAVITY * flight_time
        rny += loft

        # Normalize again
        mag = math.sqrt(rnx * rnx + rny * rny + rnz * rnz) or 1.0
        rnx /= mag
        rny /= mag
        rnz /= mag

        arrow = Arrow(
            x=sx, y=sy, z=sz,
            vx=rnx * SKELETON_ARROW_SPEED,
            vy=rny * SKELETON_ARROW_SPEED,
            vz=rnz * SKELETON_ARROW_SPEED,
        )
        self.arrows.append(arrow)

    def _tick_arrows(self):
        """Advance all arrows, handle hits, terrain collision, and pearls."""
        surviving = []
        for arrow in self.arrows:
            status = arrow.tick()
            if status == "despawn":
                continue

            # Terrain collision: arrow below ground at its position
            terrain_h = self._terrain_height_at(arrow.x, arrow.z)
            if arrow.y <= terrain_h:
                if arrow.is_pearl:
                    # Enderpearl: teleport player to landing pos
                    self.px = arrow.x
                    self.pz = arrow.z
                    self.py = terrain_h
                    self.vx = 0.0
                    self.vz = 0.0
                    self.vy = 0.0
                    self.health -= PEARL_DAMAGE
                    self._fall_start_height = terrain_h
                    if self.health <= 0:
                        self.health = 0
                        self.alive = False
                        self.total_deaths += 1
                continue  # arrow/pearl stuck in terrain

            # Out of arena bounds (far)
            if (arrow.x < -50 or arrow.x >= ARENA_SIZE + 50
                    or arrow.z < -50 or arrow.z >= ARENA_SIZE + 50):
                continue

            if arrow.is_pearl:
                surviving.append(arrow)
                continue

            if arrow.is_player:
                # Player arrow: check collision with mobs
                hit = False
                for j in range(len(self.zx)):
                    if self.zh[j] <= 0:
                        continue
                    d = math.sqrt(
                        (arrow.x - self.zx[j]) ** 2
                        + (arrow.y - self.zy[j] - 1.0) ** 2
                        + (arrow.z - self.zz[j]) ** 2
                    )
                    if d < 0.6:
                        self.zh[j] -= arrow.damage
                        self.hit_landed = True
                        if self.zh[j] <= 0:
                            self.kills_this_tick += 1
                            self.total_kills += 1
                        hit = True
                        break
                if not hit:
                    for j in range(len(self.sx)):
                        if self.sh[j] <= 0:
                            continue
                        d = math.sqrt(
                            (arrow.x - self.sx[j]) ** 2 + (arrow.z - self.sz[j]) ** 2
                        )
                        if d < 0.6:
                            self.sh[j] -= arrow.damage
                            self.hit_landed = True
                            if self.sh[j] <= 0:
                                self.kills_this_tick += 1
                                self.total_kills += 1
                            hit = True
                            break
                if not hit:
                    for j in range(len(self.cx)):
                        if self.ch[j] <= 0:
                            continue
                        d = math.sqrt(
                            (arrow.x - self.cx[j]) ** 2 + (arrow.z - self.cz[j]) ** 2
                        )
                        if d < 0.6:
                            self.ch[j] -= arrow.damage
                            if self.ch[j] <= 0:
                                self.kills_this_tick += 1
                                self.total_kills += 1
                            hit = True
                            break
                if hit:
                    continue
                surviving.append(arrow)
                continue

            # Skeleton arrow: hit detection against player (dist < 0.6 blocks)
            dx = arrow.x - self.px
            dy = arrow.y - (self.py + 1.0)  # player center
            dz = arrow.z - self.pz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 0.6:
                # Shield check
                if self._shield_raised and self._shield_disabled_ticks <= 0:
                    angle_to = math.atan2(arrow.z - self.pz, arrow.x - self.px)
                    angle_diff = (angle_to - self.yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(angle_diff) <= math.pi / 2:
                        continue  # blocked by shield
                resist_mult = max(0.0, 1.0 - 0.2 * self._effect_resistance)
                actual_damage = SKELETON_ARROW_DAMAGE * (1.0 - self._get_damage_reduction()) * resist_mult
                self.health -= actual_damage
                self._apply_player_kb(arrow.x, arrow.z)
                if self._is_eating:
                    self._is_eating = False
                    self._eat_ticks = 0
                if self.health <= 0:
                    self.health = 0
                    self.alive = False
                    self.total_deaths += 1
                continue  # arrow consumed

            surviving.append(arrow)
        self.arrows = surviving

    # ── Action Processing ────────────────────────────────────────────

    def _process_discrete_action(self, action: int):
        """Process a discrete (Phase 3/4) action."""
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        # MC convention: forward = (-sin(yaw), cos(yaw))
        fwd_x = -sin_y
        fwd_z = cos_y
        right_x = cos_y
        right_z = sin_y

        wants_sprint = False
        wants_forward = False
        do_jump = False
        do_attack = False

        if action == 0:    # FORWARD
            accel = SPRINT_ACCEL if self.is_sprinting else GROUND_ACCEL
            self._apply_accel(fwd_x, fwd_z, accel)
            wants_forward = True
            wants_sprint = self.is_sprinting
        elif action == 1:  # BACKWARD
            self._apply_accel(-fwd_x, -fwd_z, GROUND_ACCEL * 0.7)
            self.is_sprinting = False
        elif action == 2:  # STRAFE_LEFT
            self._apply_accel(-right_x, -right_z, GROUND_ACCEL)
        elif action == 3:  # STRAFE_RIGHT
            self._apply_accel(right_x, right_z, GROUND_ACCEL)
        elif action == 4:  # JUMP
            do_jump = True
        elif action == 5:  # FORWARD_JUMP
            accel = SPRINT_ACCEL if self.is_sprinting else GROUND_ACCEL
            self._apply_accel(fwd_x, fwd_z, accel)
            wants_forward = True
            wants_sprint = self.is_sprinting
            do_jump = True
        elif action == 6:  # SPRINT_FORWARD
            wants_sprint = True
            wants_forward = True
            self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)
        elif action == 7:  # LOOK_LEFT
            self.yaw -= LOOK_DELTA
        elif action == 8:  # LOOK_RIGHT
            self.yaw += LOOK_DELTA
        elif action == 11:  # ATTACK
            do_attack = True
        elif action == 18:  # SPRINT_CRIT
            # Sprint + jump + attack. Crit only if already falling.
            wants_sprint = True
            wants_forward = True
            self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)
            do_jump = True
            do_attack = True
        elif action == 19:  # W_TAP
            # Stop movement → attack (full KB) → re-sprint
            self.vx *= 0.1
            self.vz *= 0.1
            do_attack = True
            wants_sprint = True
            wants_forward = True
        elif action == 20:  # APPROACH_TARGET (macro)
            ni = self._nearest_enemy_info()
            if ni is not None:
                tx, tz, _ = ni
                dx = tx - self.px
                dz = tz - self.pz
                d = math.hypot(dx, dz)
                if d > 0.1:
                    self.yaw = math.atan2(dz, dx)
                    # Remap forward after yaw change
                    fwd_x = -math.sin(self.yaw)
                    fwd_z = math.cos(self.yaw)
                    self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)
                    wants_sprint = True
                    wants_forward = True
        elif action == 21:  # FLEE (macro)
            ni = self._nearest_enemy_info()
            if ni is not None:
                tx, tz, _ = ni
                dx = self.px - tx
                dz = self.pz - tz
                d = math.hypot(dx, dz)
                if d > 0.1:
                    flee_yaw = math.atan2(dz, dx)
                    self.yaw = flee_yaw
                    fwd_x = -math.sin(self.yaw)
                    fwd_z = math.cos(self.yaw)
                    self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)
                    wants_sprint = True
                    wants_forward = True
        elif action == 13:  # USE_ITEM
            self._process_use_item()
        elif action == 14:  # HOTBAR_NEXT
            self._cycle_hotbar(1)
        elif action == 15:  # HOTBAR_PREV
            self._cycle_hotbar(-1)
        # 9, 10, 12, 16-17, 22-24: no-ops in sim

        self._update_sprint(wants_sprint, wants_forward)
        if do_jump:
            self._jump()
        if do_attack:
            self.hit_landed = self._try_attack()

    def _process_factored_action(self, movement: int, look: int, combat: int):
        """Process a factored (composite) action's three axes."""
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        fwd_x = -sin_y
        fwd_z = cos_y
        right_x = cos_y
        right_z = sin_y

        wants_sprint = self.is_sprinting
        wants_forward = False
        do_jump = False
        do_attack = False

        # Movement axis (0-6)
        if movement == 1:    # forward
            accel = SPRINT_ACCEL if self.is_sprinting else GROUND_ACCEL
            self._apply_accel(fwd_x, fwd_z, accel)
            wants_forward = True
        elif movement == 2:  # backward
            self._apply_accel(-fwd_x, -fwd_z, GROUND_ACCEL * 0.7)
            self.is_sprinting = False
        elif movement == 3:  # left
            self._apply_accel(-right_x, -right_z, GROUND_ACCEL)
        elif movement == 4:  # right
            self._apply_accel(right_x, right_z, GROUND_ACCEL)
        elif movement == 5:  # forward+jump
            accel = SPRINT_ACCEL if self.is_sprinting else GROUND_ACCEL
            self._apply_accel(fwd_x, fwd_z, accel)
            wants_forward = True
            do_jump = True
        elif movement == 6:  # sprint
            wants_sprint = True
            wants_forward = True
            self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)

        # Look axis (0-5)
        if look == 1:  # track target
            ni = self._nearest_enemy_info()
            if ni is not None:
                tx, tz, _ = ni
                dx = tx - self.px
                dz = tz - self.pz
                if math.hypot(dx, dz) > 0.1:
                    self.yaw = math.atan2(dz, dx)
        elif look == 2:  # look left
            self.yaw -= LOOK_DELTA
        elif look == 3:  # look right
            self.yaw += LOOK_DELTA
        # look 4,5 (up/down) are no-ops in sim

        # Combat axis (0-6)
        if combat == 1:    # attack
            do_attack = True
        elif combat == 2:  # crit
            wants_sprint = True
            wants_forward = True
            self._apply_accel(fwd_x, fwd_z, SPRINT_ACCEL)
            do_jump = True
            do_attack = True
        elif combat == 3:  # wtap
            self.vx *= 0.1
            self.vz *= 0.1
            do_attack = True
            wants_sprint = True
            wants_forward = True
        elif combat == 4:  # use_start (raise shield / start charge / eat / place / pearl)
            self._process_use_item()
        elif combat == 5:  # use_stop (lower shield / release bow)
            self._process_use_stop()
        elif combat == 6:  # hotbar_next
            self._cycle_hotbar(1)

        self._update_sprint(wants_sprint, wants_forward)
        if do_jump:
            self._jump()
        if do_attack:
            self.hit_landed = self._try_attack()

    def _process_use_item(self):
        """Context-sensitive USE_ITEM based on selected slot and equipment."""
        slot = self._selected_slot
        if slot == 0:
            # Weapon slot: shield toggle or eat
            if self._has_shield and not self._is_eating:
                self._shield_raised = not self._shield_raised
            elif self.food > 0 and self.health < 20.0 and not self._shield_raised:
                if not self._is_eating:
                    self._is_eating = True
                    self._eat_ticks = 0
        elif slot == 1 and self._has_bow:
            # Bow: start charging
            if not self._is_charging_bow:
                self._is_charging_bow = True
                self._bow_charge_ticks = 0
        elif slot == 2 and self._has_blocks and self._block_count > 0:
            # Block placement
            self._place_block()
        elif slot == 3 and self._has_pearl and self._pearl_count > 0:
            # Throw enderpearl
            self._throw_pearl()

    def _process_use_stop(self):
        """Release action — fires bow or lowers shield."""
        if self._is_charging_bow and self._bow_charge_ticks > 0:
            self._fire_player_arrow()
            self._is_charging_bow = False
            self._bow_charge_ticks = 0
        elif self._shield_raised:
            self._shield_raised = False

    def _cycle_hotbar(self, direction: int):
        """Cycle selected hotbar slot. Only cycles through occupied slots."""
        slots = [0]  # weapon always present
        if self._has_bow:
            slots.append(1)
        if self._has_blocks and self._block_count > 0:
            slots.append(2)
        if self._has_pearl and self._pearl_count > 0:
            slots.append(3)
        if len(slots) <= 1:
            return
        try:
            idx = slots.index(self._selected_slot)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(slots)
        self._selected_slot = slots[idx]
        # Cancel charging/eating on slot switch
        self._is_charging_bow = False
        self._bow_charge_ticks = 0
        self._is_eating = False
        self._eat_ticks = 0

    def _fire_player_arrow(self):
        """Fire a player arrow toward the nearest enemy."""
        charge_frac = min(1.0, self._bow_charge_ticks / 20.0)
        damage = charge_frac * 6.0
        if damage < 0.5:
            return  # too weak, no arrow

        # Aim at nearest enemy
        ni = self._nearest_enemy_info()
        if ni is None:
            # Fire forward
            cos_y = math.cos(self.yaw)
            sin_y = math.sin(self.yaw)
            nx, nz = -sin_y, cos_y
            ny = 0.0
        else:
            tx, tz, td = ni
            dx = tx - self.px
            dz = tz - self.pz
            dy = 0.0
            dist = math.sqrt(dx * dx + dz * dz) or 1.0
            nx = dx / dist
            nz = dz / dist
            # Loft for distance
            flight_time = dist / SKELETON_ARROW_SPEED
            loft = 0.5 * SKELETON_ARROW_GRAVITY * flight_time
            ny = loft

        # Normalize
        mag = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx /= mag
        ny /= mag
        nz /= mag

        arrow = Arrow(
            x=self.px, y=self.py + 1.5, z=self.pz,
            vx=nx * SKELETON_ARROW_SPEED,
            vy=ny * SKELETON_ARROW_SPEED,
            vz=nz * SKELETON_ARROW_SPEED,
            is_player=True,
            damage=damage,
        )
        self.arrows.append(arrow)

    def _place_block(self):
        """Place a block for towering or bridging."""
        if self._block_count <= 0:
            return
        speed = math.hypot(self.vx, self.vz)
        gx = int(self.px)
        gz = int(self.pz)
        if speed < 0.02:
            # Towering: raise terrain at feet by 1.0
            if 0 <= gx < ARENA_INT and 0 <= gz < ARENA_INT:
                self._heights[gx, gz] += 1.0
                self.py = float(self._heights[gx, gz])
                self._fall_start_height = self.py
                self._block_count -= 1
        else:
            # Bridging: place block 1.5 blocks ahead at current Y
            cos_y = math.cos(self.yaw)
            sin_y = math.sin(self.yaw)
            fwd_x = -sin_y * 1.5
            fwd_z = cos_y * 1.5
            bx = int(self.px + fwd_x)
            bz = int(self.pz + fwd_z)
            if 0 <= bx < ARENA_INT and 0 <= bz < ARENA_INT:
                target_h = self.py
                if self._heights[bx, bz] < target_h:
                    self._heights[bx, bz] = target_h
                    self._block_count -= 1

    def _throw_pearl(self):
        """Throw an enderpearl projectile."""
        if self._pearl_count <= 0:
            return
        self._pearl_count -= 1
        # Aim at 30° upward angle in facing direction
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        fwd_x = -sin_y
        fwd_z = cos_y
        speed = SKELETON_ARROW_SPEED * 0.8
        pitch = math.radians(30)
        horiz = math.cos(pitch) * speed
        vert = math.sin(pitch) * speed
        arrow = Arrow(
            x=self.px, y=self.py + 1.5, z=self.pz,
            vx=fwd_x * horiz,
            vy=vert,
            vz=fwd_z * horiz,
            is_pearl=True,
        )
        self.arrows.append(arrow)

    def _nearest_enemy_info(self):
        """Return (x, z, dist) of nearest living enemy, or None."""
        best = None
        best_d = float("inf")
        for i in range(len(self.zx)):
            if self.zh[i] <= 0:
                continue
            d = math.hypot(self.zx[i] - self.px, self.zz[i] - self.pz)
            if d < best_d:
                best_d = d
                best = (self.zx[i], self.zz[i], d)
        for i in range(len(self.sx)):
            if self.sh[i] <= 0:
                continue
            d = math.hypot(self.sx[i] - self.px, self.sz[i] - self.pz)
            if d < best_d:
                best_d = d
                best = (self.sx[i], self.sz[i], d)
        for i in range(len(self.cx)):
            if self.ch[i] <= 0:
                continue
            d = math.hypot(self.cx[i] - self.px, self.cz[i] - self.pz)
            if d < best_d:
                best_d = d
                best = (self.cx[i], self.cz[i], d)
        return best

    # ── Main Step ────────────────────────────────────────────────────

    def step(self, action: int) -> dict:
        """Advance simulation one tick."""
        self.tick += 1
        self.hit_landed = False
        self.kills_this_tick = 0

        # Process player action (sets acceleration + intent)
        if action >= len(PHASE_4_ACTIONS) + 1:
            m, l, c = decode_composite(action)
            self._process_factored_action(m, l, c)
        else:
            self._process_discrete_action(action)

        # Physics tick (gravity, friction, position update)
        self._physics_tick()

        # Food/exhaustion drain
        self._drain_exhaustion()

        # Entity AI
        self._zombie_ai()
        self._skeleton_ai()
        self._creeper_ai()
        self._tick_arrows()

        # Shield disable countdown
        if self._shield_disabled_ticks > 0:
            self._shield_disabled_ticks -= 1
            if self._shield_disabled_ticks <= 0:
                self._shield_raised = False

        # Bow charge tick
        if self._is_charging_bow:
            self._bow_charge_ticks += 1

        # Eating tick
        if self._is_eating:
            self._eat_ticks += 1
            if self._eat_ticks >= EAT_DURATION:
                self.health = min(20.0, self.health + EAT_HEAL)
                self.food = max(0, self.food - 1)
                self._is_eating = False
                self._eat_ticks = 0

        # Natural regeneration (1 HP per 4s when food >= 18)
        if (self.tick % REGEN_INTERVAL == 0 and self.food >= REGEN_MIN_FOOD
                and self.health < 20.0 and self.alive):
            self.health = min(20.0, self.health + 1.0)
            self.exhaustion += REGEN_EXHAUSTION

        # Remove dead zombies, respawn
        dead_z = [i for i in range(len(self.zx)) if self.zh[i] <= 0]
        for i in sorted(dead_z, reverse=True):
            self.zx.pop(i); self.zz.pop(i); self.zy.pop(i)
            self.zvx.pop(i); self.zvz.pop(i)
            self.zh.pop(i); self.z_invuln.pop(i)
            self.z_atk_cd.pop(i); self.z_is_baby.pop(i)
            self.z_fire_ticks.pop(i)
            if i < len(self._prev_zombie_speeds):
                self._prev_zombie_speeds.pop(i)

        # Remove dead skeletons, respawn
        dead_s = [i for i in range(len(self.sx)) if self.sh[i] <= 0]
        for i in sorted(dead_s, reverse=True):
            self.sx.pop(i); self.sz.pop(i); self.sh.pop(i)
            self.svx.pop(i); self.svz.pop(i)
            self.s_fire_cd.pop(i); self.s_invuln.pop(i)
            self.s_fire_ticks.pop(i)

        # Remove dead creepers
        dead_c = [i for i in range(len(self.cx)) if self.ch[i] <= 0]
        for i in sorted(dead_c, reverse=True):
            self.cx.pop(i); self.cz.pop(i); self.cy.pop(i)
            self.cvx.pop(i); self.cvz.pop(i)
            self.ch.pop(i); self.c_fuse.pop(i)
            self.c_charged.pop(i); self.c_invuln.pop(i)
            self.c_fire_ticks.pop(i)

        # Respawn based on episode type
        self._respawn_entities()

        # Track combat momentum
        if self.hit_landed:
            self._recent_hits.append(self.tick)
        health_delta = self.health - self._prev_health
        if health_delta < 0:
            self._recent_damage.append(-health_delta)
        if self.kills_this_tick > 0:
            for _ in range(self.kills_this_tick):
                self._recent_kills.append(self.tick)
        # Prune old entries
        self._recent_hits = [t for t in self._recent_hits if self.tick - t < 200]
        self._recent_damage = self._recent_damage[-20:]
        self._recent_kills = [t for t in self._recent_kills if self.tick - t < 600]

        state = self._build_state()

        # Update prev tracking
        self._prev_health = self.health
        self._prev_food = self.food

        return state

    def _respawn_entities(self):
        """Respawn dead entities to maintain episode composition."""
        if self._episode_type == "zombies":
            while len(self.zx) < self._num_zombies:
                self._spawn_zombie()
        elif self._episode_type == "mixed":
            if len(self.zx) < 1:
                self._spawn_zombie()
            if len(self.sx) < 1:
                self._spawn_skeleton()
        elif self._episode_type == "horde":
            while len(self.zx) < 3:
                self._spawn_zombie()
        elif self._episode_type == "hard":
            if len(self.zx) < 1:
                self._spawn_zombie(force_baby=True)
            if len(self.sx) < 1:
                self._spawn_skeleton()
        elif self._episode_type == "creeper":
            if len(self.cx) < 1:
                self._spawn_creeper()
            if len(self.zx) < 1:
                self._spawn_zombie()

    # ── State Building ───────────────────────────────────────────────

    def _compute_spatial_from_terrain(self) -> dict:
        """Compute spatial data from actual heightmap geometry."""
        player_ground = self._terrain_height_at(self.px, self.pz)

        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        dirs = [
            (-sin_y, cos_y),    # forward
            (sin_y, -cos_y),    # backward
            (cos_y, sin_y),     # left
            (-cos_y, -sin_y),   # right
        ]

        # body_clear: how far can walk before hitting terrain wall
        body_clear = []
        for dx, dz in dirs:
            clear = 3.0
            for s in [1.0, 2.0, 3.0]:
                nx = self.px + dx * s
                nz = self.pz + dz * s
                h = self._terrain_height_at(nx, nz)
                if h - player_ground > JUMP_CLEAR_HEIGHT:
                    clear = s - 1.0
                    break
            body_clear.append(max(0.0, clear))

        # drop_depth: how far ground drops in each direction (normalized 0-1)
        drop_depth = []
        for dx, dz in dirs:
            check_x = self.px + dx * 2
            check_z = self.pz + dz * 2
            check_x = max(0.0, min(ARENA_SIZE - 1, check_x))
            check_z = max(0.0, min(ARENA_SIZE - 1, check_z))
            h = self._terrain_height_at(check_x, check_z)
            drop = max(0.0, player_ground - h)
            drop_depth.append(min(1.0, drop / 5.0))

        # overhead: always open (no ceilings in heightmap terrain)
        overhead = [1.0, 1.0, 1.0, 1.0]

        # composition: local terrain statistics from 7x7 grid
        gx = int(self.px)
        gz = int(self.pz)
        if 0 <= gx < ARENA_INT and 0 <= gz < ARENA_INT:
            x_lo = max(0, gx - 3)
            x_hi = min(ARENA_INT, gx + 4)
            z_lo = max(0, gz - 3)
            z_hi = min(ARENA_INT, gz + 4)
            local_heights = self._heights[x_lo:x_hi, z_lo:z_hi]
            total_cells = local_heights.size
            passable = int(np.sum(np.abs(local_heights - player_ground) <= JUMP_CLEAR_HEIGHT))
            air_ratio = passable / total_cells
            wall_count = int(np.sum(local_heights - player_ground > JUMP_CLEAR_HEIGHT))
            wall_density = wall_count / total_cells
            ground_count = int(np.sum(np.abs(local_heights - player_ground) <= STEP_UP_HEIGHT))
            ground_coverage = ground_count / total_cells
        else:
            # Outside terrain feature area — flat open ground
            air_ratio = 1.0
            wall_density = 0.0
            ground_coverage = 1.0

        # immediate: on_ground, front_foot, water, above_head
        fwd_x = self.px + dirs[0][0]
        fwd_z = self.pz + dirs[0][1]
        fwd_h = self._terrain_height_at(fwd_x, fwd_z)
        front_foot = 1.0 if abs(fwd_h - player_ground) <= STEP_UP_HEIGHT else 0.0

        return {
            "body_clear": body_clear,
            "drop_depth": drop_depth,
            "overhead": overhead,
            "composition": [
                float(max(0.0, min(1.0, air_ratio))),
                float(max(0.0, min(1.0, wall_density))),
                float(max(0.0, min(1.0, ground_coverage))),
                None,  # filled by danger later
            ],
            "immediate": [
                1.0 if self.on_ground else 0.0,
                front_foot,
                1.0 if self._is_in_water(self.px, self.pz) else 0.0,
                0.0,
            ],
        }

    def _build_spatial(self, fl: float, fr: float, bl: float, br: float) -> dict:
        spatial = self._compute_spatial_from_terrain()
        spatial["danger"] = [fl, fr, bl, br]
        spatial["composition"][3] = min(1.0, (fl + fr + bl + br) / 4)
        return spatial

    def _compute_hostile_accel(self, zombie_dists: list) -> float:
        """Compute speed change of nearest hostile."""
        if not zombie_dists:
            return 0.0
        _, i = zombie_dists[0]
        cur_speed = math.hypot(self.zvx[i], self.zvz[i]) * 20
        if i < len(self._prev_zombie_speeds):
            accel = cur_speed - self._prev_zombie_speeds[i]
        else:
            accel = 0.0
        while len(self._prev_zombie_speeds) <= i:
            self._prev_zombie_speeds.append(0.0)
        self._prev_zombie_speeds[i] = cur_speed
        return accel

    def _time_since_last_hit(self) -> float:
        if not self._recent_hits:
            return 10.0
        ticks_since = self.tick - self._recent_hits[-1]
        return min(10.0, ticks_since / 20.0)

    def _compute_strafing(self) -> float:
        """Compute lateral velocity component (perpendicular to facing)."""
        # Right vector
        right_x = math.cos(self.yaw)
        right_z = math.sin(self.yaw)
        return (self.vx * right_x + self.vz * right_z) * 20  # blocks/sec

    def _get_incoming_projectile(self) -> dict | None:
        """Find the most threatening incoming arrow."""
        if not self.arrows:
            return None

        best = None
        best_urgency = 0.0
        for arrow in self.arrows:
            dx = arrow.x - self.px
            dz = arrow.z - self.pz
            dy = arrow.y - (self.py + 1.0)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > 20:
                continue
            # Check if arrow is approaching player
            speed = math.sqrt(arrow.vx ** 2 + arrow.vy ** 2 + arrow.vz ** 2)
            if speed < 0.01 or dist < 0.01:
                continue
            # Approach: dot of velocity with arrow→player direction
            to_player_x = -dx / dist
            to_player_z = -dz / dist
            approach = (arrow.vx * to_player_x + arrow.vz * to_player_z)
            if approach <= 0:
                continue  # moving away
            urgency = math.exp(-dist * dist / 128)
            if urgency > best_urgency:
                # Bearing relative to player facing
                angle_to_arrow = math.atan2(dz, dx)
                rel_angle = angle_to_arrow - self.yaw
                best = {
                    "name": "arrow",
                    "distance": dist,
                    "speed": speed * 20,  # blocks/sec
                    "bearing": {
                        "sin": math.sin(rel_angle),
                        "cos": math.cos(rel_angle),
                    },
                }
                best_urgency = urgency
        return best

    def _build_inventory(self) -> dict:
        """Build dynamic inventory based on current loadout."""
        hotbar: list = [None] * 9
        slots_used = 1
        # Slot 0: weapon (always)
        hotbar[0] = {
            "category": 1, "tier": self._weapon_tier,
            "durability": 1.0, "count": 1, "max_stack": 1,
        }
        # Slot 1: bow
        if self._has_bow:
            hotbar[1] = {
                "category": 1, "tier": 0.3,
                "durability": 1.0, "count": 1, "max_stack": 1,
            }
            slots_used += 1
        # Slot 2: blocks
        if self._has_blocks and self._block_count > 0:
            hotbar[2] = {
                "category": 6, "tier": 0.2,
                "durability": 1.0, "count": self._block_count, "max_stack": 64,
            }
            slots_used += 1
        # Slot 3: enderpearl
        if self._has_pearl and self._pearl_count > 0:
            hotbar[3] = {
                "category": 7, "tier": 0.0,
                "durability": 1.0, "count": self._pearl_count, "max_stack": 16,
            }
            slots_used += 1
        return {
            "slots_used": slots_used,
            "selected_slot": self._selected_slot,
            "hotbar": hotbar,
        }

    def _build_state(self) -> dict:
        """Build state dict compatible with MinecraftStateEncoder."""
        # Sort zombies by distance
        zombie_dists = []
        for i in range(len(self.zx)):
            if self.zh[i] <= 0:
                continue
            d = math.hypot(self.zx[i] - self.px, self.zz[i] - self.pz)
            zombie_dists.append((d, i))
        zombie_dists.sort()

        # Build hostile list (zombies + skeletons merged by distance)
        all_hostiles = []
        for d, i in zombie_dists:
            dx = self.zx[i] - self.px
            dz = self.zz[i] - self.pz
            abs_angle = math.atan2(dz, dx)
            rel_angle = abs_angle - self.yaw
            speed = math.hypot(self.zvx[i], self.zvz[i]) * 20
            if d > 0.01:
                to_player_x = (self.px - self.zx[i]) / d
                to_player_z = (self.pz - self.zz[i]) / d
                approach = (self.zvx[i] * 20 * to_player_x + self.zvz[i] * 20 * to_player_z)
            else:
                approach = 0.0

            all_hostiles.append({
                "name": "zombie",
                "distance": d,
                "bearing": {"sin": math.sin(rel_angle), "cos": math.cos(rel_angle)},
                "speed": speed,
                "approach": approach,
                "facing_us": 1.0,
                "health": self.zh[i],
                "max_health": 20,
                "flags": 0,
                "hand_state": 0,
                "is_baby": self.z_is_baby[i],
                "creeper_state": -1,
                "creeper_charged": False,
                "_sort_dist": d,
            })

        for i in range(len(self.sx)):
            if self.sh[i] <= 0:
                continue
            d = math.hypot(self.sx[i] - self.px, self.sz[i] - self.pz)
            dx = self.sx[i] - self.px
            dz = self.sz[i] - self.pz
            abs_angle = math.atan2(dz, dx)
            rel_angle = abs_angle - self.yaw
            speed = math.hypot(self.svx[i], self.svz[i]) * 20
            if d > 0.01:
                to_player_x = (self.px - self.sx[i]) / d
                to_player_z = (self.pz - self.sz[i]) / d
                approach = (self.svx[i] * 20 * to_player_x + self.svz[i] * 20 * to_player_z)
            else:
                approach = 0.0

            all_hostiles.append({
                "name": "skeleton",
                "distance": d,
                "bearing": {"sin": math.sin(rel_angle), "cos": math.cos(rel_angle)},
                "speed": speed,
                "approach": approach,
                "facing_us": 1.0,
                "health": self.sh[i],
                "max_health": 20,
                "flags": 0,
                "hand_state": 0,
                "is_baby": False,
                "creeper_state": -1,
                "creeper_charged": False,
                "_sort_dist": d,
            })

        # Add creepers
        for i in range(len(self.cx)):
            if self.ch[i] <= 0:
                continue
            d = math.hypot(self.cx[i] - self.px, self.cz[i] - self.pz)
            dx = self.cx[i] - self.px
            dz = self.cz[i] - self.pz
            abs_angle = math.atan2(dz, dx)
            rel_angle = abs_angle - self.yaw
            speed = math.hypot(self.cvx[i], self.cvz[i]) * 20
            if d > 0.01:
                to_player_x = (self.px - self.cx[i]) / d
                to_player_z = (self.pz - self.cz[i]) / d
                approach = (self.cvx[i] * 20 * to_player_x + self.cvz[i] * 20 * to_player_z)
            else:
                approach = 0.0

            all_hostiles.append({
                "name": "creeper",
                "distance": d,
                "bearing": {"sin": math.sin(rel_angle), "cos": math.cos(rel_angle)},
                "speed": speed,
                "approach": approach,
                "facing_us": 1.0,
                "health": self.ch[i],
                "max_health": 20,
                "flags": 0,
                "hand_state": 0,
                "is_baby": False,
                "creeper_state": self.c_fuse[i],
                "creeper_charged": self.c_charged[i],
                "_sort_dist": d,
            })

        # Sort all hostiles by distance
        all_hostiles.sort(key=lambda h: h["_sort_dist"])
        for h in all_hostiles:
            del h["_sort_dist"]

        # Danger quadrants (from all hostiles)
        cos_neg = math.cos(-self.yaw)
        sin_neg = math.sin(-self.yaw)
        fl = fr = bl = br = 0.0
        qd_fl = qd_fr = qd_bl = qd_br = 0.0
        threat_sum_x = 0.0
        threat_sum_z = 0.0

        for h in all_hostiles:
            d = h["distance"]
            if d > 16:
                continue
            # Get raw position from bearing
            angle = self.yaw + math.atan2(
                h["bearing"]["sin"], h["bearing"]["cos"]
            )
            dx = d * math.cos(angle)
            dz = d * math.sin(angle)
            rx = dx * cos_neg - dz * sin_neg
            rz = dx * sin_neg + dz * cos_neg
            w = max(0.0, 1.0 - d / 16)
            prox_w = 1.0 / max(1.0, d * d)
            if rx >= 0:
                if rz >= 0:
                    fr += w; qd_fr += prox_w
                else:
                    fl += w; qd_fl += prox_w
            else:
                if rz >= 0:
                    br += w; qd_br += prox_w
                else:
                    bl += w; qd_bl += prox_w
            threat_sum_x += dx * prox_w
            threat_sum_z += dz * prox_w

        # Threat direction
        threat_mag = math.hypot(threat_sum_x, threat_sum_z)
        if threat_mag > 0.01:
            threat_abs_angle = math.atan2(threat_sum_z, threat_sum_x)
            threat_rel = threat_abs_angle - self.yaw
            threat_sin = math.sin(threat_rel)
            threat_cos = math.cos(threat_rel)
            threat_magnitude = min(1.0, threat_mag)
        else:
            threat_sin = 0.0
            threat_cos = 0.0
            threat_magnitude = 0.0

        # Nearest hostile for height_vs_hostile
        nearest_hostile_d = all_hostiles[0]["distance"] if all_hostiles else 64.0
        # Use real terrain-based height difference
        if zombie_dists:
            nearest_i = zombie_dists[0][1]
            height_vs_hostile = self.py - self.zy[nearest_i]
        elif all_hostiles:
            # Skeleton (no zy, use terrain at skeleton pos)
            for i in range(len(self.sx)):
                if self.sh[i] > 0:
                    height_vs_hostile = self.py - self._terrain_height_at(self.sx[i], self.sz[i])
                    break
            else:
                height_vs_hostile = 0.0
        else:
            height_vs_hostile = 0.0

        # Health/food deltas
        health_delta = self.health - self._prev_health
        food_delta = self.food - self._prev_food

        return {
            "health": self.health,
            "food": self.food,
            "alive": self.alive,
            "time_of_day": 18000,
            "light_level": 0,
            "position": {"x": self.px, "y": self.py + GROUND_Y, "z": self.pz},
            "yaw": math.degrees(self.yaw),
            "pitch": 0.0,
            "on_ground": self.on_ground,
            "is_in_water": self._is_in_water(self.px, self.pz),
            "is_raining": False,
            "altitude": self.py + GROUND_Y,
            "spatial": self._build_spatial(fl, fr, bl, br),
            "entities": {
                "hostiles": all_hostiles[:8],
                "passives": [],
                "players": [],
            },
            "crowd": {
                "quadrant_density": [
                    min(1.0, qd_fl), min(1.0, qd_fr),
                    min(1.0, qd_bl), min(1.0, qd_br),
                ],
                "hostile_count": len(all_hostiles),
                "hostile_avg_dist": (
                    sum(h["distance"] for h in all_hostiles) / len(all_hostiles)
                    if all_hostiles else 64.0
                ),
                "hostile_near": sum(1 for h in all_hostiles if h["distance"] <= 8),
                "passive_count": 0,
                "player_count": 0,
                "threat_direction": {
                    "sin": threat_sin,
                    "cos": threat_cos,
                    "magnitude": threat_magnitude,
                },
                "attacker_dist": nearest_hostile_d,
                "attacker_bearing": (
                    all_hostiles[0]["bearing"] if all_hostiles
                    else {"sin": 0.0, "cos": 0.0}
                ),
                "under_attack": 1.0 if health_delta < 0 else 0.0,
            },
            "inventory": self._build_inventory(),
            "xp_level": 0,
            "xp_points": 0,
            "attack_cooldown": max(0, MIN_ATTACK_INTERVAL - (self.tick - self.last_attack_tick)),
            "hit_landed": self.hit_landed,
            "player_hit_landed": False,
            "kills": self.kills_this_tick,
            # Self-awareness + threat dynamics
            "self_velocity": {
                "x": self.vx * 20,
                "y": self.vy * 20,
                "z": self.vz * 20,
            },
            "health_delta": health_delta,
            "food_delta": float(food_delta),
            "food_saturation": self.saturation,
            "ticks_airborne": self.ticks_airborne,
            "self_effects": {
                "speed": self._effect_speed,
                "strength": self._effect_strength,
                "resistance": self._effect_resistance,
                "regeneration": 0,
            },
            "incoming_projectile": self._get_incoming_projectile(),
            "self_armor_tier": self._armor_tier,
            "is_thundering": False,
            "nearest_hostile_accel": self._compute_hostile_accel(zombie_dists),
            "nearest_player_armor": 0.0,
            "height_vs_hostile": height_vs_hostile,
            "height_vs_player": 0.0,
            "combat_hits_5s": len([t for t in self._recent_hits if self.tick - t < 100]),
            "combat_damage_5s": sum(self._recent_damage),
            "time_since_hit": self._time_since_last_hit(),
            "kill_streak": len([t for t in self._recent_kills if self.tick - t < 600]),
            "strafing": self._compute_strafing(),
        }


# ─── Training loop ──────────────────────────────────────────────────

def run_fast_train(
    steps: int = 500_000,
    phase: int = 3,
    seed: int = 42,
    save_path: str = "pretrained.pkl",
    num_zombies: int = ZOMBIE_COUNT,
    report_interval: int = 1000,
    factored: bool = False,
) -> None:
    arena = CombatArena(num_zombies=num_zombies, seed=seed)
    encoder = MinecraftStateEncoder()
    history = HistoryTrace(decay=0.6, output_dims=16, input_dims=412, seed=seed)

    if factored:
        action_space = FACTORED_ACTIONS
    elif phase >= 4:
        action_space = PHASE_4_ACTIONS
    else:
        action_space = PHASE_3_ACTIONS

    agent = Agent(
        similarity_threshold=0.80,
        seed=seed,
        exploration_base=0.12,
        enable_compositional=True,
        patterns_per_modality=16,
        modality_slices=MinecraftEnv.MODALITY_SLICES,
        modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
        composite_similarity_threshold=0.5,
        max_associations=5000,
        association_decay_rate=0.002,
        maintenance_cost_per_pattern=0.00025,
        maintenance_cost_per_association=0.0001,
        enable_salience=True,
        enable_options=True,
        enable_episodic_memory=True,
        episodic_capacity=500,
        episodic_surprise_threshold=0.4,
        enable_lookahead=True,
        lookahead_depth=2,
        lookahead_discount=0.9,
        enable_eligibility_traces=True,
        trace_decay=0.8,
        discount_factor=0.95,
        enable_abstraction=True,
    )
    agent.vitality = Vitality(entropy_rate=0.001)

    def encode_state(state: dict, ts: int) -> Signal:
        base = encoder.encode(state, timestamp=ts)
        vision = np.zeros(16, dtype=np.float64)
        bv = np.concatenate([base.data, vision])  # 396 + 16 = 412
        h = history.update(bv)
        combined = np.concatenate([bv, h])  # 412 + 16 = 428
        vn = np.linalg.norm(combined[396:412])
        if vn > 0:
            combined[396:412] /= vn
        hn = np.linalg.norm(combined[412:428])
        if hn > 0:
            combined[412:428] /= hn
        return Signal(data=combined, timestamp=ts, modality="minecraft")

    initial_state = arena._build_state()
    prev_state = initial_state
    obs = encode_state(initial_state, 0)
    agent.step_with_action(obs, 0.0, None)

    idle_ticks = 0
    t0 = time.time()
    print(
        f"[fast_train] {steps:,} steps | phase {phase} | "
        f"{len(action_space)} actions | {num_zombies} zombies | "
        f"3D physics + facing + crits + KB + food"
    )

    for step in range(1, steps + 1):
        if not agent.vitality.alive:
            agent.vitality = Vitality(entropy_rate=0.001)

        if not arena.alive:
            arena.reset()
            history.reset()
            idle_ticks = 0

        action = agent.select_action(action_space)
        state = arena.step(action)

        if state.get("hit_landed"):
            idle_ticks = 0
        else:
            idle_ticks += 1
        state["_idle_ticks"] = idle_ticks

        energy_delta = compute_energy_delta(prev_state, state)
        prev_state = state

        obs = encode_state(state, step)
        result = agent.step_with_action(obs, energy_delta, action)
        agent.consolidate()

        if step % report_interval == 0:
            elapsed = time.time() - t0
            sps = step / elapsed
            patterns = len(agent.world_model.memory.distinction.patterns)
            assocs = agent.world_model.memory.association_count
            pos_v = sum(
                1 for pid in agent.valence._values
                if agent.valence.get(pid) > 0.01
            )
            neg_v = sum(
                1 for pid in agent.valence._values
                if agent.valence.get(pid) < -0.01
            )
            kd = arena.total_kills / max(arena.total_deaths, 1)
            print(
                f"[{step:>8,}] "
                f"{sps:>6,.0f} sps  "
                f"v={result.vitality:.3f}  "
                f"s={result.surprise:.2f}  "
                f"pat={patterns}  "
                f"asc={assocs}  "
                f"val=+{pos_v}/-{neg_v}  "
                f"K={arena.total_kills} D={arena.total_deaths} "
                f"kd={kd:.1f}  "
                f"food={arena.food}"
            )

    elapsed = time.time() - t0
    kd = arena.total_kills / max(arena.total_deaths, 1)
    print(f"\n[fast_train] Done: {steps:,} steps in {elapsed:.1f}s "
          f"({steps / elapsed:,.0f} sps)")
    print(f"  K/D: {arena.total_kills}/{arena.total_deaths} = {kd:.2f}")
    print(f"  Patterns: {len(agent.world_model.memory.distinction.patterns)}")
    print(f"  Associations: {agent.world_model.memory.association_count}")

    options_list = []
    if agent._option_executor is not None:
        options_list = agent._option_executor._options
    saved = {
        "world_model": agent.world_model,
        "valence": agent.valence,
        "options": options_list,
    }
    with open(save_path, "wb") as f:
        pickle.dump(saved, f)
    print(f"[fast_train] Saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast offline combat pre-training for FPI agent (3D physics)",
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Training steps (default: 500K)",
    )
    parser.add_argument(
        "--phase", type=int, default=3, choices=[3, 4],
        help="Action phase (3=combat combos, 4=+macros)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save", type=str, default="pretrained.pkl",
        help="Output pickle path",
    )
    parser.add_argument(
        "--zombies", type=int, default=2,
        help="Number of simultaneous zombies",
    )
    parser.add_argument(
        "--report", type=int, default=1000,
        help="Report every N steps",
    )
    parser.add_argument(
        "--factored", action="store_true",
        help="Use factored action space (294 composite actions)",
    )
    args = parser.parse_args()

    try:
        run_fast_train(
            steps=args.steps,
            phase=args.phase,
            seed=args.seed,
            save_path=args.save,
            num_zombies=args.zombies,
            report_interval=args.report,
            factored=args.factored,
        )
    except KeyboardInterrupt:
        print("\n[fast_train] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
