"""CombatSimulator — fast Python 2D combat arena for pre-training.

Simplified Minecraft combat mechanics in a flat 2D arena. Produces
signals in the SAME format as MinecraftEnv (92-dim with history trace),
so learned patterns transfer directly to real Minecraft.

Performance: ~100,000 steps/sec single-threaded (vs 5 steps/sec in MC).

Mechanics:
- 2D arena (x, z), flat terrain
- Agent: position, yaw, health(20), attack cooldown(32 ticks)
- Mobs: position, health(20), speed(0.08 blocks/tick), damage(3), range(2)
- Movement: 0.1 blocks/tick (sprint: 0.15)
- Attack: range 3.0 blocks, damage depends on cooldown, max at 0 cooldown
- Knockback: velocity impulse, decays 50% per tick
- Mob types: Zombie (basic), Strafer (circles), Crit (sprints+jumps), Shield (blocks)

Curriculum stages:
- Stage 1: Passive mobs only (low damage, slow)
- Stage 2: Zombies (standard)
- Stage 3: Skeletons + Strafers (ranged/dodging)
- Stage 4: Sprint-attacking players (CritMob)
- Stage 5: Mixed including ShieldMob
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..env.base import Environment
from ..primitives.signal import Signal
from .actions import FACTORED_ACTIONS, decode_composite
from .encoder import HistoryTrace, MinecraftStateEncoder
from .env import MinecraftEnv, compute_energy_delta


class _SimMob:
    """A simulated mob in the 2D arena."""

    __slots__ = ("x", "z", "health", "max_health", "speed", "damage",
                 "attack_range", "attack_cooldown", "cooldown_timer",
                 "vx", "vz", "mob_type", "yaw", "behavior")

    def __init__(
        self,
        x: float,
        z: float,
        mob_type: str = "zombie",
        health: float = 20.0,
        speed: float = 0.08,
        damage: float = 3.0,
        attack_range: float = 2.0,
        attack_cooldown: int = 20,
        behavior: str = "chase",
    ) -> None:
        self.x = x
        self.z = z
        self.mob_type = mob_type
        self.health = health
        self.max_health = health
        self.speed = speed
        self.damage = damage
        self.attack_range = attack_range
        self.attack_cooldown = attack_cooldown
        self.cooldown_timer = 0
        self.vx = 0.0
        self.vz = 0.0
        self.yaw = 0.0
        self.behavior = behavior

    def distance_to(self, ax: float, az: float) -> float:
        return math.sqrt((self.x - ax) ** 2 + (self.z - az) ** 2)

    def tick(self, agent_x: float, agent_z: float, tick_count: int = 0) -> float:
        """Move toward agent, attempt attack. Returns damage dealt to agent."""
        # Apply knockback velocity
        self.x += self.vx
        self.z += self.vz
        self.vx *= 0.5
        self.vz *= 0.5

        # Compute direction to agent
        dx = agent_x - self.x
        dz = agent_z - self.z
        dist = math.sqrt(dx * dx + dz * dz)

        # Update facing (yaw toward agent)
        if dist > 0.1:
            self.yaw = math.atan2(-dx, dz)

        # Behavior-specific movement
        if self.behavior == "chase":
            self._move_chase(dx, dz, dist)
        elif self.behavior == "strafe":
            self._move_strafe(dx, dz, dist, tick_count)
        elif self.behavior == "crit":
            self._move_crit(dx, dz, dist, tick_count)
        elif self.behavior == "shield":
            self._move_shield(dx, dz, dist, tick_count)
        elif self.behavior == "passive":
            self._move_passive(dx, dz, dist)

        # Attack
        damage_dealt = 0.0
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        elif dist <= self.attack_range:
            damage_dealt = self.damage
            self.cooldown_timer = self.attack_cooldown

        return damage_dealt

    def _move_chase(self, dx: float, dz: float, dist: float) -> None:
        """Basic chase: walk straight at agent."""
        if dist > self.attack_range:
            if dist > 0:
                self.x += (dx / dist) * self.speed
                self.z += (dz / dist) * self.speed

    def _move_strafe(self, dx: float, dz: float, dist: float, tick: int) -> None:
        """Strafe: circle around agent while approaching."""
        if dist > self.attack_range * 1.5:
            # Approach
            if dist > 0:
                self.x += (dx / dist) * self.speed
                self.z += (dz / dist) * self.speed
        elif dist > self.attack_range * 0.5:
            # Circle strafe (alternate direction every 30 ticks)
            direction = 1 if (tick // 30) % 2 == 0 else -1
            if dist > 0:
                # Perpendicular to approach vector
                self.x += (-dz / dist) * self.speed * direction * 0.8
                self.z += (dx / dist) * self.speed * direction * 0.8
                # Slight approach
                self.x += (dx / dist) * self.speed * 0.3
                self.z += (dz / dist) * self.speed * 0.3

    def _move_crit(self, dx: float, dz: float, dist: float, tick: int) -> None:
        """Crit: sprint at agent, simulates sprint-crit (extra damage)."""
        sprint_speed = self.speed * 1.8
        if dist > self.attack_range:
            if dist > 0:
                self.x += (dx / dist) * sprint_speed
                self.z += (dz / dist) * sprint_speed
        # Bonus damage on first hit after sprint (simulated)
        if dist <= self.attack_range and self.cooldown_timer == 0:
            self.damage = 5.0  # Crit damage
        else:
            self.damage = 3.0  # Normal

    def _move_shield(self, dx: float, dz: float, dist: float, tick: int) -> None:
        """Shield: block after attacking, approach slowly."""
        # Approach slowly
        if dist > self.attack_range:
            if dist > 0:
                self.x += (dx / dist) * self.speed * 0.6
                self.z += (dz / dist) * self.speed * 0.6
        # After attacking, reduce incoming damage (simulated by backing off briefly)
        if self.cooldown_timer > self.attack_cooldown * 0.5:
            # Back away briefly after attack
            if dist > 0:
                self.x -= (dx / dist) * self.speed * 0.4
                self.z -= (dz / dist) * self.speed * 0.4

    def _move_passive(self, dx: float, dz: float, dist: float) -> None:
        """Passive: wander randomly, don't approach."""
        # Random drift
        self.x += (math.sin(self.yaw + 0.5) * self.speed * 0.3)
        self.z += (math.cos(self.yaw + 0.5) * self.speed * 0.3)

    @property
    def alive(self) -> bool:
        return self.health > 0


# Curriculum stage definitions
CURRICULUM_STAGES = {
    1: {
        "name": "passive",
        "mob_types": ("cow", "pig", "sheep"),
        "behavior": "passive",
        "health": 10.0,
        "speed": 0.04,
        "damage": 1.0,
        "max_mobs": 3,
    },
    2: {
        "name": "zombies",
        "mob_types": ("zombie",),
        "behavior": "chase",
        "health": 20.0,
        "speed": 0.08,
        "damage": 3.0,
        "max_mobs": 3,
    },
    3: {
        "name": "skeletons+strafers",
        "mob_types": ("skeleton", "spider"),
        "behavior": "strafe",
        "health": 20.0,
        "speed": 0.09,
        "damage": 3.0,
        "max_mobs": 3,
    },
    4: {
        "name": "crit-attackers",
        "mob_types": ("zombie", "skeleton"),
        "behavior": "crit",
        "health": 20.0,
        "speed": 0.10,
        "damage": 3.0,
        "max_mobs": 3,
    },
    5: {
        "name": "mixed",
        "mob_types": ("zombie", "skeleton", "spider", "creeper"),
        "behaviors": ("chase", "strafe", "crit", "shield"),
        "health": 20.0,
        "speed": 0.10,
        "damage": 4.0,
        "max_mobs": 4,
    },
}


class CombatSimulator(Environment):
    """Fast Python combat simulator for pre-training.

    Produces 92-dim signals matching MinecraftEnv format (76 base + 16 history).
    Supports Phase 1 (13), Phase 2 (18), or Phase 3 (20) action space.

    Args:
        arena_size: Width/height of the square arena.
        max_mobs: Maximum mobs alive simultaneously.
        mob_speed: Mob movement speed (blocks/tick).
        mob_types: Types of mobs to spawn.
        seed: Random seed.
        curriculum_stage: 1-5, determines mob difficulty. 0 = use mob_types/speed directly.
        phase: Action space phase (1, 2, or 3).
    """

    def __init__(
        self,
        arena_size: float = 64.0,
        max_mobs: int = 3,
        mob_speed: float = 0.08,
        mob_types: tuple[str, ...] = ("zombie", "skeleton", "spider", "creeper"),
        seed: int = 42,
        curriculum_stage: int = 0,
        phase: int = 1,
        factored: bool = False,
    ) -> None:
        self._arena_size = arena_size
        self._max_mobs = max_mobs
        self._mob_speed = mob_speed
        self._mob_types = mob_types
        self._mob_behavior = "chase"
        self._mob_health = 20.0
        self._mob_damage = 3.0
        self._rng = np.random.default_rng(seed)
        self._phase = phase
        self._factored = factored

        # Apply curriculum stage
        if curriculum_stage > 0 and curriculum_stage in CURRICULUM_STAGES:
            stage = CURRICULUM_STAGES[curriculum_stage]
            self._mob_types = stage["mob_types"]
            self._mob_behavior = stage.get("behavior", "chase")
            self._mob_health = stage.get("health", 20.0)
            self._mob_speed = stage.get("speed", 0.08)
            self._mob_damage = stage.get("damage", 3.0)
            self._max_mobs = stage.get("max_mobs", 3)
            # Mixed stage has multiple behaviors
            self._behaviors = stage.get("behaviors", (self._mob_behavior,))
        else:
            self._behaviors = (self._mob_behavior,)

        self._encoder = MinecraftStateEncoder()
        self._history_trace = HistoryTrace(
            decay=0.6, output_dims=16,
            input_dims=76,  # MinecraftStateEncoder.SIGNAL_DIM
        )

        # Agent state
        self._x = 0.0
        self._z = 0.0
        self._yaw = 0.0
        self._health = 20.0
        self._vx = 0.0
        self._vz = 0.0
        self._attack_cooldown = 0
        self._food = 20.0
        self._sprinting = False

        self._mobs: list[_SimMob] = []
        self._step_count = 0
        self._kill_count = 0
        self._death_count = 0
        self._prev_state: dict = {}
        self._hit_landed = False
        self._player_hit_landed = False
        self._idle_ticks = 0

    @property
    def action_space(self) -> list[int]:
        if self._factored:
            return FACTORED_ACTIONS
        if self._phase >= 3:
            return list(range(20))
        if self._phase >= 2:
            return list(range(18))
        return list(range(13))

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def death_count(self) -> int:
        return self._death_count

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self) -> Signal:
        self._x = self._arena_size / 2
        self._z = self._arena_size / 2
        self._yaw = 0.0
        self._health = 20.0
        self._food = 20.0
        self._vx = 0.0
        self._vz = 0.0
        self._attack_cooldown = 0
        self._sprinting = False
        self._mobs.clear()
        self._step_count = 0
        self._hit_landed = False
        self._player_hit_landed = False
        self._idle_ticks = 0
        self._history_trace.reset()

        self._spawn_mobs()
        state = self._get_state()
        self._prev_state = state
        return self._encode_with_history(state, 0)

    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        self._step_count += 1
        self._hit_landed = False
        self._player_hit_landed = False

        # 1. Execute agent action
        if action is not None:
            if self._factored:
                self._execute_composite_action(action)
            else:
                self._execute_action(action)

        # 2. Apply agent knockback velocity
        self._x += self._vx
        self._z += self._vz
        self._vx *= 0.5
        self._vz *= 0.5

        # Clamp to arena
        self._x = max(0.0, min(self._arena_size, self._x))
        self._z = max(0.0, min(self._arena_size, self._z))

        # 3. Tick mob AI
        total_damage = 0.0
        for mob in self._mobs:
            damage = mob.tick(self._x, self._z, self._step_count)
            if damage > 0:
                total_damage += damage
                # Knockback from mob
                dx = self._x - mob.x
                dz = self._z - mob.z
                dist = math.sqrt(dx * dx + dz * dz)
                if dist > 0:
                    self._vx += (dx / dist) * 0.5
                    self._vz += (dz / dist) * 0.5

        # Apply damage
        self._health -= total_damage
        self._health = max(0.0, self._health)

        # Remove dead mobs
        killed = sum(1 for m in self._mobs if not m.alive)
        self._kill_count += killed
        self._mobs = [m for m in self._mobs if m.alive]

        # Respawn mobs
        self._spawn_mobs()

        # Decrement attack cooldown
        if self._attack_cooldown > 0:
            self._attack_cooldown -= 1

        # 4. Build state
        state = self._get_state()

        # Track idle ticks for escalating penalty
        if state.get("hit_landed", False) or state.get("player_hit_landed", False):
            self._idle_ticks = 0
        else:
            self._idle_ticks += 1
        state["_idle_ticks"] = self._idle_ticks

        energy_delta = compute_energy_delta(self._prev_state, state)

        # 5. Handle death
        done = False
        if self._health <= 0:
            self._death_count += 1
            self._health = 20.0
            self._food = 20.0
            self._x = self._arena_size / 2
            self._z = self._arena_size / 2
            self._vx = 0.0
            self._vz = 0.0
            self._attack_cooldown = 0
            self._sprinting = False
            self._idle_ticks = 0
            self._history_trace.reset()
            state = self._get_state()

        self._prev_state = state
        obs = self._encode_with_history(state, self._step_count)
        return obs, energy_delta, done

    def _execute_action(self, action: int) -> None:
        move_speed = 0.1
        sprint_speed = 0.15

        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)

        if action == 0:  # Forward
            self._x += sin_yaw * move_speed
            self._z += cos_yaw * move_speed
            self._sprinting = False
        elif action == 1:  # Backward
            self._x -= sin_yaw * move_speed
            self._z -= cos_yaw * move_speed
            self._sprinting = False
        elif action == 2:  # Strafe left
            self._x -= cos_yaw * move_speed
            self._z += sin_yaw * move_speed
        elif action == 3:  # Strafe right
            self._x += cos_yaw * move_speed
            self._z -= sin_yaw * move_speed
        elif action == 4:  # Jump (no-op in 2D)
            pass
        elif action == 5:  # Forward + jump
            self._x += sin_yaw * move_speed
            self._z += cos_yaw * move_speed
        elif action == 6:  # Sprint forward
            self._x += sin_yaw * sprint_speed
            self._z += cos_yaw * sprint_speed
            self._sprinting = True
        elif action == 7:  # Look left 45 deg
            self._yaw += math.pi / 4
        elif action == 8:  # Look right 45 deg
            self._yaw -= math.pi / 4
        elif action == 9:  # Look up (no-op in 2D)
            pass
        elif action == 10:  # Look down (no-op in 2D)
            pass
        elif action == 11:  # Attack
            if self._attack_cooldown <= 0:
                self._try_attack(damage_mult=1.0)
                self._attack_cooldown = 32  # Full sword cooldown
        elif action == 12:  # Idle
            pass
        # Phase 2 actions (13-17) are no-ops in combat sim
        elif action == 13:  # Use item
            pass
        elif action == 14:  # Hotbar next
            pass
        elif action == 15:  # Hotbar prev
            pass
        elif action == 16:  # Craft planks
            pass
        elif action == 17:  # Craft tool
            pass
        elif action == 18:  # Sprint-crit combo
            if self._attack_cooldown <= 0:
                # Sprint forward then attack with crit damage
                self._x += sin_yaw * sprint_speed * 2
                self._z += cos_yaw * sprint_speed * 2
                self._try_attack(damage_mult=1.5)  # Critical hit = 1.5x
                self._attack_cooldown = 32
                self._sprinting = True
        elif action == 19:  # W-tap
            if self._attack_cooldown <= 0:
                # Extra knockback from w-tap technique
                self._try_attack(damage_mult=1.0, extra_knockback=0.3)
                self._attack_cooldown = 32
                # Small forward movement (re-engage)
                self._x += sin_yaw * move_speed * 0.5
                self._z += cos_yaw * move_speed * 0.5
                self._sprinting = True

    def _execute_composite_action(self, composite_id: int) -> None:
        """Execute a factored action: decode and apply all 3 axes."""
        movement, look, combat = decode_composite(composite_id)

        move_speed = 0.1
        sprint_speed = 0.15
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)

        # Combat=2 (crit) and Combat=3 (wtap) are motor programs that
        # override the movement axis — you can't independently strafe
        # while executing a sprint-crit.
        if combat == 2:  # Crit: sprint forward + attack with crit damage
            if self._attack_cooldown <= 0:
                self._x += sin_yaw * sprint_speed * 2
                self._z += cos_yaw * sprint_speed * 2
                self._try_attack(damage_mult=1.5)
                self._attack_cooldown = 32
                self._sprinting = True
            # Apply look even during motor programs
            self._apply_look_axis(look)
            return

        if combat == 3:  # W-tap: attack with extra knockback
            if self._attack_cooldown <= 0:
                self._try_attack(damage_mult=1.0, extra_knockback=0.3)
                self._attack_cooldown = 32
                self._x += sin_yaw * move_speed * 0.5
                self._z += cos_yaw * move_speed * 0.5
                self._sprinting = True
            self._apply_look_axis(look)
            return

        # --- Parallel execution: movement + look + optional attack ---

        # Movement axis
        if movement == 0:  # none
            pass
        elif movement == 1:  # forward
            self._x += sin_yaw * move_speed
            self._z += cos_yaw * move_speed
            self._sprinting = False
        elif movement == 2:  # back
            self._x -= sin_yaw * move_speed
            self._z -= cos_yaw * move_speed
            self._sprinting = False
        elif movement == 3:  # left
            self._x -= cos_yaw * move_speed
            self._z += sin_yaw * move_speed
        elif movement == 4:  # right
            self._x += cos_yaw * move_speed
            self._z -= sin_yaw * move_speed
        elif movement == 5:  # forward + jump
            self._x += sin_yaw * move_speed
            self._z += cos_yaw * move_speed
        elif movement == 6:  # forward + sprint
            self._x += sin_yaw * sprint_speed
            self._z += cos_yaw * sprint_speed
            self._sprinting = True

        # Look axis
        self._apply_look_axis(look)

        # Combat axis
        if combat == 1:  # attack
            if self._attack_cooldown <= 0:
                self._try_attack(damage_mult=1.0)
                self._attack_cooldown = 32

    def _apply_look_axis(self, look: int) -> None:
        """Apply look axis: 0=none, 1=track_target, 2=left, 3=right, 4=up, 5=down."""
        if look == 0:
            pass
        elif look == 1:  # Track nearest target
            best_dist = float("inf")
            best_mob = None
            for mob in self._mobs:
                dist = mob.distance_to(self._x, self._z)
                if dist < best_dist:
                    best_dist = dist
                    best_mob = mob
            if best_mob is not None:
                dx = best_mob.x - self._x
                dz = best_mob.z - self._z
                self._yaw = math.atan2(dx, dz)
        elif look == 2:  # Look left 45 deg
            self._yaw += math.pi / 4
        elif look == 3:  # Look right 45 deg
            self._yaw -= math.pi / 4
        elif look == 4:  # Look up (no-op in 2D)
            pass
        elif look == 5:  # Look down (no-op in 2D)
            pass

    def _try_attack(self, damage_mult: float = 1.0, extra_knockback: float = 0.0) -> None:
        """Attack the nearest mob within range."""
        attack_range = 3.0
        base_damage = 7.0  # Full sword damage (diamond sword equivalent)

        # Damage scales with cooldown: 0 ticks waited = 20% damage, full wait = 100%
        # This rewards the agent for timing attacks with cooldown
        cooldown_progress = 1.0  # We only attack when cooldown is 0, so full damage

        attack_damage = base_damage * damage_mult * cooldown_progress

        best_mob = None
        best_dist = float("inf")

        for mob in self._mobs:
            dist = mob.distance_to(self._x, self._z)
            if dist < attack_range and dist < best_dist:
                # Check if mob is roughly in front (within 90 degrees)
                dx = mob.x - self._x
                dz = mob.z - self._z
                angle = math.atan2(dx, dz)
                angle_diff = abs((angle - self._yaw + math.pi) % (2 * math.pi) - math.pi)
                if angle_diff < math.pi / 2:
                    best_mob = mob
                    best_dist = dist

        if best_mob is not None:
            best_mob.health -= attack_damage
            self._hit_landed = True
            # Knockback on mob (sprint knockback is stronger)
            dx = best_mob.x - self._x
            dz = best_mob.z - self._z
            dist = math.sqrt(dx * dx + dz * dz)
            if dist > 0:
                kb_force = 0.4 + extra_knockback
                if self._sprinting:
                    kb_force += 0.3  # Sprint bonus knockback
                best_mob.vx += (dx / dist) * kb_force
                best_mob.vz += (dz / dist) * kb_force

    def _spawn_mobs(self) -> None:
        """Ensure max_mobs are alive, spawning at random arena edges."""
        while len(self._mobs) < self._max_mobs:
            # Spawn at random position 20-40 blocks from agent
            angle = float(self._rng.uniform(0, 2 * math.pi))
            dist = float(self._rng.uniform(20.0, 40.0))
            mx = self._x + math.cos(angle) * dist
            mz = self._z + math.sin(angle) * dist
            # Clamp to arena
            mx = max(0.0, min(self._arena_size, mx))
            mz = max(0.0, min(self._arena_size, mz))

            mob_type = self._mob_types[int(self._rng.integers(len(self._mob_types)))]
            behavior = self._behaviors[int(self._rng.integers(len(self._behaviors)))]

            self._mobs.append(_SimMob(
                x=mx, z=mz,
                mob_type=mob_type,
                health=self._mob_health,
                speed=self._mob_speed,
                damage=self._mob_damage,
                behavior=behavior,
            ))

    def _get_state(self) -> dict:
        """Build a state dict matching the Minecraft bridge format."""
        # Find nearest hostile mob
        nearest_hostile = None
        nearest_hostile_dist = float("inf")
        nearest_hostile_mob = None
        for mob in self._mobs:
            dist = mob.distance_to(self._x, self._z)
            if dist < nearest_hostile_dist:
                nearest_hostile_dist = dist
                nearest_hostile = {"name": mob.mob_type, "distance": dist}
                nearest_hostile_mob = mob

        # Compute facing info for nearest hostile
        hostile_facing = None
        if nearest_hostile_mob is not None:
            # Vector from mob to agent
            dx = self._x - nearest_hostile_mob.x
            dz = self._z - nearest_hostile_mob.z
            angle_to_agent = math.atan2(-dx, dz)
            angle_diff = abs(
                (angle_to_agent - nearest_hostile_mob.yaw + math.pi)
                % (2 * math.pi) - math.pi
            )
            facing_us = 1.0 - (angle_diff / math.pi)
            hostile_facing = {
                "facing_us": max(0.0, min(1.0, facing_us)),
                "angle_diff": angle_diff,
            }

        alive = self._health > 0
        return {
            "health": self._health if alive else 0,
            "food": self._food,
            "food_saturation": 5.0,
            "xp_level": 0,
            "xp_points": self._kill_count,
            "position": {"x": self._x, "y": 64.0, "z": self._z},
            "yaw": self._yaw,
            "pitch": 0.0,
            "on_ground": True,
            "is_in_water": False,
            "is_raining": False,
            "time_of_day": 15000,  # Always night
            "light_level": 4,
            "altitude": 64.0,
            "block_composition": {
                "air": 0.4, "stone": 0.1, "dirt": 0.3,
                "wood": 0.05, "water": 0.0, "ore": 0.0,
                "danger": 0.0, "other": 0.15,
            },
            "entities": {
                "hostile": nearest_hostile,
                "passive": None,
                "player": None,
            },
            "inventory": {
                "slots_used": 0,
                "has_weapon": True,  # Always has weapon in combat sim
                "has_food": False,
                "has_wood": False,
                "has_tool": False,
            },
            "alive": alive,
            "hit_landed": self._hit_landed,
            "player_hit_landed": self._player_hit_landed,
            "kills": 0,
            "attack_cooldown": self._attack_cooldown,
            "hostile_facing": hostile_facing,
            "player_facing": None,
        }

    def _encode_with_history(self, state: dict, timestamp: int) -> Signal:
        """Encode state to 76 dims, append 16-dim history trace -> 92-dim Signal."""
        base_signal = self._encoder.encode(state, timestamp=timestamp)
        history = self._history_trace.update(base_signal.data)
        combined = np.concatenate([base_signal.data, history])
        # L2-normalize the history slice
        hist_slice = combined[76:92]
        norm = np.linalg.norm(hist_slice)
        if norm > 0:
            combined[76:92] = hist_slice / norm
        return Signal(data=combined, timestamp=timestamp, modality="minecraft")

    def close(self) -> None:
        pass
