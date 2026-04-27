"""MinecraftEnv — Minecraft environment for FPI agents.

Implements the standard Environment interface so that the Agent can
interact with Minecraft using the same step_with_action() loop used
for SurvivalEnv. Communication with the Minecraft game happens
through a Mineflayer bot accessed via TCP bridge.

The environment is continuous — there are no fixed-length episodes.
On death, the bot auto-respawns and the agent's vitality is reset
externally (see runner.py), but learned knowledge persists.

Reward signal is purely primitive — the agent feels:
- Pain (health loss)
- Relief (health gain)
- Hunger (food loss)
- Satiation (food gain)
- Satisfaction (XP gain)
- Death (-1.0)
- Metabolic cost (idle drain)

All other knowledge must be learned from experience.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..env.base import Environment
from ..primitives.signal import Signal
from .actions import (
    FACTORED_ACTIONS, PHASE_1_ACTIONS, PHASE_2_ACTIONS, PHASE_3_ACTIONS,
    PHASE_4_ACTIONS, decode_composite,
)
from .bridge import MinecraftBridge
from .encoder import HistoryTrace, MinecraftStateEncoder
from .opponent import OpponentTracker

# ── State dict schema ────────────────────────────────────────────────
# Keys expected from the JS relay. Used for one-shot validation.
# Names must match the snake_case keys in controller/state.js exactly.

REQUIRED_STATE_KEYS: frozenset[str] = frozenset({
    "health", "food", "alive",
    "time_of_day", "light_level",
    "yaw", "pitch", "on_ground", "is_in_water", "is_raining", "altitude",
    "spatial", "entities", "inventory",
    "xp_level", "xp_points",
    "attack_cooldown", "hit_landed", "player_hit_landed", "kills",
})

OPTIONAL_STATE_KEYS: frozenset[str] = frozenset({
    "food_saturation", "position",
    "bot_control_active", "user_active", "is_using_item",
    "type",            # injected by bridge.js protocol wrapper
    "macro_status",    # only present after macro-action commands
    "crowd",           # aggregate entity counts + directional data
    # Self-awareness + threat dynamics (396→428 expansion)
    "self_velocity", "health_delta", "food_delta", "ticks_airborne",
    "self_effects", "incoming_projectile", "self_armor_tier", "is_thundering",
    "nearest_hostile_accel", "nearest_player_armor", "height_vs_hostile",
    "height_vs_player", "combat_hits_5s", "combat_damage_5s",
    "time_since_hit", "kill_streak", "strafing",
})

KNOWN_STATE_KEYS: frozenset[str] = REQUIRED_STATE_KEYS | OPTIONAL_STATE_KEYS

_schema_logger = logging.getLogger("fpi.minecraft.env")


def _validate_state_schema(state: dict, source: str = "bridge") -> None:
    """Validate state dict keys on first receipt. Warns on mismatches."""
    state_keys = frozenset(state.keys())

    missing = REQUIRED_STATE_KEYS - state_keys
    if missing:
        _schema_logger.warning(
            "[schema] State from %s missing %d required key(s): %s. "
            "These default to zero/None in the encoder — check state.js ↔ encoder.py sync.",
            source, len(missing), sorted(missing),
        )

    unknown = {k for k in state_keys - KNOWN_STATE_KEYS if not k.startswith("_")}
    if unknown:
        _schema_logger.warning(
            "[schema] State from %s has %d unknown key(s): %s. "
            "If new JS-side additions, add to KNOWN_STATE_KEYS in env.py.",
            source, len(unknown), sorted(unknown),
        )


def _nearest_target(state: dict):
    """Get nearest entity by priority: hostile > player > passive."""
    entities = state.get("entities", {})
    for key in ("hostiles", "players", "passives"):
        lst = entities.get(key, [])
        if lst:
            return lst[0]
    return None


def compute_energy_delta(prev: dict, curr: dict) -> float:
    """Compute FPI energy_delta from consecutive Minecraft states.

    Pure primitive signals only — no hardcoded game knowledge.

    Args:
        prev: Previous game state dict.
        curr: Current game state dict.

    Returns:
        Energy delta in roughly [-1.0, +0.5] range.
    """
    # Death overrides everything
    if not curr.get("alive", True):
        return -1.0

    delta = 0.0

    # Pain / relief: health change
    curr_health = curr.get("health") or 20.0
    prev_health = prev.get("health") or 20.0
    health_change = (float(curr_health) - float(prev_health)) / 20.0
    delta += health_change * 0.5

    # Extra damage penalty: damage feels worse than healing feels good
    if health_change < 0:
        delta += health_change * 0.3

    # Hunger / satiation: food change
    curr_food = curr.get("food") or 20.0
    prev_food = prev.get("food") or 20.0
    food_change = (float(curr_food) - float(prev_food)) / 20.0
    delta += food_change * 0.1

    # Hit confirmation: attack connected with a mob. This is the primary
    # positive signal. Real sensory feedback — mob flashes red, takes
    # knockback, makes pain sound. Immediate, attributable, frequent.
    if curr.get("hit_landed", False):
        delta += 0.1

    # Player hit: stronger reward for hitting players (PVP)
    if curr.get("player_hit_landed", False):
        delta += 0.1  # Stacks with hit_landed for 0.2 total

    # Positioning: weak signal for being at optimal attack range (2.5-3.5 blocks).
    # This is borderline "injected knowledge" so kept very weak — the agent should
    # mostly learn positioning from hit/pain primitives, not from this signal.
    nearest_target = _nearest_target(curr)
    if nearest_target is not None:
        dist = float(nearest_target.get("distance", 99.0))
        if 2.5 <= dist <= 3.5:
            delta += 0.02  # Weak bonus — in optimal attack range

    # Approach reward: getting closer to nearest mob (hostile, player, or passive).
    # Weaker than hit_landed (0.1). Clamped to avoid wild jumps from spawn/despawn.
    prev_target = _nearest_target(prev)
    curr_target = _nearest_target(curr)
    if prev_target is not None and curr_target is not None:
        prev_hdist = prev_target.get("distance")
        curr_hdist = curr_target.get("distance")
        if prev_hdist is not None and curr_hdist is not None:
            approach = float(prev_hdist) - float(curr_hdist)  # positive = got closer
            approach = max(-0.5, min(0.5, approach))  # clamp wild jumps
            delta += approach * 0.03

    # Metabolic cost: base rate + escalating idle penalty.
    # Standing around doing nothing gets progressively more painful,
    # pushing the agent to move and find new targets.
    idle_ticks = curr.get("_idle_ticks", 0)
    idle_penalty = 0.001 + min(idle_ticks / 50, 1.0) * 0.005  # ramps 0.001→0.006
    delta -= idle_penalty

    return delta


def apply_phase_bias(agent, env: MinecraftEnv) -> None:
    """Bias per-modality thresholds based on current game phase.

    Uses the agent's AbstractionLayer meta-pattern to detect game phase
    and bias thresholds accordingly:
    - Danger (negative meta-valence): tighten entity threshold for precision
    - Safety (positive meta-valence): loosen entity threshold for efficiency
    - High surprise: tighten all thresholds to capture more distinctions

    Args:
        agent: FPI Agent instance (must have abstraction_layer attribute).
        env: MinecraftEnv instance (for MODALITY_SLICES reference).
    """
    # Check if agent has abstraction and compositional distinction
    if not hasattr(agent, 'abstraction_layer') or agent.abstraction_layer is None:
        return
    wm = agent.world_model
    distinction = wm.memory.distinction
    if not hasattr(distinction, '_modal_distinctions'):
        return

    modal_dists = distinction._modal_distinctions
    al = agent.abstraction_layer

    # Get current meta-pattern valence
    meta_pattern = al.current_meta_pattern
    if meta_pattern is None:
        return
    meta_val = al.valence.get(meta_pattern.pattern_id)

    # Get current surprise level
    avg_surprise = wm.average_surprise

    # Entity modality index (index 3 in our slice layout)
    ENTITY_IDX = 3
    # Combat modality index
    COMBAT_IDX = 5

    for i, md in enumerate(modal_dists):
        if not md._adaptive:
            continue

        bias = 0.0

        if i == ENTITY_IDX:
            # Danger: tighten entity threshold for finer discrimination
            if meta_val < -0.1:
                bias -= 0.03 * min(1.0, abs(meta_val))
            # Safety: loosen for efficiency
            elif meta_val > 0.1:
                bias += 0.02 * min(1.0, meta_val)

        if i == COMBAT_IDX:
            # In combat (negative meta-valence): tighten for timing precision
            if meta_val < -0.05:
                bias -= 0.02 * min(1.0, abs(meta_val))

        # High surprise: tighten slightly across all modalities
        if avg_surprise > 0.5:
            bias -= 0.01 * min(1.0, avg_surprise - 0.5)

        if bias != 0.0:
            new_threshold = md.similarity_threshold + bias
            md.similarity_threshold = max(
                md._threshold_min,
                min(md._threshold_max, new_threshold),
            )


class MinecraftEnv(Environment):
    """Minecraft environment accessed through Mineflayer bridge.

    Implements the standard Environment interface:
    - reset(): connect to bot, return initial observation
    - step(action): execute action, return (observation, energy_delta, done)
    - action_space: available discrete actions

    Args:
        host: Mineflayer bridge host.
        port: Mineflayer bridge port.
        phase: 1 = movement only (13 actions), 2 = full (18 actions).
    """

    # Full modality slices including vision + history trace (428 dims total).
    # This is what the agent actually sees — use this for compositional patterns.
    MODALITY_SLICES: list[tuple[int, int]] = [
        (0, 12),     # body: health + food + urgency
        (12, 20),    # environment: time + light
        (20, 44),    # terrain: spatial grid (24 dims)
        (44, 332),   # entities: 8 hostile + 4 passive + 4 player + crowd
        (332, 352),  # inventory + situation (4 situation + 4 fullness + 12 hotbar)
        (352, 364),  # combat + movement: xp + orientation + cooldown
        (364, 380),  # self-awareness: velocity, deltas, effects, armor (16 dims)
        (380, 396),  # threat dynamics: projectiles, timing, momentum (16 dims)
        (396, 412),  # vision: CNN features from screen capture (16 dims)
        (412, 428),  # history: temporal context trace (16 dims)
    ]

    # Per-modality similarity thresholds: volatile modalities get looser matching
    # to produce fewer, broader patterns and prevent cascade evictions on live servers.
    MODALITY_THRESHOLDS: list[float] = [
        0.85,  # body [0:12] — stable, tighter for combat sensitivity
        0.80,  # environment [12:20] — changes slowly, fine as-is
        0.65,  # terrain [20:44] — spatial grid, loose to merge similar layouts
        0.65,  # entities [44:332] — wider modality needs looser matching
        0.80,  # inventory [332:352] — hotbar changes matter for tool selection
        0.80,  # combat [352:364] — matters for PvP timing
        0.75,  # self-awareness [364:380] — moderate, some signals volatile
        0.70,  # threat dynamics [380:396] — loose, highly volatile combat
        0.70,  # vision [396:412] — moderately volatile (lighting, camera angle)
        0.60,  # history [412:428] — exp-decay diverges naturally
    ]

    SIGNAL_DIM = 428

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        phase: int = 1,
        factored: bool = False,
        screen_capture: bool = False,
        capture_region: tuple[int, int, int, int] | None = None,
        vision_weights: str | None = None,
    ) -> None:
        self._bridge = MinecraftBridge(host=host, port=port)
        self._encoder = MinecraftStateEncoder()
        self._opponent_tracker = OpponentTracker()

        # Vision: optional screen capture + CNN encoder
        self._screen_capture_enabled = screen_capture
        self._capture_process = None
        self._vision_encoder = None
        if screen_capture:
            from .screen_capture import ScreenCaptureProcess
            from .vision import VisionEncoder
            self._capture_process = ScreenCaptureProcess(
                region=capture_region, fps=4,
            )
            self._vision_encoder = VisionEncoder(weights_path=vision_weights)

        # History trace projects from base + vision (396 + 16 = 412 dims)
        self._history_trace = HistoryTrace(
            decay=0.6,
            output_dims=16,
            input_dims=412,  # 396 base + 16 vision
        )
        self._phase = phase
        self._factored = factored
        self._prev_state: dict | None = None
        self._step_count: int = 0
        self._death_count: int = 0
        self._total_xp: int = 0
        self._kill_count: int = 0
        self._idle_ticks: int = 0
        self._schema_validated: bool = False

    @property
    def action_space(self) -> list[int]:
        """Available discrete actions."""
        if self._factored:
            return FACTORED_ACTIONS
        if self._phase >= 4:
            return PHASE_4_ACTIONS
        if self._phase >= 3:
            return PHASE_3_ACTIONS
        if self._phase >= 2:
            return PHASE_2_ACTIONS
        return PHASE_1_ACTIONS

    @property
    def death_count(self) -> int:
        return self._death_count

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def opponent_tracker(self) -> OpponentTracker:
        return self._opponent_tracker

    def _encode_with_history(
        self,
        state: dict,
        timestamp: int,
        opponent_profiles: list[tuple[float, float]] | None = None,
    ) -> Signal:
        """Encode state to 396 dims + 16 vision + 16 history -> 428-dim Signal."""
        base_signal = self._encoder.encode(
            state, timestamp=timestamp, opponent_profiles=opponent_profiles,
        )

        # Vision: encode screen frame or use zeros
        if self._screen_capture_enabled and self._vision_encoder is not None:
            frame, _ = self._capture_process.read_frame()
            vision = self._vision_encoder.encode(frame)
        else:
            vision = np.zeros(16, dtype=np.float64)

        # Combine base (396) + vision (16) = 412 dims for history projection
        base_plus_vision = np.concatenate([base_signal.data, vision])

        # History trace projects from the 412-dim combined signal
        history = self._history_trace.update(base_plus_vision)
        combined = np.concatenate([base_plus_vision, history])

        # L2-normalize vision slice [396:412]
        vis_slice = combined[396:412]
        vis_norm = np.linalg.norm(vis_slice)
        if vis_norm > 0:
            combined[396:412] = vis_slice / vis_norm

        # L2-normalize history slice [412:428]
        hist_slice = combined[412:428]
        hist_norm = np.linalg.norm(hist_slice)
        if hist_norm > 0:
            combined[412:428] = hist_slice / hist_norm

        return Signal(data=combined, timestamp=timestamp, modality="minecraft")

    def reset(self) -> Signal:
        """Connect to the Mineflayer bot and return the initial observation."""
        # Start screen capture process if enabled
        if self._capture_process is not None and not self._capture_process.is_alive:
            self._capture_process.start()

        self._bridge.connect()

        # Retry get_state — the controller may still be connecting to the game
        import time as _time
        state = None
        for attempt in range(30):
            try:
                state = self._bridge.get_state()
                break
            except Exception as e:
                if "not connected" in str(e).lower() and attempt < 29:
                    if attempt == 0:
                        print("[env] Waiting for controller to connect to game...")
                    _time.sleep(2)
                else:
                    raise
        assert state is not None
        if not self._schema_validated:
            _validate_state_schema(state, source="bridge")
            self._schema_validated = True
        self._prev_state = state
        self._step_count = 0
        self._total_xp = state.get("xp_points", 0)
        self._history_trace.reset()
        return self._encode_with_history(state, timestamp=0)

    def step(self, action: int | None = None) -> tuple[Signal, float, bool]:
        """Execute an action and return the resulting observation.

        Args:
            action: Discrete action index from action_space.

        Returns:
            (observation, energy_delta, done). Done is always False — Minecraft
            is continuous. On death, energy_delta is -1.0 and the bot is
            respawned automatically.
        """
        if action is not None:
            if self._factored:
                m, l, c = decode_composite(action)
                state = self._bridge.send_composite_action(m, l, c)
            else:
                state = self._bridge.send_action(action)
        else:
            state = self._bridge.get_state()

        self._step_count += 1

        # Track idle ticks: reset on hit or successful macro, escalate otherwise
        if state.get("hit_landed", False) or state.get("player_hit_landed", False):
            self._idle_ticks = 0
        elif state.get("macro_status") == "completed":
            self._idle_ticks = 0
        else:
            self._idle_ticks += 1

        # Inject idle ticks into state for energy delta computation
        state["_idle_ticks"] = self._idle_ticks

        # Update opponent tracker and get profiles
        if self._prev_state is not None:
            self._opponent_tracker.update(
                state, self._prev_state, self._step_count,
            )
        opponent_profiles = self._opponent_tracker.get_profiles_for_state(state)

        # Compute energy delta from state change
        energy_delta = compute_energy_delta(self._prev_state, state)

        # Track XP and kills
        xp_now = state.get("xp_points", 0)
        if xp_now > self._total_xp:
            self._total_xp = xp_now
        kills = state.get("kills", 0)
        if kills > 0:
            self._kill_count += kills

        # Handle death: respawn the bot
        if not state.get("alive", True):
            self._death_count += 1
            self._history_trace.reset()  # New life = fresh temporal context
            # Respawn and get fresh state
            state = self._bridge.respawn()

        self._prev_state = state
        observation = self._encode_with_history(
            state, timestamp=self._step_count,
            opponent_profiles=opponent_profiles,
        )

        # Never done — Minecraft is continuous
        return observation, energy_delta, False

    def set_bot_control(self, enabled: bool) -> None:
        """Enable or disable FPI agent control of the character.

        When enabled, the relay suppresses the real client's inputs and the
        agent's actions are injected.  When disabled, the user plays normally
        and the agent only observes.
        """
        self._bridge.send_bot_control(enabled)

    def close(self) -> None:
        """Disconnect from the Mineflayer bot and stop screen capture."""
        if self._capture_process is not None:
            self._capture_process.stop()
        self._bridge.close()
