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

import math

import numpy as np

from ..env.base import Environment
from ..primitives.signal import Signal
from .actions import (
    FACTORED_ACTIONS, PHASE_1_ACTIONS, PHASE_2_ACTIONS, PHASE_3_ACTIONS,
    decode_composite,
)
from .bridge import MinecraftBridge
from .encoder import HistoryTrace, MinecraftStateEncoder


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
    entities = curr.get("entities", {})
    nearest_target = entities.get("hostile") or entities.get("player") or entities.get("passive")
    if nearest_target is not None:
        dist = float(nearest_target.get("distance", 99.0))
        if 2.5 <= dist <= 3.5:
            delta += 0.02  # Weak bonus — in optimal attack range

    # Approach reward: getting closer to nearest mob (hostile, player, or passive).
    # Weaker than hit_landed (0.1). Clamped to avoid wild jumps from spawn/despawn.
    prev_entities = prev.get("entities", {})
    curr_entities = curr.get("entities", {})
    prev_target = prev_entities.get("hostile") or prev_entities.get("player") or prev_entities.get("passive")
    curr_target = curr_entities.get("hostile") or curr_entities.get("player") or curr_entities.get("passive")
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

    # Full modality slices including history trace (92 dims total).
    # This is what the agent actually sees — use this for compositional patterns.
    MODALITY_SLICES: list[tuple[int, int]] = [
        (0, 12),   # body: health + food + urgency
        (12, 20),  # environment: time + light
        (20, 28),  # terrain: block composition
        (28, 48),  # entities: hostile + passive + player
        (48, 60),  # inventory + situation
        (60, 76),  # combat + movement: xp + orientation + cooldown + facing
        (76, 92),  # history: temporal context trace
    ]

    # Per-modality similarity thresholds: volatile modalities get looser matching
    # to produce fewer, broader patterns and prevent cascade evictions on live servers.
    MODALITY_THRESHOLDS: list[float] = [
        0.85,  # body [0:12] — stable, tighter for combat sensitivity
        0.80,  # environment [12:20] — changes slowly, fine as-is
        0.65,  # terrain [20:28] — most volatile, forest/jungle/plains should merge
        0.70,  # entities [28:48] — mobs at similar distances should merge
        0.85,  # inventory [48:60] — stable
        0.80,  # combat [60:76] — matters for PvP timing
        0.60,  # history [76:92] — exp-decay diverges naturally
    ]

    SIGNAL_DIM = 92

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        phase: int = 1,
        factored: bool = False,
    ) -> None:
        self._bridge = MinecraftBridge(host=host, port=port)
        self._encoder = MinecraftStateEncoder()
        self._history_trace = HistoryTrace(
            decay=0.6,
            output_dims=16,
            input_dims=76,  # MinecraftStateEncoder.SIGNAL_DIM
        )
        self._phase = phase
        self._factored = factored
        self._prev_state: dict | None = None
        self._step_count: int = 0
        self._death_count: int = 0
        self._total_xp: int = 0
        self._kill_count: int = 0
        self._idle_ticks: int = 0

    @property
    def action_space(self) -> list[int]:
        """Available discrete actions."""
        if self._factored:
            return FACTORED_ACTIONS
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

    def _encode_with_history(self, state: dict, timestamp: int) -> Signal:
        """Encode state to 76 dims, append 16-dim history trace → 92-dim Signal."""
        base_signal = self._encoder.encode(state, timestamp=timestamp)
        history = self._history_trace.update(base_signal.data)
        combined = np.concatenate([base_signal.data, history])
        # L2-normalize the history slice
        hist_slice = combined[76:92]
        norm = np.linalg.norm(hist_slice)
        if norm > 0:
            combined[76:92] = hist_slice / norm
        return Signal(data=combined, timestamp=timestamp, modality="minecraft")

    def reset(self) -> Signal:
        """Connect to the Mineflayer bot and return the initial observation."""
        self._bridge.connect()
        state = self._bridge.get_state()
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

        # Track idle ticks: reset on hit, escalate otherwise
        if state.get("hit_landed", False) or state.get("player_hit_landed", False):
            self._idle_ticks = 0
        else:
            self._idle_ticks += 1

        # Inject idle ticks into state for energy delta computation
        state["_idle_ticks"] = self._idle_ticks

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
        observation = self._encode_with_history(state, timestamp=self._step_count)

        # Never done — Minecraft is continuous
        return observation, energy_delta, False

    def close(self) -> None:
        """Disconnect from the Mineflayer bot."""
        self._bridge.close()
