"""Discrete action space for Minecraft.

Each action maps to a Mineflayer control command. Actions are executed
for a fixed number of game ticks (default 4 = 200ms at 20 ticks/sec).

Phase 1 (13 actions): Movement, looking, attacking, idle.
Phase 2 (18 actions): Adds item use, hotbar cycling, basic crafting.
Phase 3 (20 actions): Adds combat combos (sprint-crit, w-tap).
"""

# Phase 1: Movement + Look + Attack + Idle
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
STRAFE_LEFT = 2
STRAFE_RIGHT = 3
JUMP = 4
FORWARD_JUMP = 5
SPRINT_FORWARD = 6
LOOK_LEFT = 7
LOOK_RIGHT = 8
LOOK_UP = 9
LOOK_DOWN = 10
ATTACK = 11
IDLE = 12

# Phase 2: Inventory + Crafting
USE_ITEM = 13
HOTBAR_NEXT = 14
HOTBAR_PREV = 15
CRAFT_PLANKS = 16
CRAFT_TOOL = 17

# Phase 3: Combat combos
SPRINT_CRIT = 18  # Sprint + jump + attack at fall peak (critical hit)
W_TAP = 19        # Release forward → attack → re-engage (extra knockback)

PHASE_1_ACTIONS: list[int] = list(range(13))
PHASE_2_ACTIONS: list[int] = list(range(18))
PHASE_3_ACTIONS: list[int] = list(range(20))

# Factored action space: 3 independent axes executed in parallel.
# Movement (7): none, forward, back, left, right, fwd+jump, fwd+sprint
# Look (6):     none, track_target, look_left, look_right, look_up, look_down
# Combat (4):   none, attack, crit, wtap
# Encoding: flat_id = movement * 24 + look * 4 + combat
MOVEMENT_COUNT = 7
LOOK_COUNT = 6
COMBAT_COUNT = 4
FACTORED_ACTION_COUNT = MOVEMENT_COUNT * LOOK_COUNT * COMBAT_COUNT  # 168
FACTORED_ACTIONS: list[int] = list(range(FACTORED_ACTION_COUNT))


def encode_composite(movement: int, look: int, combat: int) -> int:
    """Encode three axis choices into a single flat action ID."""
    return movement * (LOOK_COUNT * COMBAT_COUNT) + look * COMBAT_COUNT + combat


def decode_composite(flat_id: int) -> tuple[int, int, int]:
    """Decode a flat action ID into (movement, look, combat) axes."""
    combat = flat_id % COMBAT_COUNT
    remainder = flat_id // COMBAT_COUNT
    look = remainder % LOOK_COUNT
    movement = remainder // LOOK_COUNT
    return movement, look, combat


ACTION_NAMES: dict[int, str] = {
    MOVE_FORWARD: "forward",
    MOVE_BACKWARD: "backward",
    STRAFE_LEFT: "strafe_left",
    STRAFE_RIGHT: "strafe_right",
    JUMP: "jump",
    FORWARD_JUMP: "forward_jump",
    SPRINT_FORWARD: "sprint_forward",
    LOOK_LEFT: "look_left",
    LOOK_RIGHT: "look_right",
    LOOK_UP: "look_up",
    LOOK_DOWN: "look_down",
    ATTACK: "attack",
    IDLE: "idle",
    USE_ITEM: "use_item",
    HOTBAR_NEXT: "hotbar_next",
    HOTBAR_PREV: "hotbar_prev",
    CRAFT_PLANKS: "craft_planks",
    CRAFT_TOOL: "craft_tool",
    SPRINT_CRIT: "sprint_crit",
    W_TAP: "w_tap",
}
