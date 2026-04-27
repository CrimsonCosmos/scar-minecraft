"""Minecraft environment for FPI agents via Mineflayer bridge."""

from .actions import (
    FACTORED_ACTIONS,
    PHASE_1_ACTIONS,
    PHASE_2_ACTIONS,
    PHASE_3_ACTIONS,
    PHASE_4_ACTIONS,
)
from .bridge import MinecraftBridge
from .encoder import MinecraftStateEncoder
from .env import MinecraftEnv

__all__ = [
    "MinecraftBridge",
    "MinecraftEnv",
    "MinecraftStateEncoder",
    "PHASE_1_ACTIONS",
    "PHASE_2_ACTIONS",
    "PHASE_3_ACTIONS",
    "PHASE_4_ACTIONS",
    "FACTORED_ACTIONS",
]

# Neural policy agents (requires torch)
try:
    from .neural_policy import DQNAgent, PPOAgent

    __all__ += ["PPOAgent", "DQNAgent"]
except ImportError:
    pass
