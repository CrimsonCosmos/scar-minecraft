"""Society — recursive intelligence at the collective level.

A Society is to Agents what a Brain is to Neurons. It uses the same
primitives (Signal, Pattern, Association, Vitality, Valence) but at a
higher scale. Individual agents are the "neurons" — their collective
behavior is what the society senses, predicts, and acts upon.

This is composable: a Collective of Agents = Society.
A Collective of Societies = Civilization. Same code, different scale.
"""

from .bridge import SignalBridge
from .core import Society, SocietyStepResult
from .social_society import SocialSociety

__all__ = ["Society", "SignalBridge", "SocietyStepResult", "SocialSociety"]
