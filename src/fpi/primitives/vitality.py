"""Vitality — the ground of all motivation.

Vitality is finite energy that depletes under entropy. At zero, the agent
ceases to exist. This single thermodynamic constraint — persist or perish —
is the foundation from which all drives emerge.

No hunger module. No fear module. No curiosity module. Just: energy depletes,
find more or die. Everything else is a consequence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Vitality:
    """Finite energy that grounds all motivation.

    Rules:
    1. Energy depletes each tick (entropy — existence costs energy).
    2. Actions cost additional energy.
    3. Resources in the environment restore energy.
    4. At energy = 0, the agent dies.

    From these four rules, all drives emerge.

    Attributes:
        energy: Current energy level in [0, max_energy].
        max_energy: Upper bound on energy.
        entropy_rate: Energy lost per tick just by existing.
        alive: Whether the agent still exists.
    """

    energy: float = 1.0
    max_energy: float = 1.0
    entropy_rate: float = 0.01
    alive: bool = True

    def tick(self) -> float:
        """Apply entropy — existence costs energy.

        Returns energy level after depletion.
        """
        if not self.alive:
            return 0.0
        self.energy = max(0.0, self.energy - self.entropy_rate)
        if self.energy <= 0.0:
            self.alive = False
        return self.energy

    def spend(self, cost: float) -> None:
        """Spend energy on an action. Can kill the agent.

        Life doesn't ask permission. The agent can always attempt an action,
        but may die doing it.
        """
        if not self.alive:
            return
        self.energy = max(0.0, self.energy - cost)
        if self.energy <= 0.0:
            self.alive = False

    def restore(self, amount: float) -> float:
        """Gain energy from a resource.

        Returns the actual amount restored (capped at max_energy).
        """
        actual = min(amount, self.max_energy - self.energy)
        self.energy += actual
        return actual

    @property
    def fraction(self) -> float:
        """Energy as fraction of max [0, 1]."""
        return self.energy / self.max_energy

    @property
    def urgency(self) -> float:
        """How urgently the agent needs energy. 0 = full, 1 = empty.

        This is not a "hunger module." It is a direct read of the energy
        deficit. But it will naturally drive behavior: when urgency is high,
        the agent weights vitality-positive actions more heavily.
        """
        return 1.0 - self.fraction
