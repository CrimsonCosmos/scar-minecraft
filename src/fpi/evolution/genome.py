"""Genome — the heritable parameter set.

A Genome encodes the "DNA" of an agent: the parameters that define its
cognitive architecture. These are not learned during a lifetime — they are
inherited from parents and varied by mutation.

This is the bridge between internal evolution (Neural Darwinism within one
agent) and population evolution (Darwinian selection across agents). The
genome determines the capacity limits, decay rates, and learning parameters
that shape how the brain self-organizes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


# Bounds for each parameter: (min, max)
GENOME_BOUNDS: dict[str, tuple[float, float]] = {
    "similarity_threshold": (0.5, 0.95),
    "valence_learning_rate": (0.05, 0.8),
    "exploration_base": (0.02, 0.4),
    "max_patterns": (3, 30),
    "max_associations": (5, 100),
    "association_decay_rate": (0.0, 0.05),
    "maintenance_cost_per_pattern": (0.0, 0.005),
    "maintenance_cost_per_association": (0.0, 0.002),
}


@dataclass(frozen=True)
class Genome:
    """Heritable parameters that define an agent's cognitive architecture.

    Each field maps to an Agent constructor parameter. The genome determines
    HOW the brain is organized — capacity, learning rates, decay — while the
    brain's content (patterns, associations, valences) is learned.
    """

    similarity_threshold: float = 0.7
    valence_learning_rate: float = 0.3
    exploration_base: float = 0.15
    max_patterns: int = 15
    max_associations: int = 50
    association_decay_rate: float = 0.005
    maintenance_cost_per_pattern: float = 0.001
    maintenance_cost_per_association: float = 0.0003

    def mutate(self, rng: np.random.Generator, mutation_rate: float = 0.1) -> Genome:
        """Create a mutated copy. Each parameter has a chance of Gaussian perturbation."""
        kwargs: dict[str, float | int] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            lo, hi = GENOME_BOUNDS[f.name]
            if rng.random() < mutation_rate:
                # Gaussian perturbation scaled to 10% of range
                scale = (hi - lo) * 0.1
                val = val + rng.normal(0, scale)
                val = max(lo, min(hi, val))
            if f.type == "int":
                val = int(round(val))
            kwargs[f.name] = val
        return Genome(**kwargs)

    def to_agent_kwargs(self) -> dict:
        """Convert to keyword arguments for Agent constructor."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "valence_learning_rate": self.valence_learning_rate,
            "exploration_base": self.exploration_base,
            "max_patterns": self.max_patterns,
            "max_associations": self.max_associations,
            "association_decay_rate": self.association_decay_rate,
            "maintenance_cost_per_pattern": self.maintenance_cost_per_pattern,
            "maintenance_cost_per_association": self.maintenance_cost_per_association,
        }

    @staticmethod
    def random(rng: np.random.Generator) -> Genome:
        """Create a random genome within bounds."""
        kwargs: dict[str, float | int] = {}
        for name, (lo, hi) in GENOME_BOUNDS.items():
            val = rng.uniform(lo, hi)
            # Check if this field is int-typed
            f_type = next(f.type for f in fields(Genome) if f.name == name)
            if f_type == "int":
                val = int(round(val))
            kwargs[name] = val
        return Genome(**kwargs)

    @staticmethod
    def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
        """Uniform crossover: each gene comes from a random parent."""
        kwargs: dict[str, float | int] = {}
        for f in fields(Genome):
            if rng.random() < 0.5:
                kwargs[f.name] = getattr(parent_a, f.name)
            else:
                kwargs[f.name] = getattr(parent_b, f.name)
        return Genome(**kwargs)
