"""Evolution — Darwinian selection at the population level.

This package implements heritable genomes and population-level evolution.
Different agents compete in the same environment; the fittest survive and
reproduce. Parameters are not hand-tuned — they are discovered through
selection pressure.
"""

from .genome import Genome
from .population import Population

__all__ = ["Genome", "Population"]
