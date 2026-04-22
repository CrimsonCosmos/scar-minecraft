"""Population — Darwinian evolution across agents.

A population of agents with different genomes compete in the same environment.
The fittest survive, reproduce, and pass their parameters to the next
generation. Over generations, the population converges on architectures
that are well-adapted to the environment.

This is the same principle as biological evolution: variation (mutation +
crossover) + selection (fitness = survival time) + inheritance (genomes
passed to offspring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .genome import Genome
from ..agent.core import Agent
from ..env.base import SurvivalEnv


@dataclass
class FitnessResult:
    """The outcome of evaluating one genome."""

    genome: Genome
    fitness: float  # Average survival time across episodes
    survival_times: list[int]


@dataclass
class GenerationResult:
    """The outcome of one generation of evolution."""

    generation: int
    results: list[FitnessResult]  # Sorted by fitness (best first)

    @property
    def best_fitness(self) -> float:
        return self.results[0].fitness

    @property
    def avg_fitness(self) -> float:
        return sum(r.fitness for r in self.results) / len(self.results)

    @property
    def best_genome(self) -> Genome:
        return self.results[0].genome


class Population:
    """Evolutionary optimization of agent genomes.

    Runs a population of agents through selection cycles:
    1. Evaluate: each genome runs K episodes, fitness = avg survival time
    2. Select: top-K genomes become parents
    3. Reproduce: elitism (best survives), mutation, crossover
    4. Repeat

    Args:
        population_size: Number of genomes per generation.
        episodes_per_agent: How many episodes to evaluate each genome.
        steps_per_episode: Max steps per survival episode.
        top_k: Number of parents selected for reproduction.
        mutation_rate: Per-gene probability of mutation.
        crossover_prob: Probability of crossover vs single-parent reproduction.
        env_factory: Callable that creates a fresh SurvivalEnv.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 10,
        episodes_per_agent: int = 5,
        steps_per_episode: int = 200,
        top_k: int = 3,
        mutation_rate: float = 0.3,
        crossover_prob: float = 0.5,
        env_factory: Callable[[], SurvivalEnv] | None = None,
        seed: int = 42,
    ) -> None:
        self._pop_size = population_size
        self._episodes = episodes_per_agent
        self._steps = steps_per_episode
        self._top_k = top_k
        self._mutation_rate = mutation_rate
        self._crossover_prob = crossover_prob
        self._env_factory = env_factory or self._default_env
        self._rng = np.random.default_rng(seed)
        self._history: list[GenerationResult] = []

    @staticmethod
    def _default_env() -> SurvivalEnv:
        return SurvivalEnv(
            grid_size=10,
            resource_positions=[2, 8],
            move_cost=0.015,
            stay_cost=0.005,
            resource_value=0.3,
            max_steps=200,
        )

    def evaluate_genome(self, genome: Genome) -> FitnessResult:
        """Run an agent with this genome through multiple episodes.

        Fitness = average survival time. Higher is better.
        """
        agent = Agent(seed=int(self._rng.integers(0, 2**31)), **genome.to_agent_kwargs())
        survival_times: list[int] = []

        for _ in range(self._episodes):
            env = self._env_factory()
            agent.world_model.reset_stats()
            results = agent.run_survival_episode(env, max_steps=self._steps)
            survival_times.append(len(results))

        avg_survival = sum(survival_times) / len(survival_times)
        return FitnessResult(genome=genome, fitness=avg_survival, survival_times=survival_times)

    def run_generation(self, genomes: list[Genome]) -> GenerationResult:
        """Evaluate all genomes and return sorted results."""
        results = [self.evaluate_genome(g) for g in genomes]
        results.sort(key=lambda r: r.fitness, reverse=True)
        gen_num = len(self._history)
        gen_result = GenerationResult(generation=gen_num, results=results)
        self._history.append(gen_result)
        return gen_result

    def select_and_reproduce(self, gen_result: GenerationResult) -> list[Genome]:
        """Select top-K parents and produce next generation.

        - Elitism: best genome survives unchanged
        - Rest: mutation and/or crossover from parents
        """
        parents = [r.genome for r in gen_result.results[:self._top_k]]
        next_gen: list[Genome] = []

        # Elitism: best parent survives unchanged
        next_gen.append(parents[0])

        # Fill remaining slots
        while len(next_gen) < self._pop_size:
            if self._rng.random() < self._crossover_prob and len(parents) >= 2:
                # Crossover between two random parents
                p1, p2 = self._rng.choice(len(parents), size=2, replace=False)
                child = Genome.crossover(parents[p1], parents[p2], self._rng)
            else:
                # Single parent mutation
                parent = parents[int(self._rng.integers(0, len(parents)))]
                child = parent

            child = child.mutate(self._rng, self._mutation_rate)
            next_gen.append(child)

        return next_gen

    def evolve(self, num_generations: int) -> list[GenerationResult]:
        """Run the full evolutionary loop.

        Returns the history of all generations.
        """
        # Initialize with random genomes
        genomes = [Genome.random(self._rng) for _ in range(self._pop_size)]

        for _ in range(num_generations):
            gen_result = self.run_generation(genomes)
            genomes = self.select_and_reproduce(gen_result)

        return self._history

    def convergence_report(self) -> dict:
        """Report the best genome's parameters from the last generation."""
        if not self._history:
            return {}
        best = self._history[-1].best_genome
        return best.to_agent_kwargs()
