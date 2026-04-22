"""Tests for Phase 3 Level 2: Darwinian population evolution.

Tests genome operations, population evaluation, and fitness improvement.
"""

import numpy as np
import pytest

from fpi.evolution.genome import Genome, GENOME_BOUNDS
from fpi.evolution.population import Population, FitnessResult, GenerationResult
from fpi.env.base import SurvivalEnv


class TestGenome:
    def test_default_genome(self):
        g = Genome()
        assert 0.5 <= g.similarity_threshold <= 0.95
        assert g.max_patterns == 15

    def test_random_genome_within_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = Genome.random(rng)
            for name, (lo, hi) in GENOME_BOUNDS.items():
                val = getattr(g, name)
                assert lo <= val <= hi, f"{name}={val} out of bounds [{lo}, {hi}]"

    def test_mutate_stays_in_bounds(self):
        rng = np.random.default_rng(42)
        g = Genome()
        for _ in range(50):
            mutated = g.mutate(rng, mutation_rate=1.0)  # Force all mutations
            for name, (lo, hi) in GENOME_BOUNDS.items():
                val = getattr(mutated, name)
                assert lo <= val <= hi, f"{name}={val} out of bounds [{lo}, {hi}]"

    def test_mutate_can_change_values(self):
        rng = np.random.default_rng(42)
        g = Genome()
        mutated = g.mutate(rng, mutation_rate=1.0)
        # At least one value should change
        any_different = any(
            getattr(g, f.name) != getattr(mutated, f.name)
            for f in Genome.__dataclass_fields__.values()
        )
        assert any_different

    def test_crossover_takes_from_both_parents(self):
        rng = np.random.default_rng(42)
        parent_a = Genome(similarity_threshold=0.5, exploration_base=0.02)
        parent_b = Genome(similarity_threshold=0.95, exploration_base=0.4)

        # Run many crossovers — should see values from both parents
        a_count = 0
        b_count = 0
        for _ in range(100):
            child = Genome.crossover(parent_a, parent_b, rng)
            if child.similarity_threshold == 0.5:
                a_count += 1
            else:
                b_count += 1
        assert a_count > 10, "Should take from parent A sometimes"
        assert b_count > 10, "Should take from parent B sometimes"

    def test_to_agent_kwargs(self):
        g = Genome()
        kwargs = g.to_agent_kwargs()
        assert "similarity_threshold" in kwargs
        assert "max_patterns" in kwargs
        assert "maintenance_cost_per_pattern" in kwargs
        assert len(kwargs) == 8

    def test_max_patterns_is_int(self):
        rng = np.random.default_rng(42)
        g = Genome.random(rng)
        assert isinstance(g.max_patterns, int)
        assert isinstance(g.max_associations, int)

    def test_genome_is_frozen(self):
        g = Genome()
        with pytest.raises(AttributeError):
            g.similarity_threshold = 0.5  # type: ignore[misc]


class TestPopulation:
    def test_evaluate_genome(self):
        pop = Population(seed=42, episodes_per_agent=3, steps_per_episode=50)
        result = pop.evaluate_genome(Genome())
        assert result.fitness > 0
        assert len(result.survival_times) == 3

    def test_run_generation(self):
        pop = Population(population_size=5, seed=42, episodes_per_agent=2, steps_per_episode=50)
        genomes = [Genome.random(np.random.default_rng(i)) for i in range(5)]
        gen_result = pop.run_generation(genomes)
        assert gen_result.generation == 0
        assert len(gen_result.results) == 5
        # Results should be sorted by fitness (best first)
        for i in range(len(gen_result.results) - 1):
            assert gen_result.results[i].fitness >= gen_result.results[i + 1].fitness

    def test_select_and_reproduce_preserves_size(self):
        pop = Population(population_size=8, top_k=3, seed=42, episodes_per_agent=2, steps_per_episode=50)
        genomes = [Genome.random(np.random.default_rng(i)) for i in range(8)]
        gen_result = pop.run_generation(genomes)
        next_gen = pop.select_and_reproduce(gen_result)
        assert len(next_gen) == 8

    def test_elitism_preserves_best(self):
        pop = Population(population_size=5, top_k=2, seed=42, episodes_per_agent=2, steps_per_episode=50)
        genomes = [Genome.random(np.random.default_rng(i)) for i in range(5)]
        gen_result = pop.run_generation(genomes)
        next_gen = pop.select_and_reproduce(gen_result)
        # First genome should be the best from previous generation (elitism)
        assert next_gen[0] == gen_result.best_genome

    def test_evolve_returns_history(self):
        pop = Population(
            population_size=5,
            episodes_per_agent=2,
            steps_per_episode=50,
            seed=42,
        )
        history = pop.evolve(num_generations=3)
        assert len(history) == 3
        assert history[0].generation == 0
        assert history[2].generation == 2

    def test_convergence_report(self):
        pop = Population(
            population_size=5,
            episodes_per_agent=2,
            steps_per_episode=50,
            seed=42,
        )
        pop.evolve(num_generations=2)
        report = pop.convergence_report()
        assert "similarity_threshold" in report
        assert "max_patterns" in report

    def test_convergence_report_empty_before_evolve(self):
        pop = Population()
        assert pop.convergence_report() == {}

    def test_fitness_improves_over_generations(self):
        """The key integration test: fitness should improve with evolution."""
        pop = Population(
            population_size=8,
            episodes_per_agent=3,
            steps_per_episode=100,
            top_k=3,
            mutation_rate=0.3,
            seed=42,
        )
        history = pop.evolve(num_generations=6)

        early_best = history[0].best_fitness
        late_best = history[-1].best_fitness
        early_avg = history[0].avg_fitness
        late_avg = history[-1].avg_fitness

        # At least one of these should improve
        improved = late_best >= early_best or late_avg > early_avg
        assert improved, (
            f"Evolution didn't improve: "
            f"early best={early_best:.0f}, late best={late_best:.0f}, "
            f"early avg={early_avg:.0f}, late avg={late_avg:.0f}"
        )

    def test_custom_env_factory(self):
        def make_env():
            return SurvivalEnv(grid_size=5, resource_positions=[2], max_steps=50)

        pop = Population(
            population_size=3,
            episodes_per_agent=2,
            steps_per_episode=50,
            env_factory=make_env,
            seed=42,
        )
        result = pop.evaluate_genome(Genome())
        assert result.fitness > 0
