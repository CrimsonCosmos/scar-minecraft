"""Entry point — run the demos.

Demonstrates:
1. Phase 1: An agent learns to predict a repeating signal sequence.
2. Phase 2: An agent learns to survive by finding resources.
3. Phase 3: Evolution discovers optimal agent architectures.
4. Phase 4: A society of agents forms a collective intelligence.
5. Phase 5: Embodied social intelligence (leaky embodiment).

Usage:
    python -m fpi.run
"""

from __future__ import annotations

import numpy as np

from .primitives.signal import Signal
from .primitives.vitality import Vitality
from .env.base import SequencePredictionEnv, SurvivalEnv
from .env.shared import SharedGridEnv
from .env.social import SocialGridEnv
from .env.shared_2d import SharedGrid2DEnv
from .env.social_2d import SocialGrid2DEnv
from .env.patch import PatchForagingEnv
from .agent.core import Agent
from .evolution.population import Population
from .evolution.genome import Genome
from .society.core import Society
from .society.social_society import SocialSociety
from .society.bridge import SignalBridge


def run_prediction_demo(num_episodes: int = 5, steps_per_episode: int = 100) -> None:
    """Run the prediction learning demo (Phase 1)."""
    print("=" * 60)
    print("  Phase 1 — Prediction Demo")
    print("=" * 60)
    print()
    print("An agent observes a repeating signal sequence and learns")
    print("to predict what comes next. Surprise should decrease over time.")
    print()

    env = SequencePredictionEnv(
        sequence=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        num_steps=steps_per_episode,
    )

    agent = Agent(similarity_threshold=0.9)

    for episode in range(num_episodes):
        agent.world_model.reset_stats()
        results = agent.run_episode(env, max_steps=steps_per_episode)

        surprises = [r.surprise for r in results]
        correct = sum(1 for r in results if r.predicted_correctly)
        avg_surprise = sum(surprises) / len(surprises)

        window = 10
        windows = [surprises[i:i + window] for i in range(0, len(surprises), window)]
        window_avgs = [sum(w) / len(w) for w in windows]

        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Avg surprise: {avg_surprise:.3f}")
        print(f"  Correct predictions: {correct}/{len(results)}")
        print(f"  Patterns learned: {agent.pattern_count}")
        print(f"  Associations formed: {agent.association_count}")
        print(f"  Surprise over time: ", end="")
        for wa in window_avgs:
            bar = "#" * int(wa * 20)
            print(f"{wa:.2f} {bar}", end="  ")
        print()
        print()

    print("-" * 60)
    print("Summary:")
    print(f"  Total patterns: {agent.pattern_count}")
    print(f"  Total associations: {agent.association_count}")
    print(f"  Final episode avg surprise: {avg_surprise:.3f}")

    if avg_surprise < 0.3:
        print("  The agent has learned the sequence!")
    else:
        print("  Still learning. More episodes would help.")
    print()


def run_survival_demo(num_episodes: int = 30, steps_per_episode: int = 200) -> None:
    """Run the survival demo (Phase 2).

    An agent must find resources to survive. It starts with no knowledge
    of the world. Survival time should increase as it learns which actions
    lead to energy gain.
    """
    print("=" * 60)
    print("  Phase 2 — Survival Demo")
    print("=" * 60)
    print()
    print("An agent has finite energy that depletes each tick (entropy).")
    print("It must find resources on a 1D grid to survive.")
    print("No drives are hardcoded — all motivation emerges from one")
    print("principle: energy depletes, find more or die.")
    print()
    print(f"Grid: 10 cells | Resources at positions 2, 8 | Agent starts at 5")
    print(f"Actions: left / stay / right")
    print(f"Energy: move costs 0.015, staying costs 0.005, resource gives 0.3")
    print(f"Entropy: -0.01 per tick just for existing")
    print()

    env = SurvivalEnv(
        grid_size=10,
        resource_positions=[2, 8],
        move_cost=0.015,
        stay_cost=0.005,
        resource_value=0.3,
        max_steps=steps_per_episode,
    )

    agent = Agent(similarity_threshold=0.7, seed=42)

    survival_times: list[int] = []
    for episode in range(num_episodes):
        agent.world_model.reset_stats()
        results = agent.run_survival_episode(env, max_steps=steps_per_episode)

        survived = len(results)
        final_energy = results[-1].vitality if results else 0.0
        alive_at_end = agent.vitality.alive
        avg_surprise = sum(r.surprise for r in results) / len(results) if results else 1.0
        survival_times.append(survived)

        # Show positive/negative valence counts
        pos_valences = sum(1 for v in agent.valence._values.values() if v > 0.01)
        neg_valences = sum(1 for v in agent.valence._values.values() if v < -0.01)

        status = "ALIVE" if alive_at_end else "DEAD "
        print(
            f"  Ep {episode + 1:3d}/{num_episodes}: "
            f"[{status}] survived {survived:3d}/{steps_per_episode} steps, "
            f"energy {final_energy:.2f}, "
            f"surprise {avg_surprise:.2f}, "
            f"valence +{pos_valences}/-{neg_valences}"
        )

    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Patterns learned: {agent.pattern_count}")
    print(f"  Associations formed: {agent.association_count}")
    print(f"  Valenced patterns: {agent.valence.known_count}")

    early = survival_times[:5]
    late = survival_times[-5:]
    early_avg = sum(early) / len(early)
    late_avg = sum(late) / len(late)

    print(f"\n  Early avg survival (ep 1-5):   {early_avg:.0f} steps")
    print(f"  Late avg survival  (ep {num_episodes-4}-{num_episodes}): {late_avg:.0f} steps")

    if late_avg > early_avg:
        improvement = ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
        print(f"  The agent learned to survive longer! (+{improvement:.0f}%)")
    else:
        print("  More episodes needed for clear improvement.")

    # Show emergent "drives"
    print("\n  Emergent behaviors (not coded, derived from persistence):")
    print("  - 'Hunger': urgency increases as energy drops -> exploit known resources")
    print("  - 'Fear': negative-valence patterns trigger avoidance")
    print("  - 'Curiosity': exploration increases when energy is high")
    print("  - 'Satisfaction': positive-valence patterns reinforce resource-seeking")
    print()


def run_evolution_demo(
    num_generations: int = 8,
    population_size: int = 10,
    episodes_per_agent: int = 5,
    steps_per_episode: int = 150,
) -> None:
    """Run the evolution demo (Phase 3).

    A population of agents with different genomes compete. The fittest
    survive and reproduce. Parameters are not hand-tuned — they are
    discovered through selection pressure.
    """
    print("=" * 60)
    print("  Phase 3 — Evolution Demo")
    print("=" * 60)
    print()
    print("A population of agents with random genomes compete in the")
    print("same environment. The fittest survive and reproduce. Over")
    print("generations, evolution discovers optimal parameters.")
    print()
    print("Two levels of evolution operating simultaneously:")
    print("  Internal: patterns/associations compete for limited capacity")
    print("  Population: agents compete for survival, fittest reproduce")
    print()
    print(f"Population: {population_size} | Generations: {num_generations} | "
          f"Episodes/agent: {episodes_per_agent}")
    print()

    def make_env():
        return SurvivalEnv(
            grid_size=10,
            resource_positions=[2, 8],
            move_cost=0.015,
            stay_cost=0.005,
            resource_value=0.3,
            max_steps=steps_per_episode,
        )

    pop = Population(
        population_size=population_size,
        episodes_per_agent=episodes_per_agent,
        steps_per_episode=steps_per_episode,
        top_k=3,
        mutation_rate=0.3,
        crossover_prob=0.5,
        env_factory=make_env,
        seed=42,
    )

    history = pop.evolve(num_generations)

    for gen in history:
        best = gen.best_fitness
        avg = gen.avg_fitness
        bar = "#" * int(best / 3)
        print(
            f"  Gen {gen.generation + 1:2d}/{num_generations}: "
            f"best {best:5.1f}, avg {avg:5.1f}  {bar}"
        )

    print()
    print("-" * 60)
    print("Summary:")

    early_best = history[0].best_fitness
    late_best = history[-1].best_fitness
    early_avg = history[0].avg_fitness
    late_avg = history[-1].avg_fitness

    print(f"  Gen 1 — best: {early_best:.0f}, avg: {early_avg:.0f}")
    print(f"  Gen {num_generations} — best: {late_best:.0f}, avg: {late_avg:.0f}")

    if late_best > early_best:
        improvement = ((late_best - early_best) / early_best) * 100 if early_best > 0 else 0
        print(f"  Best fitness improved by {improvement:.0f}%")

    # Show converged parameters
    report = pop.convergence_report()
    print("\n  Evolved genome (best agent's DNA):")
    for name, val in report.items():
        if isinstance(val, float):
            print(f"    {name}: {val:.4f}")
        else:
            print(f"    {name}: {val}")

    print()
    print("  These parameters were not hand-tuned — they were discovered")
    print("  by the same principle that produced biological brains:")
    print("  variation + selection + inheritance.")
    print()


def run_society_demo(
    n_agents: int = 6,
    grid_size: int = 20,
    num_resources: int = 4,
    max_steps: int = 200,
) -> None:
    """Run the society demo (Phase 4).

    Multiple agents inhabit a shared environment. A Society emerges that
    observes their collective state, learns patterns, and acts to allocate
    resources — using the exact same primitives as an individual agent.
    The agents don't know the society exists.
    """
    print("=" * 60)
    print("  Phase 4 — Recursive Intelligence Demo")
    print("=" * 60)
    print()
    print("Multiple agents share an environment with finite resources.")
    print("A Society emerges that observes aggregate behavior, forms")
    print("patterns, and acts — using the SAME primitives as Agent.")
    print("Agents don't know the society exists. It's invisible from below.")
    print()
    print(f"Agents: {n_agents} | Grid: {grid_size} | Resources: {num_resources}")
    print(f"Society senses: regional density + mean vitality")
    print(f"Society acts: resource allocation bias (left / even / right)")
    print()

    # Create agents with evolved diversity
    rng = np.random.default_rng(42)
    agents = []
    for i in range(n_agents):
        genome = Genome.random(rng)
        agent = Agent(seed=int(rng.integers(0, 2**31)), **genome.to_agent_kwargs())
        agents.append(agent)

    env = SharedGridEnv(
        grid_size=grid_size,
        num_resources=num_resources,
        resource_value=0.3,
        resource_regen_rate=0.05,
        move_cost=0.015,
        stay_cost=0.005,
        max_steps=max_steps,
        seed=42,
    )
    bridge = SignalBridge(grid_size=grid_size, n_regions=4)
    society = Society(agents=agents, env=env, bridge=bridge, seed=42)

    # Run the episode
    results = society.run_episode(max_steps=max_steps)

    # Report progress in chunks
    chunk_size = 25
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    print(f"  {'Step':>6s}  {'Alive':>5s}  {'Soc.Vitality':>12s}  {'Surprise':>8s}  {'Action':>6s}")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 12}  {'-' * 8}  {'-' * 6}")

    for chunk in chunks:
        last = chunk[-1]
        avg_surprise = sum(r.surprise for r in chunk) / len(chunk)
        actions = ["left", "even", "right"]
        action_name = actions[last.action] if last.action is not None else "none"
        print(
            f"  {last.tick:6d}  {last.num_alive:5d}  "
            f"{last.collective_vitality:12.3f}  "
            f"{avg_surprise:8.3f}  {action_name:>6s}"
        )

    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Episode length: {len(results)} steps")
    print(f"  Agents alive at end: {results[-1].num_alive}/{n_agents}")
    print(f"  Society patterns learned: {society.pattern_count}")
    print(f"  Society associations formed: {society.association_count}")
    print(f"  Society vitality: {society.vitality.energy:.3f}")

    # Surprise trend
    if len(results) >= 40:
        early = results[:20]
        late = results[-20:]
        early_surprise = sum(r.surprise for r in early) / len(early)
        late_surprise = sum(r.surprise for r in late) / len(late)
        print(f"\n  Early surprise (first 20):  {early_surprise:.3f}")
        print(f"  Late surprise  (last 20):   {late_surprise:.3f}")
        if late_surprise < early_surprise:
            reduction = ((early_surprise - late_surprise) / early_surprise) * 100
            print(f"  Society learned to predict collective behavior! (-{reduction:.0f}% surprise)")
        else:
            print(f"  Surprise stable — collective state is consistent.")

    # The key insight
    print()
    print("  The recursive principle:")
    print("  - Agent uses: Signal -> Pattern -> Association -> Predict -> Act")
    print("  - Society uses: Signal -> Pattern -> Association -> Predict -> Act")
    print("  - Same code. Same primitives. Different scale.")
    print("  - Agents cannot perceive the society. It is invisible from below.")
    print("  - A Society of Societies = Civilization. Same code, higher scale.")
    print()


def run_social_demo(
    n_agents: int = 6,
    grid_size: int = 20,
    num_resources: int = 4,
    max_steps: int = 150,
    perception_radius: int = 5,
) -> None:
    """Run the social intelligence demo (Phase 5).

    Agents involuntarily leak internal state (vitality, surprise, movement
    direction). Other agents perceive these emissions as part of their
    observation — processed by the same WorldModel, no social module.
    """
    print("=" * 60)
    print("  Phase 5 — Embodied Social Intelligence Demo")
    print("=" * 60)
    print()
    print("Agents leak internal state involuntarily (vitality, surprise,")
    print("movement direction). Others perceive these emissions within a")
    print("radius. The same WorldModel processes richer 21-dim signals.")
    print("Social cognition emerges from perception, not from a social module.")
    print()
    print(f"Agents: {n_agents} | Grid: {grid_size} | Resources: {num_resources}")
    print(f"Perception radius: {perception_radius} | Obs dim: 6 (blind) -> 21 (social)")
    print()

    rng = np.random.default_rng(42)
    agents = []
    for _ in range(n_agents):
        agent = Agent(
            similarity_threshold=0.7,
            seed=int(rng.integers(0, 2**31)),
            max_patterns=20,
            max_associations=60,
        )
        agents.append(agent)

    env = SocialGridEnv(
        grid_size=grid_size,
        num_resources=num_resources,
        resource_value=0.3,
        resource_regen_rate=0.05,
        move_cost=0.015,
        stay_cost=0.005,
        max_steps=max_steps,
        seed=42,
        perception_radius=perception_radius,
    )
    bridge = SignalBridge(grid_size=grid_size, n_regions=4)
    society = SocialSociety(agents=agents, env=env, bridge=bridge, seed=42)

    results = society.run_episode(max_steps=max_steps)

    # Report in chunks
    chunk_size = 25
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    print(f"  {'Step':>6s}  {'Alive':>5s}  {'Mean Vit':>8s}  {'Surprise':>8s}  "
          f"{'Social%':>7s}  {'Action':>6s}")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 6}")

    for chunk in chunks:
        last = chunk[-1]
        avg_surprise = sum(r.surprise for r in chunk) / len(chunk)

        # Compute social%: how many agents are within perception_radius of another
        positions = env.agent_positions
        alive_ids = [i for i, a in enumerate(agents) if a.vitality.alive]
        social_count = 0
        for aid in alive_ids:
            pos = positions.get(aid)
            if pos is None:
                continue
            for other_id in alive_ids:
                if other_id == aid:
                    continue
                other_pos = positions.get(other_id)
                if other_pos is not None and abs(pos - other_pos) <= perception_radius:
                    social_count += 1
                    break
        social_pct = (social_count / max(len(alive_ids), 1)) * 100

        actions = ["left", "even", "right"]
        action_name = actions[last.action] if last.action is not None else "none"
        print(
            f"  {last.tick:6d}  {last.num_alive:5d}  "
            f"{last.mean_vitality:8.3f}  "
            f"{avg_surprise:8.3f}  "
            f"{social_pct:6.0f}%  {action_name:>6s}"
        )

    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Episode length: {len(results)} steps")
    print(f"  Agents alive at end: {results[-1].num_alive}/{n_agents}")
    print(f"  Society patterns learned: {society.pattern_count}")
    print(f"  Society associations formed: {society.association_count}")

    # Per-agent pattern counts
    for i, agent in enumerate(agents):
        status = "alive" if agent.vitality.alive else "dead"
        print(f"  Agent {i} ({status}): {agent.world_model.memory.pattern_count} patterns")

    # Surprise trend
    if len(results) >= 40:
        early = results[:20]
        late = results[-20:]
        early_surprise = sum(r.surprise for r in early) / len(early)
        late_surprise = sum(r.surprise for r in late) / len(late)
        print(f"\n  Early surprise (first 20):  {early_surprise:.3f}")
        print(f"  Late surprise  (last 20):   {late_surprise:.3f}")
        if late_surprise < early_surprise:
            reduction = ((early_surprise - late_surprise) / early_surprise) * 100
            print(f"  Society learned to predict social behavior! (-{reduction:.0f}% surprise)")

    print()
    print("  What's happening:")
    print("  - Agents leak vitality, surprise, and direction involuntarily")
    print("  - Nearby agents perceive these as part of their 21-dim observation")
    print("  - The WorldModel learns patterns from richer signals")
    print("  - No social module was added — same code, richer input")
    print("  - This is Step 1 (involuntary emission) + Step 2 (contingency detection)")
    print("  - Step 3 (instrumental control of emissions) comes next")
    print()


def run_proprioceptive_demo(
    n_agents: int = 6,
    grid_size: int = 25,
    num_resources: int = 2,
    max_steps: int = 200,
    perception_radius: int = 5,
) -> None:
    """Run the proprioceptive control demo (Phase 10, Step 3).

    Agents perceive their own leaked state (proprioceptive feedback).
    This enables pattern-level differentiation of own emission state,
    the foundation for instrumental control.

    Harder environment than Phase 9 — fewer resources, slower regen.
    """
    print("=" * 60)
    print("  Phase 10 — Proprioceptive Control Demo")
    print("=" * 60)
    print()
    print("Agents now perceive their OWN leaked state (proprioception).")
    print("Observation grows from 21 to 32 dims. The WorldModel forms")
    print("distinct patterns for 'I am sick near a healthy neighbor' vs")
    print("'I am healthy near a sick neighbor.' This is the foundation")
    print("for instrumental control of emissions.")
    print()
    print(f"Agents: {n_agents} | Grid: {grid_size} | Resources: {num_resources}")
    print(f"Perception radius: {perception_radius} | Obs dim: 32 (proprioceptive)")
    print(f"Harder env: fewer resources, slower regen rate (0.02)")
    print()

    rng = np.random.default_rng(42)
    agents = []
    for _ in range(n_agents):
        agent = Agent(
            similarity_threshold=0.6,
            seed=int(rng.integers(0, 2**31)),
            max_patterns=20,
            max_associations=60,
        )
        agents.append(agent)

    env = SocialGridEnv(
        grid_size=grid_size,
        num_resources=num_resources,
        resource_value=0.35,
        resource_regen_rate=0.02,
        move_cost=0.015,
        stay_cost=0.005,
        max_steps=max_steps,
        seed=42,
        perception_radius=perception_radius,
        include_self_emission=True,
    )
    bridge = SignalBridge(grid_size=grid_size, n_regions=4)
    society = SocialSociety(agents=agents, env=env, bridge=bridge, seed=42)

    results = society.run_episode(max_steps=max_steps)

    # Report in chunks
    chunk_size = 25
    chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

    print(f"  {'Step':>6s}  {'Alive':>5s}  {'Mean Vit':>8s}  {'Surprise':>8s}  {'Action':>6s}")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 6}")

    for chunk in chunks:
        last = chunk[-1]
        avg_surprise = sum(r.surprise for r in chunk) / len(chunk)
        actions = ["left", "even", "right"]
        action_name = actions[last.action] if last.action is not None else "none"
        print(
            f"  {last.tick:6d}  {last.num_alive:5d}  "
            f"{last.mean_vitality:8.3f}  "
            f"{avg_surprise:8.3f}  {action_name:>6s}"
        )

    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Episode length: {len(results)} steps")
    print(f"  Agents alive at end: {results[-1].num_alive}/{n_agents}")
    print(f"  Society patterns learned: {society.pattern_count}")

    # Per-agent pattern counts — 32-dim should produce more patterns than 21-dim
    for i, agent in enumerate(agents):
        status = "alive" if agent.vitality.alive else "dead"
        print(f"  Agent {i} ({status}): {agent.world_model.memory.pattern_count} patterns")

    total_patterns = sum(a.world_model.memory.pattern_count for a in agents)
    print(f"\n  Total patterns across agents: {total_patterns}")
    print(f"  (Compare to ~23 blind, ~39 social from Phase 9)")

    print()
    print("  What's new:")
    print("  - Agents see their OWN leaked state (dims [21:32])")
    print("  - Same Gaussian encoding as other-perception — no special module")
    print("  - One-tick delay: proprioception has latency, like real biology")
    print("  - WorldModel forms distinct patterns for different self-states")
    print("  - This is Step 3: instrumental control of emissions")
    print("  - Next: Step 4 (joint attention) — two agents attending to same thing")
    print()


def run_social_facilitation_2d_demo(
    n_agents: int = 6,
    grid_size: int = 15,
    num_resources: int = 4,
    num_episodes: int = 10,
    max_steps: int = 200,
    perception_radius: int = 5,
) -> None:
    """Run the social facilitation demo on a 2D grid (Phase 10, Step 4).

    2D direction carries real information: if A moves east and finds
    a resource, the cluster extends further east. Agent persistence
    allows learning to accumulate across episodes.
    """
    print("=" * 60)
    print("  Phase 10 — Social Facilitation Demo (2D Grid)")
    print("=" * 60)
    print()
    print("2D grid fixes the 1D information bottleneck. Direction now")
    print("carries real information (4 options, not 2). Agents persist")
    print("across episodes: WorldModel/Valence accumulate experience.")
    print()
    print(f"Agents: {n_agents} | Grid: {grid_size}x{grid_size} | "
          f"Resources: {num_resources}")
    print(f"Perception radius: {perception_radius} | Obs dim: 44 (proprioceptive)")
    print(f"Episodes: {num_episodes} | Clustering: regen=0.002, cluster_prob=0.8")
    print()

    rng = np.random.default_rng(42)
    agents = []
    for _ in range(n_agents):
        agent = Agent(
            similarity_threshold=0.6,
            seed=int(rng.integers(0, 2**31)),
            max_patterns=20,
            max_associations=60,
        )
        agents.append(agent)

    survival_times = []

    for episode in range(num_episodes):
        # Reset agents (preserve WorldModel/Valence)
        for agent in agents:
            agent.vitality = Vitality()
            agent._tick = 0
            agent.history = []
            agent.world_model.reset_stats()

        env = SocialGrid2DEnv(
            grid_size=grid_size,
            num_resources=num_resources,
            resource_value=0.4,
            resource_regen_rate=0.002,
            resource_cluster_prob=0.8,
            move_cost=0.015,
            stay_cost=0.005,
            max_steps=max_steps,
            seed=episode + 42,
            perception_radius=perception_radius,
            include_self_emission=True,
        )

        # Register agents
        for idx in range(n_agents):
            obs = env.register_agent(idx)
            agents[idx].step_with_action(obs, 0.0, None)
            env.update_agent_state(
                idx, agents[idx].vitality.energy,
                agents[idx].world_model.last_surprise,
            )

        # Run episode
        ticks = 0
        for tick in range(max_steps):
            alive = [i for i in range(n_agents) if agents[i].vitality.alive]
            if not alive:
                break

            rng_ep = np.random.default_rng((episode + 42) * 10000 + tick)
            order = list(alive)
            rng_ep.shuffle(order)

            for idx in order:
                action = agents[idx].select_action(env.action_space)
                obs, delta, done = env.step_agent(idx, action)
                agents[idx].step_with_action(obs, delta, action)
                env.update_agent_state(
                    idx, agents[idx].vitality.energy,
                    agents[idx].world_model.last_surprise,
                )

            env.tick()
            ticks += 1
            if done:
                break

        survival_times.append(ticks)
        alive_at_end = sum(1 for a in agents if a.vitality.alive)
        total_patterns = sum(a.world_model.memory.pattern_count for a in agents)
        print(f"  Ep {episode + 1:2d}/{num_episodes}: "
              f"survived {ticks:3d} steps, "
              f"alive {alive_at_end}/{n_agents}, "
              f"patterns {total_patterns}")

    print()
    print("-" * 60)
    print("Summary:")
    early = survival_times[:3]
    late = survival_times[-3:]
    print(f"  Early avg survival (ep 1-3):   {sum(early)/len(early):.0f} steps")
    print(f"  Late avg survival  (ep {num_episodes-2}-{num_episodes}): {sum(late)/len(late):.0f} steps")
    total_patterns = sum(a.world_model.memory.pattern_count for a in agents)
    print(f"  Total patterns (accumulated): {total_patterns}")

    if sum(late)/len(late) > sum(early)/len(early):
        print("  Agents improved with persistence! Learning accumulates across episodes.")
    print()


def run_patch_foraging_demo(
    n_agents: int = 6,
    num_patches: int = 8,
    num_episodes: int = 10,
    max_steps: int = 200,
) -> None:
    """Run the patch foraging demo (Phase 10, Step 4).

    Non-spatial environment: agents choose from N patches. Some are rich
    (high resource probability), others are poor. Social agents can see
    which patch another agent chose and whether they were rewarded.
    """
    print("=" * 60)
    print("  Phase 10 — Patch Foraging Demo")
    print("=" * 60)
    print()
    print("Agents choose from 8 discrete patches each tick. 2 patches")
    print("are 'rich' (50% chance of resource), 6 are 'poor' (5%).")
    print("Social agents see which patch others chose + their vitality.")
    print("Agent persistence: WorldModel accumulates across episodes.")
    print()
    print(f"Agents: {n_agents} | Patches: {num_patches} | Episodes: {num_episodes}")
    print(f"Obs dim: 32 (proprioceptive = 8 patch + 16 social + 8 self)")
    print()

    rng = np.random.default_rng(42)
    agents = []
    for _ in range(n_agents):
        agent = Agent(
            similarity_threshold=0.6,
            seed=int(rng.integers(0, 2**31)),
            max_patterns=20,
            max_associations=60,
        )
        agents.append(agent)

    survival_times = []

    for episode in range(num_episodes):
        # Reset agents (preserve WorldModel/Valence)
        for agent in agents:
            agent.vitality = Vitality()
            agent._tick = 0
            agent.history = []
            agent.world_model.reset_stats()

        env = PatchForagingEnv(
            num_patches=num_patches,
            num_rich=2,
            rich_prob=0.5,
            poor_prob=0.05,
            resource_value=0.4,
            visit_cost=0.01,
            max_steps=max_steps,
            seed=episode + 42,
            include_social=True,
            include_self_emission=True,
        )

        # Register agents
        for idx in range(n_agents):
            obs = env.register_agent(idx)
            agents[idx].step_with_action(obs, 0.0, None)
            env.update_agent_state(
                idx, agents[idx].vitality.energy,
                agents[idx].world_model.last_surprise,
            )

        # Run episode
        ticks = 0
        total_finds = 0
        for tick in range(max_steps):
            alive = [i for i in range(n_agents) if agents[i].vitality.alive]
            if not alive:
                break

            rng_ep = np.random.default_rng((episode + 42) * 10000 + tick)
            order = list(alive)
            rng_ep.shuffle(order)

            for idx in order:
                action = agents[idx].select_action(env.action_space)
                obs, delta, done = env.step_agent(idx, action)
                agents[idx].step_with_action(obs, delta, action)
                env.update_agent_state(
                    idx, agents[idx].vitality.energy,
                    agents[idx].world_model.last_surprise,
                )
                if delta > 0:
                    total_finds += 1

            env.tick()
            ticks += 1
            if done:
                break

        survival_times.append(ticks)
        alive_at_end = sum(1 for a in agents if a.vitality.alive)
        total_patterns = sum(a.world_model.memory.pattern_count for a in agents)
        print(f"  Ep {episode + 1:2d}/{num_episodes}: "
              f"survived {ticks:3d} steps, "
              f"alive {alive_at_end}/{n_agents}, "
              f"patterns {total_patterns}, "
              f"finds {total_finds}")

    print()
    print("-" * 60)
    print("Summary:")
    early = survival_times[:3]
    late = survival_times[-3:]
    print(f"  Early avg survival (ep 1-3):   {sum(early)/len(early):.0f} steps")
    print(f"  Late avg survival  (ep {num_episodes-2}-{num_episodes}): {sum(late)/len(late):.0f} steps")
    total_patterns = sum(a.world_model.memory.pattern_count for a in agents)
    print(f"  Total patterns (accumulated): {total_patterns}")

    if sum(late)/len(late) > sum(early)/len(early):
        print("  Agents improved with persistence! Learning accumulates across episodes.")

    print()
    print("  What's new:")
    print("  - Non-spatial environment: pure social learning test")
    print("  - A's patch choice directly observable by B (no spatial noise)")
    print("  - Rich patches have 10x the resource probability of poor ones")
    print("  - Social agents should learn 'if A just got rewarded at patch P,")
    print("    then P is rich — go there too'")
    print("  - Same Agent, same WorldModel — only the observation changes")
    print()


def main() -> None:
    run_prediction_demo()
    print("\n")
    run_survival_demo()
    print("\n")
    run_evolution_demo()
    print("\n")
    run_society_demo()
    print("\n")
    run_social_demo()
    print("\n")
    run_proprioceptive_demo()
    print("\n")
    run_social_facilitation_2d_demo()
    print("\n")
    run_patch_foraging_demo()


if __name__ == "__main__":
    main()
