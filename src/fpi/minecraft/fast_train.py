"""Fast pre-training via CombatSimulator.

Runs the FPI agent through 100K+ combat steps per second in a pure-Python
2D arena. The agent learns combat patterns (approach, attack, retreat,
knockback, sprint-crit, w-tap) that transfer directly to Minecraft via
shared signal format.

Supports curriculum training: progressive difficulty stages that teach
the agent increasingly advanced combat skills.

Usage:
    python -m fpi.minecraft.fast_train [--steps 500000] [--report 10000]
    python -m fpi.minecraft.fast_train --steps 1000000 --save pretrained.pkl
    python -m fpi.minecraft.fast_train --curriculum --steps 2000000 --save pretrained.pkl
    python -m fpi.minecraft.fast_train --transfer pretrained.pkl  # load + continue in sim
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

from ..agent.core import Agent
from ..primitives.vitality import Vitality
from .combat_sim import CURRICULUM_STAGES, CombatSimulator
from .env import MinecraftEnv


def _create_agent(seed: int = 42, phase: int = 3, factored: bool = False) -> Agent:
    """Create an FPI agent configured for combat training."""
    lookahead_depth = 3 if factored else 5
    agent = Agent(
        similarity_threshold=0.80,
        seed=seed,
        exploration_base=0.12,
        enable_compositional=True,
        patterns_per_modality=16,
        modality_slices=MinecraftEnv.MODALITY_SLICES,
        modality_thresholds=MinecraftEnv.MODALITY_THRESHOLDS,
        composite_similarity_threshold=0.5,
        max_associations=5000,
        association_decay_rate=0.002,
        maintenance_cost_per_pattern=0.00025,
        maintenance_cost_per_association=0.0001,
        enable_salience=True,
        enable_options=True,
        enable_episodic_memory=True,
        episodic_capacity=500,
        episodic_surprise_threshold=0.4,
        enable_lookahead=True,
        lookahead_depth=lookahead_depth,
        lookahead_discount=0.9,
        enable_eligibility_traces=True,
        trace_decay=0.8,
        discount_factor=0.95,
        enable_abstraction=True,
    )
    agent.vitality = Vitality(entropy_rate=0.001)
    return agent


def run_fast_train(
    steps: int = 500_000,
    report_interval: int = 10_000,
    seed: int = 42,
    save_path: str | None = None,
    transfer_path: str | None = None,
    max_mobs: int = 3,
    mob_speed: float = 0.08,
    curriculum: bool = False,
    phase: int = 3,
    factored: bool = False,
) -> Agent:
    """Run fast pre-training in the combat simulator.

    Args:
        steps: Total training steps.
        report_interval: Print metrics every N steps.
        seed: Random seed.
        save_path: If set, save agent state to this pickle file.
        transfer_path: If set, load agent state from this pickle before training.
        max_mobs: Mobs in the arena (ignored if curriculum=True).
        mob_speed: Mob movement speed (ignored if curriculum=True).
        curriculum: Use progressive difficulty stages.
        phase: Action space phase (1=13 actions, 2=18, 3=20 with combat combos).

    Returns:
        The trained Agent.
    """
    agent = _create_agent(seed=seed, phase=phase, factored=factored)

    # Load pre-trained state if provided
    if transfer_path:
        print(f"[fast-train] Loading agent state from {transfer_path}...")
        with open(transfer_path, "rb") as f:
            saved = pickle.load(f)
        agent.world_model = saved["world_model"]
        agent.valence = saved["valence"]
        if "options" in saved and agent._option_executor is not None:
            for opt in saved["options"]:
                agent._option_executor.add_option(opt)
        print(f"[fast-train] Loaded {len(agent.world_model.memory.distinction.patterns)} patterns, "
              f"{agent.world_model.memory.association_count} associations.")

    if curriculum:
        print(f"[fast-train] CURRICULUM MODE: {steps:,} steps across {len(CURRICULUM_STAGES)} stages")
        _run_curriculum(agent, steps, report_interval, seed, phase, factored)
    else:
        env = CombatSimulator(
            arena_size=64.0,
            max_mobs=max_mobs,
            mob_speed=mob_speed,
            seed=seed,
            phase=phase,
            factored=factored,
        )
        action_label = "168 factored" if factored else str(len(env.action_space))
        print(f"[fast-train] Starting {steps:,} step combat simulation...")
        print(f"[fast-train] Arena: 64x64, mobs: {max_mobs}, speed: {mob_speed}, actions: {action_label}")
        _run_stage(agent, env, steps, report_interval)

    if save_path:
        print(f"[fast-train] Saving agent state to {save_path}...")
        options_list = []
        if agent._option_executor is not None:
            options_list = agent._option_executor._options
        saved = {
            "world_model": agent.world_model,
            "valence": agent.valence,
            "options": options_list,
        }
        with open(save_path, "wb") as f:
            pickle.dump(saved, f)
        print(f"[fast-train] Saved.")

    return agent


def _run_curriculum(
    agent: Agent,
    total_steps: int,
    report_interval: int,
    seed: int,
    phase: int,
    factored: bool = False,
) -> None:
    """Run curriculum training through all stages."""
    steps_per_stage = total_steps // len(CURRICULUM_STAGES)

    for stage_num in sorted(CURRICULUM_STAGES.keys()):
        stage = CURRICULUM_STAGES[stage_num]
        print(f"\n[fast-train] === STAGE {stage_num}: {stage['name']} ===")
        print(f"  Mobs: {stage['mob_types']}, Speed: {stage.get('speed', 0.08)}, "
              f"Damage: {stage.get('damage', 3.0)}")

        env = CombatSimulator(
            arena_size=64.0,
            seed=seed + stage_num,
            curriculum_stage=stage_num,
            phase=phase,
            factored=factored,
        )
        _run_stage(agent, env, steps_per_stage, report_interval)


def _run_stage(
    agent: Agent,
    env: CombatSimulator,
    steps: int,
    report_interval: int,
) -> None:
    """Run a single training stage."""
    obs = env.reset()
    result = agent.step_with_action(obs, 0.0, None)

    t0 = time.monotonic()
    last_report_time = t0

    for step in range(1, steps + 1):
        if not agent.vitality.alive:
            agent.vitality = Vitality(entropy_rate=0.001)

        action = agent.select_action(env.action_space)
        obs, energy_delta, done = env.step(action)
        result = agent.step_with_action(obs, energy_delta, action)

        # Consolidate periodically
        if step % 10 == 0:
            agent.consolidate()

        if step % report_interval == 0:
            now = time.monotonic()
            elapsed = now - last_report_time
            sps = report_interval / elapsed if elapsed > 0 else 0
            total_elapsed = now - t0

            pattern_count = len(agent.world_model.memory.distinction.patterns)
            assoc_count = agent.world_model.memory.association_count
            option_count = 0
            if agent._option_executor is not None:
                option_count = agent._option_executor.option_count

            print(
                f"[step {step:>8,}] "
                f"{sps:,.0f} sps  "
                f"surprise={agent.average_surprise:.3f}  "
                f"patterns={pattern_count}  "
                f"assocs={assoc_count}  "
                f"options={option_count}  "
                f"kills={env.kill_count}  "
                f"deaths={env.death_count}  "
                f"K/D={env.kill_count / max(env.death_count, 1):.2f}  "
                f"elapsed={total_elapsed:.1f}s"
            )
            last_report_time = now

    total_time = time.monotonic() - t0
    avg_sps = steps / total_time if total_time > 0 else 0
    print(f"\n[fast-train] Stage complete: {steps:,} steps in {total_time:.1f}s ({avg_sps:,.0f} sps)")
    print(f"  Kills: {env.kill_count}, Deaths: {env.death_count}, "
          f"K/D: {env.kill_count / max(env.death_count, 1):.2f}")
    print(f"  Patterns: {len(agent.world_model.memory.distinction.patterns)}")
    print(f"  Avg surprise: {agent.average_surprise:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast pre-training via CombatSimulator",
    )
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps")
    parser.add_argument("--report", type=int, default=10_000, help="Report interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None, help="Save agent state to pickle")
    parser.add_argument("--transfer", type=str, default=None, help="Load agent state from pickle")
    parser.add_argument("--mobs", type=int, default=3, help="Max mobs")
    parser.add_argument("--mob-speed", type=float, default=0.08, help="Mob speed")
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Use curriculum training (progressive difficulty stages)",
    )
    parser.add_argument(
        "--phase", type=int, default=3, choices=[1, 2, 3],
        help="Action space phase (3 = includes combat combos)",
    )
    parser.add_argument(
        "--factored", action="store_true",
        help="Use factored action space (168 = 7 move × 6 look × 4 combat)",
    )

    args = parser.parse_args()

    try:
        run_fast_train(
            steps=args.steps,
            report_interval=args.report,
            seed=args.seed,
            save_path=args.save,
            transfer_path=args.transfer,
            max_mobs=args.mobs,
            mob_speed=args.mob_speed,
            curriculum=args.curriculum,
            phase=args.phase,
            factored=args.factored,
        )
    except KeyboardInterrupt:
        print("\n[fast-train] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
