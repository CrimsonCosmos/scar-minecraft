"""Runner — main loop for FPI agent in Minecraft.

Connects to a Minecraft server via the Mineflayer bridge and runs
the FPI agent's sense-predict-act loop. Knowledge persists across
deaths. The agent learns from scratch to survive and earn XP.

Usage:
    python -m fpi.minecraft.runner [--host localhost] [--port 3001]
                                   [--steps 10000] [--phase 1]
                                   [--until-converge]
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

from ..agent.core import Agent
from ..primitives.vitality import Vitality
from .env import MinecraftEnv


class ConvergenceDetector:
    """Detects when the agent has stopped improving.

    Tracks kill/death ratio and average surprise over a sliding window.
    Declares convergence when neither metric improves for `patience` steps.
    """

    def __init__(self, patience: int = 2000, window: int = 500):
        self.patience = patience
        self.window = window
        self._surprise_history: deque[float] = deque(maxlen=window)
        self._kd_history: deque[float] = deque(maxlen=window)
        self._best_surprise = float("inf")
        self._best_kd = 0.0
        self._steps_without_improvement = 0

    def update(self, avg_surprise: float, kills: int, deaths: int) -> bool:
        """Record metrics. Returns True if converged (should stop)."""
        kd = kills / max(deaths, 1)
        self._surprise_history.append(avg_surprise)
        self._kd_history.append(kd)

        # Need at least a full window before checking
        if len(self._surprise_history) < self.window:
            return False

        current_surprise = sum(self._surprise_history) / len(self._surprise_history)
        current_kd = sum(self._kd_history) / len(self._kd_history)

        improved = False
        if current_surprise < self._best_surprise - 0.005:
            self._best_surprise = current_surprise
            improved = True
        if current_kd > self._best_kd + 0.1:
            self._best_kd = current_kd
            improved = True

        if improved:
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        return self._steps_without_improvement >= self.patience

    @property
    def steps_without_improvement(self) -> int:
        return self._steps_without_improvement


def run_minecraft(
    host: str = "localhost",
    port: int = 3001,
    max_steps: int = 10000,
    phase: int = 1,
    report_interval: int = 50,
    seed: int = 42,
    until_converge: bool = False,
    transfer_path: str | None = None,
    factored: bool = False,
    save_path: str | None = None,
) -> None:
    """Run FPI agent in Minecraft.

    The agent loops: observe -> decide -> act, learning from experience.
    Knowledge (patterns, associations, valence) persists across deaths.
    Only vitality is reset on death.

    Args:
        host: Mineflayer bridge host.
        port: Mineflayer bridge port.
        max_steps: Total number of observe-act cycles to run.
        phase: Action space phase (1 = movement, 2 = full).
        report_interval: Print status every N steps.
        seed: Random seed for reproducibility.
        until_converge: Run until convergence detected (ignores max_steps).
        transfer_path: Load pre-trained agent state from this pickle file.
    """
    env = MinecraftEnv(host=host, port=port, phase=phase, factored=factored)

    # Factored action space (168 actions): reduce lookahead depth to stay fast.
    lookahead_depth = 3 if factored else 5

    agent = Agent(
        similarity_threshold=0.80,
        seed=seed,
        exploration_base=0.12,
        # Compositional patterns: per-modality matching gives exponential capacity.
        # 16 patterns/modality × 7 modalities = 112 base patterns,
        # but up to 16^7 ≈ 268M distinguishable situations (only observed combos stored).
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
        # Cognitive fixes: lookahead, eligibility traces, goal-directed exploration
        enable_lookahead=True,
        lookahead_depth=lookahead_depth,
        lookahead_discount=0.9,
        enable_eligibility_traces=True,
        trace_decay=0.8,
        discount_factor=0.95,
        # Recursive cognition: abstraction layer for generalization
        enable_abstraction=True,
    )

    # Lower entropy rate for longer exploration window
    agent.vitality = Vitality(entropy_rate=0.001)

    # Load pre-trained state if provided
    if transfer_path:
        import pickle
        print(f"[fpi-minecraft] Loading pre-trained state from {transfer_path}...")
        with open(transfer_path, "rb") as f:
            saved = pickle.load(f)
        agent.world_model = saved["world_model"]
        agent.valence = saved["valence"]
        if "options" in saved and agent._option_executor is not None:
            for opt in saved["options"]:
                agent._option_executor.add_option(opt)
        print(f"[fpi-minecraft] Loaded {len(agent.world_model.memory.distinction.patterns)} patterns, "
              f"{agent.world_model.memory.association_count} associations.")

    convergence = ConvergenceDetector(patience=2000, window=500) if until_converge else None
    effective_max = 1_000_000 if until_converge else max_steps

    print(f"[fpi-minecraft] Connecting to bridge at {host}:{port}...")
    obs = env.reset()
    print(f"[fpi-minecraft] Connected. Phase {phase}, {len(env.action_space)} actions.")
    if until_converge:
        print("[fpi-minecraft] Running until convergence (patience=2000 steps)...")
    else:
        print(f"[fpi-minecraft] Running for {max_steps} steps...")

    # First step: observe only, no action
    result = agent.step_with_action(obs, 0.0, None)

    for step in range(1, effective_max):
        # Respawn: reset vitality, keep all learned knowledge
        if not agent.vitality.alive:
            agent.vitality = Vitality(entropy_rate=0.001)

        # Decide
        action = agent.select_action(env.action_space)

        # Act and observe (with error recovery)
        try:
            obs, energy_delta, done = env.step(action)
        except Exception as exc:
            print(f"[step {step}] Bridge error: {exc}. Reconnecting...")
            import time
            for retry in range(5):
                try:
                    time.sleep(3)
                    obs = env.reset()
                    energy_delta = -0.1  # Mild penalty for disruption
                    print(f"[step {step}] Reconnected after {retry + 1} attempts.")
                    break
                except Exception:
                    pass
            else:
                print("[fpi-minecraft] Reconnect failed after 5 attempts. Exiting.")
                break
            continue

        # Learn
        result = agent.step_with_action(obs, energy_delta, action)

        # Consolidate: replay important episodes when safe
        agent.consolidate()

        # Periodic reporting
        if step % report_interval == 0:
            avg_surprise = agent.average_surprise
            pattern_count = len(agent.world_model.memory.distinction.patterns)
            assoc_count = agent.world_model.memory.association_count

            # Count valenced patterns
            positive_valence = sum(
                1 for pid in agent.valence._values
                if agent.valence.get(pid) > 0.01
            )
            negative_valence = sum(
                1 for pid in agent.valence._values
                if agent.valence.get(pid) < -0.01
            )

            print(
                f"[step {step:>6}] "
                f"vitality={result.vitality:.3f}  "
                f"surprise={result.surprise:.2f}  "
                f"avg_surprise={avg_surprise:.3f}  "
                f"patterns={pattern_count}  "
                f"assocs={assoc_count}  "
                f"valence=+{positive_valence}/-{negative_valence}  "
                f"kills={env.kill_count}  "
                f"deaths={env.death_count}  "
                f"urgency={agent.vitality.urgency:.2f}"
            )

            # Check convergence
            if convergence is not None:
                converged = convergence.update(avg_surprise, env.kill_count, env.death_count)
                if converged:
                    print(
                        f"\n[fpi-minecraft] CONVERGED at step {step}. "
                        f"No improvement for {convergence.patience} steps."
                    )
                    break

    # Summary
    print("\n[fpi-minecraft] Session complete.")
    print(f"  Total steps: {env.step_count}")
    print(f"  Deaths: {env.death_count}")
    print(f"  Kills: {env.kill_count}")
    print(f"  Kill/Death ratio: {env.kill_count / max(env.death_count, 1):.2f}")
    print(f"  Patterns learned: {len(agent.world_model.memory.distinction.patterns)}")
    print(f"  Average surprise: {agent.average_surprise:.4f}")
    if convergence is not None:
        print(f"  Best avg_surprise: {convergence._best_surprise:.4f}")
        print(f"  Best K/D ratio: {convergence._best_kd:.2f}")

    if save_path:
        import pickle
        print(f"[fpi-minecraft] Saving agent state to {save_path}...")
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
        print("[fpi-minecraft] Saved.")

    env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run FPI agent in Minecraft via Mineflayer bridge",
    )
    parser.add_argument("--host", default="localhost", help="Bridge host")
    parser.add_argument("--port", type=int, default=3001, help="Bridge port")
    parser.add_argument("--steps", type=int, default=10000, help="Max steps")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Action phase")
    parser.add_argument("--report", type=int, default=50, help="Report interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--until-converge", action="store_true",
        help="Run until learning plateaus (ignores --steps)",
    )
    parser.add_argument(
        "--transfer", type=str, default=None,
        help="Load pre-trained agent state from pickle file",
    )
    parser.add_argument(
        "--factored", action="store_true",
        help="Use factored action space (168 = 7 move × 6 look × 4 combat)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save agent state to pickle file on exit",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Load server config from JSON file (overrides --host/--port/--phase)",
    )

    args = parser.parse_args()

    # Config file overrides CLI args for host/port/phase
    host = args.host
    port = args.port
    phase = args.phase
    if args.config:
        import json
        from pathlib import Path
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"[fpi-minecraft] Config not found: {cfg_path}")
            sys.exit(1)
        with open(cfg_path) as f:
            cfg = json.load(f)
        port = cfg.get("bridgePort", port)
        phase = cfg.get("phase", phase)
        print(f"[fpi-minecraft] Loaded config from {cfg_path}")

    try:
        run_minecraft(
            host=host,
            port=port,
            max_steps=args.steps,
            phase=phase,
            report_interval=args.report,
            seed=args.seed,
            until_converge=args.until_converge,
            transfer_path=args.transfer,
            factored=args.factored,
            save_path=args.save,
        )
    except KeyboardInterrupt:
        print("\n[fpi-minecraft] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
