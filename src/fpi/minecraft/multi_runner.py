"""MultiServerRunner — run FPI agents across multiple servers in parallel.

Launches one agent thread per server config. Each instance gets its own
MinecraftEnv and Agent. All agents share the same brain configuration
but learn independently.

Usage:
    python -m fpi.minecraft.multi_runner --config servers.json [--steps 10000]

Config format (servers.json):
    [
      {"bridgePort": 3001, "label": "java-local", "phase": 1},
      {"bridgePort": 3002, "label": "bedrock-realm", "phase": 1}
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from ..agent.core import Agent
from ..primitives.vitality import Vitality
from .env import MinecraftEnv


def _create_agent(seed: int = 42) -> Agent:
    """Create an FPI agent with standard Minecraft configuration."""
    agent = Agent(
        similarity_threshold=0.80,
        seed=seed,
        exploration_base=0.12,
        enable_compositional=True,
        patterns_per_modality=16,
        modality_slices=MinecraftEnv.MODALITY_SLICES,
        max_associations=5000,
        association_decay_rate=0.002,
        maintenance_cost_per_pattern=0.001,
        maintenance_cost_per_association=0.0003,
        enable_salience=True,
        enable_options=True,
        enable_episodic_memory=True,
        episodic_capacity=500,
        episodic_surprise_threshold=0.4,
        enable_lookahead=True,
        lookahead_depth=5,
        lookahead_discount=0.9,
        enable_eligibility_traces=True,
        trace_decay=0.8,
        discount_factor=0.95,
        enable_abstraction=True,
    )
    agent.vitality = Vitality(entropy_rate=0.001)
    return agent


def _run_one(
    env: MinecraftEnv,
    agent: Agent,
    max_steps: int,
    label: str,
    report_interval: int = 50,
) -> None:
    """Run a single agent instance (called in a thread)."""
    print(f"[{label}] Connecting to bridge at port {env._bridge._port}...")
    try:
        obs = env.reset()
    except Exception as exc:
        print(f"[{label}] Failed to connect: {exc}")
        return

    print(f"[{label}] Connected. Running for {max_steps} steps...")

    # First step: observe only
    result = agent.step_with_action(obs, 0.0, None)

    for step in range(1, max_steps):
        if not agent.vitality.alive:
            agent.vitality = Vitality(entropy_rate=0.001)

        action = agent.select_action(env.action_space)

        try:
            obs, energy_delta, done = env.step(action)
        except Exception as exc:
            print(f"[{label}][step {step}] Bridge error: {exc}. Reconnecting...")
            import time
            for retry in range(5):
                try:
                    time.sleep(3)
                    obs = env.reset()
                    energy_delta = -0.1
                    print(f"[{label}][step {step}] Reconnected after {retry + 1} attempts.")
                    break
                except Exception:
                    pass
            else:
                print(f"[{label}] Reconnect failed. Stopping.")
                break
            continue

        result = agent.step_with_action(obs, energy_delta, action)
        agent.consolidate()

        if step % report_interval == 0:
            pattern_count = len(agent.world_model.memory.distinction.patterns)
            assoc_count = agent.world_model.memory.association_count
            print(
                f"[{label}][step {step:>6}] "
                f"vitality={result.vitality:.3f}  "
                f"surprise={result.surprise:.2f}  "
                f"patterns={pattern_count}  "
                f"kills={env.kill_count}  "
                f"deaths={env.death_count}"
            )

    print(f"\n[{label}] Session complete. Steps={env.step_count} "
          f"Kills={env.kill_count} Deaths={env.death_count}")
    env.close()


class MultiServerRunner:
    """Run FPI agents across multiple servers in parallel threads."""

    def __init__(self, configs: list[dict], seed: int = 42):
        self.instances: list[tuple[MinecraftEnv, Agent, str]] = []
        for i, cfg in enumerate(configs):
            port = cfg.get("bridgePort", 3001 + i)
            phase = cfg.get("phase", 1)
            label = cfg.get("label", f"server-{i}")
            env = MinecraftEnv(host="localhost", port=port, phase=phase)
            agent = _create_agent(seed=seed + i)
            self.instances.append((env, agent, label))

    def run(self, steps: int = 10000, report_interval: int = 50) -> None:
        """Launch all agents in parallel threads."""
        threads = []
        for env, agent, label in self.instances:
            t = threading.Thread(
                target=_run_one,
                args=(env, agent, steps, label, report_interval),
                daemon=True,
            )
            threads.append(t)
            t.start()
            print(f"[multi-runner] Started thread for {label}")

        for t in threads:
            t.join()

        print("[multi-runner] All instances finished.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run FPI agents across multiple Minecraft servers",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config file (array of server configs)",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Max steps per server")
    parser.add_argument("--report", type=int, default=50, help="Report interval")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[multi-runner] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        configs = [configs]

    print(f"[multi-runner] Loaded {len(configs)} server config(s) from {config_path}")

    runner = MultiServerRunner(configs, seed=args.seed)

    try:
        runner.run(steps=args.steps, report_interval=args.report)
    except KeyboardInterrupt:
        print("\n[multi-runner] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
