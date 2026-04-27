"""Training loops for neural policy agents (PPO / DQN).

Runs the neural agent through MinecraftEnv (live play), using the same
428-dim signal (396 base + 16 vision + 16 history) and reward signal
as the FPI agent.

Usage:
    # Train PPO in live Minecraft
    python -m fpi.minecraft.neural_runner --algo ppo --steps 500000 --save ppo.pt

    # Train DQN in live Minecraft
    python -m fpi.minecraft.neural_runner --algo dqn --steps 500000 --save dqn.pt

    # Transfer pre-trained weights
    python -m fpi.minecraft.neural_runner --algo ppo --transfer ppo.pt

    # DQN with custom hyperparameters
    python -m fpi.minecraft.neural_runner --algo dqn --lr 5e-4 --epsilon 0.5
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .env import MinecraftEnv
from .neural_policy import DQNAgent, PPOAgent


def _create_ppo_agent(
    n_actions: int,
    obs_dim: int = 428,
    hidden: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    epochs: int = 4,
    batch_size: int = 64,
    entropy_coef: float = 0.01,
    rollout_len: int = 2048,
    seed: int = 42,
) -> PPOAgent:
    return PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        epochs=epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        rollout_len=rollout_len,
        seed=seed,
    )


def _create_dqn_agent(
    n_actions: int,
    obs_dim: int = 428,
    hidden: int = 128,
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.9995,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    target_update_freq: int = 1000,
    seed: int = 42,
) -> DQNAgent:
    return DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=hidden,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def run_ppo_sim(
    agent: PPOAgent,
    env: MinecraftEnv,
    total_steps: int,
    report_interval: int = 10_000,
) -> None:
    """PPO training loop: collect rollout -> compute GAE -> update -> repeat."""
    obs_signal = env.reset()
    obs = obs_signal.data.astype(np.float32)

    t0 = time.monotonic()
    last_report = t0
    ep_rewards: list[float] = []
    ep_reward = 0.0
    step = 0

    while step < total_steps:
        # Collect rollout
        for _ in range(agent.rollout_len):
            action, log_prob, value = agent.select_action(obs)
            next_signal, reward, done = env.step(action)
            next_obs = next_signal.data.astype(np.float32)

            agent.store_transition(obs, action, reward, done, value, log_prob)
            ep_reward += reward
            obs = next_obs
            step += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs_signal = env.reset()
                obs = obs_signal.data.astype(np.float32)

            if step % report_interval == 0:
                now = time.monotonic()
                elapsed = now - last_report
                sps = report_interval / elapsed if elapsed > 0 else 0
                total_elapsed = now - t0

                kills = getattr(env, "kill_count", 0)
                deaths = getattr(env, "death_count", 0)
                avg_ret = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0

                print(
                    f"[step {step:>8,}] "
                    f"{sps:,.0f} sps  "
                    f"avg_return={avg_ret:.3f}  "
                    f"kills={kills}  "
                    f"deaths={deaths}  "
                    f"K/D={kills / max(deaths, 1):.2f}  "
                    f"updates={agent._update_count}  "
                    f"elapsed={total_elapsed:.1f}s"
                )
                last_report = now

            if step >= total_steps:
                break

        # PPO update
        losses = agent.update(last_obs=obs)
        if step % report_interval < agent.rollout_len:
            print(
                f"  [PPO update #{agent._update_count}] "
                f"policy={losses['policy_loss']:.4f}  "
                f"value={losses['value_loss']:.4f}  "
                f"entropy={losses['entropy']:.4f}  "
                f"kl={losses['approx_kl']:.4f}"
            )

    total_time = time.monotonic() - t0
    avg_sps = total_steps / total_time if total_time > 0 else 0
    kills = getattr(env, "kill_count", 0)
    deaths = getattr(env, "death_count", 0)
    print(f"\n[neural-ppo] Complete: {total_steps:,} steps in {total_time:.1f}s ({avg_sps:,.0f} sps)")
    print(f"  Kills: {kills}, Deaths: {deaths}, K/D: {kills / max(deaths, 1):.2f}")
    print(f"  Updates: {agent._update_count}")


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def run_dqn_sim(
    agent: DQNAgent,
    env: MinecraftEnv,
    total_steps: int,
    report_interval: int = 10_000,
    update_freq: int = 4,
) -> None:
    """DQN training loop: step -> store -> update every N steps."""
    obs_signal = env.reset()
    obs = obs_signal.data.astype(np.float32)
    action_space = env.action_space

    t0 = time.monotonic()
    last_report = t0
    ep_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs, action_space)
        next_signal, reward, done = env.step(action)
        next_obs = next_signal.data.astype(np.float32)

        agent.store_transition(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs_signal = env.reset()
            obs = obs_signal.data.astype(np.float32)

        # Update
        if step % update_freq == 0:
            agent.update()

        if step % report_interval == 0:
            now = time.monotonic()
            elapsed = now - last_report
            sps = report_interval / elapsed if elapsed > 0 else 0
            total_elapsed = now - t0

            kills = getattr(env, "kill_count", 0)
            deaths = getattr(env, "death_count", 0)
            avg_ret = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0

            print(
                f"[step {step:>8,}] "
                f"{sps:,.0f} sps  "
                f"avg_return={avg_ret:.3f}  "
                f"epsilon={agent.epsilon:.3f}  "
                f"kills={kills}  "
                f"deaths={deaths}  "
                f"K/D={kills / max(deaths, 1):.2f}  "
                f"buffer={len(agent.buffer)}  "
                f"elapsed={total_elapsed:.1f}s"
            )
            last_report = now

    total_time = time.monotonic() - t0
    avg_sps = total_steps / total_time if total_time > 0 else 0
    kills = getattr(env, "kill_count", 0)
    deaths = getattr(env, "death_count", 0)
    print(f"\n[neural-dqn] Complete: {total_steps:,} steps in {total_time:.1f}s ({avg_sps:,.0f} sps)")
    print(f"  Kills: {kills}, Deaths: {deaths}, K/D: {kills / max(deaths, 1):.2f}")
    print(f"  Updates: {agent._update_count}, Epsilon: {agent.epsilon:.4f}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_neural_train(
    algo: str = "ppo",
    steps: int = 500_000,
    report_interval: int = 10_000,
    seed: int = 42,
    save_path: str | None = None,
    transfer_path: str | None = None,
    phase: int = 3,
    factored: bool = False,
    hidden: int = 128,
    lr: float | None = None,
    gamma: float = 0.99,
    # PPO-specific
    rollout_len: int = 2048,
    batch_size: int = 64,
    entropy_coef: float = 0.01,
    # DQN-specific
    epsilon: float = 1.0,
    epsilon_decay: float = 0.9995,
    buffer_size: int = 100_000,
    # Minecraft
    host: str = "localhost",
    port: int = 3001,
    observe_only: bool = False,
) -> PPOAgent | DQNAgent:
    """Unified entry point for neural agent training.

    Args:
        algo: "ppo" or "dqn".
        steps: Total training steps.
        Other args: see CLI help.

    Returns:
        The trained agent.
    """
    env = MinecraftEnv(host=host, port=port, phase=phase, factored=factored)
    print(f"[neural-{algo}] Minecraft: {host}:{port}, phase={phase}")

    n_actions = len(env.action_space)
    obs_dim = MinecraftEnv.SIGNAL_DIM

    # Create agent
    if algo == "ppo":
        effective_lr = lr if lr is not None else 3e-4
        agent = _create_ppo_agent(
            n_actions=n_actions,
            obs_dim=obs_dim,
            hidden=hidden,
            lr=effective_lr,
            gamma=gamma,
            batch_size=batch_size,
            entropy_coef=entropy_coef,
            rollout_len=rollout_len,
            seed=seed,
        )
    elif algo == "dqn":
        effective_lr = lr if lr is not None else 1e-4
        agent = _create_dqn_agent(
            n_actions=n_actions,
            obs_dim=obs_dim,
            hidden=hidden,
            lr=effective_lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algo: {algo!r}. Use 'ppo' or 'dqn'.")

    # Load pre-trained weights
    if transfer_path:
        print(f"[neural-{algo}] Loading weights from {transfer_path}...")
        agent.load(transfer_path)
        print(f"[neural-{algo}] Loaded.")

    # Enable bot control
    if not observe_only:
        env.set_bot_control(True)
        print(f"[neural-{algo}] Bot control enabled.")

    print(f"[neural-{algo}] Training for {steps:,} steps...")

    # Run training
    if algo == "ppo":
        run_ppo_sim(agent, env, steps, report_interval)
    else:
        run_dqn_sim(agent, env, steps, report_interval)

    # Save
    if save_path:
        print(f"[neural-{algo}] Saving to {save_path}...")
        agent.save(save_path)
        print(f"[neural-{algo}] Saved.")

    # Cleanup
    if not observe_only:
        try:
            env.set_bot_control(False)
        except Exception:
            pass
    env.close()

    return agent


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train neural policy (PPO/DQN) for Minecraft combat",
    )
    parser.add_argument(
        "--algo", choices=["ppo", "dqn"], default="ppo",
        help="Algorithm: ppo or dqn (default: ppo)",
    )
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps")
    parser.add_argument("--report", type=int, default=10_000, help="Report interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None, help="Save model to path")
    parser.add_argument("--transfer", type=str, default=None, help="Load model from path")
    parser.add_argument(
        "--phase", type=int, default=3, choices=[1, 2, 3, 4],
        help="Action space phase (3 = combat combos, 4 = macro-actions)",
    )
    parser.add_argument(
        "--factored", action="store_true",
        help="Use factored action space (168 = 7 move x 6 look x 4 combat)",
    )

    # Network
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # PPO
    parser.add_argument("--rollout-len", type=int, default=2048, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus")

    # DQN
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon decay")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size")

    # Minecraft
    parser.add_argument("--host", default="localhost", help="Bridge host")
    parser.add_argument("--port", type=int, default=3001, help="Bridge port")
    parser.add_argument(
        "--observe-only", action="store_true",
        help="Watch without taking control",
    )

    args = parser.parse_args()

    try:
        run_neural_train(
            algo=args.algo,
            steps=args.steps,
            report_interval=args.report,
            seed=args.seed,
            save_path=args.save,
            transfer_path=args.transfer,
            phase=args.phase,
            factored=args.factored,
            hidden=args.hidden,
            lr=args.lr,
            gamma=args.gamma,
            rollout_len=args.rollout_len,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            host=args.host,
            port=args.port,
            observe_only=args.observe_only,
        )
    except KeyboardInterrupt:
        print(f"\n[neural] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
