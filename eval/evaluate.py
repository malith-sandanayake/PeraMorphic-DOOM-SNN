"""
Agent Evaluation Utilities.

Runs trained agents across multiple episodes, collecting statistics
on kills, deaths, survival time, and reward.

Usage:
    python main.py --mode evaluate --model models/ann_simple --episodes 100
"""

import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from env.vizdoom_env import DoomEnv


def evaluate_ann_agent(
    model,
    scenario: str = "simple",
    n_episodes: int = 100,
    screen_size: tuple = (84, 84),
    use_depth: bool = False,
    deterministic: bool = True,
) -> dict:
    """Evaluate a Stable-Baselines3 PPO model.

    Args:
        model: Trained PPO model
        scenario: ViZDoom scenario name
        n_episodes: Number of evaluation episodes
        screen_size: (width, height) for screen observations
        use_depth: Whether to include depth buffer
        deterministic: Use deterministic actions

    Returns:
        Dictionary of evaluation metrics
    """
    env = DoomEnv(
        scenario=scenario,
        screen_size=screen_size,
        use_depth=use_depth,
        seed=123,
    )

    rewards = []
    kills = []
    steps = []
    max_kills_in_ep = 0
    max_steps_in_ep = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_kills = 0
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_kills = info.get("kills", 0)
            ep_steps = info.get("episode_steps", 0)

        rewards.append(ep_reward)
        kills.append(ep_kills)
        steps.append(ep_steps)
        max_kills_in_ep = max(max_kills_in_ep, ep_kills)
        max_steps_in_ep = max(max_steps_in_ep, ep_steps)

    env.close()

    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_kills": float(np.mean(kills)),
        "std_kills": float(np.std(kills)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "max_kills": max_kills_in_ep,
        "max_steps": max_steps_in_ep,
    }

    return results


def evaluate_snn_agent(
    snn_policy,
    scenario: str = "simple",
    n_episodes: int = 100,
    screen_size: tuple = (84, 84),
    T: int = 16,
    use_depth: bool = False,
    deterministic: bool = True,
) -> dict:
    """Evaluate a trained SNN policy.

    Args:
        snn_policy: Trained SNNPolicy model
        scenario: ViZDoom scenario name
        n_episodes: Number of evaluation episodes
        screen_size: (width, height)
        T: Number of simulation time steps per env step
        use_depth: Whether to include depth buffer
        deterministic: Use highest-rate action

    Returns:
        Dictionary of evaluation metrics including spike statistics
    """
    from utils.preprocess import FrameProcessor

    env = DoomEnv(
        scenario=scenario,
        screen_size=screen_size,
        use_depth=use_depth,
        seed=123,
    )

    processor = FrameProcessor(
        target_size=screen_size,
        stack_size=4,
        normalize=True,
    )

    snn_policy.eval()

    rewards = []
    kills = []
    steps = []
    firing_rates = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        processor.reset()
        done = False
        ep_reward = 0
        ep_kills = 0
        ep_steps = 0
        total_spikes = []

        while not done:
            # Preprocess frame
            processed = processor.process(obs)

            # Convert to tensor (C, H, W) then reshape to (1, T, C, H, W)
            # Use T from the SNN policy, not hardcoded T=1
            frame_tensor = torch.from_numpy(
                processed.transpose(2, 0, 1)
            ).unsqueeze(0).float()  # (1, C, H, W)
            T_actual = getattr(snn_policy, "T", T)
            # Expand to (1, T, C, H, W) — repeat same frame across T steps
            frame_tensor = frame_tensor.view(1, 1, *frame_tensor.shape[1:])
            frame_tensor = frame_tensor.expand(1, T_actual, *frame_tensor.shape[2:])

            # Forward pass through SNN
            with torch.no_grad():
                policy_rate, value, hidden = snn_policy.network(frame_tensor)

                if deterministic:
                    action_idx = torch.argmax(policy_rate, dim=-1).item()
                else:
                    import torch.nn.functional as F
                    dist = torch.distributions.Categorical(
                        F.softmax(policy_rate, dim=-1)
                    )
                    action_idx = dist.sample().item()

                # Track firing rates
                total_spikes.append(policy_rate.mean().item())

            # Execute action in environment
            action = env.action_space.__class__(action_idx)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_kills = info.get("kills", 0)
            ep_steps = info.get("episode_steps", 0)

        rewards.append(ep_reward)
        kills.append(ep_kills)
        steps.append(ep_steps)
        firing_rates.append(np.mean(total_spikes) if total_spikes else 0)

    env.close()
    snn_policy.train()  # Back to training mode

    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_kills": float(np.mean(kills)),
        "std_kills": float(np.std(kills)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "avg_firing_rate": float(np.mean(firing_rates)),
        "max_kills": max(kills),
        "max_steps": max(steps),
    }

    return results


def save_results(results: dict, output_path: str) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Evaluation metrics dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def print_results(results: dict, label: str = "Agent") -> None:
    """Print formatted evaluation results.

    Args:
        results: Evaluation metrics dictionary
        label: Name for the agent being evaluated
    """
    print(f"\n{'='*50}")
    print(f"  {label} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Mean Reward:  {results['mean_reward']:.2f} ± {results.get('std_reward', 0):.2f}")
    print(f"  Mean Kills:   {results['mean_kills']:.2f} ± {results.get('std_kills', 0):.2f}")
    print(f"  Mean Steps:   {results['mean_steps']:.0f} ± {results.get('std_steps', 0):.0f}")
    print(f"  Max Kills:    {results.get('max_kills', 0)}")
    print(f"  Max Steps:    {results.get('max_steps', 0)}")

    if "avg_firing_rate" in results:
        print(f"  Avg Firing:   {results['avg_firing_rate']:.4f}")

    print(f"{'='*50}\n")


def main(args: argparse.Namespace):
    """CLI entry point for evaluation."""
    checkpoint_path = args.checkpoint

    # Check if it's an ANN or SNN checkpoint
    is_snn = "snn" in checkpoint_path.lower()

    if is_snn:
        # Load SNN
        print(f"Loading SNN checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        from models.snn.snn_policy import SNNPolicy
        snn = SNNPolicy(
            n_actions=checkpoint["n_actions"],
            T=checkpoint["T"],
            tau=checkpoint["tau"],
            input_channels=checkpoint["input_channels"],
        )
        snn.load_state_dict(checkpoint["state_dict"])
        snn.eval()

        results = evaluate_snn_agent(
            snn_policy=snn,
            scenario=getattr(args, "scenario", "simple"),
            n_episodes=args.episodes,
            screen_size=tuple(args.screen_size),
            T=checkpoint["T"],
            use_depth=args.use_depth,
        )
        print_results(results, "SNN")

    else:
        # Load ANN (SB3 model)
        from stable_baselines3 import PPO
        print(f"Loading ANN checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path)

        results = evaluate_ann_agent(
            model=model,
            scenario=getattr(args, "scenario", "simple"),
            n_episodes=args.episodes,
            screen_size=tuple(args.screen_size),
            use_depth=args.use_depth,
            deterministic=True,
        )
        print_results(results, "ANN PPO")

    # Save results
    output_path = f"eval/results_{os.path.basename(checkpoint_path)}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main(None)  # Called via main.py CLI
