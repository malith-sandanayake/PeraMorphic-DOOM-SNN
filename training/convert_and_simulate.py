"""
ANN-to-SNN Conversion Pipeline.

Converts a trained PPO ANN model (DoomNet) to a rate-coded SNN (SpikingCNN)
by directly copying convolutional and linear layer weights.

The conversion preserves the network connectivity and weights, replacing
ReLU activations with LIF neurons. The SNN is then simulated for
evaluation.

Usage:
    python main.py --mode convert --checkpoint models/ann_simple
"""

import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from models.ann.ppo_network import DoomNet
from models.snn.snn_policy import SNNPolicy


def convert(
    ann_checkpoint_path: str,
    snn_save_path: str = "models/snn_converted",
    T: int = 16,
    tau: float = 16.0,
    n_eval_episodes: int = 50,
    scenario: str = "simple",
    screen_size: tuple = (84, 84),
    use_depth: bool = False,
) -> dict:
    """Convert trained ANN to SNN and evaluate.

    Args:
        ann_checkpoint_path: Path to trained ANN checkpoint
        snn_save_path: Path to save converted SNN
        T: Number of simulation time steps per environment step
        tau: LIF neuron time constant
        n_eval_episodes: Number of evaluation episodes
        scenario: ViZDoom scenario name
        screen_size: (width, height) for screen observations
        use_depth: Whether to include depth buffer

    Returns:
        Dict with ANN and SNN performance metrics
    """
    print("=== ANN-to-SNN Conversion ===")

    # Step 1: Load ANN model
    print("\n[1/4] Loading ANN checkpoint...")
    ann_model = PPO.load(ann_checkpoint_path)
    print(f"  Loaded: {ann_checkpoint_path}")

    # Extract network architecture details
    ann_net = ann_model.policy
    n_actions = ann_net.action_net.out_features
    input_channels = ann_net.features[0].in_channels
    hidden_dim = ann_net.mlp_extractor.policy_net.out_features

    print(f"  Architecture: input_ch={input_channels}, "
          f"actions={n_actions}, hidden={hidden_dim}")

    # Step 2: Create SNN model
    print("\n[2/4] Creating SNN model...")
    snn_policy = SNNPolicy(
        n_actions=n_actions,
        T=T,
        tau=tau,
        input_channels=input_channels,
    )
    print(f"  SNN: T={T}, tau={tau}, input_ch={input_channels}")

    # Step 3: Convert weights
    print("\n[3/4] Converting ANN weights to SNN...")
    snn_policy.convert_from_ann(ann_net)

    # Verify conversion: check that weights were copied
    with torch.no_grad():
        # Compare conv1 weights
        ann_conv1 = ann_net.features[0].weight
        snn_conv1 = snn_policy.network.conv1.weight
        weight_diff = (ann_conv1 - snn_conv1).abs().max().item()
        print(f"  Max weight difference after copy: {weight_diff:.6f}")

    # Save converted SNN
    torch.save({
        "n_actions": n_actions,
        "T": T,
        "tau": tau,
        "input_channels": input_channels,
        "state_dict": snn_policy.state_dict(),
    }, snn_save_path)
    print(f"  SNN saved to: {snn_save_path}")

    # Step 4: Evaluate SNN
    print("\n[4/4] Evaluating converted SNN...")
    from env.vizdoom_env import DoomEnv
    from eval.evaluate import evaluate_snn_agent

    results = evaluate_snn_agent(
        snn_policy=snn_policy,
        scenario=scenario,
        n_episodes=n_eval_episodes,
        screen_size=screen_size,
        T=T,
        use_depth=use_depth,
    )

    print(f"\n  SNN Evaluation Results:")
    print(f"    Mean Reward: {results['mean_reward']:.2f}")
    print(f"    Mean Kills:  {results['mean_kills']:.2f}")
    print(f"    Avg Steps:   {results['mean_steps']:.0f}")
    print(f"    Firing Rate: {results['avg_firing_rate']:.3f}")

    print("\n  ✅ Conversion complete!")
    return results


def main(args: argparse.Namespace):
    """CLI entry point for conversion."""
    scenario = getattr(args, "scenario", "simple")
    screen_size = tuple(args.screen_size)

    convert(
        ann_checkpoint_path=args.checkpoint,
        snn_save_path="models/snn_converted",
        T=args.simulation_steps,
        tau=args.snn_tau,
        scenario=scenario,
        screen_size=screen_size,
        use_depth=args.use_depth,
    )


if __name__ == "__main__":
    main(None)  # Called via main.py CLI
