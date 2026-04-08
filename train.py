"""
Training Loop for DQN Routing Agent
Trains the agent on the transportation environment
"""

import numpy as np
import torch
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from environment import TransportationEnv
from dqn_agent import DQNAgent


def train_dqn(num_episodes=2000, eval_interval=100):
    """
    Main training function.
    Trains the DQN agent for a bunch of episodes.
    """
    print("Initializing environment and agent...")

    # 1200 nodes = ~city-sized network. Watts-Strogatz gives realistic "small-world" topology
    env = TransportationEnv(num_nodes=1200, seed=42)
    max_actions = max(dict(env.graph.degree()).values()) 
    print(f"Max neighbors found in graph: {max_actions}")
    # State is 2400 dims because we one hot encode both current position AND destination
    # (need to know where we're going, not just where we are!)
    # k=4 from Watts-Strogatz means max 4 neighbors per node
    agent = DQNAgent(
        state_size=2400,
        max_actions=max_actions,
        learning_rate=0.0005  # lowered from 0.001, loss was exploding with bigger state space
    )

    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon_history = []

    print(f"Starting training for {num_episodes} episodes...")

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            action = agent.select_action(state, valid_actions, training=True)

            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            agent.store_transition(
                state, action, reward, next_state, done, next_valid_actions
            )

            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward
            episode_steps += 1

        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        epsilon_history.append(agent.epsilon)

        # Log progress every 100 episodes so we can see if it's actually learning
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            avg_loss = np.mean(losses[-1000:]) if losses else 0

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")

    agent.save("dqn_routing_model.pt")
    print("\nModel saved to dqn_routing_model.pt")

    history = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "epsilon": epsilon_history
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    plot_training_curves(episode_rewards, episode_lengths, epsilon_history)

    return agent, env, history


def plot_training_curves(rewards, lengths, epsilon):
    """Plot training metrics over time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Rewards should trend upward as agent learns better routes
    axes[0].plot(rewards, alpha=0.3, label="Raw")
    window = 50
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, label=f"MA({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode lengths should decrease as agent finds shorter paths
    axes[1].plot(lengths, alpha=0.3)
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps to Goal")
    axes[1].set_title("Episode Lengths")
    axes[1].grid(True, alpha=0.3)

    # Epsilon decay shows exploration dropping off
    axes[2].plot(epsilon)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_title("Exploration Rate")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")
    plt.close()


if __name__ == "__main__":
    agent, env, history = train_dqn(num_episodes=2000)
