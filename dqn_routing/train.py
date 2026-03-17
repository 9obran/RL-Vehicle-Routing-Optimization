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

    # create environment with 1200 nodes
    env = TransportationEnv(num_nodes=1200, seed=42)

    # create agent
    # state is 2400 dims (1200 for current + 1200 for goal)
    # max actions is 4 (from watts_strogatz k=4)
    agent = DQNAgent(
        state_size=2400,
        max_actions=4,
        learning_rate=0.0005  # lowered a bit since state is big
    )

    # tracking metrics
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
            # get number of valid actions
            valid_actions = env.get_valid_actions()

            # select action
            action = agent.select_action(state, valid_actions, training=True)

            # take step
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            # store transition
            agent.store_transition(
                state, action, reward, next_state, done, next_valid_actions
            )

            # train
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward
            episode_steps += 1

        # decay epsilon after each episode
        agent.decay_epsilon()

        # log metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        epsilon_history.append(agent.epsilon)

        # periodic evaluation
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            avg_loss = np.mean(losses[-1000:]) if losses else 0

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")

    # save the trained model
    agent.save("dqn_routing_model.pt")
    print("\nModel saved to dqn_routing_model.pt")

    # save training history
    history = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "epsilon": epsilon_history
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    # plot training curves
    plot_training_curves(episode_rewards, episode_lengths, epsilon_history)

    return agent, env, history


def plot_training_curves(rewards, lengths, epsilon):
    """Plot training metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # plot rewards
    axes[0].plot(rewards, alpha=0.3, label="Raw")
    # moving average
    window = 50
    if len(rewards) > window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, label=f"MA({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # plot lengths
    axes[1].plot(lengths, alpha=0.3)
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(lengths)), smoothed)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps to Goal")
    axes[1].set_title("Episode Lengths")
    axes[1].grid(True, alpha=0.3)

    # plot epsilon
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
    # run training
    agent, env, history = train_dqn(num_episodes=2000)
