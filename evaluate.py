"""
Evaluation Script
Compares DQN agent against a greedy baseline algorithm
"""

import numpy as np
import time
from environment import TransportationEnv
from dqn_agent import DQNAgent


def greedy_baseline(env):
    """
    Simple greedy algorithm - always pick the neighbor closest to goal.
    This is a common heuristic for routing problems.
    """
    state = env.reset()
    total_time = 0
    steps = 0
    max_steps = 100
    done = False

    while not done and steps < max_steps:
        neighbors = env.get_neighbors()

        if not neighbors:
            break  # dead end

        # find neighbor with minimum edge weight (greedy choice)
        best_neighbor = None
        best_time = float('inf')

        for neighbor in neighbors:
            travel_time = env.graph[env.current_node][neighbor]['weight']
            if travel_time < best_time:
                best_time = travel_time
                neighbor_idx = neighbors.index(neighbor)
                best_neighbor = neighbor_idx

        if best_neighbor is None:
            break

        # take the step
        next_state, reward, done, _ = env.step(best_neighbor)
        total_time += -reward  # reward is negative of travel time
        steps += 1

    return total_time, steps, env.current_node == env.goal_node


def dqn_evaluate(env, agent, num_episodes=100):
    """
    Evaluate the trained DQN agent.
    """
    total_times = []
    total_steps = []
    successes = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_time = 0
        episode_steps = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            next_state, reward, done, _ = env.step(action)

            episode_time += -reward
            episode_steps += 1
            state = next_state

            if episode_steps > 100:  # safety limit
                break

        total_times.append(episode_time)
        total_steps.append(episode_steps)
        successes.append(env.current_node == env.goal_node)

    return total_times, total_steps, successes


def run_comparison():
    """
    Run comparison between DQN and greedy baseline.
    This is the main evaluation that shows if DQN learned anything useful.
    """
    print("=" * 60)
    print("DQN vs Greedy Baseline Evaluation")
    print("=" * 60)

    # create environment
    env = TransportationEnv(num_nodes=1200, seed=123)  # different seed for eval

    # load trained agent
    print("\nLoading trained DQN agent...")
    agent = DQNAgent(state_size=2400, max_actions=4)
    try:
        agent.load("dqn_routing_model.pt")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Warning: No trained model found. Running with random weights.")
        print("Train the model first with: python train.py")

    # evaluate DQN
    print("\nEvaluating DQN agent...")
    dqn_times, dqn_steps, dqn_success = dqn_evaluate(env, agent, num_episodes=100)

    # evaluate greedy
    print("Evaluating greedy baseline...")
    greedy_times = []
    greedy_steps = []
    greedy_success = []

    for _ in range(100):
        time_taken, steps, success = greedy_baseline(env)
        greedy_times.append(time_taken)
        greedy_steps.append(steps)
        greedy_success.append(success)

    # calculate statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # DQN stats
    dqn_avg_time = np.mean(dqn_times)
    dqn_std_time = np.std(dqn_times)
    dqn_avg_steps = np.mean(dqn_steps)
    dqn_success_rate = np.mean(dqn_success) * 100

    # Greedy stats
    greedy_avg_time = np.mean(greedy_times)
    greedy_std_time = np.std(greedy_times)
    greedy_avg_steps = np.mean(greedy_steps)
    greedy_success_rate = np.mean(greedy_success) * 100

    print(f"\nDQN Agent:")
    print(f"  Average route time: {dqn_avg_time:.2f} ± {dqn_std_time:.2f}")
    print(f"  Average steps: {dqn_avg_steps:.2f}")
    print(f"  Success rate: {dqn_success_rate:.1f}%")

    print(f"\nGreedy Baseline:")
    print(f"  Average route time: {greedy_avg_time:.2f} ± {greedy_std_time:.2f}")
    print(f"  Average steps: {greedy_avg_steps:.2f}")
    print(f"  Success rate: {greedy_success_rate:.1f}%")

    # calculate improvement
    time_improvement = ((greedy_avg_time - dqn_avg_time) / greedy_avg_time) * 100
    step_improvement = ((greedy_avg_steps - dqn_avg_steps) / greedy_avg_steps) * 100

    print(f"\nImprovement:")
    print(f"  Time reduction: {time_improvement:.1f}%")
    print(f"  Step reduction: {step_improvement:.1f}%")

    # check if we hit the target
    if time_improvement >= 28:
        print(f"\n[OK] Target achieved! {time_improvement:.1f}% improvement > 28%")
    else:
        print(f"\n[WARNING] Target not reached. {time_improvement:.1f}% < 28%")
        print("  (Might need more training or parameter tuning)")

    # save results
    results = {
        "dqn": {
            "avg_time": float(dqn_avg_time),
            "std_time": float(dqn_std_time),
            "avg_steps": float(dqn_avg_steps),
            "success_rate": float(dqn_success_rate)
        },
        "greedy": {
            "avg_time": float(greedy_avg_time),
            "std_time": float(greedy_std_time),
            "avg_steps": float(greedy_avg_steps),
            "success_rate": float(greedy_success_rate)
        },
        "improvement": {
            "time_percent": float(time_improvement),
            "step_percent": float(step_improvement)
        }
    }

    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to evaluation_results.json")

    return results


if __name__ == "__main__":
    results = run_comparison()
