"""
Demo Script - Shows the DQN agent navigating the network
Run this after training to see the agent in action
"""

import numpy as np
from environment import TransportationEnv
from dqn_agent import DQNAgent


def visualize_route(env, agent):
    """
    Show a single episode with the trained agent.
    Prints step-by-step what the agent does.
    """
    state = env.reset()

    print("=" * 60)
    print("DQN Agent Route Demo")
    print("=" * 60)
    print(f"Start: Node {env.current_node}")
    print(f"Goal:  Node {env.goal_node}")
    print(f"Optimal path length: {env.get_shortest_path_length()[0]:.2f}")
    print("-" * 60)

    total_time = 0
    steps = 0
    path = [env.current_node]
    done = False

    while not done and steps < 50:
        valid_actions = env.get_valid_actions()
        neighbors = env.get_neighbors()

        # get Q-values for all actions
        import torch
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor).cpu().numpy()[0]

        action = agent.select_action(state, valid_actions, training=False)
        chosen_neighbor = neighbors[action]

        # get travel time for this edge
        travel_time = env.graph[env.current_node][chosen_neighbor]['weight']

        print(f"Step {steps + 1}:")
        print(f"  At node: {env.current_node}")
        print(f"  Neighbors: {neighbors}")
        print(f"  Q-values: {[f'{q:.2f}' for q in q_values[:valid_actions]]}")
        print(f"  Chose: {chosen_neighbor} (time: {travel_time:.2f})")

        next_state, reward, done, _ = env.step(action)
        total_time += -reward
        path.append(env.current_node)
        state = next_state
        steps += 1

        print()

    print("-" * 60)
    print(f"Route completed in {steps} steps")
    print(f"Total time: {total_time:.2f}")
    print(f"Path: {' -> '.join(map(str, path))}")

    if env.current_node == env.goal_node:
        print("[SUCCESS] Successfully reached goal!")
    else:
        print("[FAIL] Did not reach goal (hit step limit)")

    return total_time, steps, env.current_node == env.goal_node


def compare_single_episode():
    """
    Run one episode with both DQN and greedy to compare side-by-side.
    """
    env = TransportationEnv(num_nodes=1200, seed=999)

    # load agent
    agent = DQNAgent(state_size=2400, max_actions=4)
    try:
        agent.load("dqn_routing_model.pt")
        print("Loaded trained model\n")
    except FileNotFoundError:
        print("No trained model found, using random weights\n")

    # run DQN
    print("\n" + "=" * 60)
    print("DQN AGENT")
    print("=" * 60)
    dqn_time, dqn_steps, dqn_success = visualize_route(env, agent)

    # reset environment with same seed to get same start/goal
    env = TransportationEnv(num_nodes=1200, seed=999)

    # run greedy
    print("\n" + "=" * 60)
    print("GREEDY BASELINE")
    print("=" * 60)
    print(f"Start: Node {env.current_node}")
    print(f"Goal:  Node {env.goal_node}")
    print(f"Optimal path length: {env.get_shortest_path_length()[0]:.2f}")
    print("-" * 60)

    greedy_time = 0
    greedy_steps = 0
    path = [env.current_node]
    done = False

    while not done and greedy_steps < 50:
        neighbors = env.get_neighbors()

        # find neighbor with minimum edge weight
        best_idx = 0
        best_time = float('inf')
        for i, neighbor in enumerate(neighbors):
            travel_time = env.graph[env.current_node][neighbor]['weight']
            if travel_time < best_time:
                best_time = travel_time
                best_idx = i

        print(f"Step {greedy_steps + 1}:")
        print(f"  At node: {env.current_node}")
        print(f"  Neighbor times: {[env.graph[env.current_node][n]['weight']:.2f' for n in neighbors]}")
        print(f"  Chose: {neighbors[best_idx]} (time: {best_time:.2f})")
        print()

        next_state, reward, done, _ = env.step(best_idx)
        greedy_time += -reward
        path.append(env.current_node)
        greedy_steps += 1

    print("-" * 60)
    print(f"Route completed in {greedy_steps} steps")
    print(f"Total time: {greedy_time:.2f}")
    print(f"Path: {' -> '.join(map(str, path))}")

    # comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"DQN:    {dqn_steps} steps, {dqn_time:.2f} time")
    print(f"Greedy: {greedy_steps} steps, {greedy_time:.2f} time")

    if dqn_time < greedy_time:
        improvement = ((greedy_time - dqn_time) / greedy_time) * 100
        print(f"\nDQN was {improvement:.1f}% faster!")
    else:
        print("\nGreedy was faster on this particular route")
        print("(DQN should win on average over many routes)")


if __name__ == "__main__":
    compare_single_episode()
