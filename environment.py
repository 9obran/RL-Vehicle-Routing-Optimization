"""
Transportation Network Environment for DQN Routing
Simple implementation - generates a random graph with dynamic edge weights
"""

import numpy as np
import networkx as nx
import random


class TransportationEnv:
    """
    Simulates a transportation network with 1000+ nodes.
    Edge weights change over time to simulate traffic/dynamic conditions.
    """

    def __init__(self, num_nodes=1200, connection_prob=0.05, seed=42):
        # not sure if 1200 is the right size but it meets the 1000+ requirement
        self.num_nodes = num_nodes
        self.connection_prob = connection_prob
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # create the graph - using watts_strogatz for small-world properties
        # (kinda like real road networks have shortcuts)
        self.graph = nx.connected_watts_strogatz_graph(
            n=num_nodes,
            k=4,  # each node connects to 4 neighbors
            p=0.1,  # rewiring probability
            seed=seed
        )

        # initialize edge weights (travel times)
        self._initialize_weights()

        # state
        self.current_node = None
        self.goal_node = None
        self.steps = 0
        self.max_steps = 100  # prevent infinite loops

    def _initialize_weights(self):
        """Set initial travel times for all edges."""
        for u, v in self.graph.edges():
            # base travel time between 1 and 10 units
            base_time = random.uniform(1.0, 10.0)
            self.graph[u][v]['weight'] = base_time
            self.graph[u][v]['base_weight'] = base_time  # store original

    def _update_dynamic_weights(self):
        """
        Simulate traffic changes - some edges get slower/faster.
        Called periodically to create dynamic environment.
        """
        # randomly select 10% of edges to change
        edges = list(self.graph.edges())
        num_to_change = max(1, len(edges) // 10)
        edges_to_change = random.sample(edges, num_to_change)

        for u, v in edges_to_change:
            base = self.graph[u][v]['base_weight']
            # add some randomness - traffic can be 0.5x to 3x normal
            multiplier = random.uniform(0.5, 3.0)
            self.graph[u][v]['weight'] = base * multiplier

    def reset(self):
        """Reset environment for new episode."""
        # update traffic conditions
        self._update_dynamic_weights()

        # pick random start and goal (make sure they're different)
        self.current_node = random.randint(0, self.num_nodes - 1)
        self.goal_node = random.randint(0, self.num_nodes - 1)
        while self.goal_node == self.current_node:
            self.goal_node = random.randint(0, self.num_nodes - 1)

        self.steps = 0

        return self._get_state()

    def _get_state(self):
        """
        Return current state representation.
        Using one-hot encoding for current and goal position.
        Could probably be more efficient but this is clear.
        """
        # create one-hot vectors
        current = np.zeros(self.num_nodes, dtype=np.float32)
        goal = np.zeros(self.num_nodes, dtype=np.float32)
        current[self.current_node] = 1.0
        goal[self.goal_node] = 1.0

        # concatenate them - this is our state
        return np.concatenate([current, goal])

    def step(self, action):
        """
        Take action (move to neighboring node).
        Action is the index of the neighbor to move to.
        """
        self.steps += 1

        # get valid neighbors
        neighbors = list(self.graph.neighbors(self.current_node))

        if action >= len(neighbors):
            # invalid action - stay in place with penalty
            # this shouldn't happen often with proper masking
            reward = -10.0
            done = self.steps >= self.max_steps
            return self._get_state(), reward, done, {"invalid": True}

        next_node = neighbors[action]

        # calculate reward (negative of travel time - we want to minimize)
        travel_time = self.graph[self.current_node][next_node]['weight']
        reward = -travel_time

        # bonus for reaching goal
        if next_node == self.goal_node:
            reward += 100.0  # big bonus for completing the route

        self.current_node = next_node

        # check if done
        done = (self.current_node == self.goal_node) or (self.steps >= self.max_steps)

        return self._get_state(), reward, done, {}

    def get_valid_actions(self):
        """
        Return number of valid actions (neighbors) for current state.
        Used for action masking.
        """
        return len(list(self.graph.neighbors(self.current_node)))

    def get_neighbors(self):
        """Return list of neighbor nodes for current position."""
        return list(self.graph.neighbors(self.current_node))

    def get_shortest_path_length(self):
        """
        Get the optimal path length using Dijkstra.
        Used for evaluation - not available to the agent during training.
        """
        try:
            path = nx.shortest_path(
                self.graph, self.current_node, self.goal_node, weight='weight'
            )
            length = nx.path_weight(self.graph, path, weight='weight')
            return length, len(path) - 1  # return weight and num edges
        except nx.NetworkXNoPath:
            return float('inf'), 0


if __name__ == "__main__":
    # quick test
    env = TransportationEnv(num_nodes=100)  # small for testing
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Current: {env.current_node}, Goal: {env.goal_node}")
    print(f"Neighbors: {env.get_neighbors()}")
