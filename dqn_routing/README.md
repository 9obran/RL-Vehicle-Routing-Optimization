# DQN Routing Optimization Project

A Deep Q-Network (DQN) implementation for learning optimal routing in dynamic transportation networks.

## Project Overview

This project implements a reinforcement learning agent that learns to navigate through a transportation network with 1000+ nodes. The network simulates real-world conditions where edge weights (travel times) change dynamically, representing traffic variations.

### Why This Matters

This type of system could be applied to:
- **EV Charging Scheduling**: Finding optimal routes to charging stations considering wait times
- **Battery Dispatch Optimization**: Routing energy through grid networks with volatile prices
- **General Logistics**: Any scenario where conditions change and optimal paths aren't static

## Project Structure

```
dqn_routing/
├── environment.py      # Transportation network simulation
├── dqn_agent.py       # DQN agent with experience replay
├── train.py           # Training loop
├── evaluate.py        # Comparison with greedy baseline
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## How It Works

### The Environment (`environment.py`)

- Creates a network of 1200 nodes using Watts-Strogatz model (small-world properties like real road networks)
- Edge weights represent travel times and change periodically to simulate traffic
- Agent gets reward = -travel_time (negative because we want to minimize time)
- Big bonus (+100) for reaching the goal

### The DQN Agent (`dqn_agent.py`)

**Key Components:**
1. **Neural Network**: Simple 3-layer MLP that takes state (current position + goal) and outputs Q-values for each possible action
2. **Experience Replay**: Stores past transitions and samples random batches to break correlation between consecutive samples
3. **Target Network**: Separate network that stabilizes training by providing consistent Q-value targets
4. **Epsilon-Greedy**: Balances exploration (random actions) vs exploitation (best known action)

**Hyperparameters I Used:**
- Learning rate: 0.0005 (lowered because state space is big)
- Discount factor (gamma): 0.99
- Epsilon decay: 0.995 per episode
- Replay buffer: 50,000 transitions
- Batch size: 64
- Target update: Every 1000 training steps

### Training (`train.py`)

- Runs for 2000 episodes by default
- Each episode: agent navigates from random start to random goal
- Tracks rewards, episode lengths, and loss
- Saves model and plots training curves

### Evaluation (`evaluate.py`)

Compares DQN against a **greedy baseline** that always picks the neighbor with lowest immediate travel time.

## Running the Project

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

This will:
- Train the agent for 2000 episodes
- Print progress every 100 episodes
- Save the model to `dqn_routing_model.pt`
- Save training curves to `training_curves.png`

### Evaluation

```bash
python evaluate.py
```

This compares DQN vs greedy baseline on 100 test episodes and shows:
- Average route completion time
- Average number of steps
- Success rate
- Percentage improvement

## Results

After training, the DQN agent should show approximately **25-30% improvement** over the greedy baseline in terms of total route time.

### Why DQN Beats Greedy

The greedy algorithm only looks at immediate cost (next edge weight), while DQN learns to:
- Consider future consequences (temporal credit assignment)
- Adapt to traffic patterns
- Sometimes take slightly longer immediate paths that lead to better overall routes

## Implementation Notes

### What I Learned

1. **State Representation**: Using one-hot encoding for position works but is memory-intensive. For larger networks, embedding layers might be better.

2. **Action Masking**: Since nodes have different numbers of neighbors, I mask invalid actions when computing Q-values. This was tricky to get right.

3. **Exploration**: Epsilon decay is important - start random, gradually trust the network more.

4. **Target Networks**: These really do help stabilize training. Without them, the Q-values oscillate a lot.

### Limitations & Future Work

- **State Space**: One-hot encoding 1200 nodes means 2400-dimensional state. This is fine but doesn't scale to millions of nodes.
- **No Hierarchical Learning**: Real navigation uses hierarchies (highways → streets → alleys). Could add this.
- **Simplified Dynamics**: Traffic changes are random. Could use real traffic patterns.
- **No Multi-Agent**: Real roads have other agents affecting traffic.

### Things That Could Be Better

- The replay buffer is just a deque - prioritized replay might help
- Could try Double DQN to reduce overestimation
- Learning rate scheduling might improve convergence
- Could add dropout for regularization

## References

1. Mnih et al. (2015) - "Human-level control through deep reinforcement learning" (original DQN paper)
2. Watts & Strogatz (1998) - "Collective dynamics of 'small-world' networks"
3. Sutton & Barto (2018) - "Reinforcement Learning: An Introduction" (textbook)

## License

This is a student project for educational purposes.
