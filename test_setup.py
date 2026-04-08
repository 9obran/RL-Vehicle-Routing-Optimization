"""
Quick test to make sure everything is set up correctly
Run this before training to check for errors
"""

import sys


def test_imports():
    """Check all required packages are installed."""
    print("Checking imports...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
    except ImportError:
        print("  [FAIL] PyTorch not found - run: pip install torch")
        return False

    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError:
        print("  [FAIL] NumPy not found - run: pip install numpy")
        return False

    try:
        import networkx as nx
        print(f"  [OK] NetworkX {nx.__version__}")
    except ImportError:
        print("  [FAIL] NetworkX not found - run: pip install networkx")
        return False

    try:
        import tqdm
        print(f"  [OK] tqdm installed")
    except ImportError:
        print("  [FAIL] tqdm not found - run: pip install tqdm")
        return False

    try:
        import matplotlib
        print(f"  [OK] Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  [FAIL] Matplotlib not found - run: pip install matplotlib")
        return False

    return True


def test_environment():
    """Test the environment works."""
    print("\nTesting environment...")
    try:
        from environment import TransportationEnv

        env = TransportationEnv(num_nodes=100, seed=42)  # small for testing
        state = env.reset()

        assert state.shape == (200,), f"Expected state shape (200,), got {state.shape}"
        print(f"  [OK] Environment created")
        print(f"  [OK] State shape: {state.shape}")
        print(f"  [OK] Current node: {env.current_node}")
        print(f"  [OK] Goal node: {env.goal_node}")

        # test step
        valid_actions = env.get_valid_actions()
        state, reward, done, _ = env.step(0)
        print(f"  [OK] Step works, reward: {reward:.2f}")

        return True
    except Exception as e:
        print(f"  [FAIL] Environment test failed: {e}")
        return False


def test_agent():
    """Test the agent works."""
    print("\nTesting agent...")
    try:
        from dqn_agent import DQNAgent

        agent = DQNAgent(state_size=200, max_actions=4, learning_rate=0.001)
        print(f"  [OK] Agent created")
        print(f"  [OK] Device: {agent.device}")
        print(f"  [OK] Parameters: {sum(p.numel() for p in agent.q_network.parameters())}")

        # test action selection
        import numpy as np
        state = np.zeros(200, dtype=np.float32)
        action = agent.select_action(state, valid_actions=4, training=False)
        print(f"  [OK] Action selection works (action: {action})")

        return True
    except Exception as e:
        print(f"  [FAIL] Agent test failed: {e}")
        return False


def test_training():
    """Test a single training step."""
    print("\nTesting training loop...")
    try:
        from environment import TransportationEnv
        from dqn_agent import DQNAgent
        import numpy as np

        env = TransportationEnv(num_nodes=100, seed=42)
        agent = DQNAgent(state_size=200, max_actions=4)

        # run one episode
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 10:  # limit steps for test
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            next_state, reward, done, _ = env.step(action)
            next_valid = env.get_valid_actions()

            agent.store_transition(state, action, reward, next_state, done, next_valid)
            loss = agent.train()

            state = next_state
            total_reward += reward
            steps += 1

        print(f"  [OK] Training step works")
        print(f"  [OK] Episode completed in {steps} steps")
        print(f"  [OK] Total reward: {total_reward:.2f}")
        print(f"  [OK] Buffer size: {len(agent.replay_buffer)}")

        return True
    except Exception as e:
        print(f"  [FAIL] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DQN Routing Project - Setup Test")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Agent", test_agent()))
    results.append(("Training", test_training()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n[OK] All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Wait for training to complete (~45 minutes)")
        print("  3. Run: python evaluate.py")
        print("  4. Run: python demo.py")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
