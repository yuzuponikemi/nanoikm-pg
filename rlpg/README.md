# Reinforcement Learning Playground - Inverted Pendulum

A comprehensive educational repository for learning reinforcement learning fundamentals through the classic **Inverted Pendulum** (Cart-Pole) control problem.

## Overview

This project provides hands-on Jupyter notebooks and Python code to teach you reinforcement learning from the ground up. By the end, you'll understand:

- Core RL concepts (agent, environment, state, action, reward, policy)
- How to build a physics simulation
- Different types of policies (random, linear, neural network)
- Training algorithms (random search, hill climbing, evolution strategies)
- How to visualize and analyze learned behaviors

## The Inverted Pendulum Problem

The inverted pendulum is a classic control problem where you must balance a pole attached to a cart by applying horizontal forces. It's perfect for learning RL because:

- **Simple to understand**: Clear physics and intuitive goal
- **Challenging enough**: Unstable equilibrium requires active control
- **Quick feedback**: Episodes are short, training is fast
- **Scalable**: Works with simple and complex policies

```
        |
       /|\      <- Pole (keep upright!)
      / | \
    [=====]     <- Cart (apply force)
    \_____/
================  <- Track
```

## Installation

### Prerequisites

- Python 3.8+
- pip or uv package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rlpg.git
cd rlpg

# Install dependencies
pip install -r requirements.in

# Or using uv
uv pip install -r requirements.in
```

## Project Structure

```
rlpg/
├── notebooks/                    # Interactive Jupyter notebooks
│   ├── 01_introduction_to_rl.ipynb
│   ├── 02_inverted_pendulum_physics.ipynb
│   ├── 03_simulation_environment.ipynb
│   ├── 04_understanding_policies.ipynb
│   ├── 05_training_policies.ipynb
│   └── 06_experiments_and_visualization.ipynb
├── src/
│   ├── environments/             # Simulation environments
│   │   └── pendulum.py          # Inverted pendulum implementation
│   ├── policies/                 # Policy implementations
│   │   ├── base.py              # Abstract base policy
│   │   ├── random_policy.py     # Random action selection
│   │   ├── linear_policy.py     # Linear controller
│   │   └── neural_policy.py     # Neural network policy
│   └── utils/                    # Helper functions
│       ├── visualization.py     # Plotting and animation
│       └── training.py          # Training algorithms
├── requirements.in               # Package dependencies
└── README.md
```

## Notebooks

Work through the notebooks in order for a complete learning experience:

### 1. Introduction to RL
Learn the fundamental concepts of reinforcement learning: agent, environment, state, action, reward, and policy.

### 2. Inverted Pendulum Physics
Understand the physics of the cart-pole system: equations of motion, state variables, and why balancing is difficult.

### 3. Simulation Environment
Explore the environment API: reset, step, customization, and data collection.

### 4. Understanding Policies
Compare different policy types: random (baseline), linear (simple but effective), and neural networks (deep RL).

### 5. Training Policies
Learn training algorithms: random search, hill climbing, and evolution strategies. Train your own policies!

### 6. Experiments and Visualization
Run comprehensive experiments, create animations, and analyze what the agent learned.

## Quick Start

```python
import sys
sys.path.append('.')

from src.environments import InvertedPendulumEnv
from src.policies import LinearPolicy
from src.utils import train_policy, evaluate_policy

# Create environment and policy
env = InvertedPendulumEnv()
policy = LinearPolicy()

# Train the policy
result = train_policy(
    env, policy,
    algorithm='evolutionary',
    n_iterations=100,
    verbose=True
)

# Evaluate
eval_result = evaluate_policy(env, policy, n_episodes=50)
print(f"Mean reward: {eval_result['mean_reward']:.1f}")
```

## Key Concepts

### Environment

The `InvertedPendulumEnv` simulates the cart-pole system:

```python
env = InvertedPendulumEnv(
    gravity=9.81,        # m/s²
    cart_mass=1.0,       # kg
    pole_mass=0.1,       # kg
    pole_length=0.5,     # m (half-length)
    force_mag=10.0,      # N (max force)
    max_steps=500        # steps per episode
)

state = env.reset()  # [x, x_dot, theta, theta_dot]
state, reward, done, info = env.step(action)
```

### Policies

Three policy types are provided:

```python
# Random - baseline
policy = RandomPolicy(action_low=-10, action_high=10)

# Linear - simple but effective
policy = LinearPolicy(weights=np.array([0, 0, 10, 3]))

# Neural network - for deep RL
policy = NeuralNetworkPolicy(hidden_sizes=[64, 64])
```

### Training

Three training algorithms:

```python
result = train_policy(
    env, policy,
    algorithm='evolutionary',  # or 'random_search', 'hill_climbing'
    n_iterations=100,
    population_size=20,
    noise_scale=0.5
)
```

## Visualization

Create beautiful visualizations of your experiments:

```python
from src.utils import plot_trajectory, animate_pendulum

# Plot episode trajectory
history = env.get_history()
plot_trajectory(history, title="My Episode")

# Create animation
animate_pendulum(history, save_path="pendulum.gif")
```

## Customization

### Different Challenges

```python
# Easy (longer pole)
easy_env = InvertedPendulumEnv(pole_length=1.0)

# Hard (shorter pole, less force)
hard_env = InvertedPendulumEnv(pole_length=0.25, force_mag=5.0)

# Moon gravity!
moon_env = InvertedPendulumEnv(gravity=1.62)
```

### Custom Rewards

Modify the environment's reward function in `src/environments/pendulum.py` to change learning objectives.

## Learning Path

After completing this tutorial, consider:

1. **Policy Gradients**: REINFORCE, actor-critic
2. **Value Functions**: Q-learning, DQN
3. **Advanced Algorithms**: PPO, SAC, DDPG
4. **Harder Environments**: MuJoCo, Atari
5. **Advanced Topics**: Model-based RL, multi-agent RL

### Recommended Resources

- Sutton & Barto: "Reinforcement Learning: An Introduction"
- OpenAI Spinning Up: https://spinningup.openai.com
- HuggingFace Deep RL Course

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Visualization
- **jupyter**: Interactive notebooks
- **torch**: Neural network policies
- **tqdm**: Progress bars

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Acknowledgments

This project is inspired by:
- OpenAI Gym's CartPole environment
- Classic control theory literature
- The RL research community

---

Happy learning! If you balance the pendulum, you've taken your first step into reinforcement learning.
