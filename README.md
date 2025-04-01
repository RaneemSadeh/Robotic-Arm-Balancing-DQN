# Robotic-Arm-Balancing-DQN
This repository contains a Python implementation of a Deep Q-Learning (DQN) model designed to control a robotic arm in the "Pendulum-v1" environment from the Gymnasium library by OpenAI. The project focuses on training a neural network to balance a pendulum (simulating a robotic arm) using reinforcement learning techniques.
# Robotic Arm Balancing with Deep Q-Learning (DQN)

## Overview

This repository contains a Python implementation of a Deep Q-Learning (DQN) model designed to control a robotic arm in the "Pendulum-v1" environment from the Gymnasium library by OpenAI. The goal of the project is to train a neural network to balance a pendulum (simulating a robotic arm) using reinforcement learning techniques.

The project demonstrates the use of:
- **Deep Q-Learning (DQN)**: A reinforcement learning algorithm that uses a neural network to approximate the Q-value function.
- **Experience Replay**: A technique to store and sample past experiences to improve learning stability.
- **Neural Networks**: Built using PyTorch, the neural network learns to predict the optimal actions to balance the pendulum.
- **Gymnasium**: A library for simulating reinforcement learning environments.

  ![image](https://github.com/user-attachments/assets/9cbdc740-4d1a-43a9-bf09-8f57fc4a5b1f)


## Table of Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Results](#results)
6. [Future Improvements](#future-improvements)
7. [References](#references)

## Project Description

The project focuses on training a robotic arm (simulated as a pendulum) to balance itself using a Deep Q-Learning (DQN) algorithm. The environment used is the "Pendulum-v1" from the Gymnasium library, which simulates a pendulum that can rotate around a fixed point. The goal is to apply torque to the pendulum to keep it upright.

![image](https://github.com/user-attachments/assets/74750f6b-5fcb-42cd-82a9-b9688354cb43)

The DQN agent uses a neural network to approximate the Q-value function, which estimates the expected cumulative reward for taking a particular action in a given state. The agent learns by interacting with the environment, storing experiences in a replay buffer, and updating the neural network using batches of experiences.

### Key Features:
- **Neural Network Architecture**: The neural network consists of three hidden layers with ReLU activation functions and a final layer with a sigmoid activation function to map the output to the action space.
- **Experience Replay**: The agent stores past experiences (state, action, reward, next state) in a replay buffer and samples from it to train the neural network.
- **Epsilon-Greedy Strategy**: The agent uses an epsilon-greedy strategy to balance exploration and exploitation during training.
- **Training and Testing**: The agent is trained over 500 episodes, and its performance is evaluated by testing it on the environment.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- `gymnasium`
- `torch`
- `numpy`
- `matplotlib`

You can install the required libraries using pip:

```bash
pip install gymnasium torch numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/RaneemSadeh/robotic-arm-balancing-dqn.git
cd robotic-arm-balancing-dqn
```

2. Run the main script to train the DQN agent:

```bash
python DQN_AI_system_used_to_control_the_robotic_arm.py
```

3. After training, the script will plot the following:
   - Average episodic rewards over time.
   - Loss values during training.
   - Epsilon values (exploration rate) over episodes.
   - Average test rewards over the last 40 episodes.

## Code Structure

The main script `DQN_AI_system_used_to_control_the_robotic_arm.py` contains the following components:

1. **Neural Network Class (`NeuralNetwork_QL`)**: Defines the architecture of the neural network used for Q-learning.
2. **Experience Replay Buffer (`memoryBuffer`)**: Stores and samples experiences for training.
3. **DQN Agent Class (`DQLearningAgent`)**: Implements the DQN algorithm, including action selection, experience storage, and learning from batches.
4. **Training Loop**: Trains the agent over 500 episodes, updating the neural network and plotting the results.
5. **Testing Function (`test_dqn_agent`)**: Evaluates the trained agent on the environment and plots the test results.

## Results

The training process produces several plots to visualize the agent's performance:

1. **Average Episodic Rewards**: Shows the average reward obtained per episode over time. The goal is to see an increasing trend, indicating that the agent is learning to balance the pendulum.
2. **Loss Values**: Displays the loss values during training. A decreasing trend indicates that the neural network is converging.
3. **Epsilon Values**: Tracks the exploration rate (epsilon) over episodes. The epsilon value decays over time, reducing exploration as the agent learns.
4. **Average Test Rewards**: Plots the average rewards obtained during the testing phase, showing the agent's performance after training.

### Example Plots:
- **Training Progress**: The average reward increases over time, indicating successful learning.
- **Loss Values**: The loss decreases as the neural network converges.
- **Test Performance**: The agent maintains a stable performance during testing, demonstrating its ability to balance the pendulum.

## Future Improvements

1. **Hyperparameter Tuning**: Experiment with different learning rates, discount factors, and exploration rates to improve performance.
2. **Advanced Architectures**: Implement more advanced reinforcement learning algorithms like DDPG "Deep Deterministic Policy Gradient" or SAC (Soft Actor-Critic) for continuous action spaces.
3. **Prioritized Experience Replay**: Use prioritized experience replay to focus on important experiences and improve learning efficiency.
4. **Multi-Agent Learning**: Explore multi-agent reinforcement learning to train multiple agents simultaneously and improve robustness.
5. **Reward Shaping**: Adjust the reward function to provide more informative feedback to the agent, potentially speeding up learning.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Reinforcement Learning: An Introduction by Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Q-Learning (DQN) Explained](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc)
