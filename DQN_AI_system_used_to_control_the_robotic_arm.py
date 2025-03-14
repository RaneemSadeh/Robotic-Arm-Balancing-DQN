#Importing the libraries 
import matplotlib.pyplot as plt # Is used for plotting graphs
import gymnasium as gym # used For the gym environment
import random as random_library # Is used For random number generation
import numpy as np # Is used For numerical operations
import torch # PyTorch library for deep learning
import torch.nn as nn # Neural network module in PyTorch
import torch.optim as optim # Optimization module in PyTorch
import collections as colections_model # The Python collections module
from torch.optim.lr_scheduler import StepLR # I did Import the learning rate scheduler from PyTorch

# Initializing The Neural Network class for Q-learning
class NeuralNetwork_QL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork_QL, self).__init__()
        # Define the layers of the neural network
        self.Layer1 = nn.Linear(input_size, hidden_size) # Here I defined the first Linear layer
        self.activationfanction_relu = nn.ReLU() # Here I defined the Activation function Relu
        self.Layer2 = nn.Linear(hidden_size, hidden_size) # Here I defined the second Linear layer
        self.Layer3 = nn.Linear(hidden_size, hidden_size) # Here I defined the third Linear layer
        self.Layer4 = nn.Linear(hidden_size, output_size) # Here I defined the fourth Linear layer
        self.activationfanction_sigmoid = nn.Sigmoid() # Here I defined the Activation function Sigmoid

    def forward(self, x):
        # Forward pass through the neural network
        x = self.Layer1(x)
        x = self.activationfanction_relu(x)
        x = self.Layer2(x)
        x = self.activationfanction_relu(x)
        x = self.Layer3(x)
        x = self.activationfanction_relu(x)
        x = self.Layer4(x)
        x = self.activationfanction_sigmoid(x) * 4 - 2 # The output range [-2, 2] if I want to use the sigmoid I should * 4 and then -2
        return x
# Initializing The Class: Experience Replay Buffer
class memoryBuffer:
    def __init__(self, capacity):
        # Initializing a deque as a buffer with a maximum capacity
        self.buffer_storage = colections_model.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # I added a tuple that represent an experience to the buffer
        self.buffer_storage.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        state, action, reward, next_state, done = zip(*random_library.sample(self.buffer_storage, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        # It's Going To Return the current length of the buffer
        return len(self.buffer_storage)
# Initializing The Class: Deep Q-Learning Agent
class DQLearningAgent:
    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.99, epsilon=1, buffer_size=10000):
         # Initialize the DQN agent with neural network, hyperparameters, optimizer, loss function, and replay buffer
        self.neural_network = NeuralNetwork_QL(state_size, hidden_size, action_size)
        self.gamma = gamma # the discount factor
        self.epsilon_value = epsilon #  the exploration rate
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.0001) # The Adam optimizer will update the neural network parameters
        self.criterion = nn.MSELoss() # criterion for computing the loss during training by the mean sequare error
        self.memory_buffer = memoryBuffer(buffer_size) # It will store and sample the experiences for training.

    def choose_action(self, state):
        # This fanction work  as: Choosing action based on epsilon-greedy strategy
        if random_library.uniform(0, 1) < self.epsilon_value: 
            # Explore: Random action within the action space
            return [4 * random_library.random() - 2]  
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                # Exploit: Will Choose action based on the learned model
                action_result = self.neural_network(state).item()  
                return [action_result]

    def store_in_the_buffer(self, state, action, reward, next_state, done):
        # This Fanction Is Going To Store the current experience in the replay buffer
        self.memory_buffer.push(state, action, reward, next_state, done)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)  # Adjust step_size and gamma as needed
    def learn_from_batch(self, batch_size):
        # This Fanction Will Learn from a batch of experiences sampled from the replay buffer
        if len(self.memory_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory_buffer.sample_batch(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

       
        predicted_actions = self.neural_network(states)

        # Compute the loss using the predicted actions and update the neural network
        loss_Value = ((predicted_actions - actions) ** 2 * (-rewards)).mean()

        self.optimizer.zero_grad() # Gradients need to be clear all the optimized parameters before running the backward pass
        loss_Value.backward() # Will Computes the gradient of the loss
        self.optimizer.step() # Updates the parameters based on the computed gradients using the optimization algorithm (Adam)
        self.scheduler.step()
        return loss_Value # Returning the loss value

    def update_epsilon(self, decay_rate):
        # This Fanction Will Update epsilon with decay for exploration-exploitation trade-off
        self.epsilon_value *= decay_rate

# Using the Penulum Environment From The OpenAI Gymnasium
env = gym.make("Pendulum-v1")
number_of_episodes = 500 # Initializing the number of episodes = 500
# intializing them through the training
Overall_Scores = []
cumulative_rewards_list = []
loss_value_list = []
epsilons_list = []
state_dimension_size = env.observation_space.shape[0]
action_dinension_size = env.action_space.shape[0]

# I Created an object of the DQLearningAgent class with the dimensions of the state and action spaces
agent = DQLearningAgent(state_dimension_size, action_dinension_size)
batch_size_value = 256


episode_reward_list = []
average_reward_list = []

# Training loop over all the 500 episodes
for current_episode in range(number_of_episodes):
    observation, info = env.reset()
    total_reward = 0
    cumulative_reward = 0
    state = observation

    # Loop over time steps within an episode
    while True:
        action_taken = agent.choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action_taken)
        next_state_value = observation

        agent.store_in_the_buffer(state, action_taken, reward, next_state_value, terminated or truncated)
        if len(agent.memory_buffer) > batch_size_value:
            loss_value = agent.learn_from_batch(batch_size_value)
            if loss_value is not None:
                loss_value_list.append(loss_value.item())

        total_reward += reward
        cumulative_reward += reward
        state = next_state_value

        if terminated or truncated:
            break

    # Updatung the epsilon
    agent.update_epsilon(0.999)
    epsilons_list.append(agent.epsilon_value)

    
    episode_reward_list.append(total_reward)
    average_reward = np.mean(episode_reward_list[-40:])
    print("-The Average * {} * For this Episode ==> {}".format(average_reward, current_episode))
    average_reward_list.append(average_reward)

# Plotting the average of the episodic rewards
plt.plot(average_reward_list)
plt.xlabel("-The Number Of Episode-")
plt.ylabel("-The Avgerage For The Episodic Reward-")
plt.title("-The Average Episodic Reward Over All The Episodes-")
plt.show()



# Plotting the loss and the epsilon value
plt.figure(figsize=(12, 6))  
plt.subplot(2, 1, 1)  
plt.plot(loss_value_list)
plt.title("-The Loss For Every Training Step-")
plt.xlabel("-The Number Of Buffer Size-")



plt.subplot(2, 1, 2)  
plt.plot(epsilons_list)
plt.title("-Epsilon Values per Episode-")
plt.xlabel("-The Number Of Episode-")

plt.tight_layout()
plt.show()

# here I will calculate average training score then I will print the result
average_score = sum(episode_reward_list) / number_of_episodes
print(f"-The Average Score over --> {number_of_episodes} -Episodes --> {average_score}")

# Intializing a function to test the DQN agent remains the same
def test_dqn_agent(env, agent, test_episodes_count):
    test_scores_list = []

    for current_episode in range(test_episodes_count):
        observation, _ = env.reset()
        total_reward_value = 0
        state = observation # sets the current state to the initial observation

        while True:
            action = agent.choose_action(state)  # The agent will choose an action based on the current state
            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation

            total_reward_value += reward
            state = next_state

            if terminated or truncated:
                break

        test_scores_list.append(total_reward_value)

    return test_scores_list


test_episodes_count = 500

original_epsilon = agent.epsilon_value
# setting epsilon to 0 during testing
agent.epsilon_value = 0


test_scores_list = test_dqn_agent(env, agent, test_episodes_count)


agent.epsilon_value = original_epsilon

# This lines will plot the average test rewards for the last 40 episodes during testing
plt.plot([np.mean(test_scores_list[max(0, i - 39):(i + 1)]) for i in range(len(test_scores_list))])
plt.xlabel("-The Test For The Episode-")
plt.ylabel("-For The Last 40 Episodes: The Average Reward-")
plt.title('-The mean rewards from the recent 40 episodes evaluated for the DQN algorithm in the Pendulum-v1 environment-')
plt.show()

# Calculating and print the average of the test score for all the testing episodes
average_test_score = sum(test_scores_list) / 500  

print(f"-The Mean performance evaluation score across 500 episodes: {average_test_score}")

env.close() # Close the environment