import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from abc import ABC, abstractmethod
import numpy as np
import config
from exploration_strategies import EpsilonGreedyStrategy, SoftmaxStrategy

class ControlAlgorithm(ABC):
    def __init__(self, control_params, exploration_strategy):
        self.params = control_params
        self.exploration_strategy = exploration_strategy

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def _discretize_state(self, state):
        pass

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) ###
        x = torch.relu(self.fc2(x))
        #return torch.tanh(self.fc3(x))
        return self.fc3(x)

# 2. Define the Replay Buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Continue with your other classes like QLearningControl or DQNControl
# 3. Define the DQNControl class (As described in the previous step)
class DQNControl(ControlAlgorithm):
    def __init__(self, control_params, exploration_strategy, state_dim, action_dim):
        super().__init__(control_params, exploration_strategy)
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=control_params['learning_rate'])
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(1000)
        self.batch_size = control_params.get('batch_size', 64)
        self.gamma = control_params['discount_factor']
        self.epsilon = control_params['epsilon']
        self.min_epsilon = control_params.get('min_epsilon', 0.01)
        self.input_dim = state_dim        ############  
        self.decay_rate = control_params.get('decay_rate', 0.995)
        self.update_target_steps = control_params.get('update_target_steps', 1000)
        self.steps = 0
        self.model = DQNNetwork(state_dim, action_dim)

    def get_action(self, state, explore=True):
        """Decide action based on exploration or exploitation."""
        if explore and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.q_network.fc3.out_features)
        else:
            # Exploitation: best action
            try:
                # FIX: Flatten the array to ensure it is always 1D (e.g., shape (2,))
                state_array = np.array(state, dtype=object)
                state_array = state_array.flatten().astype(np.float32) 
                
                # Defensive check for malformed state (e.g., empty array)
                if state_array.size == 0:
                    raise ValueError("State array is empty or malformed.")
                
                # Secondary check: ensure the state has the expected input size (2 for CartPole)
                if state_array.shape[0] != self.input_dim:
                    raise ValueError(f"State array has incorrect size after flattening: {state_array.shape}")
                    
                state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
            except Exception as e:
                # If anything goes wrong, fall back to exploration (random action)
                print(f"Warning: Failed to get best action (Error: {e}). Falling back to random action. State: {state}.")
                # Fallback action
                return np.random.choice(self.q_network.fc3.out_features)

    def update(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        self.replay_buffer.add((state, action, reward, next_state, done))

        # Only start learning once we have enough samples in the buffer
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)


        states = [np.asarray(s, dtype=np.float32) for s in states]      #############
        next_states = [np.asarray(s, dtype=np.float32) for s in next_states]   ############

        # Convert to torch tensors
        #states = torch.tensor(states, dtype=torch.float32)
        states = torch.tensor(np.stack(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        #next_states = torch.tensor(next_states, dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Calculate the current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate the next Q values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

        # Calculate the target Q values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network periodically
        if self.steps % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps += 1

    def decay_epsilon(self):
        """Decay the epsilon value over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def _discretize_state(self, state):
        return state


class QLearningControl(ControlAlgorithm):
    def __init__(self, control_params, exploration_strategy):
        super().__init__(control_params, exploration_strategy)
        self.q_table = {}
        self.epsilon = control_params['epsilon']
        self.min_epsilon = control_params.get('min_epsilon', 0.01)
        self.decay_rate = control_params.get('decay_rate', 0.995)

    def get_action(self, state, epsilon=0.1):
        
        state = self._discretize_state(state)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]  # Initialize Q-values for unseen state
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.q_table[state]))  # Exploration: random action
        else:
            action = np.argmax(self.q_table[state])  # Exploitation: best action

        return action

    def update(self, state, action, reward, next_state, done):
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.params['learning_rate'] * (reward + self.params['discount_factor'] * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def update_learning_rate(self, new_lr):
        """Dynamically update the learning rate during training."""
        self.learning_rate = new_lr

    def decay_epsilon(self):
        """Decay the epsilon value over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def _discretize_state(self, state):
        return tuple(np.round(x, 1) for x in state)

class SarsaControl(ControlAlgorithm):
    def __init__(self, control_params, exploration_strategy):
        super().__init__(control_params, exploration_strategy)
        self.q_table = {}
        self.state_bins = self._create_bins()  # Create bins for discretization
        self.epsilon = self.params['epsilon']  # Initial epsilon
        self.epsilon_decay = self.params.get('epsilon_decay', 0.99)  # Decay factor
        self.min_epsilon = self.params.get('min_epsilon', 0.01)  # Minimum epsilon

    def _create_bins(self):
        # Example binning for CartPole state
        bins = {
            'x': np.linspace(-4.8, 4.8, 10),  # Cart position
            'x_dot': np.linspace(-3.0, 3.0, 10),  # Cart velocity
            'theta': np.linspace(-0.418, 0.418, 10),  # Pole angle (in radians)
            'theta_dot': np.linspace(-2.0, 2.0, 10)  # Pole angular velocity
        }
        return bins

    '''def _discretize_state(self, state):
        #x, x_dot, theta, theta_dot = state
        theta, theta_dot = state
        state_discrete = (
            np.digitize(x, self.state_bins['x']) - 1,
            np.digitize(x_dot, self.state_bins['x_dot']) - 1,
            np.digitize(theta, self.state_bins['theta']) - 1,
            np.digitize(theta_dot, self.state_bins['theta_dot']) - 1
        )
        state_discrete = (
            np.digitize(theta, self.state_bins['theta']) - 1,
            np.digitize(theta_dot, self.state_bins['theta_dot']) - 1
        )
        return state_discrete'''

    def _discretize_state(self, state):
    # Detect if we are looking at 2D or 4D
      if len(state) == 2:
        theta, theta_dot = state
        state_discrete = (
            np.digitize(theta, self.state_bins['theta']) - 1,
            np.digitize(theta_dot, self.state_bins['theta_dot']) - 1
        )
      else:
        # 4D Mode
        x, x_dot, theta, theta_dot = state
        state_discrete = (
            np.digitize(x, self.state_bins['x']) - 1,
            np.digitize(x_dot, self.state_bins['x_dot']) - 1,
            np.digitize(theta, self.state_bins['theta']) - 1,
            np.digitize(theta_dot, self.state_bins['theta_dot']) - 1
        )
      return state_discrete

    def get_action(self, state):
        state = self._discretize_state(state)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]  # Initialize Q-values for unseen state

        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.q_table[state]))  # Exploration
        else:
            action = np.argmax(self.q_table[state])  # Exploitation

        return action

    def update(self, state, action, reward, next_state, next_action, done):
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]

        # SARSA update rule
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.params['learning_rate'] * (reward + self.params['discount_factor'] * next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


