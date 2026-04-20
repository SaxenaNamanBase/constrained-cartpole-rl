import numpy as np

class EpsilonGreedyStrategy:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(len(q_values))
        else:
            # Exploitation: choose the action with the highest Q-value
            return np.argmax(q_values)

# Code for the Softmax not integrated directly, need to make changes to use this
class SoftmaxStrategy:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select_action(self, q_values):
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probabilities)