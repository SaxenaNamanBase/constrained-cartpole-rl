import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import csv

class DataLogger:
    def __init__(self, log_params, session_dir):
        self.params = log_params
        self.save_path = session_dir
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_q_values = []
        self.evaluation_rewards = []  # List of (episode_num, reward_value)
        self.epsilons = []
        self.learning_rates = [] 

    def log_episode(self, episode, reward, length, average_q_value=None, epsilon=None, lr=None):
        # Logs standard metrics for an episode.
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if average_q_value is not None:
            self.episode_q_values.append(average_q_value)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        if lr is not None:
            self.learning_rates.append(lr)

    def log_eval(self, episode, eval_reward):
        # Specifically logs evaluation rewards.
        self.evaluation_rewards.append((episode, eval_reward))

    def save_logs_as_csv(self, filename_prefix='training'):
        # Saves all collected data to a CSV in the session folder.
        log_path = os.path.join(self.save_path, f'{filename_prefix}_logs.csv')
        
        # Merge eval rewards with training rewards based on episode number
        eval_dict = {ep: rew for ep, rew in self.evaluation_rewards}

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['episode', 'reward', 'length']
            if len(self.episode_q_values) > 0: header.append('avg_q')
            if len(self.epsilons) > 0: header.append('epsilon')
            if len(self.learning_rates) > 0: header.append('lr') # <--- NEW
            header.append('eval_reward')
            writer.writerow(header)

            for i in range(len(self.episode_rewards)):
                ep_num = i + 1
                row = [ep_num, self.episode_rewards[i], self.episode_lengths[i]]
                
                # Append Q-values
                if len(self.episode_q_values) > 0:
                    row.append(self.episode_q_values[i] if i < len(self.episode_q_values) else '')
                
                # Append Epsilon
                if len(self.epsilons) > 0:
                    row.append(self.epsilons[i] if i < len(self.epsilons) else '')
                
                # Append Learning Rate
                if len(self.learning_rates) > 0:
                    row.append(self.learning_rates[i] if i < len(self.learning_rates) else '')
                
                # Check if this episode had an evaluation
                row.append(eval_dict.get(ep_num, '')) 
                
                writer.writerow(row)
        
        print(f"📁 Logs saved to: {log_path}")

    def get_average_reward(self, last_n_episodes=100):
        if not self.episode_rewards: return 0
        return np.mean(self.episode_rewards[-last_n_episodes:])

    @staticmethod
    def moving_average(data, window_size):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size), 'valid') / window_size
