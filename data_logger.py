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
        self.evaluation_rewards = []
        self.current_episode_data = []

    def log(self, state, action, reward, next_state):
        self.current_episode_data.append((state, action, reward, next_state))

    def log_episode(self, episode, reward, length, average_q_value=None):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if average_q_value is not None:
            self.episode_q_values.append(average_q_value)
        self.current_episode_data = []

    def log_rewards(self, episode, train_reward, eval_reward):
        # Log both training and evaluation rewards for a specific episode.
        self.episode_rewards.append(train_reward)
        self.evaluation_rewards.append((episode, eval_reward))

    def save_model(self, model, model_name):
        os.makedirs(self.params['save_path'], exist_ok=True)
        model_path = os.path.join(self.params['save_path'], f'{model_name}_model.pkl')
        joblib.dump(model, model_path)

    def save_metrics(self, state=''):

        os.makedirs(self.params['save_path'], exist_ok=True)
        metrics_path = os.path.join(self.params['save_path'], f'{state}_metrics.txt')
        
        with open(metrics_path, 'w') as f:
            f.write(f"Average Reward (last 100 episodes): {self.get_average_reward()}\n")
            f.write(f"Success Rate (last 100 episodes): {self.get_success_rate()}\n")
            f.write(f"Average Episode Length (last 100 episodes): {self.get_average_episode_length()}\n")

    def get_average_reward(self, last_n_episodes=100):
        return np.mean(self.episode_rewards[-last_n_episodes:])

    def get_success_rate(self, last_n_episodes=100, success_threshold=195):
        successes = [1 if r >= success_threshold else 0 for r in self.episode_rewards[-last_n_episodes:]]
        return np.mean(successes)

    def get_average_episode_length(self, last_n_episodes=100):
        return np.mean(self.episode_lengths[-last_n_episodes:])

    def plot_results(self, model_name='QLearning', window_size=100):
      os.makedirs(self.params['save_path'], exist_ok=True)

      plt.figure(figsize=(12, 10))
    
      # Episode Rewards plot
      plt.subplot(2, 1, 1)
      plt.plot(self.episode_rewards, label='Episode Rewards')
    
      # Calculate and plot moving average
      if len(self.episode_rewards) >= window_size:
          moving_avg = self.moving_average(self.episode_rewards, window_size)
          plt.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 
                   label=f'Moving Average (window={window_size})', color='red')
    
      plt.title(f'{model_name} - Episode Rewards')
      plt.xlabel('Episode')
      plt.ylabel('Total Reward')
      plt.legend()

      # Episode Lengths plot
      plt.subplot(2, 1, 2)
      plt.plot(self.episode_lengths, label='Episode Lengths')
    
      # Calculate and plot moving average for episode lengths
      if len(self.episode_lengths) >= window_size:
          moving_avg_lengths = self.moving_average(self.episode_lengths, window_size)
          plt.plot(range(window_size-1, len(self.episode_lengths)), moving_avg_lengths, 
                   label=f'Moving Average (window={window_size})', color='red')
    
      plt.title(f'{model_name} - Episode Lengths')
      plt.xlabel('Episode')
      plt.ylabel('Steps')
      plt.legend()

      plt.tight_layout()
      plt.savefig(self.params['save_path'] + f'{model_name}_results.png')
      plt.show()


    def save_logs_as_csv(self, state=''):
     
        os.makedirs(self.params['save_path'], exist_ok=True)
        log_path = os.path.join(self.params['save_path'], f'{state}_logs.csv')

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
        
            # Determine if Q-values are available, as was used sometimes for testing
            include_q_values = len(self.episode_q_values) > 0

            if include_q_values:
                writer.writerow(['Episode', 'Reward', 'Length', 'Average Q-Value'])  # Include Q-Value header
                min_length = min(len(self.episode_rewards), len(self.episode_lengths), len(self.episode_q_values))
            else:
                writer.writerow(['Episode', 'Reward', 'Length'])  # Skip Q-Value header
                min_length = min(len(self.episode_rewards), len(self.episode_lengths))

            for i in range(min_length):
                if include_q_values:
                    avg_q_value = self.episode_q_values[i] if i < len(self.episode_q_values) else None
                    writer.writerow([i+1, self.episode_rewards[i], self.episode_lengths[i], avg_q_value])
                else:
                    writer.writerow([i+1, self.episode_rewards[i], self.episode_lengths[i]])

    @staticmethod
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size
