import joblib
from control_algorithm import DQNControl, QLearningControl, SarsaControl
import torch.optim as optim
import numpy as np
import gymnasium as gym
import config
import matplotlib.pyplot as plt
from gym_wrapper import GymWrapper
from data_logger import DataLogger
from exploration_strategies import EpsilonGreedyStrategy
import os

def performance_based_lr_update(episode, recent_rewards, control_params, current_lr):
    patience = control_params['patience']
    min_delta = control_params['min_delta']
    decay_factor = control_params['decay_factor']
    min_lr = control_params.get('min_lr', 1e-4)  # Set a minimum learning rate, defaulting to 1e-4
    
    if len(recent_rewards) >= patience:
        avg_reward_recent = np.mean(recent_rewards[-patience:])

        #avg_reward_past = np.mean(recent_rewards[:-patience])

        past_rewards = recent_rewards[:-patience]
        if len(past_rewards) == 0:
            return current_lr # Or use a default value for avg_reward_past

        avg_reward_past = np.mean(past_rewards)
        

        if avg_reward_recent < avg_reward_past + min_delta:
            new_lr = max(current_lr * decay_factor, min_lr)
            #print(f"Performance plateau detected at episode {episode + 1}. Reducing learning rate to {new_lr}")
            return new_lr

    return current_lr

def cross_validation_qlearning(model_class, exploration_strategy_class, config, k_folds=5):
    best_avg_reward = -np.inf
    best_controller = None
    best_logger = None
    best_learning_rate = None
    for fold in range(k_folds):
        initial_lr = config.CONTROL_PARAMS['learning_rate']  # Get the initial learning rate
        current_lr = initial_lr
        print(f"Fold {fold + 1}/{k_folds}: Training on {config.NUM_EPISODES} episodes")
        print(f"Initial Learning Rate for Fold {fold + 1}: {initial_lr}")

        # Reset environment and logger for each fold
        #env = GymWrapper(gym.make('CartPole-v1'))
        #env = GymWrapper(gym.make('CustomCartPoleEnv-v0'))
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                 mode=config.STATE_MODE, 
                 action_mode=config.ACTION_MODE)
        exploration_strategy = exploration_strategy_class(epsilon=config.CONTROL_PARAMS['epsilon'])
        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)
        logger = DataLogger(config.LOG_PARAMS)

        patience_counter = 0
        best_fold_reward = -np.inf
        episodes_completed = 0  # Track the number of episodes completed in this fold
        recent_rewards = []  # Track recent rewards for learning rate adjustment

        # Training loop for the fold
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            episodes_completed += 1  # Increment the episode counter

            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                #next_state, reward, done, _ = env.step(action)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                logger.log(state, action, reward, next_state)

                state = next_state
                episode_reward += reward
                if done:
                    break

            # Log the episode data
            logger.log_episode(episode, episode_reward, t + 1)
            recent_rewards.append(episode_reward)

            #controller.decay_epsilon()

            # Adjust the learning rate based on performance
            new_lr = performance_based_lr_update(episode, recent_rewards, config.CONTROL_PARAMS, current_lr)
            if new_lr != current_lr:  # Update only if there's a change
                controller.update_learning_rate(new_lr)
                current_lr = new_lr

            # Track the best single episode reward for this fold
            if episode_reward > best_fold_reward:
                best_fold_reward = episode_reward
                #best_controller = controller  # Track the best controller within this fold
                #best_logger = logger
                #best_learning_rate = current_lr  # Track the best learning rate within this fold

            # Early stopping logic
            '''if episode_reward > best_fold_reward + config.EARLY_STOPPING['min_delta']:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.EARLY_STOPPING['patience']:
                print(f"Early stopping in fold {fold + 1} at episode {episode + 1} with best reward {best_fold_reward}")
                break'''

        # Print the final learning rate for this fold
        print(f"Final Learning Rate for Fold {fold + 1}: {current_lr}")
        print(f"Best Episode Reward in Fold {fold + 1}: {best_fold_reward}")

        avg_reward = logger.get_average_reward()
        avg_length = logger.get_average_episode_length()
        success_rate = logger.get_success_rate()

        print(f"Fold {fold + 1} Results: Average Reward: {avg_reward}, Success Rate: {success_rate * 100}%")
        print(f"Fold {fold + 1} was early stopped after {episodes_completed} episodes")

        save_path = config.LOG_PARAMS['save_path']
        os.makedirs(save_path, exist_ok=True)

        model_file_path = os.path.join(save_path, 'sarsa_model.pkl')
        joblib.dump(controller, model_file_path)

        # Save the best-performing model for this fold
        #best_logger.save_model(best_controller, f'best_fold_{fold + 1}')
        #best_logger.save_logs_as_csv(state=f'best_fold_{fold + 1}')
        #best_logger.save_metrics(state=f'best_fold_{fold + 1}')
        #best_logger.plot_results(model_name=f'Best_Model_Fold_{fold + 1}')

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_controller = controller  # Track the best controller within this fold
            best_logger = logger
            best_learning_rate = current_lr

        env.close()

    if best_logger and best_controller:
        best_logger.save_model(best_controller, 'best')
        best_logger.save_logs_as_csv(state='best')
        best_logger.save_metrics(state='best')
        best_logger.plot_results(model_name='Best_Model')

    print("Cross-Validation Complete.")
    print(f"Best Average Reward across all folds: {best_avg_reward}")
    return best_avg_reward, controller

def train_dqn(model_class, exploration_strategy_class, config):
    # Load your custom environment
    '''env = GymWrapper(gym.make('CustomCartPoleEnv-v0'))  # Use your custom CartPole env
    eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0'))  # Same for evaluation'''

    env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                 mode=config.STATE_MODE, 
                 action_mode=config.ACTION_MODE)
    eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                      mode=config.STATE_MODE, 
                      action_mode=config.ACTION_MODE)

    # Updated dimensions: state_dim now only includes pole_angle and pole_velocity (2 dimensions), action_dim is motor speed (1 dimension)
    #state_dim = env.observation_space[0]  # pole_angle, pole_velocity
    #action_dim = env.env.action_space.n  # motor speed (continuous action)

    state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.n
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        # For Continuous (Box) spaces, we need the number of output nodes
        action_dim = env.action_space.shape[0]

    # Initialize DQN controller
    controller = model_class(config.CONTROL_PARAMS, exploration_strategy_class(config.CONTROL_PARAMS['epsilon']), state_dim, action_dim)
    logger = DataLogger(config.LOG_PARAMS)

    optimizer_name = 'Adam'
    patience = 50
    min_delta = 1.0
    best_avg_reward = -np.inf
    patience_counter = 0
    recent_rewards = []
    training_rewards = []  # Track training rewards
    evaluation_rewards = []  # Track evaluation rewards

    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        t = 0

        while not done:
            action = controller.get_action(state)  # Get continuous action (motor speed)
            '''next_state, reward, done, _ = env.step(action)'''  #########
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Add this check for debugging ---
            #print(f"Current state type: {type(state)}, shape: {state.shape if isinstance(state, np.ndarray) else 'N/A'}")
            #print(f"Next state type: {type(next_state)}, shape: {next_state.shape if isinstance(next_state, np.ndarray) else 'N/A'}")
            # --- End of check ---

            controller.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            t += 1

        # Decay epsilon after the episode
        controller.decay_epsilon()

        # Log the episode data
        logger.log_episode(episode, episode_reward, t + 1)
        training_rewards.append(episode_reward)

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > patience:
            recent_rewards.pop(0)

        # Calculate average reward over the last `patience` episodes
        avg_reward = np.mean(recent_rewards)

        # Check for improvement
        if avg_reward > best_avg_reward + min_delta:
            best_avg_reward = avg_reward
            patience_counter = 0  # Reset the patience counter if there's improvement
        else:
            patience_counter += 1  # Increment the patience counter if no improvement

        # Output progress every few episodes
        if (episode + 1) % 10 == 0:
            eval_reward = evaluate_agent(controller, eval_env)  # Evaluate the model
            evaluation_rewards.append(eval_reward)
            print(f"Episode {episode + 1}/{config.NUM_EPISODES} - Training Reward: {episode_reward:.2f} - Epsilon: {controller.epsilon:.4f} - Avg Reward: {avg_reward:.2f} - Evaluation Reward: {eval_reward:.2f}")

            logger.log_rewards(episode + 1, episode_reward, eval_reward)

        # Learning rate update if performance drops
        current_lr = controller.optimizer.param_groups[0]['lr']
        new_lr = performance_based_lr_update(episode, recent_rewards, config.CONTROL_PARAMS, current_lr)
        if new_lr != current_lr:
            controller.update_learning_rate(new_lr)

    print("DQN Training Complete.")
    env.close()

    # Log the model and hyperparameters
    model_info = {
        'model': controller,
        'hyperparameters': {
            'learning_rate': config.CONTROL_PARAMS['learning_rate'],
            'discount_factor': config.CONTROL_PARAMS['discount_factor'],
            'epsilon': config.CONTROL_PARAMS['epsilon'],
            'min_epsilon': config.CONTROL_PARAMS.get('min_epsilon', 0.01),
            'decay_rate': config.CONTROL_PARAMS.get('decay_rate', 0.995),
            'buffer_size': config.CONTROL_PARAMS.get('buffer_size', 10000),
            'batch_size': config.CONTROL_PARAMS.get('batch_size', 64),
            'update_target_steps': config.CONTROL_PARAMS.get('update_target_steps', 1000),
            'optimizer': optimizer_name  # Log the optimizer name
        }
    }

    # Save model and hyperparameters
    eval_interval = 10  # Example interval for evaluation
    save_path = config.LOG_PARAMS['save_path']
    os.makedirs(save_path, exist_ok=True)
    plot_rewards(training_rewards, evaluation_rewards, eval_interval, save_path=save_path)
    joblib.dump(model_info, os.path.join(save_path, 'trained_dqn_model_with_params.pkl'))

    logger.save_logs_as_csv(state='train')
    logger.save_metrics(state='train')

    return controller

'''def train_dqn_until_overfitting(model_class, exploration_strategy_class, config, max_episodes=5000):
    
    
    env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                 mode=config.STATE_MODE, 
                 action_mode=config.ACTION_MODE)
    eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                      mode=config.STATE_MODE, 
                      action_mode=config.ACTION_MODE)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if config.ACTION_MODE == 'discrete' else 1
    controller = model_class(config.CONTROL_PARAMS, exploration_strategy_class(config.CONTROL_PARAMS['epsilon']), state_dim, action_dim)
    logger = DataLogger(config.LOG_PARAMS)

    patience = 50  # Number of episodes to wait before considering early stopping
    min_delta = 1.0  # Minimum change to qualify as an improvement
    best_avg_reward = -np.inf
    patience_counter = 0
    recent_rewards = []
    training_rewards = []  # List to store training rewards
    evaluation_rewards = []  # List to store evaluation rewards

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        t = 0
        while not done:
            action = controller.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            controller.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            t += 1

        # Decay epsilon after the episode
        controller.decay_epsilon()

        logger.log_episode(episode, episode_reward, t + 1)
        training_rewards.append(episode_reward)

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > patience:
            recent_rewards.pop(0)

        # Calculate average reward over the last `patience` episodes
        avg_reward = np.mean(recent_rewards)

        # Check for improvement
        if avg_reward > best_avg_reward + min_delta:
            best_avg_reward = avg_reward
            patience_counter = 0  # Reset the patience counter if there's an improvement
        else:
            patience_counter += 1  # Increment the patience counter if no improvement

        # Evaluate the model periodically
        if (episode + 1) % 10 == 0:
            eval_reward = evaluate_agent(controller, eval_env)
            evaluation_rewards.append(eval_reward)
            print(f"Episode {episode + 1} - Training Reward: {episode_reward:.2f} - Avg Reward: {avg_reward:.2f} - Evaluation Reward: {eval_reward:.2f}")

            logger.log_rewards(episode + 1, episode_reward, eval_reward)

            # Check for overfitting
            if len(evaluation_rewards) > patience and evaluation_rewards[-1] < np.max(evaluation_rewards[:-patience]):
                print(f"Overfitting detected at episode {episode + 1}. Stopping training.")
                break

    print("DQN Training Complete.")
    env.close()

    # Log the model along with its hyperparameters
    model_info = {
        'model': controller,
        'hyperparameters': {
            'learning_rate': config.CONTROL_PARAMS['learning_rate'],
            'discount_factor': config.CONTROL_PARAMS['discount_factor'],
            'epsilon': config.CONTROL_PARAMS['epsilon'],
            'min_epsilon': config.CONTROL_PARAMS.get('min_epsilon', 0.01),
            'decay_rate': config.CONTROL_PARAMS.get('decay_rate', 0.995),
            'buffer_size': config.CONTROL_PARAMS.get('buffer_size', 10000),
            'batch_size': config.CONTROL_PARAMS.get('batch_size', 64),
            'update_target_steps': config.CONTROL_PARAMS.get('update_target_steps', 1000),
            'optimizer': config.CONTROL_PARAMS.get('optimizer_name', 'Adam')  # Log the optimizer name
        }
    }

    # Save the model and hyperparameters together
    eval_interval = 10  # Example value, adjust based on your needs
    save_path = config.LOG_PARAMS['save_path']
    os.makedirs(save_path, exist_ok=True)
    plot_rewards(training_rewards, evaluation_rewards, eval_interval, save_path=save_path)
    joblib.dump(model_info, os.path.join(save_path, 'trained_dqn_model_with_params.pkl'))

    logger.save_logs_as_csv(state='train')
    logger.save_metrics(state='train')

    return controller'''

def train_dqn_until_overfitting(model_class, exploration_strategy_class, config, max_episodes=1000, stop_logic='overfitting'):
    env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                     mode=config.STATE_MODE, 
                     action_mode=config.ACTION_MODE)
    eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                          mode=config.STATE_MODE, 
                          action_mode=config.ACTION_MODE)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if config.ACTION_MODE == 'discrete' else 1
    
    controller = model_class(config.CONTROL_PARAMS, 
                             exploration_strategy_class(config.CONTROL_PARAMS['epsilon']), 
                             state_dim, action_dim)
    logger = DataLogger(config.LOG_PARAMS)

    # 2. Tracking Variables
    training_rewards = []
    evaluation_rewards = []
    eval_window = []
    
    best_window_avg = -np.inf
    save_path = config.LOG_PARAMS['save_path']
    os.makedirs(save_path, exist_ok=True)

    print(f"Starting Training | Mode: {stop_logic} | Max Episodes: {max_episodes}")

    # 3. Training Loop
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        t = 0
        
        while not done:
            action = controller.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            controller.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            t += 1

        controller.decay_epsilon()
        training_rewards.append(episode_reward)
        logger.log_episode(episode, episode_reward, t)

        # 4. Periodic Evaluation (Every 10 Episodes)
        if (episode + 1) % 10 == 0:
            eval_reward = evaluate_agent(controller, eval_env)
            evaluation_rewards.append(eval_reward)
            eval_window.append(eval_reward)
            
            if len(eval_window) > 5:  # Keep a sliding window of 5 evaluations
                eval_window.pop(0)
            
            current_window_avg = np.mean(eval_window)
            logger.log_rewards(episode + 1, episode_reward, eval_reward)

            # --- BEST MODEL CHECKPOINTING ---
            # Scientifically safer than just saving the "last" model
            if current_window_avg > best_window_avg:
                best_window_avg = current_window_avg
                checkpoint = {
                    'model': controller,
                    'episode': episode + 1,
                    'avg_reward': current_window_avg,
                    'config': config.CONTROL_PARAMS
                }
                joblib.dump(checkpoint, os.path.join(save_path, 'best_dqn_model.pkl'))
                print(f"⭐ New Best Model Saved (Avg: {current_window_avg:.2f})")

            # --- STOP LOGIC: OVERFITTING / DIVERGENCE ---
            if stop_logic == 'overfitting':
                # Rule 1: Must have a "Best" baseline (at least 100 reward)
                # Rule 2: Must drop 30% below that baseline to trigger stop
                if best_window_avg > 100 and current_window_avg < (best_window_avg * 0.5):
                    print(f"🛑 Overfitting/Divergence detected at episode {episode + 1}.")
                    print(f"Current Avg {current_window_avg:.2f} is < 70% of Best {best_window_avg:.2f}")
                    break

            print(f"Episode {episode + 1} | Train: {episode_reward} | Eval Avg: {current_window_avg:.2f}")

    # 5. Final Logging & Cleanup
    print("Training Process Finalized.")
    env.close()

    # Save final model info
    model_info = {'model': controller, 'hyperparameters': config.CONTROL_PARAMS}
    joblib.dump(model_info, os.path.join(save_path, 'final_dqn_model.pkl'))
    
    plot_rewards(training_rewards, evaluation_rewards, 10, save_path=save_path)
    logger.save_logs_as_csv(state='train')
    
    return controller


def train_sarsa(model_class, exploration_strategy_class, config):
    env = GymWrapper(gym.make('CartPole-v1'))
    exploration_strategy = exploration_strategy_class(config.CONTROL_PARAMS['epsilon'])
    controller = model_class(config.CONTROL_PARAMS, exploration_strategy)
    logger = DataLogger(config.LOG_PARAMS)

    reward_history = []

    for episode in range(config.NUM_EPISODES_SARSA):
        state, _ = env.reset()
        action = controller.get_action(state)  # Select initial action
        episode_reward = 0

        for t in range(config.MAX_STEPS):
            #next_state, reward, done, _ = env.step(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = controller.get_action(next_state)  # Select next action
            
            controller.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action  # Move to the next state-action pair
            episode_reward += reward
            
            if done:
                break

        # Decay epsilon after each episode
        controller.decay_epsilon()

        logger.log_episode(episode, episode_reward, t + 1)

        reward_history.append(episode_reward)

        # Calculate the average reward over the last 100 episodes
        if len(reward_history) > 100:
            reward_history.pop(0)  # Keep only the last 100 rewards

        average_reward = sum(reward_history) / len(reward_history)

        print(f"Episode {episode}: Reward = {episode_reward}, Average Reward (last 100 episodes) = {average_reward:.2f}, Steps = {t + 1}")

    env.close()

    # Create results directory if it doesn't exist
    save_path = config.LOG_PARAMS['save_path']
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    plot_sarsa_training(logger, config, save_path=save_path)

    # Save the model
    model_file_path = os.path.join(save_path, 'sarsa_model.pkl')
    joblib.dump(controller, model_file_path)

    # Save the logs as CSV
    logger.save_logs_as_csv(state='train')

    # Save the metrics
    logger.save_metrics(state='train')

    return controller


def evaluate_agent(controller, eval_env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state,_ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Set explore=False to disable exploration during evaluation
            action = controller.get_action(state, explore=False)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    avg_eval_reward = total_reward / num_episodes
    return avg_eval_reward

def plot_rewards(training_rewards, evaluation_rewards, eval_interval, save_path=None):
    # Debugging prints
    print(f"Training Rewards: {training_rewards}")
    print(f"Evaluation Rewards: {evaluation_rewards}")
    
    if not training_rewards or not evaluation_rewards:
        print("One of the reward lists is empty. Skipping plot.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    plt.plot(training_rewards, label='Training Reward', color='blue')
    
    # Plot evaluation rewards at the appropriate intervals
    eval_episodes = list(range(eval_interval, len(training_rewards) + 1, eval_interval))
    if len(evaluation_rewards) > 0 and len(eval_episodes) == len(evaluation_rewards):
        plt.plot(eval_episodes, evaluation_rewards, label='Evaluation Reward', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards Over Time')
    plt.legend()
    if save_path:
        plot_path = os.path.join(save_path, 'training_plot_dqn.png')
        plt.savefig(plot_path)
    plt.show()

'''def run_cross_validation_or_training(algorithm='qlearning'):
    if algorithm == 'qlearning':
        best_reward, controller = cross_validation_qlearning(QLearningControl, EpsilonGreedyStrategy, config)
        print(f"Best Average Reward from Q-Learning Cross-Validation: {best_reward}")
        if controller is None:
           print("Empty")
        return controller  # or return best_reward if you want
    elif algorithm == 'dqn':
        #controller = train_dqn_until_overfitting(DQNControl, EpsilonGreedyStrategy, config, max_episodes=5000)
        controller = train_dqn(DQNControl, EpsilonGreedyStrategy, config)
        print(f"DQN Training completed. You can now test the agent.")
        return controller
    elif algorithm == 'sarsa':
        controller = train_sarsa(SarsaControl, EpsilonGreedyStrategy, config)
        print(f"SARSA Training completed. You can now test the agent.")
        return controller
    else:
        raise ValueError("Unknown algorithm specified. Use 'qlearning' or 'dqn'.")'''
  
def run_cross_validation_or_training(algorithm, mode, action_mode, stop_logic):
    # 1. Update global config variables based on dropdown choices
    config.STATE_MODE = mode
    config.STOP_LOGIC = stop_logic
    config.ACTION_MODE = action_mode
    config.STATE_DIM = 2 if mode == '2D' else 4

    print(f"--- Configuration Updated ---")
    print(f"Algorithm: {algorithm} | Mode: {config.STATE_MODE} | State Dim: {config.STATE_DIM} | Action: {action_mode}")

    # 2. Q-LEARNING BLOCK
    if algorithm == 'qlearning':
        # Uses your existing CV logic
        best_reward, controller = cross_validation_qlearning(QLearningControl, EpsilonGreedyStrategy, config)
        print(f"Best Average Reward from Q-Learning Cross-Validation: {best_reward}")
        return controller

    # 3. DQN BLOCK
    elif algorithm == 'dqn':
        # Check stop_logic from dropdown to decide which function to run
        if config.STOP_LOGIC == 'overfitting':
            controller = train_dqn_until_overfitting(DQNControl, EpsilonGreedyStrategy, config)
        else:
            controller = train_dqn(DQNControl, EpsilonGreedyStrategy, config)
            
        print(f"DQN Training completed.")
        return controller

    # 4. SARSA BLOCK
    elif algorithm == 'sarsa':
        # Sarsa will now use the dynamic bins we set up earlier
        controller = train_sarsa(SarsaControl, EpsilonGreedyStrategy, config)
        print(f"SARSA Training completed.")
        return controller

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def plot_sarsa_training(logger, config, save_path=None):
    episodes = range(len(logger.episode_rewards))

    # Calculate Rolling Average of Rewards (e.g., over the last 100 episodes)
    window_size = 100
    rolling_avg = [sum(logger.episode_rewards[i:i+window_size]) / window_size 
                   for i in range(len(logger.episode_rewards) - window_size + 1)]

    # Plot Training Rewards and Rolling Average in a single plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(episodes, logger.episode_rewards, label='Training Reward', color='blue')
    plt.plot(episodes[:len(rolling_avg)], rolling_avg, label=f'Average Reward (last {window_size} episodes)', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SARSA Training and Average Reward Over Time')
    plt.legend()
    if save_path:
        plot_path = os.path.join(save_path, 'training_plot.png')
        plt.savefig(plot_path)
    plt.show()