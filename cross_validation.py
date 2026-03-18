import joblib
import copy
from control_algorithm import DQNControl, QLearningControl, SarsaControl
import torch.optim as optim
import numpy as np
import gymnasium as gym
import config
import matplotlib.pyplot as plt
import pandas as pd
from gym_wrapper import GymWrapper
from data_logger import DataLogger
from exploration_strategies import EpsilonGreedyStrategy
import os
import datetime

def create_session_folder(algorithm_name, config):
    """Creates a unique directory for the current training session."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{algorithm_name}_{config.STATE_MODE}_{timestamp}"
    
    # Ensure the base Results directory exists
    base_path = config.LOG_PARAMS['save_path']
    session_dir = os.path.join(base_path, folder_name)
    
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

'''def performance_based_lr_update(episode, recent_rewards, control_params, current_lr):
    patience = control_params['patience']
    min_delta = control_params['min_delta']
    decay_factor = control_params['decay_factor']
    min_lr = control_params.get('min_lr', 1e-4)
    
    if len(recent_rewards) >= patience:
        avg_reward_recent = np.mean(recent_rewards[-patience:])
        past_rewards = recent_rewards[:-patience]
        if len(past_rewards) == 0: return current_lr

        avg_reward_past = np.mean(past_rewards)
        if avg_reward_recent < avg_reward_past + min_delta:
            return max(current_lr * decay_factor, min_lr)
    return current_lr'''

def performance_based_lr_update(episode, recent_rewards, control_params, current_lr):

    patience = control_params.get('patience', 100)
    min_delta = control_params.get('min_delta', 0.01)
    decay_factor = control_params.get('decay_factor', 0.5)
    min_lr = control_params.get('min_lr', 1e-3) # Default to 0.001 if missing

    if episode > 0 and episode % patience == 0:
        if len(recent_rewards) >= patience * 2:
            avg_recent = np.mean(recent_rewards[-patience:])
            avg_previous = np.mean(recent_rewards[-2*patience:-patience])

            # If improvement is less than min_delta, decay the LR
            if avg_recent < avg_previous + min_delta:
                new_lr = max(current_lr * decay_factor, min_lr)
                if new_lr < current_lr:
                    print(f"📉 [Ep {episode}] Plateau detected. LR: {current_lr:.6f} -> {new_lr:.6f}")
                return new_lr
                
    return current_lr

'''def cross_validation_qlearning(model_class, exploration_strategy_class, config, session_dir, k_folds=5):
    best_avg_reward = -np.inf
    best_controller = None
    best_logger_data = None

    winning_fold = 0
    winning_episode = 0
    win_final_eps = 0

    init_lr = config.CONTROL_PARAMS['learning_rate_qlearning']
    init_eps = config.CONTROL_PARAMS['epsilon']
    print(f"🚀 Initial Params | LR: {init_lr} | Start Epsilon: {init_eps}")
    
    for fold in range(k_folds):
        best_eval_in_fold = -np.inf
        #current_lr = config.CONTROL_PARAMS['learning_rate_qlearning']
        peak_episode_in_fold = 0
        current_lr = init_lr

        print(f"\n--- Fold {fold + 1}/{k_folds} | Session: {os.path.basename(session_dir)} ---")

        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                         mode=config.STATE_MODE, 
                         action_mode=config.ACTION_MODE)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                      mode=config.STATE_MODE, action_mode=config.ACTION_MODE)
        
        controller = model_class(config.CONTROL_PARAMS, exploration_strategy_class(init_eps))
        # Pass session_dir to DataLogger
        logger = DataLogger(config.LOG_PARAMS, session_dir=session_dir)
        #fold_rewards = [] 

        recent_rewards = []
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done: break

            logger.log_episode(episode+1, episode_reward, t + 1, epsilon=controller.epsilon)
            #recent_rewards.append(episode_reward)
            controller.decay_epsilon()

            if (episode + 1) % 20 == 0:  # Evaluate every 20 episodes
              eval_reward = evaluate_agent(controller, eval_env)
              logger.log_eval(episode + 1, eval_reward)

              if eval_reward >= best_eval_in_fold:
                    best_eval_in_fold = eval_reward
                    peak_episode_in_fold = episode + 1
                    fold_model_path = os.path.join(session_dir, f'temp_fold_{fold+1}_best.pkl')
                    joblib.dump(controller, fold_model_path)

            # LR Update
            new_lr = performance_based_lr_update(episode, recent_rewards, config.CONTROL_PARAMS, current_lr)
            if new_lr != current_lr:
                controller.update_learning_rate(new_lr)
                current_lr = new_lr

        #logger.save_to_csv(fold_name=f"fold_{fold+1}")
        #logger.save_logs_as_csv(filename_prefix=f"fold_{fold+1}")

        avg_reward = logger.get_average_reward()
        final_fold_lr = current_lr
        final_fold_eps = controller.epsilon
        
        print(f" Fold {fold+1} Finished")
        print(f"   - Avg Reward: {avg_reward:.2f}")
        print(f"   - Final LR: {final_fold_lr:.6f}")
        print(f"   - Final Epsilon: {final_fold_eps:.4f}")
        #print(f"Fold {fold+1} Finished. Avg Reward: {avg_reward:.2f} | Final Epsilon: {controller.epsilon:.3f}")

        #if avg_reward > best_avg_reward:
            #best_avg_reward = avg_reward
            #best_controller = controller
            # Save the best model of the CV into the session folder
            #best_logger_data = logger
            #joblib.dump({'model': best_controller, 'config': config.CONTROL_PARAMS}, 
                        #os.path.join(session_dir, 'qlearning_best_model.pkl'))

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            winning_fold = fold + 1
            best_controller = joblib.load(os.path.join(session_dir, f'temp_fold_{fold+1}_best.pkl'))
            best_logger_data = logger
            win_final_lr = final_fold_lr
            win_final_eps = final_fold_eps

    if best_logger_data is not None:
        print(f"\n Saving Best Fold (Avg Reward: {best_avg_reward:.2f})")
        
        joblib.dump({'model': best_controller, 'config': config.CONTROL_PARAMS}, 
                    os.path.join(session_dir, 'qlearning_best_model.pkl'))
        
        best_logger_data.save_logs_as_csv(filename_prefix="qlearning_best_fold")
        
        plot_rewards_from_csv(session_dir, csv_filename="qlearning_best_fold_logs.csv")
    env.close()
    return best_avg_reward, best_controller'''

def cross_validation_qlearning(model_class, exploration_strategy_class, config, session_dir, k_folds=5):
    best_avg_reward = -np.inf
    best_controller = None
    best_logger_data = None

    winning_fold = 0
    winning_episode = 0
    win_final_lr = 0
    win_final_eps = 0

    init_lr = config.CONTROL_PARAMS['learning_rate_qlearning']
    init_eps = config.CONTROL_PARAMS['epsilon']
    
    print(f"🚀 Initial Params | LR: {init_lr} | Start Epsilon: {init_eps}")
    
    for fold in range(k_folds):
        best_eval_in_fold = -np.inf
        peak_episode_in_fold = 0
        current_lr = init_lr

        print(f"\n--- Fold {fold + 1}/{k_folds} | Session: {os.path.basename(session_dir)} ---")

        # Initialize Environments
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                         mode=config.STATE_MODE, action_mode=config.ACTION_MODE)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                              mode=config.STATE_MODE, action_mode=config.ACTION_MODE)
        
        # Initialize Controller and Logger
        controller = model_class(config.CONTROL_PARAMS, exploration_strategy_class(init_eps))
        logger = DataLogger(config.LOG_PARAMS, session_dir=session_dir)

        recent_rewards = []
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done: break

            # LOGGING: Include LR in the logs
            logger.log_episode(episode + 1, episode_reward, t + 1, 
                               epsilon=controller.epsilon, lr=current_lr)
            
            controller.decay_epsilon()
            recent_rewards.append(episode_reward)

            # EVALUATION BLOCK
            if (episode + 1) % 20 == 0:
                eval_reward = evaluate_agent(controller, eval_env)
                logger.log_eval(episode + 1, eval_reward)

                # Capture the peak performance of this fold
                if eval_reward >= best_eval_in_fold:
                    best_eval_in_fold = eval_reward
                    peak_episode_in_fold = episode + 1
                    #fold_model_path = os.path.join(session_dir, f'temp_fold_{fold+1}_best.pkl')
                    #joblib.dump(controller, fold_model_path)
                    session_champion_brain = copy.deepcopy(controller)

            # PERFORMANCE-BASED LR UPDATE
            #new_lr = performance_based_lr_update(episode, recent_rewards, config.CONTROL_PARAMS, current_lr)
            #if new_lr != current_lr:
                #controller.update_learning_rate(new_lr)
                #current_lr = new_lr

            current_lr = performance_based_lr_update(episode, recent_rewards, config.CONTROL_PARAMS, current_lr)
            if current_lr != controller.learning_rate:
                controller.update_learning_rate(current_lr)

        # FOLD SUMMARY
        avg_reward = logger.get_average_reward()
        final_fold_lr = current_lr
        final_fold_eps = controller.epsilon
        
        print(f"✅ Fold {fold+1} Finished")
        print(f"   - Avg Reward: {avg_reward:.2f}")
        print(f"   - Peak Eval: {best_eval_in_fold:.2f} (at Ep {peak_episode_in_fold})")
        print(f"   - Final LR: {final_fold_lr:.6f}")
        print(f"   - Final Epsilon: {final_fold_eps:.4f}")

        fold_final_path = os.path.join(session_dir, f'qlearning_fold_{fold+1}_final.pkl')
        joblib.dump({'model': controller, 'config': config.CONTROL_PARAMS}, fold_final_path)
        
        print(f" Fold {fold+1} Finished. Final model saved to: {os.path.basename(fold_final_path)}")

        # UPDATE OVERALL SESSION WINNER
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            winning_fold = fold + 1
            winning_episode = peak_episode_in_fold
            win_final_lr = final_fold_lr
            win_final_eps = final_fold_eps
            best_logger_data = logger
            # Load the peak version of the winner
            #best_controller = joblib.load(os.path.join(session_dir, f'temp_fold_{fold+1}_best.pkl'))
            best_controller = session_champion_brain

    # FINAL CHAMPION SUMMARY
    if best_logger_data is not None:
        print(f"\n")
        print(f" SESSION WINNER: Fold {winning_fold}")
        print(f" Best Avg Reward: {best_avg_reward:.2f}")
        print(f" Peak Performance: Episode {winning_episode}")
        print(f" Final Learning Rate: {win_final_lr:.6f}")
        print(f" Final Epsilon: {win_final_eps:.4f}")
        print(f"\n")

        # Save the final model dictionary
        joblib.dump({'model': best_controller, 'config': config.CONTROL_PARAMS}, 
                    os.path.join(session_dir, 'qlearning_best_model.pkl'))
        
        # Finalize Logs and Generate Plot
        best_logger_data.save_logs_as_csv(filename_prefix="qlearning_best_fold")
        plot_rewards_from_csv(session_dir, csv_filename="qlearning_best_fold_logs.csv")
    
    env.close()
    return best_avg_reward, best_controller

def train_dqn(model_class, exploration_strategy_class, config, session_dir):
    env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                     mode=config.STATE_MODE, action_mode=config.ACTION_MODE)
    eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                          mode=config.STATE_MODE, action_mode=config.ACTION_MODE)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    controller = model_class(config.CONTROL_PARAMS, exploration_strategy_class(config.CONTROL_PARAMS['epsilon']), state_dim, action_dim)
    logger = DataLogger(config.LOG_PARAMS, session_dir=session_dir)

    best_eval_reward = -np.inf
    training_rewards = []
    evaluation_rewards = []

    for episode in range(config.NUM_EPISODES_DQN):
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
        logger.log_episode(episode + 1, episode_reward, t, epsilon=controller.epsilon)

        if (episode + 1) % 10 == 0:
            eval_reward = evaluate_agent(controller, eval_env)
            logger.log_eval(episode + 1, eval_reward)
            if eval_reward >= best_eval_reward:
                best_eval_reward = eval_reward
                joblib.dump({'model': controller, 'episode': episode}, os.path.join(session_dir, 'dqn_best_model.pkl'))
            
            print(f"Ep {episode+1}/{config.NUM_EPISODES_DQN} | Train: {episode_reward:.1f} | Eval: {eval_reward:.1f} | Eps: {controller.epsilon:.3f}")

    # Final Save
    joblib.dump({'model': controller, 'config': config.CONTROL_PARAMS}, os.path.join(session_dir, 'dqn_final_model.pkl'))
    logger.save_logs_as_csv(filename_prefix="dqn")
    plot_rewards_from_csv(session_dir, csv_filename="dqn_logs.csv")
    env.close()
    return controller

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


'''def evaluate_agent(controller, eval_env, num_episodes=10):
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
    return avg_eval_reward'''

def evaluate_agent(controller, eval_env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Check if the controller supports the 'explore' argument
            if hasattr(controller, 'q_table'):
                # Q-Learning/SARSA: Just pass the state
                action = controller.get_action(state)
            else:
                # DQN: Disable exploration explicitly
                action = controller.get_action(state, explore=False)
                
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
        
    avg_eval_reward = total_reward / num_episodes
    return avg_eval_reward

'''def plot_rewards(training_rewards, evaluation_rewards, eval_interval, save_path=None):
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
    plt.show()'''

'''def plot_rewards_from_csv(session_dir, csv_filename):
    csv_path = os.path.join(session_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    
    # Plot raw rewards
    plt.plot(df['episode'], df['reward'], label='Raw Reward', alpha=0.3, color='blue')
    
    # Plot moving average (Trendline)
    if len(df) > 20:
        df['ma'] = df['reward'].rolling(window=20).mean()
        plt.plot(df['episode'], df['ma'], label='20-Ep Moving Avg', color='red', linewidth=2)

    plt.title(f"Best Fold Training Performance ({os.path.basename(session_dir)})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_img = os.path.join(session_dir, "best_fold_curve.png")
    plt.savefig(save_img)
    plt.show()
    print(f"📊 Plot saved to {save_img}")'''

def plot_rewards_from_csv(session_dir, csv_filename):
    csv_path = os.path.join(session_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"❌ Plotting Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Training Rewards (Light blue background)
    plt.plot(df['episode'], df['reward'], label='Training Reward (Raw)', alpha=0.3, color='#3498db')
    
    # 2. Plot Moving Average (The trendline)
    if len(df) > 20:
        df['smooth'] = df['reward'].rolling(window=20).mean()
        plt.plot(df['episode'], df['smooth'], label='Trend (20-Ep MA)', color='#2980b9', linewidth=2)

    # 3. Plot Evaluation Rewards (Distinct dots)
    # This checks if the column exists and has data
    if 'eval_reward' in df.columns:
        # Filter rows where eval_reward is not empty
        eval_data = df.dropna(subset=['eval_reward'])
        if not eval_data.empty:
            plt.scatter(eval_data['episode'], eval_data['eval_reward'], 
                        color='#e74c3c', label='Evaluation Score (Best Model)', zorder=5, s=40)

    plt.title(f"Performance: {os.path.basename(session_dir)}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot inside the session folder
    plot_name = csv_filename.replace('.csv', '.png')
    plt.savefig(os.path.join(session_dir, plot_name))
    plt.show()

def plot_rewards(training_rewards, evaluation_rewards, eval_interval, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(training_rewards, label='Training Reward', alpha=0.6)
    if evaluation_rewards:
        eval_episodes = list(range(eval_interval, (len(evaluation_rewards)*eval_interval) + 1, eval_interval))
        plt.plot(eval_episodes, evaluation_rewards, label='Evaluation Reward', color='red', linewidth=2)
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'learning_curve.png'))
    plt.close()
  
'''def run_cross_validation_or_training(algorithm, mode, action_mode, stop_logic):
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
        raise ValueError(f"Unknown algorithm: {algorithm}")'''

def run_cross_validation_or_training(algorithm, mode, action_mode, stop_logic):
    config.STATE_MODE = mode
    config.STOP_LOGIC = stop_logic
    config.ACTION_MODE = action_mode
    config.STATE_DIM = 2 if mode == '2D' else 4

    # Create the session directory here
    session_dir = create_session_folder(algorithm_name=algorithm, config=config)
    print(f"\n🚀 Starting {algorithm.upper()} | Mode: {mode} | Folder: {session_dir}")

    if algorithm == 'qlearning':
        best_reward, controller = cross_validation_qlearning(QLearningControl, EpsilonGreedyStrategy, config, session_dir)
    elif algorithm == 'dqn':
        controller = train_dqn(DQNControl, EpsilonGreedyStrategy, config, session_dir)
    elif algorithm == 'sarsa':
        # You would update train_sarsa similarly to accept session_dir
        controller = train_sarsa(SarsaControl, EpsilonGreedyStrategy, config, session_dir)
    
    return controller


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