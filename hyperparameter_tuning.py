# This file aimed to find the best hyperparameters for the initial run of the model training.
# The issues comes with the Reinforcement Learning is that the environments are always noisy like the real environment.
# This makes it not possible to find the accurate parameters, rather the code in this file gives a general idea of where the intial parameters should lie.
# For the above mentioned reasons it is also not feasible to use overfitting logic.

from skopt import gp_minimize
from gym_wrapper import GymWrapper
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import gymnasium as gym
import config
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# For Qlearning
def tune_hyperparameters_qlearning(model_class, exploration_strategy_class, config):
    
    # Defining the space for the parameters to be tuned
    space = [
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform'),
        Real(0.8, 0.999, name='discount_factor'),
        Real(0.95, 0.999, name='decay_rate')
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, decay_rate):
        # Setup Env (Always use v0 for consistency)
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0'), mode=config.STATE_MODE)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0'), mode=config.STATE_MODE)
        
        # Force start epsilon to 1.0 for the decay to be meaningful
        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': 1.0, 
            'decay_rate': decay_rate
        })

        exploration_strategy = exploration_strategy_class(epsilon=1.0)
        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)

        # Training Phase
        for episode in range(200): # Fixed short episodes for tuning speed
            state, _ = env.reset()
            done = False
            while not done:
                action = controller.get_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                controller.update(state, action, reward, next_state, done)
                state = next_state
            controller.decay_epsilon()

        # Greedy Evaluation Phase
        eval_scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            total_reward = 0
            done = False
            # Force greedy
            temp_eps = controller.epsilon
            controller.epsilon = 0.0 
            while not done:
                action = controller.get_action(state)
                state, reward, term, trunc, _ = eval_env.step(action)
                total_reward += reward
                done = term or trunc
            eval_scores.append(total_reward)
            controller.epsilon = temp_eps

        return -np.mean(eval_scores)

    result = gp_minimize(objective, space, n_calls=30, random_state=42)
    
    # Update config with best results
    best_params = {param.name: val for param, val in zip(space, result.x)}
    config.CONTROL_PARAMS.update(best_params)
    config.CONTROL_PARAMS['epsilon'] = 1.0 # Ensure main training starts at 1.0
    return result

# For DQN
def tune_hyperparameters_dqn(model_class, exploration_strategy_class, cfg):
    # Define Search Space
    space = [
        Real(1e-4, 5e-3, name='learning_rate', prior='log-uniform'),
        Real(0.98, 0.999, name='discount_factor'),
        Integer(32, 128, name='batch_size'),
        Real(0.99, 0.999, name='decay_rate'),
        Integer(10000, 30000, name='buffer_size'),
    ]

    @use_named_args(space)
    def objective(**params):
        cfg.CONTROL_PARAMS.update(params)
        
        act_mode = getattr(cfg, 'ACTION_MODE', 'discrete')
        st_mode = getattr(cfg, 'STATE_MODE', '2D')

        # Setup Env
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=act_mode), 
                         mode=st_mode, action_mode=act_mode)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=act_mode), 
                              mode=st_mode, action_mode=act_mode)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if act_mode == 'discrete' else 1

        cfg.CONTROL_PARAMS['epsilon'] = 1.0 
        cfg.CONTROL_PARAMS['decay_rate'] = params['decay_rate']
        
        # Initialize Controller
        exploration_strategy = exploration_strategy_class(epsilon=1.0)
        controller = model_class(cfg.CONTROL_PARAMS, exploration_strategy, state_dim, action_dim)
        
        # Training Phase
        tuning_episodes = 200 # Reduced for faster optimization
        for episode in range(tuning_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = controller.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                state = next_state
            controller.decay_epsilon()

        # Evaluation Phase
        # Evaluate without exploration to get the "true" performance
        eval_scores = []
        
        # Store the original epsilon so we can restore it after evaluation
        original_epsilon = controller.exploration_strategy.epsilon
        
        for _ in range(10): 
            state, _ = eval_env.reset()
            total_reward = 0
            done = False
            
            # Manually force greedy by setting epsilon to 0 temporarily
            controller.exploration_strategy.epsilon = 0.0
            
            while not done:
                action = controller.get_action(state)
                state, reward, term, trunc, _ = eval_env.step(action)
                total_reward += reward
                done = term or trunc
            eval_scores.append(total_reward)
        
        # Restore the epsilon for the next potential training step
        controller.exploration_strategy.epsilon = original_epsilon
        
        avg_eval_score = np.mean(eval_scores)

        print(f"Trial Result: {avg_eval_score:.1f} | "
              f"LR: {params['learning_rate']:.5f}, "
              f"Gamma: {params['discount_factor']:.3f}, "
              f"Batch: {params['batch_size']}, "
              f"Decay: {params['decay_rate']:.4f}, "
              f"Buf: {params['buffer_size']}")
        
        return -avg_eval_score

    # Run Optimization
    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    # Update cfg with best params
    best_params = {param.name: val for param, val in zip(space, result.x)}
    cfg.CONTROL_PARAMS.update(best_params)

    print("\n✅ Optimization Complete!")
    print(f"Best Eval Score: {-result.fun:.2f}")
    return result

# For SARSA
def tune_hyperparameters_sarsa(model_class, exploration_strategy_class, config):
    # Define Space
    space = [
        Real(1e-4, 1e-1, name='learning_rate_sarsa', prior='log-uniform'),
        Real(0.9, 0.999, name='discount_factor'),
        Real(0.95, 0.999, name='decay_rate')
    ]

    @use_named_args(space)
    def objective(learning_rate_sarsa, discount_factor, decay_rate):
        # Setup Environment
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0'), mode=config.STATE_MODE)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0'), mode=config.STATE_MODE)
        
        config.CONTROL_PARAMS.update({
            'learning_rate_sarsa': learning_rate_sarsa,
            'discount_factor': discount_factor,
            'epsilon': 1.0,
            'decay_rate': decay_rate
        })

        exploration_strategy = exploration_strategy_class(epsilon=1.0)
        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)

        # Training
        for episode in range(200):
            state, _ = env.reset()
            action = controller.get_action(state)
            done = False
            while not done:
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                next_action = controller.get_action(next_state)
                controller.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
            controller.decay_epsilon()

        # Greedy Evaluation
        eval_scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            total_reward, done = 0, False
            temp_eps = controller.epsilon
            controller.epsilon = 0.0
            while not done:
                action = controller.get_action(state)
                state, reward, term, trunc, _ = eval_env.step(action)
                total_reward += reward
                done = term or trunc
            eval_scores.append(total_reward)
            controller.epsilon = temp_eps

        return -np.mean(eval_scores)

    result = gp_minimize(objective, space, n_calls=30, random_state=42)
    
    # Final Update
    best_params = {param.name: val for param, val in zip(space, result.x)}
    config.CONTROL_PARAMS.update(best_params)
    return result