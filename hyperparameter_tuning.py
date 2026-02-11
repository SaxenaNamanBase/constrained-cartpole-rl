from skopt import gp_minimize
from gym_wrapper import GymWrapper
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import gymnasium as gym
import config
import numpy as np

def tune_hyperparameters_qlearning(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform'),
        Real(0.8, 0.999, name='discount_factor'),
        Real(0.01, 0.2, name='epsilon')
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon):
        # Initialize environment
        env = GymWrapper(gym.make('CartPole-v1'))
        state_dim = env.env.observation_space.shape[0]
        action_dim = env.env.action_space.n
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)
        '''control_params = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        }'''

        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        })

        #controller = model_class(control_params, exploration_strategy)

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)

        '''controller = model_class(
        control_params={'learning_rate': learning_rate, 'discount_factor': discount_factor},
        exploration_strategy=exploration_strategy
        )'''
        
        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                #next_state, reward, done, _ = env.step(action)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'epsilon': result.x[2]
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Best Score: {-result.fun}")

'''def tune_hyperparameters_dqn(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
    Real(1e-3, 1e-2, name='learning_rate', prior='log-uniform'),
    Real(0.9, 0.999, name='discount_factor'),
    Integer(64, 128, name='batch_size'),
    Real(0.05, 0.2, name='epsilon'),
    Integer(10000, 30000, name='buffer_size'),
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon, batch_size, buffer_size):
        # Initialize environment
        #env = GymWrapper(gym.make('CartPole-v1'))
        #state_dim = env.env.observation_space.shape[0]
        #state_dim = env.observation_space[0]
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=config.ACTION_MODE), 
                 mode=config.STATE_MODE, 
                 action_mode=config.ACTION_MODE)
        state_dim = env.observation_space.shape[0]
        #action_dim = env.env.action_space.n
        action_dim = env.action_space.n if config.ACTION_MODE == 'discrete' else 1
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)

        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'epsilon': epsilon,
            'buffer_size': buffer_size
        })

        #controller = model_class(control_params, exploration_strategy)

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy, state_dim, action_dim)
        
        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                action = controller.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                controller.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    print("Optimization result:", result.x)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'batch_size': result.x[2],
        'epsilon': result.x[3],
        'buffer_size': result.x[4]
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Batch Size: {config.CONTROL_PARAMS['batch_size']}")
    print(f"Buffer Size: {config.CONTROL_PARAMS['buffer_size']}")
    print(f"Best Score: {-result.fun}")'''


def tune_hyperparameters_dqn(model_class, exploration_strategy_class, cfg):
    # 1. Define Search Space
    space = [
        Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform'),
        Real(0.95, 0.999, name='discount_factor'),
        Integer(32, 256, name='batch_size'),
        Real(0.01, 0.1, name='epsilon'), 
        Integer(5000, 50000, name='buffer_size'),
    ]

    @use_named_args(space)
    def objective(**params):
        # Update cfg with the suggested hyperparameters
        cfg.CONTROL_PARAMS.update(params)
        
        # Determine Action Mode safely
        # If it's not in cfg, we assume 'discrete' for DQN
        act_mode = getattr(cfg, 'ACTION_MODE', 'discrete')
        st_mode = getattr(cfg, 'STATE_MODE', '2D')

        # Setup Env
        env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=act_mode), 
                         mode=st_mode, action_mode=act_mode)
        eval_env = GymWrapper(gym.make('CustomCartPoleEnv-v0', action_mode=act_mode), 
                              mode=st_mode, action_mode=act_mode)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if act_mode == 'discrete' else 1
        
        # Initialize Controller
        exploration_strategy = exploration_strategy_class(epsilon=params['epsilon'])
        controller = model_class(cfg.CONTROL_PARAMS, exploration_strategy, state_dim, action_dim)
        
        # --- TRAINING PHASE (Shortened for speed) ---
        tuning_episodes = 100 # Reduced for faster optimization
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

        # --- EVALUATION PHASE ---
        # Evaluate without exploration to get the "true" performance
        eval_scores = []
        for _ in range(3): 
            # We use an evaluation function (ensure evaluate_agent is imported/defined)
            state, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                # Use a greedy action (epsilon=0) for evaluation
                action = controller.get_action(state, force_greedy=True) 
                state, reward, term, trunc, _ = eval_env.step(action)
                total_reward += reward
                done = term or trunc
            eval_scores.append(total_reward)
        
        avg_eval_score = np.mean(eval_scores)
        print(f"Testing: LR={params['learning_rate']:.4f} | Result: {avg_eval_score:.1f}")
        
        return -avg_eval_score # Negative because we minimize

    # 2. Run Optimization
    result = gp_minimize(objective, space, n_calls=15, random_state=42)

    # 3. Update cfg with best params
    best_params = {param.name: val for param, val in zip(space, result.x)}
    cfg.CONTROL_PARAMS.update(best_params)

    print("\n✅ Optimization Complete!")
    print(f"Best Eval Score: {-result.fun:.2f}")
    return result


def tune_hyperparameters_sarsa(model_class, exploration_strategy_class, config):
    # Define the hyperparameter search space
    space = [
        Real(1e-3, 1e-1, name='learning_rate', prior='log-uniform'),
        Real(0.9, 0.999, name='discount_factor'),
        Real(0.01, 0.2, name='epsilon'),
    ]

    @use_named_args(space)
    def objective(learning_rate, discount_factor, epsilon):
        # Initialize environment
        env = GymWrapper(gym.make('CartPole-v1'))
        exploration_strategy = exploration_strategy_class(epsilon=epsilon)
        
        # Update config with the current hyperparameters
        config.CONTROL_PARAMS.update({
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon': epsilon
        })

        controller = model_class(config.CONTROL_PARAMS, exploration_strategy)

        total_reward = 0
        # Training loop (simplified for tuning purposes)
        for episode in range(config.NUM_EPISODES):
            state, _ = env.reset()
            action = controller.get_action(state)
            episode_reward = 0
            for t in range(config.MAX_STEPS):
                #next_state, reward, done, _ = env.step(action)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_action = controller.get_action(next_state)
                controller.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward

        # Return the negative total reward (since we are minimizing)
        return -total_reward / config.NUM_EPISODES  # Minimize the negative average reward

    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Update config with the best parameters
    config.CONTROL_PARAMS.update({
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'epsilon': result.x[2],
    })

    print("Best Parameters Updated in CONTROL_PARAMS:")
    print(f"Learning Rate: {config.CONTROL_PARAMS['learning_rate']}")
    print(f"Discount Factor: {config.CONTROL_PARAMS['discount_factor']}")
    print(f"Epsilon: {config.CONTROL_PARAMS['epsilon']}")
    print(f"Best Score: {-result.fun}")

