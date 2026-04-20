STATE_MODE = '2D'
STOP_LOGIC = 'fixed'
ACTION_MODE = 'discrete'
STATE_DIM = 2

# Simulation parameters
NUM_EPISODES = 2000
NUM_EPISODES_Hyperparameter = 2000
NUM_EPISODES_DQN = 1000
NUM_EPISODES_SARSA=2000
MAX_STEPS = 500

# Control algorithm parameters and early stopping

CONTROL_PARAMS = {
    'learning_rate': 0.001,
    'learning_rate_qlearning': 0.03,
    'learning_rate_sarsa': 0.03,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'min_epsilon': 0.01,
    'decay_rate': 0.995,
    'buffer_size': 10000,  # For DQN
    'batch_size': 64,      # For DQN
    'update_target_steps': 1000,  # For DQN
    'patience': 150,          
    'min_delta': 0.05,       
    'decay_factor': 0.5,
    'min_lr': 0.001,
    'num_bins': [12, 12, 12, 12],  # Add this!
    'state_bounds': [              # Add this!
        [-2.4, 2.4],     # x
        [-3.0, 3.0],     # x_dot
        [-0.418, 0.418], # theta
        [-3.5, 3.5]      # theta_dot
    ]
}

# Logging parameters
LOG_PARAMS = {
    'log_frequency': 10,
    'save_path': './Results/'
}

# Hardware interface parameters (when using real hardware)
HARDWARE_PARAMS = {
    'motor_pins': [18, 23],  #
    'encoder_pins': [24, 25],
    'update_frequency': 50  # Hz
}