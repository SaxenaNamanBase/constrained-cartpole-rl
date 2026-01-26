STATE_MODE = '2D'  # Default
STOP_LOGIC = 'fixed' # Default
STATE_DIM = 2

# Simulation parameters
NUM_EPISODES = 2000
NUM_EPISODES_DQN = 500
NUM_EPISODES_SARSA=2000
MAX_STEPS = 500

# Control algorithm parameters

CONTROL_PARAMS = {
    'learning_rate': 0.001,
    'discount_factor': 0.99,
    'epsilon': 1.0,
    'min_epsilon': 0.01,
    'decay_rate': 0.995,
    'buffer_size': 10000,  # For DQN
    'batch_size': 64,      # For DQN
    'update_target_steps': 1000,  # For DQN
    'patience': 100,          
    'min_delta': 0.01,       
    'decay_factor': 0.9
}

# Logging parameters
LOG_PARAMS = {
    'log_frequency': 10,
    'save_path': './Results/'
}

# Hardware interface parameters (when using real hardware)
HARDWARE_PARAMS = {
    'motor_pins': [18, 23],  # Example GPIO pins
    'encoder_pins': [24, 25],
    'update_frequency': 50  # Hz
}

EARLY_STOPPING = {
    'patience': 10,  # Number of episodes with no improvement after which training will be stopped
    'min_delta': 1e-5  # Minimum change to consider as an improvement
}