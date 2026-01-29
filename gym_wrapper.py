import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.registration import register
import torch

# Register the custom environment
register(
    id='CustomCartPoleEnv-v0',
    entry_point='gym_wrapper:CustomCartPoleEnv',
    max_episode_steps=500,
)

class CustomCartPoleEnv(gym.Env):
    def __init__(self, action_mode='discrete', render_mode=None):
        super(CustomCartPoleEnv, self).__init__()
        self.render_mode = render_mode
        self.action_mode = action_mode
        # Define action space (continuous motor speed, e.g., from -1.0 to 1.0)
        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

        if self.action_mode == 'discrete':
            # The agent sees 2 choices: 0 and 1
            self.action_space = gym.spaces.Discrete(2)
        else:
            # The agent sees a continuous range: [-1.0, 1.0]
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define observation space
        self.observation_space = gym.spaces.Box(low=np.array([-2.4, -np.inf, -0.418, -np.inf]),
                                                high=np.array([2.4, np.inf, 0.418, np.inf]),
                                                dtype=np.float32)
        
        # Environment constants (copied from your original code)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps_beyond_done = None
        self.step_count = 0

    def step(self, action):
        #motor_speed = action

        if self.action_mode == 'discrete':
            # Map 0 -> -1.0 (Full Left) and 1 -> 1.0 (Full Right)
            # We use float() to ensure it's a number for the physics
            motor_speed = float(action * 2.0 - 1.0)
        else:
            # For continuous, we extract the number from the array or use directly
            motor_speed = float(action) if not isinstance(action, np.ndarray) else float(action.item())

        force = self.force_mag * motor_speed
        
        x, x_dot, theta, theta_dot = self.state
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        self.step_count += 1
        truncated = self.step_count >= 500

        reward = 1.0 if not terminated else 0.0
        
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        #self.steps_beyond_done = None
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        return None

    def close(self):
        # Your close logic here
        pass

# The wrapper class
class GymWrapper:
    '''def __init__(self, env, mode='2D'):
        self.env = env
        self.mode = mode
        self.observation_space = self._filter_state(self.env.observation_space.sample()).shape
        self.action_space = self.env.action_space'''

    def __init__(self, env, mode='2D', action_mode='discrete'):
        self.env = env
        self.mode = mode
        self.action_mode= action_mode
        
        # We need to define a proper Gymnasium Box space so 
        # that the agents can "read" the limits and shape.
        if self.mode == '2D':
            # Angle and Angular Velocity limits
            low = np.array([-0.418, -np.inf], dtype=np.float32)
            high = np.array([0.418, np.inf], dtype=np.float32)
        else:
            # Full 4D limits (Cart Pos, Cart Vel, Angle, Angle Vel)
            low = np.array([-2.4, -np.inf, -0.418, -np.inf], dtype=np.float32)
            high = np.array([2.4, np.inf, 0.418, np.inf], dtype=np.float32)
            
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        #self.action_space = self.env.action_space
        # 2. Define Action Space (Output)
        if self.action_mode == 'discrete':
            # This tells Q-Learning/SARSA/DQN there are 2 choices: 0 and 1
            self.action_space = gym.spaces.Discrete(2)
        else:
            # This tells a Continuous agent (like PPO) it can use range [-1, 1]
            self.action_space = self.env.action_space

    def _filter_state(self, state):
        if self.mode == '2D':
            return np.array(state[2:4], dtype=np.float32)  # Keep only the pole angle and angular velocity
        return np.array(state, dtype=np.float32)

    '''def reset(self):
        state, info = self.env.reset()
        return self._filter_state(state), info'''

    def reset(self, seed=None, options=None):
        # Unpack the Gymnasium 0.26+ reset tuple
        state, info = self.env.reset(seed=seed, options=options)
        return self._filter_state(state), info

    '''def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_state(next_state), reward, terminated, truncated, info'''

    '''def step(self, action):
    # 3. Handle Action Mapping
      if self.action_mode == 'discrete':
        # Map 0 -> -1.0 and 1 -> 1.0
        # We handle it safely regardless of if action is int or array([int])
        val = int(action) if not isinstance(action, np.ndarray) else int(action.item())
        continuous_speed = float(val * 2.0 - 1.0)
      else:
        # For continuous, just ensure it's a float
        continuous_speed = float(action)  

      # Wrap in a numpy array as expected by the Box action space
      act_for_env = np.array([continuous_speed], dtype=np.float32)

      # Call the original environment
      next_state, reward, terminated, truncated, info = self.env.step(act_for_env)
    
      return self._filter_state(next_state), reward, terminated, truncated, info'''

    def step(self, action):
      # 1. HANDLE INPUT FROM AGENT
      # Agents often return actions as numpy arrays or tensors; we need the raw value.
      if isinstance(action, (np.ndarray, torch.Tensor)):
          action_val = action.item()
      else:
          action_val = action

      # 2. FORMAT FOR YOUR CUSTOM ENV
      if self.action_mode == 'discrete':
          # If your custom env expects 0 or 1:
          act_for_env = int(action_val)
      else:
          # If your custom env expects a float between -1 and 1:
          # This ensures it's a float32, which Gymnasium prefers for Box spaces
          act_for_env = np.array([float(action_val)], dtype=np.float32)

      # 3. EXECUTE STEP
      try:
          next_state, reward, terminated, truncated, info = self.env.step(act_for_env)
      except AssertionError as e:
          # Fallback: If your CustomEnv is strictly expecting a scalar in discrete mode
          if self.action_mode == 'discrete':
              next_state, reward, terminated, truncated, info = self.env.step(int(action_val))
          else:
              raise e

      # 4. FILTER AND RETURN
      return self._filter_state(next_state), reward, terminated, truncated, info

      '''# If the agent sends 0, we translate to -1.0 (Left)
    # If the agent sends 1, we translate to 1.0 (Right)
    # This works for Q-Learning, SARSA, and DQN
      if isinstance(action, (int, np.integer)):
        # Formula: (0 * 2) - 1 = -1.0 | (1 * 2) - 1 = 1.0
        continuous_speed = float(action * 2.0 - 1.0)
      else:
        # If the action is already a float (from a different agent)
        continuous_speed = action'''

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()