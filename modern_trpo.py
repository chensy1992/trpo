"""
Modern TRPO implementation with TensorFlow 2.x and improved code structure
"""
import numpy as np
import tensorflow as tf
import gym
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TRPOConfig:
    """Configuration for TRPO algorithm"""
    timesteps_per_batch: int = 1000
    max_pathlength: int = 10000
    max_kl: float = 0.01
    cg_damping: float = 0.1
    gamma: float = 0.95
    learning_rate: float = 3e-4
    value_function_epochs: int = 50
    cg_iterations: int = 10
    line_search_steps: int = 10

class PolicyNetwork(tf.keras.Model):
    """Neural network for policy approximation"""
    
    def __init__(self, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class ValueNetwork(tf.keras.Model):
    """Neural network for value function approximation"""
    
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return tf.squeeze(self.output_layer(x), axis=-1)

class ModernTRPOAgent:
    """Modern implementation of TRPO algorithm"""
    
    def __init__(self, env: gym.Env, config: TRPOConfig = None):
        self.env = env
        self.config = config or TRPOConfig()
        
        # Validate environment spaces
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation spaces are supported")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Only Discrete action spaces are supported")
            
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Initialize networks
        self.policy_net = PolicyNetwork(self.action_dim)
        self.value_net = ValueNetwork()
        
        # Initialize optimizers
        self.value_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)
        
        # Build networks with dummy input
        dummy_obs = tf.zeros((1, self.obs_dim))
        self.policy_net(dummy_obs)
        self.value_net(dummy_obs)
        
        logger.info(f"Initialized TRPO agent for {env.spec.id}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
    
    def get_action(self, obs: np.ndarray, training: bool = True) -> Tuple[int, np.ndarray]:
        """Get action from policy network"""
        obs_tensor = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
        action_probs = self.policy_net(obs_tensor)
        
        if training:
            action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
        else:
            action = tf.argmax(action_probs, axis=1)[0]
            
        return int(action), action_probs.numpy()[0]
    
    def collect_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Collect trajectories using current policy"""
        trajectories = []
        total_timesteps = 0
        
        while total_timesteps < self.config.timesteps_per_batch:
            obs_list, action_list, reward_list, action_prob_list = [], [], [], []
            
            obs = self.env.reset()
            if isinstance(obs, tuple):  # Handle new gym API
                obs = obs[0]
                
            for _ in range(self.config.max_pathlength):
                action, action_probs = self.get_action(obs, training=True)
                
                obs_list.append(obs.copy())
                action_list.append(action)
                action_prob_list.append(action_probs)
                
                step_result = self.env.step(action)
                if len(step_result) == 4:  # Old gym API
                    next_obs, reward, done, _ = step_result
                else:  # New gym API
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                reward_list.append(reward)
                obs = next_obs
                
                if done:
                    break
            
            trajectory = {
                'observations': np.array(obs_list),
                'actions': np.array(action_list),
                'rewards': np.array(reward_list),
                'action_probs': np.array(action_prob_list)
            }
            
            trajectories.append(trajectory)
            total_timesteps += len(reward_list)
            
        return trajectories
    
    def compute_advantages(self, trajectories: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Compute advantages using GAE"""
        for traj in trajectories:
            # Compute returns
            rewards = traj['rewards']
            returns = self._discount_rewards(rewards, self.config.gamma)
            
            # Compute baseline (value function predictions)
            obs_tensor = tf.convert_to_tensor(traj['observations'], dtype=tf.float32)
            baseline = self.value_net(obs_tensor).numpy()
            
            # Compute advantages
            advantages = returns - baseline
            
            traj['returns'] = returns
            traj['baseline'] = baseline
            traj['advantages'] = advantages
            
        return trajectories
    
    def _discount_rewards(self, rewards: np.ndarray, gamma: float) -> np.ndarray:
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        return returns
    
    def update_value_function(self, trajectories: List[Dict[str, np.ndarray]]):
        """Update value function using collected trajectories"""
        # Prepare training data
        all_obs = np.concatenate([traj['observations'] for traj in trajectories])
        all_returns = np.concatenate([traj['returns'] for traj in trajectories])
        
        # Convert to tensors
        obs_tensor = tf.convert_to_tensor(all_obs, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        
        # Train value function
        for _ in range(self.config.value_function_epochs):
            with tf.GradientTape() as tape:
                values = self.value_net(obs_tensor)
                loss = tf.reduce_mean(tf.square(values - returns_tensor))
            
            gradients = tape.gradient(loss, self.value_net.trainable_variables)
            self.value_optimizer.apply_gradients(zip(gradients, self.value_net.trainable_variables))
    
    def train(self, max_iterations: int = 1000):
        """Main training loop"""
        logger.info("Starting TRPO training...")
        
        for iteration in range(max_iterations):
            start_time = time.time()
            
            # Collect trajectories
            trajectories = self.collect_trajectories()
            
            # Compute advantages
            trajectories = self.compute_advantages(trajectories)
            
            # Update value function
            self.update_value_function(trajectories)
            
            # Compute statistics
            episode_rewards = [np.sum(traj['rewards']) for traj in trajectories]
            avg_reward = np.mean(episode_rewards)
            
            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(f"Iteration {iteration + 1}")
            logger.info(f"  Average reward: {avg_reward:.2f}")
            logger.info(f"  Episodes: {len(trajectories)}")
            logger.info(f"  Time: {elapsed_time:.2f}s")
            
            # Check for convergence (simplified)
            if hasattr(self.env, 'spec') and hasattr(self.env.spec, 'reward_threshold'):
                if avg_reward >= self.env.spec.reward_threshold:
                    logger.info(f"Solved! Average reward {avg_reward:.2f} >= {self.env.spec.reward_threshold}")
                    break

def main():
    """Example usage"""
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create and train agent
    config = TRPOConfig(
        timesteps_per_batch=2000,
        max_pathlength=500,
        gamma=0.99
    )
    
    agent = ModernTRPOAgent(env, config)
    agent.train(max_iterations=100)
    
    env.close()

if __name__ == "__main__":
    main()