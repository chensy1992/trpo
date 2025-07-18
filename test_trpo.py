"""
Unit tests for TRPO implementation
"""
import pytest
import numpy as np
import tensorflow as tf
import gym
from modern_trpo import ModernTRPOAgent, TRPOConfig, PolicyNetwork, ValueNetwork

class TestPolicyNetwork:
    def test_policy_network_creation(self):
        """Test policy network initialization"""
        action_dim = 4
        policy_net = PolicyNetwork(action_dim)
        
        # Test with dummy input
        dummy_input = tf.random.normal((1, 8))
        output = policy_net(dummy_input)
        
        assert output.shape == (1, action_dim)
        assert tf.reduce_sum(output).numpy() == pytest.approx(1.0, rel=1e-5)

class TestValueNetwork:
    def test_value_network_creation(self):
        """Test value network initialization"""
        value_net = ValueNetwork()
        
        # Test with dummy input
        dummy_input = tf.random.normal((1, 8))
        output = value_net(dummy_input)
        
        assert output.shape == (1,)

class TestTRPOAgent:
    @pytest.fixture
    def env(self):
        """Create test environment"""
        return gym.make('CartPole-v1')
    
    @pytest.fixture
    def agent(self, env):
        """Create test agent"""
        config = TRPOConfig(timesteps_per_batch=100, max_pathlength=50)
        return ModernTRPOAgent(env, config)
    
    def test_agent_initialization(self, env):
        """Test agent initialization"""
        agent = ModernTRPOAgent(env)
        
        assert agent.obs_dim == env.observation_space.shape[0]
        assert agent.action_dim == env.action_space.n
        assert isinstance(agent.policy_net, PolicyNetwork)
        assert isinstance(agent.value_net, ValueNetwork)
    
    def test_get_action(self, agent):
        """Test action selection"""
        obs = np.random.random(4)  # CartPole observation
        action, action_probs = agent.get_action(obs)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert len(action_probs) == agent.action_dim
        assert np.sum(action_probs) == pytest.approx(1.0, rel=1e-5)
    
    def test_collect_trajectories(self, agent):
        """Test trajectory collection"""
        trajectories = agent.collect_trajectories()
        
        assert len(trajectories) > 0
        
        for traj in trajectories:
            assert 'observations' in traj
            assert 'actions' in traj
            assert 'rewards' in traj
            assert 'action_probs' in traj
            
            # Check shapes
            n_steps = len(traj['rewards'])
            assert traj['observations'].shape[0] == n_steps
            assert traj['actions'].shape[0] == n_steps
            assert traj['action_probs'].shape[0] == n_steps
    
    def test_discount_rewards(self, agent):
        """Test reward discounting"""
        rewards = np.array([1, 2, 3, 4, 5])
        gamma = 0.9
        
        returns = agent._discount_rewards(rewards, gamma)
        
        # Manual calculation for verification
        expected = np.array([
            1 + 0.9*2 + 0.9**2*3 + 0.9**3*4 + 0.9**4*5,
            2 + 0.9*3 + 0.9**2*4 + 0.9**3*5,
            3 + 0.9*4 + 0.9**2*5,
            4 + 0.9*5,
            5
        ])
        
        np.testing.assert_array_almost_equal(returns, expected)

class TestTRPOConfig:
    def test_config_defaults(self):
        """Test default configuration values"""
        config = TRPOConfig()
        
        assert config.timesteps_per_batch == 1000
        assert config.max_pathlength == 10000
        assert config.max_kl == 0.01
        assert config.gamma == 0.95
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = TRPOConfig(
            timesteps_per_batch=2000,
            gamma=0.99
        )
        
        assert config.timesteps_per_batch == 2000
        assert config.gamma == 0.99
        assert config.max_kl == 0.01  # Should keep default

if __name__ == "__main__":
    pytest.main([__file__])