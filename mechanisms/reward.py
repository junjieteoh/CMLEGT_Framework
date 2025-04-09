"""
Reward mechanism implementations for the EGT Framework.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from core.mechanisms import RewardMechanism


class FixedReward(RewardMechanism):
    """
    Simple fixed reward mechanism that provides consistent rewards based on strategy.
    Rewards are predetermined and do not change based on performance.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        # Get strategy-specific rewards (honest, withholding, adversarial)
        self.strategy_rewards = self.config.get('strategy_rewards', [10.0, 5.0, 1.0])
        self.flat_reward = self.config.get('flat_reward', 10.0)  # Default flat reward amount
        self.use_flat = self.config.get('use_flat', False)  # Whether to use flat rewards
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """No updates needed for fixed rewards."""
        # Fixed rewards don't change based on performance
        pass
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute rewards based on predefined fixed values for each strategy."""
        rewards = np.zeros(len(strategies))
        
        if self.use_flat:
            # Everyone gets the same flat reward
            rewards = np.ones(len(strategies)) * self.flat_reward
        else:
            # Different rewards based on strategy
            for i, strategy_idx in enumerate(strategies):
                if strategy_idx < len(self.strategy_rewards):
                    rewards[i] = self.strategy_rewards[strategy_idx]
        
        return rewards
    
    def distribute(self, strategies: np.ndarray, contributions: np.ndarray) -> np.ndarray:
        """Simply return the computed rewards, ignoring contributions."""
        # For fixed rewards, we typically don't factor in contribution quality
        return self.compute(strategies)


class AdaptiveReward(RewardMechanism):
    """
    Reward mechanism that adapts based on model performance.
    The reward pool grows or shrinks dynamically based on accuracy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.accuracy_threshold = self.config.get('accuracy_threshold', 0.7)
        self.strategy_weights = self.config.get('strategy_weights', [1.0, 0.5, 0.0])
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update reward pool based on accuracy improvement."""
        accuracy = metrics.get('accuracy', 0.0)
        
        # Calculate reward adjustment
        adjustment = (accuracy - self.accuracy_threshold) / (1.0 - self.accuracy_threshold)
        adjustment = max(-0.5, min(0.5, adjustment))  # Limit adjustment range
        
        # Update reward pool
        old_pool = self.reward_pool
        self.reward_pool *= (1 + self.learning_rate * adjustment)
        
        # Log the state change
        self.log_state({
            'old_pool': old_pool,
            'new_pool': self.reward_pool,
            'adjustment': adjustment,
            'accuracy': accuracy
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute base rewards for each client based on strategy."""
        # Map strategy indices to weights
        rewards = np.zeros(len(strategies))
        for i, strategy_idx in enumerate(strategies):
            if strategy_idx < len(self.strategy_weights):
                rewards[i] = self.reward_pool * self.strategy_weights[strategy_idx]
        
        return rewards
    
    def distribute(self, strategies: np.ndarray, contributions: np.ndarray) -> np.ndarray:
        """Distribute rewards based on contribution quality."""
        base_rewards = self.compute(strategies)
        
        # Normalize contributions (avoid division by zero)
        if np.sum(contributions) > 0:
            normalized_contrib = contributions / np.sum(contributions)
        else:
            normalized_contrib = np.ones_like(contributions) / len(contributions)
        
        # Distribute rewards proportionally
        distributed_rewards = base_rewards * normalized_contrib
        
        return distributed_rewards


class PerformanceBasedReward(RewardMechanism):
    """
    Reward mechanism based on client contribution to model performance.
    Rewards are proportional to the quality of contributions.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.performance_weight = self.config.get('performance_weight', 0.8)
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update based on overall model performance."""
        performance = metrics.get('accuracy', 0.0)
        
        # Simple linear scaling of reward pool based on performance
        old_pool = self.reward_pool
        target_pool = 500 + 1000 * performance  # Scale from 500 to 1500 based on performance
        self.reward_pool = old_pool * 0.8 + target_pool * 0.2  # Smooth adjustment
        
        self.log_state({
            'old_pool': old_pool,
            'new_pool': self.reward_pool,
            'performance': performance
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute rewards based on strategy types."""
        # Use strategy weightings (honest: 1.0, withholding: 0.5, adversarial: 0.0)
        strategy_weights = [1.0, 0.5, 0.0]
        rewards = np.zeros(len(strategies))
        
        for i, strategy in enumerate(strategies):
            if strategy < len(strategy_weights):
                rewards[i] = self.reward_pool * strategy_weights[strategy]
        
        return rewards
    
    def distribute(self, strategies: np.ndarray, contributions: np.ndarray) -> np.ndarray:
        """Distribute rewards based on strategies and contributions."""
        base_rewards = self.compute(strategies)
        
        # Adjust rewards based on normalized contributions
        total_contribution = max(np.sum(contributions), 1e-10)
        contribution_factor = contributions / total_contribution
        
        # Blend base rewards with contribution-weighted rewards
        rewards = (1 - self.performance_weight) * base_rewards + \
                 self.performance_weight * self.reward_pool * contribution_factor
        
        return rewards


class ShapleyValueReward(RewardMechanism):
    """
    Reward mechanism based on approximate Shapley values.
    Tries to fairly allocate rewards based on marginal contributions.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.samples = self.config.get('samples', 100)  # Monte Carlo samples
        self.contribution_history = []
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update reward pool based on model improvement."""
        accuracy = metrics.get('accuracy', 0.0)
        accuracy_delta = metrics.get('accuracy_delta', 0.0)
        
        # Save current contribution data for Shapley calculation
        if 'client_contributions' in metrics:
            self.contribution_history.append(metrics['client_contributions'])
            if len(self.contribution_history) > 5:  # Keep limited history
                self.contribution_history.pop(0)
        
        # Adjust reward pool based on improvement
        old_pool = self.reward_pool
        self.reward_pool *= (1 + 0.2 * accuracy_delta)
        self.reward_pool = max(100, min(2000, self.reward_pool))  # Keep in reasonable range
        
        self.log_state({
            'old_pool': old_pool,
            'new_pool': self.reward_pool,
            'accuracy': accuracy,
            'accuracy_delta': accuracy_delta
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute base rewards considering strategy types."""
        # Base rewards by strategy
        strategy_weights = [1.0, 0.3, 0.0]  # Strongly favor honest behavior
        base_rewards = np.zeros(len(strategies))
        
        for i, strategy in enumerate(strategies):
            if strategy < len(strategy_weights):
                base_rewards[i] = self.reward_pool * strategy_weights[strategy]
        
        return base_rewards
    
    def _approximate_shapley(self, contributions: np.ndarray) -> np.ndarray:
        """Approximate Shapley values using Monte Carlo sampling."""
        n = len(contributions)
        shapley_values = np.zeros(n)
        
        # Simple approximation using marginal contributions
        if len(self.contribution_history) > 0:
            # Use history if available
            mean_contrib = np.mean([np.mean(c) for c in self.contribution_history if len(c) > 0])
            for i, contrib in enumerate(contributions):
                # Approximate marginal value vs. average
                marginal = (contrib - mean_contrib) / max(mean_contrib, 1e-10)
                shapley_values[i] = 0.5 + 0.5 * marginal  # Normalize to positive range
        else:
            # Fallback if no history
            mean_contrib = np.mean(contributions)
            for i, contrib in enumerate(contributions):
                marginal = (contrib - mean_contrib) / max(mean_contrib, 1e-10)
                shapley_values[i] = 0.5 + 0.5 * marginal
        
        # Ensure non-negative values
        shapley_values = np.maximum(0.1, shapley_values)
        
        # Normalize to sum to 1
        total = np.sum(shapley_values)
        if total > 0:
            shapley_values /= total
        else:
            shapley_values = np.ones(n) / n
        
        return shapley_values
    
    def distribute(self, strategies: np.ndarray, contributions: np.ndarray) -> np.ndarray:
        """Distribute rewards based on approximate Shapley values."""
        # Get base rewards
        base_rewards = self.compute(strategies)
        
        # Calculate Shapley values
        shapley_values = self._approximate_shapley(contributions)
        
        # Apply Shapley-based distribution
        shapley_rewards = self.reward_pool * shapley_values
        
        # Combine base rewards with Shapley rewards (80% Shapley, 20% base)
        rewards = 0.2 * base_rewards + 0.8 * shapley_rewards
        
        return rewards
