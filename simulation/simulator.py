"""
Core EGT Simulator for evolutionary game dynamics.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

from core.mechanisms import RewardMechanism, CostMechanism, PunishmentMechanism


class EGTSimulator:
    """
    Core simulator for Evolutionary Game Theory dynamics.
    Tracks population strategy distribution and evolution.
    """
    
    def __init__(self, 
                num_clients: int,
                strategy_labels: List[str] = ["honest", "withholding", "adversarial"],
                initial_distribution: Optional[np.ndarray] = None,
                config: Dict[str, Any] = None):
        """
        Initialize the simulator.
        
        Args:
            num_clients: Number of clients to simulate
            strategy_labels: Labels for each strategy
            initial_distribution: Optional initial strategy distribution
            config: Configuration parameters
        """
        self.num_clients = num_clients
        self.strategy_labels = strategy_labels
        self.num_strategies = len(strategy_labels)
        self.config = config or {}
        
        # Initialize distribution
        if initial_distribution is not None and len(initial_distribution) == self.num_strategies:
            self.strategy_distribution = initial_distribution.copy()
        else:
            self.strategy_distribution = np.ones(self.num_strategies) / self.num_strategies
        
        # Generate client strategies based on distribution
        self.client_strategies = np.random.choice(
            range(self.num_strategies),
            size=num_clients,
            p=self.strategy_distribution
        )
        
        # Random client IDs
        self.client_ids = [f"client_{i}" for i in range(num_clients)]
        
        # For convergence analysis
        self.history = {
            'strategy_distribution': [self.strategy_distribution.copy()],
            'timestamp': [time.time()],
            'metrics': [],
            'rewards': [],
            'costs': [],
            'penalties': [],
            'payoffs': [],
            'contributions': []
        }
        
        # Configure replicator dynamics
        self.replicator_config = self.config.get('replicator', {})
        self.learning_rate = self.replicator_config.get('learning_rate', 0.1)
        self.noise_level = self.replicator_config.get('noise_level', 0.01)
        self.memory_length = self.replicator_config.get('memory_length', 3)
        
        # Store recent payoffs for each strategy
        self.strategy_payoff_history = [[] for _ in range(self.num_strategies)]
    
    def _generate_contributions(self, honest_quality: float = 0.8, 
                               withholding_quality: float = 0.2,
                               adversarial_quality: float = -0.4,
                               noise_std: float = 0.1) -> np.ndarray:
        """
        Generate contribution quality values for each client.
        
        Args:
            honest_quality: Base quality for honest clients
            withholding_quality: Base quality for withholding clients
            adversarial_quality: Base quality for adversarial clients
            noise_std: Standard deviation of random noise
            
        Returns:
            Array of contribution quality values
        """
        # Base contribution by strategy
        strategy_to_quality = {
            0: honest_quality,
            1: withholding_quality,
            2: adversarial_quality
        }
        
        # Generate base contributions
        contributions = np.array([
            strategy_to_quality.get(s, 0.0) for s in self.client_strategies
        ])
        
        # Add random noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=len(contributions))
            contributions += noise
        
        # Ensure reasonable bounds
        contributions = np.clip(contributions, -1.0, 1.0)
        
        return contributions
    
    def update(self,
              reward_mechanism: RewardMechanism,
              cost_mechanism: CostMechanism,
              punishment_mechanism: Optional[PunishmentMechanism] = None,
              training_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update the simulation based on training metrics and mechanisms.
        
        Args:
            reward_mechanism: Mechanism for calculating rewards
            cost_mechanism: Mechanism for calculating costs
            punishment_mechanism: Optional mechanism for calculating penalties
            training_metrics: Optional metrics from ML training
            
        Returns:
            Dictionary with update results
        """
        # Generate client contributions
        honest_quality = self.config.get('honest_quality', 0.8)
        withholding_quality = self.config.get('withholding_quality', 0.2)
        adversarial_quality = self.config.get('adversarial_quality', -0.4)
        
        contributions = self._generate_contributions(
            honest_quality, withholding_quality, adversarial_quality
        )
        
        # Update training metrics with contribution data
        metrics = training_metrics.copy() if training_metrics else {}
        metrics['client_contributions'] = contributions
        metrics['client_ids'] = self.client_ids
        metrics['client_strategies'] = self.client_strategies
        
        # Apply contribution impact on accuracy if not provided
        if 'accuracy' not in metrics:
            avg_contrib = np.mean(contributions)
            metrics['accuracy'] = max(0.0, min(1.0, 0.5 + avg_contrib * 0.3))
        
        # Update mechanisms with current metrics
        reward_mechanism.update(metrics)
        cost_mechanism.update(metrics)
        if punishment_mechanism:
            punishment_mechanism.update(metrics)
            if hasattr(punishment_mechanism, 'set_client_ids'):
                punishment_mechanism.set_client_ids(self.client_ids)
        
        # Calculate payoffs
        rewards = reward_mechanism.distribute(self.client_strategies, contributions)
        costs = cost_mechanism.compute(self.client_strategies, contributions)
        penalties = (punishment_mechanism.compute(self.client_strategies, contributions) 
                    if punishment_mechanism else np.zeros_like(costs))
        
        payoffs = rewards - costs - penalties
        
        # Store history
        self.history['rewards'].append(rewards.copy())
        self.history['costs'].append(costs.copy())
        self.history['penalties'].append(penalties.copy())
        self.history['payoffs'].append(payoffs.copy())
        self.history['contributions'].append(contributions.copy())
        if training_metrics:
            self.history['metrics'].append(training_metrics.copy())
        
        # Calculate average payoff for each strategy
        strategy_payoffs = np.zeros(self.num_strategies)
        strategy_counts = np.zeros(self.num_strategies)
        
        for i, strategy in enumerate(self.client_strategies):
            strategy_payoffs[strategy] += payoffs[i]
            strategy_counts[strategy] += 1
        
        # Avoid division by zero
        for i in range(self.num_strategies):
            if strategy_counts[i] > 0:
                strategy_payoffs[i] /= strategy_counts[i]
        
        # Update strategy payoff history
        for i in range(self.num_strategies):
            self.strategy_payoff_history[i].append(strategy_payoffs[i])
            if len(self.strategy_payoff_history[i]) > self.memory_length:
                self.strategy_payoff_history[i].pop(0)
        
        # Apply replicator dynamics with memory
        avg_strategy_payoffs = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            if self.strategy_payoff_history[i]:
                avg_strategy_payoffs[i] = np.mean(self.strategy_payoff_history[i])
        
        # Average payoff across strategies
        avg_payoff = np.sum(self.strategy_distribution * avg_strategy_payoffs)
        
        # Update strategy distribution using replicator equation
        delta_dist = self.strategy_distribution * (avg_strategy_payoffs - avg_payoff)
        delta_dist *= self.learning_rate  # Apply learning rate
        
        # Apply update
        self.strategy_distribution += delta_dist
        
        # Add small noise to avoid getting stuck
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=self.num_strategies)
            self.strategy_distribution += noise
        
        # Ensure the distribution remains valid
        self.strategy_distribution = np.maximum(0, self.strategy_distribution)
        sum_dist = np.sum(self.strategy_distribution)
        if sum_dist > 0:
            self.strategy_distribution /= sum_dist
        else:
            self.strategy_distribution = np.ones(self.num_strategies) / self.num_strategies
        
        # Store updated distribution
        self.history['strategy_distribution'].append(self.strategy_distribution.copy())
        self.history['timestamp'].append(time.time())
        
        # Resample client strategies
        old_strategies = self.client_strategies.copy()
        self.client_strategies = np.random.choice(
            range(self.num_strategies),
            size=self.num_clients,
            p=self.strategy_distribution
        )
        
        # Return update summary
        return {
            'old_distribution': self.history['strategy_distribution'][-2],
            'new_distribution': self.strategy_distribution,
            'delta_distribution': delta_dist,
            'avg_strategy_payoffs': avg_strategy_payoffs,
            'avg_payoff': avg_payoff,
            'strategy_changes': np.sum(old_strategies != self.client_strategies)
        }
    
    def get_history(self) -> Dict[str, List]:
        """Get the full simulation history."""
        return self.history
    
    def get_current_distribution(self) -> Dict[str, float]:
        """Get current strategy distribution as a dictionary."""
        return {
            self.strategy_labels[i]: self.strategy_distribution[i]
            for i in range(self.num_strategies)
        }
    
    def analyze_convergence(self, window_size: int = 5, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Analyze whether the simulation has converged.
        
        Args:
            window_size: Window of distributions to check
            threshold: Maximum change allowed to consider converged
            
        Returns:
            Dictionary with convergence analysis
        """
        distributions = self.history['strategy_distribution']
        
        if len(distributions) <= window_size:
            return {
                'converged': False,
                'stable_steps': 0,
                'dominant_strategy': None,
                'distribution_type': 'evolving'
            }
        
        # Check if recent distributions are stable
        recent = distributions[-window_size:]
        max_change = 0
        
        for i in range(1, len(recent)):
            change = np.max(np.abs(recent[i] - recent[i-1]))
            max_change = max(max_change, change)
        
        converged = max_change < threshold
        
        # Analyze final distribution
        final_dist = distributions[-1]
        max_prop = np.max(final_dist)
        dominant_idx = np.argmax(final_dist)
        
        # Determine distribution type
        if max_prop > 0.9:
            dist_type = 'pure_strategy'
            dominant = self.strategy_labels[dominant_idx]
        elif max_prop > 0.6:
            dist_type = 'dominant_strategy'
            dominant = self.strategy_labels[dominant_idx]
        else:
            dist_type = 'mixed_strategy'
            strategies_present = [
                self.strategy_labels[i] for i in range(self.num_strategies)
                if final_dist[i] > 0.1
            ]
            dominant = None
        
        # Calculate convergence time if converged
        convergence_time = None
        if converged:
            for i in range(len(distributions) - window_size):
                window = distributions[i:i+window_size]
                stable = True
                
                for j in range(1, len(window)):
                    if np.max(np.abs(window[j] - window[j-1])) >= threshold:
                        stable = False
                        break
                
                if stable:
                    convergence_time = i
                    break
        
        return {
            'converged': converged,
            'stable_steps': 0 if not converged else len(distributions) - (convergence_time or 0),
            'max_change': max_change,
            'distribution_type': dist_type,
            'dominant_strategy': dominant,
            'final_distribution': {
                self.strategy_labels[i]: final_dist[i] for i in range(self.num_strategies)
            },
            'convergence_time': convergence_time
        }
    
    def reset(self, initial_distribution: Optional[np.ndarray] = None) -> None:
        """
        Reset the simulator state.
        
        Args:
            initial_distribution: Optional new initial distribution
        """
        if initial_distribution is not None and len(initial_distribution) == self.num_strategies:
            self.strategy_distribution = initial_distribution.copy()
        else:
            self.strategy_distribution = np.ones(self.num_strategies) / self.num_strategies
        
        self.client_strategies = np.random.choice(
            range(self.num_strategies),
            size=self.num_clients,
            p=self.strategy_distribution
        )
        
        # Reset history
        self.history = {
            'strategy_distribution': [self.strategy_distribution.copy()],
            'timestamp': [time.time()],
            'metrics': [],
            'rewards': [],
            'costs': [],
            'penalties': [],
            'payoffs': [],
            'contributions': []
        }
        
        # Reset payoff history
        self.strategy_payoff_history = [[] for _ in range(self.num_strategies)]
