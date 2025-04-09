"""
Cost mechanism implementations for the EGT Framework.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from core.mechanisms import CostMechanism


class ComputationalCost(CostMechanism):
    """
    Cost mechanism based on computational resources used by clients.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Default costs for each strategy
        self.base_costs = self.config.get('base_costs', {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0})
        self.scaling_factor = self.config.get('scaling_factor', 0.2)
        self.current_loss = 1.0  # Initial value
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update costs based on training metrics."""
        # Update based on loss
        old_loss = self.current_loss
        self.current_loss = metrics.get('loss', self.current_loss)
        
        self.log_state({
            'old_loss': old_loss,
            'new_loss': self.current_loss
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute costs for each client based on their strategy."""
        # Map strategies to cost types
        strategy_to_cost = {
            0: self.base_costs['honest'],
            1: self.base_costs['withholding'],
            2: self.base_costs['adversarial']
        }
        
        # Loss-based adjustment factor
        loss_factor = 1.0 + (self.current_loss - 1.0) * self.scaling_factor
        loss_factor = max(0.5, min(1.5, loss_factor))  # Limit range
        
        # Calculate costs
        costs = np.zeros(len(strategies))
        for i, strategy in enumerate(strategies):
            base_cost = strategy_to_cost.get(strategy, 0.0)
            
            # Adjust honest costs based on loss (higher loss → higher cost)
            if strategy == 0:  # honest
                costs[i] = base_cost * loss_factor
            elif strategy == 1:  # withholding
                costs[i] = base_cost
            elif strategy == 2:  # adversarial
                costs[i] = base_cost * (2.0 - loss_factor)  # Lower cost when model performs poorly
        
        return costs


class PrivacyCost(CostMechanism):
    """
    Cost mechanism that incorporates privacy loss considerations.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Base costs including privacy component
        self.base_costs = self.config.get('base_costs', {'honest': 8.0, 'withholding': 4.0, 'adversarial': 1.0})
        self.privacy_factor = self.config.get('privacy_factor', 0.3)
        self.data_sensitivity = self.config.get('data_sensitivity', 1.0)
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update privacy costs based on training progress."""
        # Privacy risk might increase over time as more data is shared
        epoch = metrics.get('epoch', 0)
        
        # Optionally adjust privacy factor based on training metrics
        if 'privacy_leakage' in metrics:
            self.data_sensitivity = min(2.0, self.data_sensitivity * (1 + 0.01 * metrics['privacy_leakage']))
        
        self.log_state({
            'epoch': epoch,
            'data_sensitivity': self.data_sensitivity
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute privacy-aware costs for each client."""
        # Base computational costs
        strategy_to_base_cost = {
            0: self.base_costs['honest'],
            1: self.base_costs['withholding'],
            2: self.base_costs['adversarial']
        }
        
        # Additional privacy costs (highest for honest contribution that shares most data)
        strategy_to_privacy_cost = {
            0: self.data_sensitivity * self.privacy_factor,  # Honest: Full privacy cost
            1: self.data_sensitivity * self.privacy_factor * 0.5,  # Withholding: Partial privacy cost
            2: 0.0  # Adversarial: No significant additional privacy cost
        }
        
        # Calculate total costs
        costs = np.zeros(len(strategies))
        for i, strategy in enumerate(strategies):
            # Computational cost
            base_cost = strategy_to_base_cost.get(strategy, 0.0)
            
            # Privacy cost
            privacy_cost = strategy_to_privacy_cost.get(strategy, 0.0)
            
            # Total cost
            costs[i] = base_cost + privacy_cost
        
        return costs
