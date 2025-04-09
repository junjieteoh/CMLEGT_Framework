"""
Core abstractions for dynamic mechanisms in the EGT Framework.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class DynamicMechanism(ABC):
    """Base abstract class for all dynamic mechanisms in the framework."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the dynamic mechanism.
        
        Args:
            name: Name of this mechanism instance
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.history = []
    
    @abstractmethod
    def update(self, metrics: Dict[str, Any]) -> None:
        """
        Update internal state based on training metrics.
        
        Args:
            metrics: Dictionary of training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute values based on client strategies and contributions.
        
        Args:
            strategies: Array of client strategy indices
            contributions: Optional array of client contribution values
            
        Returns:
            Array of computed values for each client
        """
        pass
    
    def reset(self) -> None:
        """Reset the mechanism's internal state."""
        self.history = []
    
    def log_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Log current state for history tracking.
        
        Args:
            state_dict: Dictionary with state information to log
        """
        self.history.append({
            'timestamp': time.time(),
            **state_dict
        })
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get the mechanism's state history."""
        return self.history
    
    def to_dict(self) -> Dict[str, Any]:
        """Export mechanism configuration as a dictionary."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicMechanism':
        """Create a mechanism instance from dictionary configuration."""
        return cls(name=data['name'], config=data['config'])


class RewardMechanism(DynamicMechanism):
    """
    Base class for reward mechanisms that allocate rewards to clients.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.reward_pool = self.config.get('initial_pool', 1000.0)
    
    @abstractmethod
    def distribute(self, strategies: np.ndarray, contributions: np.ndarray) -> np.ndarray:
        """
        Distribute rewards based on strategies and contributions.
        
        Args:
            strategies: Array of client strategy indices
            contributions: Array of client contribution values
            
        Returns:
            Array of rewards for each client
        """
        pass


class CostMechanism(DynamicMechanism):
    """
    Base class for cost mechanisms that calculate costs for clients.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.base_costs = self.config.get('base_costs', {})


class PunishmentMechanism(DynamicMechanism):
    """
    Base class for punishment mechanisms that penalize client behaviors.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.penalty_strength = self.config.get('penalty_strength', 1.0)
