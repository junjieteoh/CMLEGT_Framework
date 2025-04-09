"""
Registry for different mechanism types in the EGT framework.
"""

from typing import Dict, List, Any, Optional

from core.mechanisms import RewardMechanism, CostMechanism, PunishmentMechanism


class MechanismRegistry:
    """
    Registry for reward, cost, and punishment mechanisms.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.reward_mechanisms = {}
        self.cost_mechanisms = {}
        self.punishment_mechanisms = {}
    
    def register_reward(self, name: str, cls, default_config: Dict[str, Any] = None) -> None:
        """
        Register a reward mechanism.
        
        Args:
            name: Mechanism name
            cls: Mechanism class
            default_config: Default configuration
        """
        self.reward_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def register_cost(self, name: str, cls, default_config: Dict[str, Any] = None) -> None:
        """
        Register a cost mechanism.
        
        Args:
            name: Mechanism name
            cls: Mechanism class
            default_config: Default configuration
        """
        self.cost_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def register_punishment(self, name: str, cls, default_config: Dict[str, Any] = None) -> None:
        """
        Register a punishment mechanism.
        
        Args:
            name: Mechanism name
            cls: Mechanism class
            default_config: Default configuration
        """
        self.punishment_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def get_reward(self, name: str, custom_config: Dict[str, Any] = None) -> RewardMechanism:
        """
        Get a reward mechanism instance.
        
        Args:
            name: Mechanism name
            custom_config: Custom configuration to override defaults
            
        Returns:
            Reward mechanism instance
        """
        if name not in self.reward_mechanisms:
            raise ValueError(f"Reward mechanism '{name}' not found in registry")
        
        # Merge default and custom configurations
        config = self.reward_mechanisms[name]['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        # Create instance
        return self.reward_mechanisms[name]['class'](name, config)
    
    def get_cost(self, name: str, custom_config: Dict[str, Any] = None) -> CostMechanism:
        """
        Get a cost mechanism instance.
        
        Args:
            name: Mechanism name
            custom_config: Custom configuration to override defaults
            
        Returns:
            Cost mechanism instance
        """
        if name not in self.cost_mechanisms:
            raise ValueError(f"Cost mechanism '{name}' not found in registry")
        
        # Merge default and custom configurations
        config = self.cost_mechanisms[name]['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        # Create instance
        return self.cost_mechanisms[name]['class'](name, config)
    
    def get_punishment(self, name: str, custom_config: Dict[str, Any] = None) -> Optional[PunishmentMechanism]:
        """
        Get a punishment mechanism instance.
        
        Args:
            name: Mechanism name
            custom_config: Custom configuration to override defaults
            
        Returns:
            Punishment mechanism instance or None if 'none'
        """
        if name == 'none':
            return None
        
        if name not in self.punishment_mechanisms:
            raise ValueError(f"Punishment mechanism '{name}' not found in registry")
        
        # Merge default and custom configurations
        config = self.punishment_mechanisms[name]['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        # Create instance
        return self.punishment_mechanisms[name]['class'](name, config)
    
    def list_mechanisms(self) -> Dict[str, List[str]]:
        """
        List all registered mechanisms.
        
        Returns:
            Dictionary with mechanism categories and names
        """
        return {
            'reward': list(self.reward_mechanisms.keys()),
            'cost': list(self.cost_mechanisms.keys()),
            'punishment': list(self.punishment_mechanisms.keys()) + ['none']
        }
