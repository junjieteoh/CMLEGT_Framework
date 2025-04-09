"""
Utility functions for the EGT framework.
"""

import torch
from typing import Dict, List, Tuple, Callable, Any

from registry.registry import MechanismRegistry
from mechanisms.reward import AdaptiveReward, PerformanceBasedReward, ShapleyValueReward, FixedReward
from mechanisms.cost import ComputationalCost, PrivacyCost
from mechanisms.punishment import AdversarialPunishment, ReputationBasedPunishment
from ml.pytorch import PyTorchIntegration


def create_default_registry() -> MechanismRegistry:
    """Create and configure a default mechanism registry."""
    registry = MechanismRegistry()
    
    # Register reward mechanisms
    registry.register_reward('adaptive', AdaptiveReward, {
        'initial_pool': 1000.0,
        'learning_rate': 0.1,
        'accuracy_threshold': 0.7
    })
    
    registry.register_reward('performance', PerformanceBasedReward, {
        'initial_pool': 1000.0,
        'performance_weight': 0.8
    })
    
    registry.register_reward('shapley', ShapleyValueReward, {
        'initial_pool': 1000.0,
        'samples': 100
    })
    
    registry.register_reward('fixed', FixedReward, {
        'flat_reward': 10.0,
        'use_flat': True,
        'strategy_rewards': [10.0, 5.0, 1.0]
    })
    
    # Register cost mechanisms
    registry.register_cost('computational', ComputationalCost, {
        'base_costs': {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0},
        'scaling_factor': 0.2
    })
    
    registry.register_cost('privacy', PrivacyCost, {
        'base_costs': {'honest': 8.0, 'withholding': 4.0, 'adversarial': 1.0},
        'privacy_factor': 0.3,
        'data_sensitivity': 1.0
    })
    
    # Register punishment mechanisms
    registry.register_punishment('adversarial', AdversarialPunishment, {
        'penalty_strength': 1.0,
        'detection_threshold': 0.7,
        'gradient_variance_weight': 0.5
    })
    
    registry.register_punishment('reputation', ReputationBasedPunishment, {
        'memory_factor': 0.8,
        'reputation_threshold': 0.5,
        'max_penalty': 20.0
    })
    
    return registry


def create_pytorch_experiment(model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             loss_fn: Callable,
                             experiment_config: Dict[str, Any],
                             device: str = 'cpu') -> Tuple:
    """
    Create a PyTorch experiment with EGT integration.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        experiment_config: Experiment configuration
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Tuple of (trainer, simulator)
    """
    # Create registry and builder
    from registry.experiment import ExperimentBuilder
    
    registry = create_default_registry()
    builder = ExperimentBuilder(registry)
    
    # Create simulator and integration
    simulator, integration = builder.create_experiment(experiment_config)
    
    # Create PyTorch integration
    pytorch_integration = PyTorchIntegration(
        simulator=simulator,
        reward_mechanism=integration.reward_mechanism,
        cost_mechanism=integration.cost_mechanism,
        punishment_mechanism=integration.punishment_mechanism,
        update_frequency=integration.update_frequency,
        config=integration.config
    )
    
    # Create trainer
    trainer = pytorch_integration.create_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )
    
    return trainer, simulator
