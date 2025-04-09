"""
Experiment builder for creating and running EGT simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt

from mechanisms.reward import AdaptiveReward, PerformanceBasedReward, ShapleyValueReward, FixedReward
from mechanisms.cost import ComputationalCost, PrivacyCost
from mechanisms.punishment import AdversarialPunishment, ReputationBasedPunishment
from simulation.simulator import EGTSimulator
from ml.integration import MLIntegration
from visualization.visualizer import EGTVisualizer
from .registry import MechanismRegistry


class ExperimentBuilder:
    """
    Builder for EGT experiments with different configurations.
    """
    
    def __init__(self, registry: MechanismRegistry = None):
        """
        Initialize experiment builder.
        
        Args:
            registry: Optional mechanism registry
        """
        # Create or use registry
        self.registry = registry if registry else MechanismRegistry()
        
        # Register default mechanisms if using a new registry
        if not registry:
            self._register_default_mechanisms()
    
    def _register_default_mechanisms(self) -> None:
        """Register default mechanisms in the registry."""
        # Reward mechanisms
        self.registry.register_reward('adaptive', AdaptiveReward, {
            'initial_pool': 1000.0,
            'learning_rate': 0.1,
            'accuracy_threshold': 0.7
        })
        
        self.registry.register_reward('performance', PerformanceBasedReward, {
            'initial_pool': 1000.0,
            'performance_weight': 0.8
        })
        
        self.registry.register_reward('shapley', ShapleyValueReward, {
            'initial_pool': 1000.0,
            'samples': 100
        })
        
        self.registry.register_reward('fixed', FixedReward, {
            'flat_reward': 10.0,
            'use_flat': True,
            'strategy_rewards': [10.0, 5.0, 1.0]
        })
        
        # Cost mechanisms
        self.registry.register_cost('computational', ComputationalCost, {
            'base_costs': {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0},
            'scaling_factor': 0.2
        })
        
        self.registry.register_cost('privacy', PrivacyCost, {
            'base_costs': {'honest': 8.0, 'withholding': 4.0, 'adversarial': 1.0},
            'privacy_factor': 0.3,
            'data_sensitivity': 1.0
        })
        
        # Punishment mechanisms
        self.registry.register_punishment('adversarial', AdversarialPunishment, {
            'penalty_strength': 1.0,
            'detection_threshold': 0.7,
            'gradient_variance_weight': 0.5
        })
        
        self.registry.register_punishment('reputation', ReputationBasedPunishment, {
            'memory_factor': 0.8,
            'reputation_threshold': 0.5,
            'max_penalty': 20.0
        })
    
    def create_experiment(self, 
                         experiment_config: Dict[str, Any]) -> Tuple[EGTSimulator, MLIntegration]:
        """
        Create an experiment based on configuration.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Tuple of (simulator, ml_integration)
        """
        # Extract configuration
        num_clients = experiment_config.get('num_clients', 100)
        strategy_labels = experiment_config.get('strategy_labels', ['honest', 'withholding', 'adversarial'])
        initial_distribution = experiment_config.get('initial_distribution', None)
        simulator_config = experiment_config.get('simulator_config', {})
        
        # Create simulator
        simulator = EGTSimulator(
            num_clients=num_clients,
            strategy_labels=strategy_labels,
            initial_distribution=initial_distribution,
            config=simulator_config
        )
        
        # Create mechanisms
        reward_name = experiment_config.get('reward_mechanism', 'adaptive')
        reward_config = experiment_config.get('reward_config', {})
        reward = self.registry.get_reward(reward_name, reward_config)
        
        cost_name = experiment_config.get('cost_mechanism', 'computational')
        cost_config = experiment_config.get('cost_config', {})
        cost = self.registry.get_cost(cost_name, cost_config)
        
        punishment_name = experiment_config.get('punishment_mechanism', 'none')
        punishment_config = experiment_config.get('punishment_config', {})
        punishment = self.registry.get_punishment(punishment_name, punishment_config)
        
        # Create ML integration
        update_frequency = experiment_config.get('update_frequency', 1)
        ml_config = experiment_config.get('ml_config', {})
        
        integration = MLIntegration(
            simulator=simulator,
            reward_mechanism=reward,
            cost_mechanism=cost,
            punishment_mechanism=punishment,
            update_frequency=update_frequency,
            config=ml_config
        )
        
        return simulator, integration
    
    def run_experiment(self, 
                      experiment_config: Dict[str, Any],
                      num_epochs: int = 100,
                      visualize: bool = True,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run an experiment and generate results.
        
        Args:
            experiment_config: Experiment configuration
            num_epochs: Number of epochs to run
            visualize: Whether to generate visualizations
            output_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary with experiment results
        """
        # Create experiment
        simulator, integration = self.create_experiment(experiment_config)
        
        # Run simulation
        results = integration.run_simulation(num_epochs=num_epochs)
        
        # Generate visualizations if requested
        if visualize:
            visualizer = EGTVisualizer(simulator)
            
            if output_dir:
                visualizer.create_summary_report(output_dir)
            else:
                visualizer.plot_strategy_evolution()
                visualizer.plot_ternary_trajectories()
                visualizer.plot_3d_convergence()
                plt.show()
        
        return results
    
    def get_available_mechanisms(self) -> Dict[str, List[str]]:
        """
        Get list of available mechanisms.
        
        Returns:
            Dictionary with mechanism categories and names
        """
        return self.registry.list_mechanisms()
