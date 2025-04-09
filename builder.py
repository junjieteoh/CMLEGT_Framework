"""
Experiment Builder and Mechanism Registry.
Provides tools to define, configure, and run EGT experiments.
"""

import os
import matplotlib.pyplot as plt # Needed for the visualize=True default in run_experiment
from typing import Dict, List, Tuple, Optional, Any

# Core components
from .core import RewardMechanism, CostMechanism, PunishmentMechanism
from .simulator import EGTSimulator
from .integration import MLIntegration
from .visualization import EGTVisualizer

# Specific mechanism implementations
from .mechanisms.rewards import AdaptiveReward, PerformanceBasedReward, ShapleyValueReward
from .mechanisms.costs import ComputationalCost, PrivacyCost
from .mechanisms.punishments import AdversarialPunishment, ReputationBasedPunishment

# ============================================================
# MECHANISM REGISTRY AND EXPERIMENT BUILDER
# ============================================================

class MechanismRegistry:
    """
    Registry for reward, cost, and punishment mechanisms.
    Allows registering and retrieving mechanism classes and configurations.
    """
    
    def __init__(self):
        """Initialize the registry.
        Stores mechanism classes and their default configurations.
        """
        self.reward_mechanisms: Dict[str, Dict[str, Any]] = {}
        self.cost_mechanisms: Dict[str, Dict[str, Any]] = {}
        self.punishment_mechanisms: Dict[str, Dict[str, Any]] = {}
    
    def register_reward(self, name: str, cls, default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a reward mechanism class.
        
        Args:
            name: Unique name for the mechanism (e.g., 'adaptive').
            cls: The mechanism class (e.g., AdaptiveReward).
            default_config: Optional dictionary of default configuration parameters.
        """
        if not issubclass(cls, RewardMechanism):
             raise TypeError(f"Class {cls.__name__} is not a subclass of RewardMechanism")
        self.reward_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def register_cost(self, name: str, cls, default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a cost mechanism class.
        
        Args:
            name: Unique name for the mechanism (e.g., 'computational').
            cls: The mechanism class (e.g., ComputationalCost).
            default_config: Optional dictionary of default configuration parameters.
        """
        if not issubclass(cls, CostMechanism):
             raise TypeError(f"Class {cls.__name__} is not a subclass of CostMechanism")
        self.cost_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def register_punishment(self, name: str, cls, default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a punishment mechanism class.
        
        Args:
            name: Unique name for the mechanism (e.g., 'reputation').
            cls: The mechanism class (e.g., ReputationBasedPunishment).
            default_config: Optional dictionary of default configuration parameters.
        """
        if not issubclass(cls, PunishmentMechanism):
             raise TypeError(f"Class {cls.__name__} is not a subclass of PunishmentMechanism")
        self.punishment_mechanisms[name] = {
            'class': cls,
            'config': default_config or {}
        }
    
    def get_reward(self, name: str, custom_config: Optional[Dict[str, Any]] = None) -> RewardMechanism:
        """
        Get an instantiated reward mechanism by name, merging configs.
        
        Args:
            name: Name of the registered mechanism.
            custom_config: Custom configuration to override/add to defaults.
            
        Returns:
            An instance of the requested RewardMechanism.
            
        Raises:
            ValueError: If the mechanism name is not found.
        """
        if name not in self.reward_mechanisms:
            raise ValueError(f"Reward mechanism '{name}' not found in registry. Available: {list(self.reward_mechanisms.keys())}")
        
        entry = self.reward_mechanisms[name]
        # Merge default and custom configurations (custom overrides default)
        config = entry['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        # Create instance
        return entry['class'](name=name, config=config)
    
    def get_cost(self, name: str, custom_config: Optional[Dict[str, Any]] = None) -> CostMechanism:
        """
        Get an instantiated cost mechanism by name, merging configs.
        
        Args:
            name: Name of the registered mechanism.
            custom_config: Custom configuration to override/add to defaults.
            
        Returns:
            An instance of the requested CostMechanism.
            
        Raises:
            ValueError: If the mechanism name is not found.
        """
        if name not in self.cost_mechanisms:
            raise ValueError(f"Cost mechanism '{name}' not found in registry. Available: {list(self.cost_mechanisms.keys())}")
        
        entry = self.cost_mechanisms[name]
        config = entry['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        return entry['class'](name=name, config=config)
    
    def get_punishment(self, name: str, custom_config: Optional[Dict[str, Any]] = None) -> Optional[PunishmentMechanism]:
        """
        Get an instantiated punishment mechanism by name, merging configs.
        Returns None if name is 'none'.
        
        Args:
            name: Name of the registered mechanism or 'none'.
            custom_config: Custom configuration to override/add to defaults.
            
        Returns:
            An instance of the requested PunishmentMechanism or None.
            
        Raises:
            ValueError: If the mechanism name is not found (and not 'none').
        """
        if name == 'none' or name is None:
            return None
        
        if name not in self.punishment_mechanisms:
            raise ValueError(f"Punishment mechanism '{name}' not found in registry. Available: {list(self.punishment_mechanisms.keys()) + ['none']}")
        
        entry = self.punishment_mechanisms[name]
        config = entry['config'].copy()
        if custom_config:
            config.update(custom_config)
        
        return entry['class'](name=name, config=config)
    
    def list_mechanisms(self) -> Dict[str, List[str]]:
        """
        List all registered mechanisms by category.
        
        Returns:
            Dictionary with keys 'reward', 'cost', 'punishment' and lists of names.
        """
        return {
            'reward': list(self.reward_mechanisms.keys()),
            'cost': list(self.cost_mechanisms.keys()),
            'punishment': list(self.punishment_mechanisms.keys()) + ['none'] # Include 'none' as an option
        }


class ExperimentBuilder:
    """
    Builder pattern for creating and running EGT simulation experiments.
    Uses a MechanismRegistry to configure components.
    """
    
    def __init__(self, registry: Optional[MechanismRegistry] = None):
        """
        Initialize the experiment builder.
        
        Args:
            registry: An optional pre-configured MechanismRegistry. 
                      If None, a default registry with standard mechanisms is created.
        """
        # Use provided registry or create a default one
        self.registry = registry if registry else self._create_default_registry()
            
    def _create_default_registry(self) -> MechanismRegistry:
        """Creates and populates a default mechanism registry."""
        registry = MechanismRegistry()
        
        # Register default reward mechanisms
        registry.register_reward('adaptive', AdaptiveReward, {
            'initial_pool': 1000.0,
            'learning_rate': 0.1,
            'accuracy_threshold': 0.7,
            'strategy_weights': [1.0, 0.5, 0.0]  # Default weights for 3 strategies
        })
        registry.register_reward('performance', PerformanceBasedReward, {
            'initial_pool': 1000.0,
            'performance_weight': 0.8
        })
        registry.register_reward('shapley', ShapleyValueReward, {
            'initial_pool': 1000.0,
            'samples': 100,
            # Add default strategy weights if needed by its compute method
            'strategy_weights': [1.0, 0.3, 0.0] 
        })
        
        # Register default cost mechanisms
        registry.register_cost('computational', ComputationalCost, {
            'base_costs': {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0},
            'scaling_factor': 0.2
        })
        registry.register_cost('privacy', PrivacyCost, {
            'base_costs': {'honest': 8.0, 'withholding': 4.0, 'adversarial': 1.0},
            'privacy_factor': 0.3,
            'data_sensitivity': 1.0
        })
        
        # Register default punishment mechanisms
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
    
    def create_experiment(self, 
                         experiment_config: Dict[str, Any]) -> Tuple[EGTSimulator, MLIntegration]:
        """
        Create the core components for an experiment based on a configuration dictionary.
        
        Args:
            experiment_config: Dictionary specifying the experiment setup. Expected keys:
                - num_clients (int, default: 100)
                - strategy_labels (List[str], default: ['honest', 'withholding', 'adversarial'])
                - initial_distribution (Optional[np.ndarray], default: uniform)
                - simulator_config (Dict, optional): Config passed to EGTSimulator
                - reward_mechanism (str, default: 'adaptive'): Name of reward mechanism
                - reward_config (Dict, optional): Custom config for reward mechanism
                - cost_mechanism (str, default: 'computational'): Name of cost mechanism
                - cost_config (Dict, optional): Custom config for cost mechanism
                - punishment_mechanism (str, default: 'none'): Name of punishment mechanism or 'none'
                - punishment_config (Dict, optional): Custom config for punishment mechanism
                - update_frequency (int, default: 1): How often EGT updates during ML training
                - ml_config (Dict, optional): Config passed to MLIntegration
            
        Returns:
            A tuple containing the configured (EGTSimulator, MLIntegration) instances.
        """
        # --- Extract configuration with defaults --- 
        num_clients = experiment_config.get('num_clients', 100)
        strategy_labels = experiment_config.get('strategy_labels', ['honest', 'withholding', 'adversarial'])
        initial_distribution = experiment_config.get('initial_distribution', None)
        simulator_config = experiment_config.get('simulator_config', {})
        
        reward_name = experiment_config.get('reward_mechanism', 'adaptive')
        reward_config = experiment_config.get('reward_config', {})
        
        cost_name = experiment_config.get('cost_mechanism', 'computational')
        cost_config = experiment_config.get('cost_config', {})
        
        punishment_name = experiment_config.get('punishment_mechanism', 'none')
        punishment_config = experiment_config.get('punishment_config', {})
        
        update_frequency = experiment_config.get('update_frequency', 1)
        ml_config = experiment_config.get('ml_config', {})
        
        # --- Create Simulator --- 
        simulator = EGTSimulator(
            num_clients=num_clients,
            strategy_labels=strategy_labels,
            initial_distribution=initial_distribution,
            config=simulator_config
        )
        
        # --- Create Mechanisms using Registry --- 
        try:
            reward = self.registry.get_reward(reward_name, reward_config)
            cost = self.registry.get_cost(cost_name, cost_config)
            punishment = self.registry.get_punishment(punishment_name, punishment_config)
        except ValueError as e:
            print(f"Error creating mechanisms: {e}")
            raise  # Re-raise the error to stop execution
        
        # --- Create ML Integration --- 
        integration = MLIntegration(
            simulator=simulator,
            reward_mechanism=reward,
            cost_mechanism=cost,
            punishment_mechanism=punishment,
            update_frequency=update_frequency,
            config=ml_config
        )
        
        print(f"Experiment created with: Reward='{reward_name}', Cost='{cost_name}', Punishment='{punishment_name or 'None'}'")
        return simulator, integration
    
    def run_experiment(self, 
                      experiment_config: Dict[str, Any],
                      num_epochs: int = 100,
                      visualize: bool = True,
                      output_dir: Optional[str] = None,
                      report_filename_base: str = "egt_run") -> Dict[str, Any]:
        """
        Run a full experiment (pure simulation) based on configuration and generate results.
        
        Args:
            experiment_config: Configuration dictionary passed to create_experiment.
            num_epochs: Number of simulation epochs (steps) to run.
            visualize: If True, generate and either show or save visualizations.
            output_dir: Directory to save visualizations and report (if visualize=True). 
                       If None and visualize=True, plots are shown interactively.
            report_filename_base: Base name for the report HTML and image files.
            
        Returns:
            Dictionary containing the simulation results (history, convergence, etc.).
        """
        # Create experiment components
        try:
            simulator, integration = self.create_experiment(experiment_config)
        except Exception as e:
             print(f"Failed to create experiment: {e}")
             return {'error': "Experiment creation failed", 'details': str(e)}

        # Run the simulation using the integration's simulation runner
        try:
            results = integration.run_simulation(num_epochs=num_epochs)
            print(f"Simulation completed. Converged: {results.get('convergence', {}).get('converged', 'N/A')}")
        except Exception as e:
             print(f"Simulation run failed after {integration.current_epoch} epochs: {e}")
             # Return partial results if possible
             results = {
                 'error': "Simulation run failed", 'details': str(e),
                 'history': simulator.get_history(), # Log history up to failure
                 'metrics_history': integration.metrics_history,
                 'current_distribution': simulator.get_current_distribution(),
                 'epoch_failed': integration.current_epoch
             }
             return results
        
        # Generate visualizations if requested
        if visualize:
            print("Generating visualizations...")
            visualizer = EGTVisualizer(simulator) # Create visualizer with the final simulator state
            
            if output_dir:
                print(f"Saving report and plots to: {output_dir}")
                report_path = visualizer.create_summary_report(output_dir, base_filename=report_filename_base)
                if report_path:
                    print(f"Summary report generated at: {report_path}")
                else:
                    print("Failed to generate summary report.")
            else:
                # Show plots interactively if no output directory is specified
                print("Displaying plots interactively...")
                try:
                    visualizer.show_plots() # Uses plt.show() internally
                except Exception as e:
                    print(f"Error displaying plots: {e}")
                    # Ensure plots generated so far are closed if plt.show fails
                    plt.close('all') 
        
        return results
    
    def get_available_mechanisms(self) -> Dict[str, List[str]]:
        """
        Get a list of available mechanisms registered with the builder.
        
        Returns:
            Dictionary with mechanism categories (reward, cost, punishment) and names.
        """
        return self.registry.list_mechanisms() 