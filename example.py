"""
Example usage of the EGT Framework.
Demonstrates setting up and running a pure simulation experiment.
"""

import logging

# Import the main components needed to run an experiment
from .builder import ExperimentBuilder
from .utils import create_default_registry # Optional: can also use builder's default

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('egt_example')

# ============================================================
# EXAMPLE USAGE
# ============================================================

def run_example(num_epochs: int = 100, visualize: bool = True, output_dir: str = "./egt_example_output"):
    """
    Run an example simulation to demonstrate framework usage.

    Args:
        num_epochs: Number of simulation steps to run.
        visualize: Whether to generate and save/show visualizations.
        output_dir: Directory to save the report and plots if visualize is True.
                    If visualize is True and output_dir is None, plots are shown interactively.
    """
    logger.info("--- Starting EGT Framework Example --- ")
    
    # Option 1: Create a builder which uses its internal default registry
    builder = ExperimentBuilder()
    
    # Option 2: Explicitly create and use a default registry (more verbose)
    # registry = create_default_registry()
    # builder = ExperimentBuilder(registry)
    
    # --- Configure the Experiment --- 
    # This dictionary defines all parameters for the simulation.
    experiment_config = {
        'num_clients': 50, # Number of agents in the simulation
        'strategy_labels': ["honest", "withholding", "adversarial"], # Names of strategies
        # 'initial_distribution': [0.8, 0.1, 0.1], # Optional: Start with a specific distribution
        
        # Mechanism Selection and Configuration
        'reward_mechanism': 'adaptive', # Use the AdaptiveReward mechanism
        'reward_config': { # Custom parameters for AdaptiveReward
            'initial_pool': 1200.0,
            'learning_rate': 0.15,
            'accuracy_threshold': 0.65
        },
        'cost_mechanism': 'computational', # Use ComputationalCost
        'cost_config': { # Custom parameters for ComputationalCost
            'base_costs': {'honest': 12.0, 'withholding': 6.0, 'adversarial': 3.0},
            'scaling_factor': 0.15 
        },
        'punishment_mechanism': 'adversarial', # Use AdversarialPunishment
        'punishment_config': { # Custom parameters for AdversarialPunishment
            'penalty_strength': 1.5,
            'detection_threshold': 0.6
        },
        
        # Simulator Configuration (EGT dynamics parameters)
        'simulator_config': {
            # Parameters for _generate_contributions function (simulating client quality)
            'honest_quality': 0.7, 
            'withholding_quality': 0.15,
            'adversarial_quality': -0.5,
            'contribution_noise_std': 0.05, # Std deviation of noise added to contributions
            # Parameters for the replicator dynamics update rule
            'replicator': {
                'learning_rate': 0.08, # Speed of strategy adaptation
                'noise_level': 0.005,  # Small random perturbation to distribution
                'memory_length': 5     # Number of past rounds to average payoffs over
            }
        },

        # ML Integration Configuration (relevant for pure simulation too)
        'update_frequency': 1, # How often EGT dynamics update (e.g., every epoch)
        'ml_config': {
             # Parameters used by MLIntegration.simulate_metrics
             'simulation_params': {
                 'base_accuracy': 0.4,
                 'max_accuracy': 0.9,
                 'noise_level': 0.01
            }
        }
    }
    
    logger.info(f"Running experiment with configuration:\n{experiment_config}")
    
    # --- Run the Experiment --- 
    # The builder handles creating the simulator and integration, 
    # running the simulation loop, and optionally visualizing.
    results = builder.run_experiment(
        experiment_config,
        num_epochs=num_epochs,
        visualize=visualize,
        output_dir=output_dir, # Pass the output directory
        report_filename_base="example_run" # Base name for output files
    )
    
    # --- Process Results --- 
    if 'error' in results:
        logger.error(f"Experiment failed: {results['error']} - {results.get('details')}")
    else:
        logger.info("--- Experiment Finished Successfully --- ")
        # Print final distribution from results
        final_dist = results.get('final_distribution', {})
        dist_str = ", ".join([f"{k}={v:.4f}" for k, v in final_dist.items()])
        logger.info(f"Final distribution: [{dist_str}]")
        
        # Print convergence analysis from results
        convergence = results.get('convergence', {})
        logger.info(f"Convergence Status: {convergence.get('converged', 'N/A')}")
        if convergence.get('converged'):
             logger.info(f"  Converged at step: {convergence.get('convergence_time', 'N/A')}")
        logger.info(f"  Distribution Type: {convergence.get('distribution_type', 'N/A')}")
        logger.info(f"  Dominant Strategy: {convergence.get('dominant_strategy', 'N/A')}")
    
    return results


# This block allows running the example directly from the command line
# e.g., python -m src.egt_framework.example
if __name__ == "__main__":
    # You can modify parameters here for quick tests
    run_example(num_epochs=150, visualize=True, output_dir="./egt_example_output") 