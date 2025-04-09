"""
Basic EGT Framework Example

This example demonstrates how to:
1. Set up a basic EGT simulation
2. Configure reward, cost, and punishment mechanisms
3. Run the simulation and visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from modules with adjusted paths for GitHub structure
from registry.experiment import ExperimentBuilder
from visualization.visualizer import EGTVisualizer
from mechanisms.reward import AdaptiveReward
from mechanisms.cost import ComputationalCost
from mechanisms.punishment import AdversarialPunishment

def run_basic_simulation():
    """Run a basic EGT simulation without ML integration."""
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"basic_sim_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiment configuration
    experiment_config = {
        'name': 'Basic EGT Simulation',
        'num_clients': 100,  # Number of participants
        
        # Configure reward mechanism
        'reward_mechanism': 'adaptive',
        'reward_config': {
            'initial_pool': 1000.0,  # Initial reward pool
            'learning_rate': 0.1,    # How fast rewards adapt
            'accuracy_threshold': 0.7,  # Target accuracy
        },
        
        # Configure cost mechanism
        'cost_mechanism': 'computational',
        'cost_config': {
            'base_costs': {
                'honest': 10.0,       # Higher cost for honest behavior
                'withholding': 5.0,   # Medium cost for withholding
                'adversarial': 2.0    # Low cost for adversarial
            },
            'scaling_factor': 0.2  # How much costs scale with performance
        },
        
        # Configure punishment mechanism
        'punishment_mechanism': 'adversarial',
        'punishment_config': {
            'penalty_strength': 1.0,  # Strength of punishment
            'detection_threshold': 0.7  # Threshold for detecting bad behavior
        },
        
        # Initial strategy distribution
        'initial_distribution': [0.33, 0.33, 0.34],  # [honest, withholding, adversarial]
        
        # Simulator settings
        'simulator_config': {
            'honest_quality': 0.6,     # Quality of honest contributions
            'withholding_quality': 0.2,  # Quality of withholding contributions
            'adversarial_quality': -0.4,  # Quality of adversarial contributions
            'replicator': {
                'learning_rate': 0.1,  # How fast strategies evolve
                'noise_level': 0.01    # Random noise in evolution
            }
        }
    }
    
    # Create experiment builder
    builder = ExperimentBuilder()
    
    # Create simulator and integration
    print("Creating EGT simulator...")
    simulator, integration = builder.create_experiment(experiment_config)
    
    # Run simulation
    print("Running simulation for 50 epochs...")
    results = integration.run_simulation(num_epochs=50)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = EGTVisualizer(simulator)
    
    # Plot strategy evolution over time
    visualizer.plot_strategy_evolution(
        save_path=os.path.join(output_dir, "strategy_evolution.png")
    )
    
    # Plot strategy trajectories in ternary space
    visualizer.plot_ternary_trajectories(
        save_path=os.path.join(output_dir, "ternary_plot.png")
    )
    
    # Analyze convergence
    visualizer.plot_convergence_analysis(
        save_path=os.path.join(output_dir, "convergence.png")
    )
    
    # Analyze payoffs
    visualizer.plot_payoff_analysis(
        save_path=os.path.join(output_dir, "payoff_analysis.png")
    )
    
    # Get final results
    convergence = simulator.analyze_convergence()
    final_dist = simulator.get_current_distribution()
    
    print("\nSimulation Results:")
    print(f"Convergence analysis: {convergence}")
    print(f"Final strategy distribution: {final_dist}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    run_basic_simulation() 