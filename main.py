"""
Main entry point for the EGT Framework for Collaborative Machine Learning.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# Import for backwards compatibility
from .registry.experiment import ExperimentBuilder
from .visualization.visualizer import EGTVisualizer
from .ml.integration import MLIntegration
from .ml.pytorch import PyTorchIntegration, PyTorchTrainer
from .registry.registry import MechanismRegistry
from .core.mechanisms import RewardMechanism, CostMechanism, PunishmentMechanism

def run_example():
    """
    Run a simple example without dependencies.
    This is a placeholder that simulates a basic EGT process.
    """
    print("Running EGT Framework Example")
    print("-----------------------------")
    
    # Simple simulation parameters
    num_strategies = 3
    num_steps = 100
    
    # Initial distribution
    distribution = np.array([0.33, 0.33, 0.34])
    
    # Simple payoff matrix (3x3 game)
    payoff_matrix = np.array([
        [3, 1, 0],   # Honest vs others
        [4, 2, 0.5], # Withholding vs others
        [5, 1, 1]    # Adversarial vs others
    ])
    
    # Track history
    history = [distribution.copy()]
    
    # Simple replicator dynamics
    for step in range(num_steps):
        # Calculate payoffs
        payoffs = np.dot(payoff_matrix, distribution)
        
        # Average payoff
        avg_payoff = np.sum(distribution * payoffs)
        
        # Update using replicator equation
        delta = distribution * (payoffs - avg_payoff)
        distribution += 0.1 * delta
        
        # Ensure valid distribution
        distribution = np.maximum(0, distribution)
        distribution /= np.sum(distribution)
        
        # Store
        history.append(distribution.copy())
        
        # Print progress occasionally
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step}: Honest={distribution[0]:.3f}, "
                  f"Withholding={distribution[1]:.3f}, "
                  f"Adversarial={distribution[2]:.3f}")
    
    # Plot results
    strategy_labels = ["Honest", "Withholding", "Adversarial"]
    plt.figure(figsize=(10, 6))
    
    history_array = np.array(history)
    for i in range(num_strategies):
        plt.plot(history_array[:, i], label=strategy_labels[i])
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Strategy Proportion')
    plt.title('Strategy Evolution in EGT Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check if we're running interactively or in a terminal
    if plt.isinteractive():
        plt.show()
    else:
        # Save to file
        plt.savefig('egt_simulation.png')
        print("Saved plot to egt_simulation.png")
    
    # Final analysis
    final_dist = distribution
    dominant_idx = np.argmax(final_dist)
    dominant_strategy = strategy_labels[dominant_idx]
    
    print("\nFinal Results:")
    print(f"Dominant strategy: {dominant_strategy} ({final_dist[dominant_idx]:.3f})")
    
    return {
        'history': history,
        'final_distribution': {label: final_dist[i] for i, label in enumerate(strategy_labels)}
    }


if __name__ == "__main__":
    run_example()