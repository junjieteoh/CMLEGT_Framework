"""
Simple ML Integration Example

This example demonstrates how to:
1. Integrate a PyTorch ML model with the EGT framework
2. Train the model while simulating strategic behavior
3. Visualize both ML and EGT results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
from ml.pytorch import PyTorchIntegration

def run_ml_integration():
    """Run an EGT simulation with PyTorch ML integration."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ml_integration_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MNIST dataset (as an example)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use a subset for faster training
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Define a simple CNN model
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.relu = nn.ReLU()
            self.max_pool = nn.MaxPool2d(2)
            self.dropout = nn.Dropout2d(0.5)
            
        def forward(self, x):
            x = self.relu(self.max_pool(self.conv1(x)))
            x = self.relu(self.max_pool(self.dropout(self.conv2(x))))
            x = x.view(-1, 320)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Create model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Define experiment configuration
    experiment_config = {
        'name': 'MNIST with EGT',
        'num_clients': 100,
        
        # Reward mechanism configuration
        'reward_mechanism': 'adaptive',
        'reward_config': {
            'initial_pool': 1000.0,
            'learning_rate': 0.1,
            'accuracy_threshold': 0.7,
        },
        
        # Cost mechanism configuration
        'cost_mechanism': 'computational',
        'cost_config': {
            'base_costs': {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0},
            'scaling_factor': 0.2
        },
        
        # Punishment mechanism configuration
        'punishment_mechanism': 'adversarial',
        'punishment_config': {
            'penalty_strength': 1.0,
            'detection_threshold': 0.7
        },
        
        'update_frequency': 1,  # Update EGT every epoch
        
        # Initial strategy distribution
        'initial_distribution': [0.33, 0.33, 0.34],  # [honest, withholding, adversarial]
        
        # Simulator configuration
        'simulator_config': {
            'honest_quality': 0.6,
            'withholding_quality': 0.2,
            'adversarial_quality': -0.4,
            'replicator': {
                'learning_rate': 0.1,
                'noise_level': 0.01
            }
        }
    }
    
    # Create experiment builder and simulator
    builder = ExperimentBuilder()
    simulator, integration = builder.create_experiment(experiment_config)
    
    # Create PyTorch integration
    print("Creating PyTorch integration...")
    pytorch_integration = PyTorchIntegration(
        simulator=simulator,
        reward_mechanism=integration.reward_mechanism,
        cost_mechanism=integration.cost_mechanism,
        punishment_mechanism=integration.punishment_mechanism,
        update_frequency=integration.update_frequency,
        config=integration.config
    )
    
    # Create trainer with EGT integration
    trainer = pytorch_integration.create_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )
    
    # Train model with EGT integration
    print("Training model with EGT integration...")
    num_epochs = 10
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        verbose=True
    )
    
    # Create EGT visualizations
    print("Creating visualizations...")
    visualizer = EGTVisualizer(simulator)
    visualizer.plot_strategy_evolution(save_path=os.path.join(output_dir, "strategy_evolution.png"))
    visualizer.plot_ternary_trajectories(save_path=os.path.join(output_dir, "ternary_plot.png"))
    visualizer.plot_convergence_analysis(save_path=os.path.join(output_dir, "convergence.png"))
    visualizer.plot_payoff_analysis(save_path=os.path.join(output_dir, "payoff_analysis.png"))
    
    # Plot ML metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['history']['train_loss'], label='Train Loss')
    plt.plot(history['history']['val_loss'], label='Val Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['history']['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ml_metrics.png"))
    
    # Print final results
    convergence = simulator.analyze_convergence()
    final_dist = simulator.get_current_distribution()
    
    print("\nFinal Results:")
    print(f"Convergence analysis: {convergence}")
    print(f"Final strategy distribution: {final_dist}")
    print(f"Final validation accuracy: {history['history']['val_accuracy'][-1]:.4f}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    run_ml_integration() 