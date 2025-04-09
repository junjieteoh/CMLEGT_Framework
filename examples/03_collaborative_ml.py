"""
Collaborative ML with EGT Framework Example

This example demonstrates how to:
1. Set up a collaborative ML environment with multiple participants
2. Simulate different participant strategies (honest, withholding, adversarial)
3. Integrate EGT framework to model strategy evolution
4. Analyze and visualize both ML and strategic behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import random
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from modules with adjusted paths for GitHub structure
from registry.experiment import ExperimentBuilder
from visualization.visualizer import EGTVisualizer
from ml.pytorch import PyTorchIntegration
from mechanisms.reward import AdaptiveReward
from mechanisms.cost import ComputationalCost
from mechanisms.punishment import AdversarialPunishment

# Custom collate function to handle different types in the dataset
def custom_collate(batch):
    """Custom collate function that handles different data types."""
    if len(batch) == 0:
        return []
    
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)
    elif isinstance(elem, tuple) and len(elem) == 2:
        # For (data, target) pairs
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        
        # Handle data
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data, 0)
        
        # Handle target
        if all(isinstance(t, torch.Tensor) for t in target):
            target = torch.stack(target, 0)
        else:
            target = torch.tensor(target)
            
        return data, target
    else:
        raise TypeError(f"Batch contains unsupported type: {type(elem)}")

class CollaborativeMLSimulation:
    """Simulates a collaborative ML scenario with multiple participants."""
    
    def __init__(self, num_clients=100, device='cpu'):
        self.num_clients = num_clients
        self.device = device
        self.client_data = []
        self.client_strategies = []
        self.client_contributions = []
        self.strategy_to_idx = {'honest': 0, 'withholding': 1, 'adversarial': 2}
        
    def prepare_data(self, dataset_name='mnist'):
        """Prepare and partition dataset among clients."""
        print(f"Preparing {dataset_name} dataset for {self.num_clients} clients...")
        
        # Load and transform dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        # Create test loader for evaluation
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=custom_collate)
        
        # Partition training data among clients
        samples_per_client = len(train_dataset) // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else len(train_dataset)
            client_subset = Subset(train_dataset, range(start_idx, end_idx))
            self.client_data.append(client_subset)
            
        print(f"Dataset partitioned: ~{samples_per_client} samples per client")
    
    def setup_initial_strategies(self, dist=None):
        """Assign initial strategies to clients."""
        if dist is None:
            dist = [0.33, 0.33, 0.34]  # Equal distribution by default
            
        strategy_counts = [int(p * self.num_clients) for p in dist]
        strategy_counts[-1] += self.num_clients - sum(strategy_counts)
        
        strategies = []
        for i, count in enumerate(['honest', 'withholding', 'adversarial']):
            strategies.extend([count] * strategy_counts[i])
            
        random.shuffle(strategies)
        self.client_strategies = strategies
        
        print(f"Initial strategies: {strategy_counts[0]} honest, {strategy_counts[1]} withholding, {strategy_counts[2]} adversarial")
    
    def apply_contribution_effects(self, strategy_distribution):
        """Apply contribution effects based on current strategy distribution."""
        # Convert distribution to counts
        honest_count = int(strategy_distribution['honest'] * self.num_clients)
        withholding_count = int(strategy_distribution['withholding'] * self.num_clients)
        adversarial_count = self.num_clients - honest_count - withholding_count
        
        # Reassign strategies
        strategies = ['honest'] * honest_count + ['withholding'] * withholding_count + ['adversarial'] * adversarial_count
        random.shuffle(strategies)
        self.client_strategies = strategies
        
        # Generate contributions based on strategy
        self.client_contributions = []
        for strategy in strategies:
            if strategy == 'honest':
                contribution = random.uniform(0.7, 1.0)  # High quality
            elif strategy == 'withholding':
                contribution = random.uniform(0.3, 0.6)  # Medium quality
            else:  # adversarial
                contribution = random.uniform(-0.5, 0.1)  # Poor/harmful quality
            self.client_contributions.append(contribution)
        
        # Return strategy distribution for logging
        current_dist = {
            'honest': honest_count / self.num_clients,
            'withholding': withholding_count / self.num_clients,
            'adversarial': adversarial_count / self.num_clients
        }
            
        return np.array(self.client_contributions), current_dist
    
    def create_collaborative_loader(self, batch_size=64):
        """Create a DataLoader that simulates collaborative training."""
        if not self.client_data or not self.client_strategies:
            raise ValueError("Must call prepare_data() and setup_initial_strategies() first")
        
        modified_datasets = []
        
        for subset, strategy in zip(self.client_data, self.client_strategies):
            if strategy == 'honest':
                # Use all data as is
                modified_datasets.append(subset)
                
            elif strategy == 'withholding':
                # Contribute fewer samples
                indices = list(range(len(subset)))
                num_samples = int(len(subset) * random.uniform(0.3, 0.6))
                selected_indices = random.sample(indices, num_samples)
                modified_datasets.append(Subset(subset, selected_indices))
                
            else:  # adversarial
                # Process the adversarial data
                temp_loader = DataLoader(subset, batch_size=len(subset))
                processed = False
                
                for images, labels in temp_loader:
                    if random.random() < 0.7:  # Label flipping
                        corrupt_ratio = random.uniform(0.3, 0.8)
                        num_corrupt = int(len(labels) * corrupt_ratio)
                        corrupt_indices = random.sample(range(len(labels)), num_corrupt)
                        
                        corrupted_labels = labels.clone()
                        for idx in corrupt_indices:
                            new_label = (int(labels[idx].item()) + random.randint(1, 9)) % 10
                            corrupted_labels[idx] = torch.tensor(new_label).to(labels.dtype)
                            
                        modified_datasets.append(TensorDataset(images, corrupted_labels))
                    else:  # Add noise
                        noise_level = random.uniform(0.2, 0.5)
                        noisy_images = images + noise_level * torch.randn_like(images)
                        noisy_images = torch.clamp(noisy_images, 0, 1)
                        modified_datasets.append(TensorDataset(noisy_images, labels))
                    
                    processed = True
                    break  # Only process one batch
                
                # Fallback if the subset was empty
                if not processed:
                    modified_datasets.append(subset)
        
        combined_dataset = ConcatDataset(modified_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

def run_collaborative_ml():
    """Run a collaborative ML experiment with EGT integration."""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"collaborative_ml_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collaborative ML simulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 100
    collaborative_sim = CollaborativeMLSimulation(num_clients=num_clients, device=device)
    collaborative_sim.prepare_data(dataset_name='mnist')
    
    # Define CNN model
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
    
    # Create model and optimizer
    model = SimpleConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Define experiment configuration
    experiment_config = {
        'name': 'Collaborative MNIST with EGT',
        'num_clients': num_clients,
        
        'reward_mechanism': 'adaptive',
        'reward_config': {
            'initial_pool': 1000.0,
            'learning_rate': 0.1,
            'accuracy_threshold': 0.7,
        },
        
        'cost_mechanism': 'computational',
        'cost_config': {
            'base_costs': {'honest': 10.0, 'withholding': 5.0, 'adversarial': 2.0},
            'scaling_factor': 0.2
        },
        
        'punishment_mechanism': 'adversarial',
        'punishment_config': {
            'penalty_strength': 1.0,
            'detection_threshold': 0.7
        },
        
        'update_frequency': 1,
        'initial_distribution': [0.33, 0.33, 0.34],
        
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
    
    # Set up initial client strategies
    collaborative_sim.setup_initial_strategies(experiment_config['initial_distribution'])
    
    # Create EGT simulator and integration
    builder = ExperimentBuilder()
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
    
    # Training loop
    num_epochs = 10
    history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'honest_ratio': [], 'withholding_ratio': [], 'adversarial_ratio': []
    }
    
    print("Starting collaborative training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Update client contributions
        contributions, strategy_dist = collaborative_sim.apply_contribution_effects(
            simulator.get_current_distribution()
        )
        
        # Log strategy distribution
        history['honest_ratio'].append(strategy_dist['honest'])
        history['withholding_ratio'].append(strategy_dist['withholding'])
        history['adversarial_ratio'].append(strategy_dist['adversarial'])
        
        print(f"Strategy distribution: {strategy_dist}")
        
        # Create collaborative loader
        train_loader = collaborative_sim.create_collaborative_loader(batch_size=64)
        
        # Train one epoch
        train_metrics = trainer.train_epoch(train_loader, collaborative_sim.test_loader)
        
        # Log metrics
        history['train_loss'].append(train_metrics.get('train_loss', 0.0))
        history['val_loss'].append(train_metrics.get('val_loss', 0.0))
        history['val_accuracy'].append(train_metrics.get('val_accuracy', 0.0))
        
        print(f"Train Loss: {train_metrics.get('train_loss', 0.0):.4f}, "
              f"Val Loss: {train_metrics.get('val_loss', 0.0):.4f}, "
              f"Val Accuracy: {train_metrics.get('val_accuracy', 0.0):.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = EGTVisualizer(simulator)
    
    # EGT visualizations
    visualizer.plot_strategy_evolution(save_path=os.path.join(output_dir, "strategy_evolution.png"))
    visualizer.plot_ternary_trajectories(save_path=os.path.join(output_dir, "ternary_plot.png"))
    visualizer.plot_convergence_analysis(save_path=os.path.join(output_dir, "convergence.png"))
    visualizer.plot_payoff_analysis(save_path=os.path.join(output_dir, "payoff_analysis.png"))
    
    # ML performance plots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance.png"))
    
    # Strategy distribution vs performance
    plt.figure(figsize=(12, 6))
    epochs = range(1, num_epochs + 1)
    
    plt.subplot(1, 2, 1)
    plt.stackplot(epochs, 
                 [history['honest_ratio'], 
                  history['withholding_ratio'], 
                  history['adversarial_ratio']], 
                 labels=['Honest', 'Withholding', 'Adversarial'],
                 colors=['green', 'orange', 'red'])
    plt.title('Strategy Distribution Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Proportion')
    plt.legend(loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'b-', label='Accuracy')
    plt.plot(epochs, [history['honest_ratio'][i] for i in range(num_epochs)], 'g--', label='Honest Ratio')
    plt.title('Accuracy vs. Honest Participation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_vs_performance.png"))
    
    # Save results
    convergence = simulator.analyze_convergence()
    final_dist = simulator.get_current_distribution()
    
    print("\nFinal Results:")
    print(f"Convergence analysis: {convergence}")
    print(f"Final strategy distribution: {final_dist}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Save history
    import json
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(serializable_history, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    run_collaborative_ml() 