"""
Simple example of Collaborative Machine Learning with EGT Framework integration.

This demonstrates how to set up:
1. A collaborative learning scenario with multiple contributing parties
2. How to integrate EGT dynamics to analyze strategy evolution
3. The feedback loop between model performance and participant strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from datetime import datetime
import argparse
from multiprocessing import Pool, cpu_count
import json

# Add parent directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from framework - with adjusted imports for GitHub structure
from registry.experiment import ExperimentBuilder
from visualization.visualizer import EGTVisualizer
from ml.pytorch import PyTorchIntegration
from simulation.simulator import EGTSimulator
from ml.integration import MLIntegration
from core.mechanisms import RewardMechanism

# Add a custom FixedReward mechanism
class FixedReward(RewardMechanism):
    """
    A simple fixed reward mechanism that assigns constant rewards based on strategy.
    
    This is useful for experiments where you want to control for reward variability
    and isolate the effects of other mechanisms (like costs or punishments).
    """
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config or {})
        # Default fixed rewards for each strategy
        self.fixed_rewards = config.get('fixed_rewards', {
            'honest': 10.0,
            'withholding': 5.0,
            'adversarial': 1.0
        })
        self.strategy_to_idx = {'honest': 0, 'withholding': 1, 'adversarial': 2}
        
    def update(self, metrics: dict) -> None:
        """No updates needed for fixed rewards."""
        # We need to implement this abstract method but no action is needed
        pass
        
    def compute(self, strategies, contributions=None) -> np.ndarray:
        """Compute fixed rewards based on strategy type."""
        rewards = np.zeros(len(strategies))
        for i, strategy_idx in enumerate(strategies):
            if strategy_idx == 0:  # honest
                rewards[i] = self.fixed_rewards['honest']
            elif strategy_idx == 1:  # withholding
                rewards[i] = self.fixed_rewards['withholding']
            else:  # adversarial
                rewards[i] = self.fixed_rewards['adversarial']
        return rewards
        
    def distribute(self, strategies, contributions) -> np.ndarray:
        """Simply return the fixed rewards based on strategy."""
        return self.compute(strategies)

#-------------------------------------------------------------
# PART 1: COLLABORATIVE ML DATA SIMULATION
#-------------------------------------------------------------

class CollaborativeDataset:
    """Simulates a collaborative dataset with multiple contributing parties."""
    
    def __init__(self, num_clients=10):
        self.num_clients = num_clients
        self.client_datasets = []
        self.client_strategies = []
        self.base_dataset = None
        self.test_dataset = None
        # Add tracking for client contributions
        self.client_contributions = []
        self.client_contribution_metrics = {}
        
    def load_and_partition_dataset(self, dataset_name='mnist'):
        """Load dataset and partition it among clients."""
        print(f"Loading {dataset_name} dataset...")
        
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        self.base_dataset = train_full
        
        # Fixed number of samples per client (100) instead of dividing the full dataset
        data_per_client = 100
        
        # Create dataset for each client with exactly 100 samples
        total_data = len(train_full)
        
        # Clear any existing datasets
        self.client_datasets = []
        
        for i in range(self.num_clients):
            # Calculate indices, ensuring we don't exceed dataset size
            start_idx = (i * data_per_client) % (total_data - data_per_client)
            end_idx = start_idx + data_per_client
            
            # Create subset with exactly 100 samples
            client_data = Subset(train_full, range(start_idx, end_idx))
            self.client_datasets.append(client_data)
        
        print(f"Dataset partitioned: {data_per_client} samples per client (fixed size)")
        
    def assign_initial_strategies(self, honest_ratio=0.33, withholding_ratio=0.33, adversarial_ratio=0.34):
        """Assign initial strategies to clients."""
        # Calculate number of clients for each strategy
        num_honest = int(self.num_clients * honest_ratio)
        num_withholding = int(self.num_clients * withholding_ratio)
        num_adversarial = self.num_clients - num_honest - num_withholding
        
        # Assign strategies
        strategies = ['honest'] * num_honest + ['withholding'] * num_withholding + ['adversarial'] * num_adversarial
        random.shuffle(strategies)
        self.client_strategies = strategies
        
        print(f"Assigned strategies: {num_honest} honest, {num_withholding} withholding, {num_adversarial} adversarial")
        
    def update_strategies(self, new_distribution):
        """Update client strategies based on new distribution."""
        honest_ratio = new_distribution['honest']
        withholding_ratio = new_distribution['withholding']
        adversarial_ratio = new_distribution['adversarial']
        
        num_honest = int(self.num_clients * honest_ratio)
        num_withholding = int(self.num_clients * withholding_ratio)
        num_adversarial = self.num_clients - num_honest - num_withholding
        
        strategies = ['honest'] * num_honest + ['withholding'] * num_withholding + ['adversarial'] * num_adversarial
        random.shuffle(strategies)
        self.client_strategies = strategies
        
    def create_combined_dataset(self):
        """
        Create a combined dataset based on client strategies.
        
        Returns:
            combined_train_loader: DataLoader for combined training data
            test_loader: DataLoader for testing
        """
        if not self.client_datasets or not self.client_strategies:
            raise ValueError("Must load data and assign strategies first")
        
        # Reset contribution tracking for this round
        self.client_contributions = []
        self.client_contribution_metrics = {
            'total_samples': [],
            'contributed_samples': [],
            'withholding_rate': [],
            'corrupted_samples': [],
            'corruption_rate': [],
            'noise_level': []
        }
        
        # Metrics by strategy
        strategy_metrics = {
            'honest': {'samples': 0, 'corrupted': 0, 'total_possible': 0},
            'withholding': {'samples': 0, 'corrupted': 0, 'total_possible': 0},
            'adversarial': {'samples': 0, 'corrupted': 0, 'total_possible': 0}
        }
        
        # Process data based on client strategies
        processed_data = []
        processed_labels = []
        
        for client_idx, (client_data, strategy) in enumerate(zip(self.client_datasets, self.client_strategies)):
            # Get client data
            loader = DataLoader(client_data, batch_size=len(client_data))
            for data, labels in loader:  # There will be only one batch
                client_metrics = {
                    'client_id': client_idx,
                    'strategy': strategy,
                    'total_samples': len(data),
                    'contributed_samples': 0,
                    'withholding_rate': 0.0,
                    'corrupted_samples': 0,
                    'corruption_rate': 0.0,
                    'noise_level': 0.0
                }
                
                strategy_metrics[strategy]['total_possible'] += len(data)
                
                if strategy == 'honest':
                    # Honest: Contribute all data with correct labels
                    processed_data.append(data)
                    processed_labels.append(labels)
                    
                    # Track metrics
                    client_metrics['contributed_samples'] = len(data)
                    client_metrics['withholding_rate'] = 0.0
                    client_metrics['corrupted_samples'] = 0
                    client_metrics['corruption_rate'] = 0.0
                    client_metrics['noise_level'] = 0.0
                    
                    strategy_metrics['honest']['samples'] += len(data)
                    
                elif strategy == 'withholding':
                    # Withholding: Contribute only a subset of data
                    withholding_rate = random.uniform(0.3, 0.7)
                    subset_size = int(len(data) * (1 - withholding_rate))
                    indices = random.sample(range(len(data)), subset_size)
                    processed_data.append(data[indices])
                    processed_labels.append(labels[indices])
                    
                    # Track metrics
                    client_metrics['contributed_samples'] = subset_size
                    client_metrics['withholding_rate'] = withholding_rate
                    client_metrics['corrupted_samples'] = 0
                    client_metrics['corruption_rate'] = 0.0
                    client_metrics['noise_level'] = 0.0
                    
                    strategy_metrics['withholding']['samples'] += subset_size
                    
                elif strategy == 'adversarial':
                    # Adversarial: Contribute data with some incorrect labels
                    # Determine corruption rate (percentage of labels to flip)
                    corruption_rate = random.uniform(0.2, 0.4)
                    corrupt_indices = random.sample(range(len(labels)), int(corruption_rate * len(labels)))
                    corrupt_labels = labels.clone()
                    for idx in corrupt_indices:
                        # Assign random incorrect label - ensure it's a tensor, not an int
                        new_label = (labels[idx].item() + random.randint(1, 9)) % 10
                        corrupt_labels[idx] = torch.tensor(new_label, dtype=labels.dtype)
                    
                    # Add noise to images
                    noise_level = random.uniform(0.1, 0.3)
                    noisy_data = data + noise_level * torch.randn_like(data)
                    noisy_data = torch.clamp(noisy_data, 0, 1)
                    
                    processed_data.append(noisy_data)
                    processed_labels.append(corrupt_labels)
                    
                    # Track metrics
                    client_metrics['contributed_samples'] = len(data)
                    client_metrics['withholding_rate'] = 0.0
                    client_metrics['corrupted_samples'] = len(corrupt_indices)
                    client_metrics['corruption_rate'] = corruption_rate
                    client_metrics['noise_level'] = noise_level
                    
                    strategy_metrics['adversarial']['samples'] += len(data)
                    strategy_metrics['adversarial']['corrupted'] += len(corrupt_indices)
                
                # Save metrics for this client
                self.client_contributions.append(client_metrics)
                
                # Update the running metrics lists
                for key in ['total_samples', 'contributed_samples', 'withholding_rate', 
                           'corrupted_samples', 'corruption_rate', 'noise_level']:
                    self.client_contribution_metrics[key].append(client_metrics[key])
        
        # Calculate aggregated metrics
        self.strategy_contribution_summary = {
            'honest': {
                'clients': self.client_strategies.count('honest'),
                'contribution_rate': strategy_metrics['honest']['samples'] / strategy_metrics['honest']['total_possible'] if strategy_metrics['honest']['total_possible'] > 0 else 0,
                'corruption_rate': 0,  # Honest clients don't corrupt data
                'total_samples': strategy_metrics['honest']['samples']
            },
            'withholding': {
                'clients': self.client_strategies.count('withholding'),
                'contribution_rate': strategy_metrics['withholding']['samples'] / strategy_metrics['withholding']['total_possible'] if strategy_metrics['withholding']['total_possible'] > 0 else 0,
                'corruption_rate': 0,  # Withholding clients don't corrupt data
                'total_samples': strategy_metrics['withholding']['samples']
            },
            'adversarial': {
                'clients': self.client_strategies.count('adversarial'),
                'contribution_rate': strategy_metrics['adversarial']['samples'] / strategy_metrics['adversarial']['total_possible'] if strategy_metrics['adversarial']['total_possible'] > 0 else 0,
                'corruption_rate': strategy_metrics['adversarial']['corrupted'] / strategy_metrics['adversarial']['samples'] if strategy_metrics['adversarial']['samples'] > 0 else 0,
                'total_samples': strategy_metrics['adversarial']['samples']
            }
        }
        
        # Calculate the overall data quality score (higher is better)
        honest_data_proportion = strategy_metrics['honest']['samples'] / sum(strategy_metrics[s]['samples'] for s in ['honest', 'withholding', 'adversarial']) if sum(strategy_metrics[s]['samples'] for s in ['honest', 'withholding', 'adversarial']) > 0 else 0
        corruption_proportion = strategy_metrics['adversarial']['corrupted'] / sum(strategy_metrics[s]['samples'] for s in ['honest', 'withholding', 'adversarial']) if sum(strategy_metrics[s]['samples'] for s in ['honest', 'withholding', 'adversarial']) > 0 else 0
        self.data_quality_score = honest_data_proportion * (1 - corruption_proportion)
        
        # Combine all data
        combined_data = torch.cat(processed_data, dim=0)
        combined_labels = torch.cat(processed_labels, dim=0)
        
        # Create TensorDataset
        from torch.utils.data import TensorDataset
        combined_dataset = TensorDataset(combined_data, combined_labels)
        
        # Create loaders
        train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        
        return train_loader, test_loader
    
    def get_contribution_summary(self):
        """Return a summary of client contributions for the current epoch."""
        total_clients = len(self.client_contributions)
        total_samples = sum(metric['contributed_samples'] for metric in self.client_contributions)
        corrupted_samples = sum(metric['corrupted_samples'] for metric in self.client_contributions)
        
        summary = {
            'total_clients': total_clients,
            'total_samples': total_samples,
            'corrupted_samples': corrupted_samples,
            'corruption_rate': corrupted_samples / total_samples if total_samples > 0 else 0,
            'honest_clients': self.client_strategies.count('honest'),
            'withholding_clients': self.client_strategies.count('withholding'),
            'adversarial_clients': self.client_strategies.count('adversarial'),
            'strategy_summary': self.strategy_contribution_summary,
            'data_quality_score': self.data_quality_score,
            'client_details': self.client_contributions
        }
        
        return summary

#-------------------------------------------------------------
# PART 2: ML MODEL DEFINITION
#-------------------------------------------------------------

class SimpleModel(nn.Module):
    """A simple CNN model for MNIST classification."""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#-------------------------------------------------------------
# PART 3: INTEGRATION WITH EGT FRAMEWORK
#-------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_scenario_config(scenario: str) -> dict:
    """Load scenario configuration from JSON file."""
    # Use absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "configs")
    
    possible_filenames = [
        f"{scenario}.json",
        f"{scenario}_config.json"
    ]
    
    for filename in possible_filenames:
        config_path = os.path.join(base_path, filename)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config {config_path}: {e}")
                
    raise FileNotFoundError(f"Could not find configuration file for scenario '{scenario}' in {base_path}")

def run_collaborative_ml_experiment(num_clients=10, num_epochs=5, scenario='baseline', experiment_id=None, epoch_callback=None, custom_config=None):
    """
    Run a collaborative machine learning experiment with EGT integration.
    
    Args:
        num_clients: Number of clients in the simulation
        num_epochs: Number of training epochs
        scenario: Name of predefined scenario to use (if custom_config is None)
        experiment_id: Optional identifier for this experiment run
        epoch_callback: Optional callback function after each epoch
        custom_config: Optional custom configuration (overrides scenario)
        
    Returns:
        (simulator, training_history): Tuple with final simulator and training metrics
    """
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = experiment_id or f"exp_{timestamp}"
    output_dir = os.path.join(script_dir, "results", experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Get configuration (custom config or from scenario)
    if custom_config:
        config = custom_config
        print("Using custom configuration")
    else:
        config = get_scenario_config(scenario)
        print(f"Using scenario configuration: {scenario}")
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 1. Create collaborative dataset
    collab_data = CollaborativeDataset(num_clients=num_clients)
    collab_data.load_and_partition_dataset('mnist')
    
    # Get strategy distribution from config if available, otherwise use defaults
    strategy_dist = config.get('strategy_distribution', {'honest': 0.4, 'withholding': 0.3, 'adversarial': 0.3})
    honest_ratio = strategy_dist.get('honest', 0.4)
    withholding_ratio = strategy_dist.get('withholding', 0.3)
    adversarial_ratio = strategy_dist.get('adversarial', 0.3)
    
    collab_data.assign_initial_strategies(
        honest_ratio=honest_ratio,
        withholding_ratio=withholding_ratio,
        adversarial_ratio=adversarial_ratio
    )
    
    # 2. Create model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Create EGT experiment with scenario-based configuration
    print("\n=== Setting up EGT Framework ===")
    experiment_config = config
    
    # Create EGT experiment - register custom mechanisms if needed
    builder = ExperimentBuilder()
    
    # For 'fixed' and 'cyclic' reward mechanisms, we need to register them
    if scenario == 'fixed_rewards' or experiment_config.get('reward_mechanism') == 'fixed':
        builder.registry.register_reward('fixed', FixedReward)
    
    if scenario == 'cyclic_environment' or experiment_config.get('reward_mechanism') == 'cyclic':
        from cyclic_reward import CyclicReward
        builder.registry.register_reward('cyclic', CyclicReward)
    
    # Create the experiment
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
    
    # Create trainer with EGT callback
    trainer = pytorch_integration.create_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )
    
    # 4. Run the collaborative ML training loop
    print("\n=== Starting Collaborative Training with EGT ===")
    # Enhanced history with detailed tracking
    history = {
        'train_loss': [],
        'test_acc': [],
        'honest_ratio': [],
        'withholding_ratio': [],
        'adversarial_ratio': [],
        # Reward and cost tracking
        'honest_reward': [],
        'withholding_reward': [],
        'adversarial_reward': [],
        'honest_cost': [],
        'withholding_cost': [],
        'adversarial_cost': [],
        # Detailed metrics for client contributions
        'data_quality_score': [],
        'total_contributed_samples': [],
        'corrupted_samples': [],
        'overall_corruption_rate': [],
        # Per-strategy metrics
        'honest_contribution_rate': [],
        'withholding_contribution_rate': [],
        'adversarial_contribution_rate': [],
        'adversarial_corruption_rate': [],
        # Detailed client-level data for each epoch
        'client_contributions': [],
        # Strategy distribution history for plotting
        'strategy_distribution': []
    }
    
    # Store initial distribution
    initial_dist = list(simulator.get_current_distribution().values())
    history['strategy_distribution'].append(initial_dist)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Get current strategy distribution from simulator
        strategy_dist = simulator.get_current_distribution()
        print(f"Strategy distribution: {strategy_dist}")
        
        # Update client strategies based on EGT simulation
        collab_data.update_strategies(strategy_dist)
        
        # Create combined dataset with updated strategies
        train_loader, test_loader = collab_data.create_combined_dataset()
        
        # Get contribution summary before training
        contribution_summary = collab_data.get_contribution_summary()
        
        # Store detailed client contributions for this epoch
        history['client_contributions'].append({
            'epoch': epoch + 1,
            'summary': contribution_summary,
            'client_details': collab_data.client_contributions
        })
        
        # Store aggregated metrics
        history['data_quality_score'].append(contribution_summary['data_quality_score'])
        history['total_contributed_samples'].append(contribution_summary['total_samples'])
        history['corrupted_samples'].append(contribution_summary['corrupted_samples'])
        history['overall_corruption_rate'].append(contribution_summary['corruption_rate'])
        
        # Store per-strategy metrics
        history['honest_contribution_rate'].append(
            contribution_summary['strategy_summary']['honest']['contribution_rate'])
        history['withholding_contribution_rate'].append(
            contribution_summary['strategy_summary']['withholding']['contribution_rate'])
        history['adversarial_contribution_rate'].append(
            contribution_summary['strategy_summary']['adversarial']['contribution_rate'])
        history['adversarial_corruption_rate'].append(
            contribution_summary['strategy_summary']['adversarial']['corruption_rate'])
        
        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, test_loader)
        
        # Get current payoffs from simulator to track reward/cost evolution
        try:
            # First try to get payoffs through the simulator's public methods
            if hasattr(simulator, 'get_payoffs'):
                payoffs = simulator.get_payoffs()
            elif hasattr(simulator, 'payoff_matrix'):
                payoffs = simulator.payoff_matrix
            else:
                # If there's no direct way to get the payoff matrix, calculate them manually
                # Import numpy inside the function to ensure it's available in this scope
                import numpy as np
                strategies = np.array([0, 1, 2])  # honest, withholding, adversarial
                rewards = integration.reward_mechanism.compute(strategies)
                costs = integration.cost_mechanism.compute(strategies) 
                punishments = integration.punishment_mechanism.compute(strategies)
                
                # Calculate net payoffs (reward - cost - punishment)
                payoffs = np.zeros((3, 3))
                for i in range(3):
                    payoffs[i, i] = rewards[i] - costs[i] - punishments[i]
            
            # Extract rewards and costs if available
            history['honest_reward'].append(float(payoffs[0, 0]))
            history['withholding_reward'].append(float(payoffs[1, 1]))
            history['adversarial_reward'].append(float(payoffs[2, 2]))
            
            # Get costs (this is implementation specific, so we'll try to approximate)
            reward_mech = integration.reward_mechanism
            cost_mech = integration.cost_mechanism
            
            # This would be more accurate if we had direct access to the cost values
            import numpy as np
            strategies = np.array([0, 1, 2])  # honest, withholding, adversarial
            
            try:
                costs = cost_mech.compute(strategies)
                history['honest_cost'].append(float(costs[0]))
                history['withholding_cost'].append(float(costs[1]))
                history['adversarial_cost'].append(float(costs[2]))
            except Exception as e:
                print(f"Warning: Could not access costs - {e}")
                history['honest_cost'].append(0.0)
                history['withholding_cost'].append(0.0)
                history['adversarial_cost'].append(0.0)
                
        except Exception as e:
            print(f"Warning: Could not access payoffs - {e}")
            # Add placeholder values
            history['honest_reward'].append(0.0)
            history['withholding_reward'].append(0.0)
            history['adversarial_reward'].append(0.0)
            history['honest_cost'].append(0.0)
            history['withholding_cost'].append(0.0)
            history['adversarial_cost'].append(0.0)
        
        # Store metrics from training - handle different possible key names
        # The PyTorchTrainer might use different key names for loss and accuracy
        train_loss = None
        test_acc = None
        
        # Check possible keys for train loss
        for loss_key in ['train_loss', 'loss', 'training_loss']:
            if loss_key in metrics:
                train_loss = metrics[loss_key]
                break
        
        # Check possible keys for test accuracy
        for acc_key in ['test_acc', 'val_accuracy', 'accuracy', 'test_accuracy', 'val_acc']:
            if acc_key in metrics:
                test_acc = metrics[acc_key]
                break
        
        # If keys not found, use default values and log a warning
        if train_loss is None:
            print(f"Warning: Could not find training loss in metrics dictionary. Keys available: {list(metrics.keys())}")
            train_loss = 0.0
        
        if test_acc is None:
            print(f"Warning: Could not find test accuracy in metrics dictionary. Keys available: {list(metrics.keys())}")
            test_acc = 0.0
        
        # Append values to history
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        # Store strategy distributions and update history
        dist_values = list(strategy_dist.values())
        history['honest_ratio'].append(dist_values[0])
        history['withholding_ratio'].append(dist_values[1])
        history['adversarial_ratio'].append(dist_values[2])
        history['strategy_distribution'].append(dist_values)
        
        # Call the epoch callback if provided
        if epoch_callback is not None:
            try:
                # Create a copy of the current state to send to the callback
                current_state = {
                    'epoch': epoch + 1,
                    'simulator': simulator,  # Pass the simulator object directly
                    'strategy_distribution': dist_values,
                    'train_loss': train_loss,
                    'test_acc': test_acc,
                    'honest_ratio': dist_values[0],
                    'withholding_ratio': dist_values[1],
                    'adversarial_ratio': dist_values[2],
                    'honest_reward': history['honest_reward'][-1] if history['honest_reward'] else 0.0,
                    'withholding_reward': history['withholding_reward'][-1] if history['withholding_reward'] else 0.0,
                    'adversarial_reward': history['adversarial_reward'][-1] if history['adversarial_reward'] else 0.0,
                    'honest_cost': history['honest_cost'][-1] if history['honest_cost'] else 0.0,
                    'withholding_cost': history['withholding_cost'][-1] if history['withholding_cost'] else 0.0,
                    'adversarial_cost': history['adversarial_cost'][-1] if history['adversarial_cost'] else 0.0
                }
                # Call the callback with the current state
                epoch_callback(current_state)
            except Exception as e:
                print(f"Warning: Error calling epoch callback - {e}")
        
        # Update the history with simulator history data
        simulator_history = simulator.get_history()
        for key, values in simulator_history.items():
            if key not in history:
                history[key] = values
                
        # Save the current history state after each epoch
        with open(os.path.join(output_dir, f"history_epoch_{epoch+1}.json"), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, list) and len(value) > 0:
                    if hasattr(value[0], 'tolist'):  # Check if it's a numpy array
                        serializable_history[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                    else:
                        serializable_history[key] = value
                else:
                    serializable_history[key] = value
            
            json.dump(serializable_history, f, indent=2)
    
    # Save final history to JSON file
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list) and len(value) > 0:
                if hasattr(value[0], 'tolist'):  # Check if it's a numpy array
                    serializable_history[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
                else:
                    serializable_history[key] = value
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=2)
    
    # Create visualizer
    visualizer = EGTVisualizer(simulator)
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Test Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'test_accuracy.png'))
    plt.close()
    
    # Generate EGT-related plots
    try:
        fig_strategy = visualizer.plot_strategy_evolution()
        fig_strategy.savefig(os.path.join(output_dir, 'strategy_evolution.png'))
        plt.close(fig_strategy)
        
        fig_convergence = visualizer.plot_convergence_analysis()
        fig_convergence.savefig(os.path.join(output_dir, 'convergence_analysis.png'))
        plt.close(fig_convergence)
        
        fig_ternary = visualizer.plot_ternary_trajectories()
        fig_ternary.savefig(os.path.join(output_dir, 'ternary_trajectories.png'))
        plt.close(fig_ternary)
        
        fig_payoff = visualizer.plot_payoff_analysis()
        fig_payoff.savefig(os.path.join(output_dir, 'payoff_analysis.png'))
        plt.close(fig_payoff)
    except Exception as e:
        print(f"Warning: Could not generate EGT plots - {e}")
    
    # Generate data quality plots
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, history['data_quality_score'], label='Data Quality Score')
    plt.title('Data Quality Score vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Quality Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'data_quality.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['overall_corruption_rate'], label='Corruption Rate')
    plt.title('Data Corruption Rate vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Corruption Rate')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'corruption_rate.png'))
    plt.close()
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Final Data Quality Score: {history['data_quality_score'][-1]:.4f}")
    print(f"Final Strategy Distribution:")
    print(f"  - Honest: {history['honest_ratio'][-1]:.4f}")
    print(f"  - Withholding: {history['withholding_ratio'][-1]:.4f}")
    print(f"  - Adversarial: {history['adversarial_ratio'][-1]:.4f}")
    
    # Return the simulator and history
    return simulator, history

def run_experiment_wrapper(args):
    """Wrapper function for running experiments in parallel"""
    if len(args) == 4:
        scenario, num_clients, num_epochs, experiment_id = args
        custom_config = None
    elif len(args) == 5:
        scenario, num_clients, num_epochs, experiment_id, custom_config = args
    else:
        print(f"Error: Invalid argument format {args}")
        return False
        
    try:
        print(f"Starting experiment: {scenario if not custom_config else 'custom'} (ID: {experiment_id})")
        run_collaborative_ml_experiment(
            num_clients=num_clients,
            num_epochs=num_epochs,
            scenario=scenario,
            experiment_id=experiment_id,
            custom_config=custom_config
        )
        print(f"Completed experiment: {scenario if not custom_config else 'custom'} (ID: {experiment_id})")
        return True
    except Exception as e:
        print(f"Error in experiment {scenario if not custom_config else 'custom'} (ID: {experiment_id}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run collaborative ML experiments with EGT framework")
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--scenario', type=str, default='baseline',
                      help='Experiment scenario to run (default: baseline)')
    parser.add_argument('--experiment_id', type=str, help='Optional experiment identifier')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per scenario')
    
    args = parser.parse_args()
    
    # Import experiment_scenarios to get available scenarios
    import experiment_scenarios
    available_scenarios = [s["id"] for s in experiment_scenarios.SCENARIOS]
    
    if args.scenario not in available_scenarios:
        print(f"Error: Unknown scenario '{args.scenario}'")
        print("Available scenarios:")
        for s in available_scenarios:
            print(f"  - {s}")
        return
    
    if args.parallel:
        # Run experiments in parallel
        scenarios = [args.scenario] if args.scenario != 'all' else available_scenarios
        run_parallel_experiments(scenarios, args.num_clients, args.num_epochs, args.runs)
    else:
        # Run single experiment
        run_collaborative_ml_experiment(
            num_clients=args.num_clients,
            num_epochs=args.num_epochs,
            scenario=args.scenario,
            experiment_id=args.experiment_id
        )

if __name__ == "__main__":
    main() 