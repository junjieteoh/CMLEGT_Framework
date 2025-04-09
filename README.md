# EGT Framework - Collaborative Machine Learning with Evolutionary Game Theory

A framework for studying strategic interactions in collaborative machine learning systems using Evolutionary Game Theory (EGT).

## Requirements

- **Python 3.11** (required, will not work with Python 3.13+)
- **NumPy < 2.0** (required, PyTorch is not compatible with NumPy 2.x)
- PyTorch 2.x
- Other dependencies listed in requirements.txt

## Overview

This project provides tools to model, simulate, and analyze the behavior of participants in collaborative machine learning settings such as federated learning. It uses evolutionary game theory to understand how different incentive mechanisms affect participants' strategies over time.

## Framework Features

- **EGT Simulator**: Core simulation engine for evolutionary game dynamics
- **Mechanism Design**: Configurable reward, cost, and punishment mechanisms
- **ML Integration**: Interface with real ML training processes (PyTorch support)
- **Visualization**: Tools for analyzing and visualizing simulation results
- **Experiment Builder**: Easy configuration and execution of experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/egt-framework.git
cd egt-framework

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from registry.experiment import ExperimentBuilder
from core.simulator import EGTSimulator

# Create experiment builder
builder = ExperimentBuilder()

# Configure experiment
experiment_config = {
    'name': 'Basic EGT Simulation',
    'num_clients': 100,  
    'reward_mechanism': 'adaptive',
    'reward_config': {
        'initial_pool': 1000.0,  
        'learning_rate': 0.1,    
        'accuracy_threshold': 0.7, 
    },
    'cost_mechanism': 'computational',
    'punishment_mechanism': 'adversarial',
    'initial_distribution': [0.33, 0.33, 0.34],  # [honest, withholding, adversarial]
}

# Run simulation
simulator, integration = builder.create_experiment(experiment_config)
results = integration.run_simulation(num_epochs=50)

# Access results
print(f"Final distribution: {simulator.get_current_distribution()}")
print(f"Converged: {simulator.analyze_convergence()}")
```

## Using the Makefile

The repository includes a Makefile with useful commands:

```bash
# Install dependencies
make install

# Run examples
make run-basic     # Basic EGT simulation
make run-ml        # ML integration example
make run-collab    # Collaborative ML example

# Run multi-distribution experiments
make run-multi

# Run a specific scenario with custom parameters
make run-scenario SCENARIO=balanced_incentives POINTS=50 EPOCHS=10 PROCS=4

# Clean up temporary files
make clean
```

## Directory Structure

```
egt-framework/
├── core/             # Core abstractions
├── mechanisms/       # Mechanism implementations
├── simulation/       # Simulation engine
├── ml/               # ML integration
├── visualization/    # Visualization tools
├── registry/         # Registry and experiment builder
├── utils/            # Utilities
├── examples/         # Example scripts
│   ├── 01_basic_egt_simulation.py
│   ├── 02_simple_ml_integration.py
│   ├── 03_collaborative_ml.py
│   ├── multi_starting_distribution.py
│   └── configs/      # Configuration files for different scenarios
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 