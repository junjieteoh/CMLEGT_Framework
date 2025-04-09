# EGT Framework Examples

This directory contains example scripts demonstrating how to use the EGT Framework for different scenarios.

## Available Examples

### 1. Basic EGT Simulation (`01_basic_egt_simulation.py`)
A simple example showing how to:
- Set up a basic EGT simulation
- Configure reward, cost, and punishment mechanisms
- Run the simulation and visualize results

```bash
# From the src directory
python -m examples.01_basic_egt_simulation
```

### 2. Simple ML Integration (`02_simple_ml_integration.py`)
Demonstrates how to:
- Integrate a PyTorch ML model with the EGT framework
- Train the model while simulating strategic behavior
- Visualize both ML and EGT results

```bash
# From the src directory
python -m examples.02_simple_ml_integration
```

### 3. Full Collaborative ML (`03_collaborative_ml.py`)
A comprehensive example showing:
- Setting up a collaborative ML environment with multiple participants
- Simulating different participant strategies (honest, withholding, adversarial)
- Integrating EGT framework to model strategy evolution
- Analyzing and visualizing both ML and strategic behavior

```bash
# From the src directory
python -m examples.03_collaborative_ml
```

## Output and Visualizations

Each example script creates its own timestamped output directory containing:
- Strategy evolution plots
- Ternary trajectory plots
- Convergence analysis
- Payoff analysis
- Model performance metrics (for ML examples)
- JSON files with detailed results

## Dependencies

Make sure you have the following dependencies installed:
- PyTorch
- NumPy
- Matplotlib
- torchvision (for ML examples)

## Module Structure

The examples use the refactored module structure:
- `egt_framework.mechanisms` - Contains reward, cost, and punishment mechanisms
- `egt_framework.registry` - Contains the mechanisms registry and experiment builder
- `egt_framework.simulation` - Contains the EGT simulator
- `egt_framework.visualization` - Contains visualization tools
- `egt_framework.ml` - Contains ML integration tools
- `egt_framework.utils` - Contains helper utilities

## Configuration

Each example includes detailed configuration options that you can modify:
- Number of participants/clients
- Reward mechanisms and parameters
- Cost functions
- Punishment mechanisms
- Initial strategy distributions
- ML model architecture (for ML examples)

## Next Steps

After understanding these examples, you can:
1. Modify the configurations to test different scenarios
2. Create your own reward/cost/punishment mechanisms
3. Integrate different ML models or tasks
4. Extend the framework for your specific use case 