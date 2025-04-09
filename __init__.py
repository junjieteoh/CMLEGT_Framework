"""
EGT Framework for Collaborative Machine Learning
================================================

A flexible, modular system for simulating evolutionary game dynamics
with integration to real machine learning training processes.
"""

# Version information
__version__ = '0.1.0'

# For backward compatibility
# This allows old code using 'from src.egt_framework.xxx import yyy' to still work
import sys
import importlib.util
import os
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

# Use relative imports for our own modules
from .core.mechanisms import (
    DynamicMechanism,
    RewardMechanism,
    CostMechanism,
    PunishmentMechanism
)

from .mechanisms.reward import (
    AdaptiveReward,
    PerformanceBasedReward, 
    ShapleyValueReward
)
from .mechanisms.cost import (
    ComputationalCost,
    PrivacyCost
)
from .mechanisms.punishment import (
    AdversarialPunishment,
    ReputationBasedPunishment
)

# Import simulator
from .simulation.simulator import EGTSimulator

# Import registry components
from .registry.registry import MechanismRegistry
from .registry.experiment import ExperimentBuilder

# Import ML components
from .ml.integration import MLIntegration
from .ml.pytorch import PyTorchIntegration, PyTorchTrainer

# Import visualization
from .visualization.visualizer import EGTVisualizer

class SrcCompatibilityFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('src.egt_framework'):
            # Convert src.egt_framework.xxx to egt_framework.xxx
            new_name = fullname.replace('src.', '', 1)
            try:
                # Try to find the module relative to the main package
                if new_name == 'egt_framework':
                    # Handle the main package import
                    parent_dir = os.path.dirname(os.path.dirname(__file__))
                    spec = ModuleSpec(
                        name=fullname,
                        loader=None,
                        origin=__file__,
                        is_package=True
                    )
                    return spec
                elif new_name.startswith('egt_framework.'):
                    # Handle submodule imports
                    submodule = new_name.split('egt_framework.')[1]
                    parts = submodule.split('.')
                    path = os.path.join(os.path.dirname(__file__), *parts)
                    
                    if os.path.isdir(path):
                        # It's a package
                        path = os.path.join(path, '__init__.py')
                        if os.path.exists(path):
                            return importlib.util.spec_from_file_location(
                                fullname, path, submodule_search_locations=[os.path.dirname(path)]
                            )
                    else:
                        # It's a module
                        path = path + '.py'
                        if os.path.exists(path):
                            return importlib.util.spec_from_file_location(fullname, path)
            except (ImportError, AttributeError, TypeError) as e:
                print(f"Error handling import {fullname}: {e}")
        return None

# Install the import hook
sys.meta_path.insert(0, SrcCompatibilityFinder()) 