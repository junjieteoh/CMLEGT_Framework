"""
Integration of EGT simulation with machine learning training.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Callable, Optional, Any

from core.mechanisms import RewardMechanism, CostMechanism, PunishmentMechanism
from simulation.simulator import EGTSimulator

# Configure logging
logger = logging.getLogger('egt_framework.ml')


class MLIntegration:
    """
    Interface for integrating ML training with EGT simulation.
    """
    
    def __init__(self, 
                simulator: EGTSimulator,
                reward_mechanism: RewardMechanism,
                cost_mechanism: CostMechanism,
                punishment_mechanism: Optional[PunishmentMechanism] = None,
                update_frequency: int = 1,
                config: Dict[str, Any] = None):
        """
        Initialize ML integration.
        
        Args:
            simulator: EGT simulator instance
            reward_mechanism: Reward mechanism to use
            cost_mechanism: Cost mechanism to use
            punishment_mechanism: Optional punishment mechanism
            update_frequency: How often to update the simulator (epochs)
            config: Additional configuration
        """
        self.simulator = simulator
        self.reward_mechanism = reward_mechanism
        self.cost_mechanism = cost_mechanism
        self.punishment_mechanism = punishment_mechanism
        self.update_frequency = update_frequency
        self.config = config or {}
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.metrics_history = []
    
    def create_callback(self, model=None) -> Callable:
        """
        Create a callback function for ML frameworks.
        
        Args:
            model: Optional model reference
            
        Returns:
            Callback function that updates the simulation
        """
        def callback(metrics: Dict[str, Any]) -> None:
            """
            Update EGT simulation based on training metrics.
            
            Args:
                metrics: Dictionary of training metrics
            """
            # Enhance metrics with deltas
            enhanced_metrics = self._enhance_metrics(metrics)
            
            # Only update at specified frequency
            if self.current_epoch % self.update_frequency == 0:
                # Update the simulator
                update_result = self.simulator.update(
                    self.reward_mechanism,
                    self.cost_mechanism,
                    self.punishment_mechanism,
                    enhanced_metrics
                )
                
                # Log update
                logger.info(f"Updated simulation at epoch {self.current_epoch}")
                logger.info(f"New distribution: {self.simulator.get_current_distribution()}")
            
            # Store metrics
            self.metrics_history.append(enhanced_metrics)
            
            # Update training state
            self.current_epoch += 1
            if 'accuracy' in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics['accuracy'])
        
        return callback
    
    def _enhance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance training metrics with additional derived metrics.
        
        Args:
            metrics: Raw training metrics
            
        Returns:
            Enhanced metrics dictionary
        """
        enhanced = metrics.copy()
        enhanced['epoch'] = self.current_epoch
        
        # Add deltas if we have history
        if self.metrics_history:
            prev_metrics = self.metrics_history[-1]
            
            if 'accuracy' in metrics and 'accuracy' in prev_metrics:
                enhanced['accuracy_delta'] = metrics['accuracy'] - prev_metrics['accuracy']
                enhanced['accuracy_drop'] = max(0, -enhanced['accuracy_delta'])
            
            if 'loss' in metrics and 'loss' in prev_metrics:
                enhanced['loss_delta'] = metrics['loss'] - prev_metrics['loss']
                enhanced['loss_increase'] = max(0, enhanced['loss_delta'])
        
        return enhanced
    
    def simulate_metrics(self, 
                        base_accuracy: float = 0.5,
                        max_accuracy: float = 0.95,
                        noise_level: float = 0.02) -> Dict[str, float]:
        """
        Simulate training metrics based on client strategies.
        
        Args:
            base_accuracy: Starting accuracy level
            max_accuracy: Maximum achievable accuracy
            noise_level: Random noise in metrics
            
        Returns:
            Dictionary of simulated metrics
        """
        # Get current strategy distribution
        dist = self.simulator.strategy_distribution
        
        # Honest contributions improve accuracy
        honest_ratio = dist[0]
        withholding_ratio = dist[1]
        adversarial_ratio = dist[2]
        
        # Calculate performance impact
        honest_impact = honest_ratio * 0.5  # Positive impact
        withholding_impact = withholding_ratio * 0.1  # Small positive impact
        adversarial_impact = adversarial_ratio * -0.3  # Negative impact
        
        # Combined impact on accuracy
        impact = honest_impact + withholding_impact + adversarial_impact
        
        # Add learning curve effect
        progress_factor = min(1.0, self.current_epoch / 50.0)  # Saturate after 50 epochs
        
        # Calculate current accuracy
        target_accuracy = base_accuracy + progress_factor * (max_accuracy - base_accuracy)
        current_accuracy = target_accuracy * (1.0 + impact)
        current_accuracy = max(0.1, min(0.99, current_accuracy))
        
        # Add noise
        if noise_level > 0:
            current_accuracy += np.random.normal(0, noise_level)
            current_accuracy = max(0.1, min(0.99, current_accuracy))
        
        # Calculate loss (inversely related to accuracy)
        current_loss = 0.5 * (1.0 - current_accuracy) + 0.1
        
        # Estimate gradient variance (higher with more adversaries)
        gradient_variance = 0.1 + 0.5 * adversarial_ratio
        
        metrics = {
            'accuracy': current_accuracy,
            'loss': current_loss,
            'gradient_variance': gradient_variance
        }
        
        return metrics
    
    def run_simulation(self, num_epochs: int = 100) -> Dict[str, Any]:
        """
        Run a pure simulation without real ML training.
        
        Args:
            num_epochs: Number of epochs to simulate
            
        Returns:
            Dictionary with simulation results
        """
        # Reset state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.metrics_history = []
        
        for epoch in range(num_epochs):
            # Simulate training metrics
            metrics = self.simulate_metrics()
            
            # Update simulation
            callback = self.create_callback()
            callback(metrics)
            
            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                dist = self.simulator.get_current_distribution()
                logger.info(f"Epoch {epoch}/{num_epochs}: acc={metrics['accuracy']:.4f}, "
                           f"dist={dist}")
        
        # Analyze convergence
        convergence = self.simulator.analyze_convergence()
        
        return {
            'history': self.simulator.get_history(),
            'metrics_history': self.metrics_history,
            'final_distribution': self.simulator.get_current_distribution(),
            'best_accuracy': self.best_accuracy,
            'convergence': convergence
        }
