"""
Punishment mechanism implementations for the EGT Framework.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from core.mechanisms import PunishmentMechanism


class AdversarialPunishment(PunishmentMechanism):
    """
    Punishment mechanism that penalizes adversarial behavior.
    Detects and penalizes potentially harmful contributions.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.penalty_strength = self.config.get('penalty_strength', 1.0)
        self.detection_threshold = self.config.get('detection_threshold', 0.7)
        self.gradient_variance_weight = self.config.get('gradient_variance_weight', 0.5)
        self.current_penalty = 0.0
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update penalties based on training anomalies."""
        # Metrics that might indicate adversarial behavior
        grad_var = metrics.get('gradient_variance', 0.0)
        loss_increase = metrics.get('loss_increase', 0.0)
        accuracy_drop = metrics.get('accuracy_drop', 0.0)
        
        # Combined adversarial detection score
        adversarial_score = (
            self.gradient_variance_weight * grad_var + 
            (1 - self.gradient_variance_weight) * max(loss_increase, accuracy_drop)
        )
        
        # Update current penalty level
        old_penalty = self.current_penalty
        if adversarial_score > self.detection_threshold:
            # Increase penalty when adversarial behavior detected
            self.current_penalty = min(
                self.penalty_strength * 2.0,
                self.current_penalty + 0.2 * self.penalty_strength
            )
        else:
            # Gradually decrease penalty otherwise
            self.current_penalty = max(0.0, self.current_penalty - 0.1 * self.penalty_strength)
        
        self.log_state({
            'old_penalty': old_penalty,
            'new_penalty': self.current_penalty,
            'adversarial_score': adversarial_score,
            'grad_var': grad_var,
            'loss_increase': loss_increase,
            'accuracy_drop': accuracy_drop
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute penalties based on client strategies and contributions."""
        penalties = np.zeros(len(strategies))
        
        # Apply penalties to adversarial strategies, reduced penalties to withholding
        for i, strategy in enumerate(strategies):
            if strategy == 2:  # Adversarial
                penalties[i] = self.current_penalty
            elif strategy == 1:  # Withholding
                penalties[i] = self.current_penalty * 0.3
        
        # If contributions are provided, increase penalties for negative contributions
        if contributions is not None:
            for i, contrib in enumerate(contributions):
                if contrib < 0:  # Negative contribution
                    penalties[i] += abs(contrib) * self.penalty_strength
        
        return penalties


class ReputationBasedPunishment(PunishmentMechanism):
    """
    Punishment mechanism based on client reputation and history.
    Maintains reputation scores and applies penalties accordingly.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.memory_factor = self.config.get('memory_factor', 0.8)
        self.reputation_threshold = self.config.get('reputation_threshold', 0.5)
        self.max_penalty = self.config.get('max_penalty', 20.0)
        
        # Initialize reputation tracker (client_id -> reputation_score)
        self.reputation_scores = {}
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update reputation scores based on client behavior."""
        # Extract client contributions and IDs
        if 'client_contributions' in metrics and 'client_ids' in metrics:
            contributions = metrics['client_contributions']
            client_ids = metrics['client_ids']
            
            for i, client_id in enumerate(client_ids):
                if i < len(contributions):
                    contrib = contributions[i]
                    
                    # Initialize reputation if new client
                    if client_id not in self.reputation_scores:
                        self.reputation_scores[client_id] = 0.7  # Start with moderate reputation
                    
                    # Update reputation based on contribution quality
                    # Positive contributions improve reputation, negative ones decrease it
                    old_rep = self.reputation_scores[client_id]
                    
                    # Normalized contribution impact on reputation
                    rep_impact = min(0.3, max(-0.3, contrib * 0.2))
                    
                    # Update with memory factor (previous reputation has more weight)
                    self.reputation_scores[client_id] = (
                        self.memory_factor * old_rep + 
                        (1 - self.memory_factor) * (old_rep + rep_impact)
                    )
                    
                    # Keep within [0, 1] range
                    self.reputation_scores[client_id] = min(1.0, max(0.0, self.reputation_scores[client_id]))
        
        # Log summary of reputation state
        self.log_state({
            'avg_reputation': np.mean(list(self.reputation_scores.values())) if self.reputation_scores else 0.0,
            'min_reputation': min(self.reputation_scores.values()) if self.reputation_scores else 0.0,
            'max_reputation': max(self.reputation_scores.values()) if self.reputation_scores else 0.0,
            'num_clients': len(self.reputation_scores)
        })
    
    def compute(self, strategies: np.ndarray, contributions: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute penalties based on client reputations and strategies."""
        penalties = np.zeros(len(strategies))
        
        # If client_ids are provided, use reputation scores
        if hasattr(self, 'current_client_ids') and len(self.reputation_scores) > 0:
            for i, client_id in enumerate(self.current_client_ids):
                if i < len(penalties) and client_id in self.reputation_scores:
                    # Calculate penalty based on reputation
                    reputation = self.reputation_scores[client_id]
                    if reputation < self.reputation_threshold:
                        # Low reputation leads to higher penalties
                        rep_factor = (self.reputation_threshold - reputation) / self.reputation_threshold
                        penalties[i] = self.max_penalty * rep_factor
        else:
            # Fallback to strategy-based penalties if no reputation data
            for i, strategy in enumerate(strategies):
                if strategy == 2:  # Adversarial
                    penalties[i] = self.max_penalty * 0.8
                elif strategy == 1:  # Withholding
                    penalties[i] = self.max_penalty * 0.3
        
        return penalties
    
    def set_client_ids(self, client_ids: List[str]) -> None:
        """Set current client IDs for reputation tracking."""
        self.current_client_ids = client_ids
