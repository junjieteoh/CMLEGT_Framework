"""
PyTorch-specific integration for the EGT framework.
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Any

from .integration import MLIntegration


class PyTorchIntegration(MLIntegration):
    """
    Integration with PyTorch training.
    """
    
    def create_trainer(self, 
                      model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable,
                      device: str = 'cpu') -> 'PyTorchTrainer':
        """
        Create a PyTorch trainer with EGT integration.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            device: Device to use ('cpu' or 'cuda')
            
        Returns:
            PyTorch trainer with EGT integration
        """
        return PyTorchTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            egt_callback=self.create_callback(model)
        )


class PyTorchTrainer:
    """
    PyTorch training wrapper with EGT integration.
    """
    
    def __init__(self, 
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: Callable,
                device: str = 'cpu',
                egt_callback: Optional[Callable] = None):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            device: Device to use ('cpu' or 'cuda')
            egt_callback: Callback function for EGT updates
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.egt_callback = egt_callback
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """
        Train for one epoch and update EGT simulation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Training metrics
        total_loss = 0
        correct = 0
        total = 0
        batch_losses = []
        
        # Train loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Calculate accuracy
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += len(data)
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        gradient_variance = np.var(batch_losses) if len(batch_losses) > 1 else 0
        
        # Validation if provided
        val_loss = None
        val_accuracy = None
        
        if val_loader:
            self.model.eval()
            val_total_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    
                    # Track metrics
                    val_total_loss += loss.item()
                    
                    # Calculate accuracy
                    preds = output.argmax(dim=1, keepdim=True)
                    val_correct += preds.eq(target.view_as(preds)).sum().item()
                    val_total += len(data)
            
            val_loss = val_total_loss / len(val_loader)
            val_accuracy = val_correct / val_total
        
        # Prepare metrics for EGT
        metrics = {
            'accuracy': train_accuracy,
            'loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'gradient_variance': gradient_variance
        }
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_accuracy is not None:
            self.history['val_accuracy'].append(val_accuracy)
        
        # Call EGT callback
        if self.egt_callback:
            self.egt_callback(metrics)
        
        return metrics
    
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             num_epochs: int = 10,
             verbose: bool = True) -> Dict[str, Any]:
        """
        Run training for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        for epoch in range(num_epochs):
            # Train one epoch
            metrics = self.train_epoch(train_loader, val_loader)
            
            # Print progress
            if verbose:
                val_str = f", val_acc={metrics['val_accuracy']:.4f}" if metrics['val_accuracy'] is not None else ""
                print(f"Epoch {epoch+1}/{num_epochs}: "
                     f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}{val_str}")
        
        return {
            'history': self.history,
            'final_metrics': metrics
        }
