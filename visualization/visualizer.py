"""
Visualization tools for EGT simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, List, Optional, Any
import os
import json

from simulation.simulator import EGTSimulator


class EGTVisualizer:
    """
    Visualization tools for EGT simulation results.
    """
    
    def __init__(self, simulator: EGTSimulator):
        """
        Initialize visualizer.
        
        Args:
            simulator: EGT simulator instance
        """
        self.simulator = simulator
        self.history = simulator.get_history()
        
        # Setup default configuration
        self.config = {
            'colors': ['green', 'orange', 'red'],
            'figsize': (12, 8),
            'dpi': 100,
            'ternary_figsize': (10, 8),
            'trajectory_color': 'blue',
            'highlight_points': True
        }
    
    def plot_strategy_evolution(self, 
                               title: str = "Strategy Evolution Over Time",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the evolution of strategy distributions over time.
        
        Args:
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        distributions = np.array(self.history['strategy_distribution'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Plot each strategy
        for i, strategy in enumerate(self.simulator.strategy_labels):
            color = self.config['colors'][i % len(self.config['colors'])]
            ax.plot(distributions[:, i], label=strategy, color=color, linewidth=2)
        
        # Add labels and legend
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Proportion')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'])
        
        return fig
    
    def plot_ternary_trajectories(self, 
                                 title: str = "Strategy Evolution Trajectories",
                                 save_path: Optional[str] = None,
                                 start_points: Optional[List[np.ndarray]] = None) -> plt.Figure:
        """
        Plot strategy evolution trajectories on a ternary diagram.
        
        Args:
            title: Plot title
            save_path: Optional path to save the figure
            start_points: Optional list of starting points to plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import ternary
        except ImportError:
            print("Please install the python-ternary package: pip install python-ternary")
            return None
        
        # Get strategy history
        distributions = np.array(self.history['strategy_distribution'])
        
        # Create ternary plot
        fig, tax = ternary.figure(scale=1.0)
        fig.set_size_inches(self.config['ternary_figsize'])
        
        # Configure the plot
        tax.boundary(linewidth=2.0)
        tax.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        
        # Set labels
        tax.left_axis_label(self.simulator.strategy_labels[2])  # adversarial
        tax.right_axis_label(self.simulator.strategy_labels[1])  # withholding
        tax.bottom_axis_label(self.simulator.strategy_labels[0])  # honest
        
        # Prepare trajectory data
        trajectory = [(d[0], d[1], d[2]) for d in distributions]
        
        # Plot trajectories
        tax.plot(trajectory, linewidth=1.5, color=self.config['trajectory_color'], alpha=0.7)
        
        # Highlight start and end points
        if self.config['highlight_points']:
            tax.scatter([trajectory[0]], marker='o', color='green', s=60, label='Start')
            tax.scatter([trajectory[-1]], marker='s', color='red', s=60, label='End')
        
        # Plot additional starting points if provided
        if start_points:
            points = [(p[0], p[1], p[2]) for p in start_points]
            tax.scatter(points, marker='^', color='purple', s=40, label='Starting Points')
        
        # Add title and legend
        plt.title(title)
        tax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'])
        
        return fig
    
    def plot_convergence_analysis(self, title: str = "Convergence Analysis",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot analysis of convergence properties.
        
        Args:
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        distributions = np.array(self.history['strategy_distribution'])
        
        # Calculate step-by-step changes
        changes = []
        for i in range(1, len(distributions)):
            change = np.max(np.abs(distributions[i] - distributions[i-1]))
            changes.append(change)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Plot strategy distributions
        for i, strategy in enumerate(self.simulator.strategy_labels):
            color = self.config['colors'][i % len(self.config['colors'])]
            ax1.plot(distributions[:, i], label=strategy, color=color, linewidth=2)
        
        # Add labels and legend for first subplot
        ax1.set_ylabel('Proportion')
        ax1.set_title(f"{title} - Strategy Evolution")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1)
        
        # Plot changes over time
        ax2.plot(changes, label='Distribution Change', color='blue', linewidth=2)
        
        # Add convergence threshold line
        ax2.axhline(y=0.01, color='red', linestyle='--', label='Convergence Threshold')
        
        # Add labels and legend for second subplot
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Max Change')
        ax2.set_title('Strategy Distribution Change per Step')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, max(max(changes) * 1.1, 0.05))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'])
        
        return fig
    
    def plot_3d_convergence(self, 
                           title: str = "3D Strategy Space Trajectories",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 3D plot of strategy space similar to the reference image.
        
        Args:
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        distributions = np.array(self.history['strategy_distribution'])
        
        # Create figure
        fig = plt.figure(figsize=self.config['ternary_figsize'], dpi=self.config['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(distributions[:, 0], distributions[:, 1], distributions[:, 2], 
               color=self.config['trajectory_color'], linewidth=2)
        
        # Highlight start and end points
        if self.config['highlight_points']:
            ax.scatter(distributions[0, 0], distributions[0, 1], distributions[0, 2],
                      color='green', s=100, label='Start')
            ax.scatter(distributions[-1, 0], distributions[-1, 1], distributions[-1, 2],
                      color='red', s=100, label='End')
        
        # Draw simplex (triangle)
        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='black', s=50)
        
        # Plot edges
        for i in range(3):
            j = (i + 1) % 3
            ax.plot([vertices[i, 0], vertices[j, 0]],
                   [vertices[i, 1], vertices[j, 1]],
                   [vertices[i, 2], vertices[j, 2]], 'k-', lw=1)
        
        # Create triangle faces with slight transparency
        tri = np.array([[0, 1, 2]])
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                       triangles=tri, color='lightblue', alpha=0.2)
        
        # Set labels
        ax.set_xlabel(self.simulator.strategy_labels[0])
        ax.set_ylabel(self.simulator.strategy_labels[1])
        ax.set_zlabel(self.simulator.strategy_labels[2])
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Title and legend
        ax.set_title(title)
        ax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'])
        
        return fig
    
    def plot_payoff_analysis(self, title: str = "Payoff Analysis",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot analysis of payoffs for different strategies.
        
        Args:
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get payoff history
        payoffs = self.history['payoffs']
        
        # Compute average payoffs per strategy over time
        avg_payoffs = []
        for i, payoff_array in enumerate(payoffs):
            strategies = self.simulator.client_strategies
            
            # Compute average for each strategy
            strategy_payoffs = {}
            for j, strategy in enumerate(strategies):
                if strategy not in strategy_payoffs:
                    strategy_payoffs[strategy] = []
                strategy_payoffs[strategy].append(payoff_array[j])
            
            # Calculate averages
            avg_per_strategy = {}
            for strategy, values in strategy_payoffs.items():
                avg_per_strategy[strategy] = np.mean(values) if values else 0
            
            avg_payoffs.append(avg_per_strategy)
        
        # Create arrays for plotting
        step_indices = range(len(avg_payoffs))
        strategy_avg_payoffs = {
            i: [avg.get(i, 0) for avg in avg_payoffs]
            for i in range(len(self.simulator.strategy_labels))
        }
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Plot average payoffs per strategy
        for i, strategy in enumerate(self.simulator.strategy_labels):
            color = self.config['colors'][i % len(self.config['colors'])]
            ax1.plot(step_indices, strategy_avg_payoffs[i], label=strategy, color=color, linewidth=2)
        
        # Add labels and legend
        ax1.set_ylabel('Average Payoff')
        ax1.set_title(f"{title} - Average Payoffs by Strategy")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot distribution and payoff correlation
        distributions = np.array(self.history['strategy_distribution'])
        
        # Ensure the arrays are the same length
        min_length = min(len(distributions), len(strategy_avg_payoffs[0]))
        distributions = distributions[:min_length]
        
        # Plot on second subplot
        for i, strategy in enumerate(self.simulator.strategy_labels):
            color = self.config['colors'][i % len(self.config['colors'])]
            payoffs = strategy_avg_payoffs[i][:min_length]  # Ensure same length
            ax2.plot(distributions[:, i], payoffs, 'o', 
                    label=strategy, color=color, alpha=0.7)
        
        # Add labels and legend
        ax2.set_xlabel('Strategy Proportion')
        ax2.set_ylabel('Average Payoff')
        ax2.set_title('Payoff vs Strategy Proportion')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'])
        
        return fig
    
    def create_summary_report(self, output_dir: str) -> str:
        """
        Create a comprehensive summary report with multiple visualizations.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to main report file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        self.plot_strategy_evolution(save_path=os.path.join(output_dir, 'strategy_evolution.png'))
        self.plot_ternary_trajectories(save_path=os.path.join(output_dir, 'ternary_trajectories.png'))
        self.plot_convergence_analysis(save_path=os.path.join(output_dir, 'convergence_analysis.png'))
        self.plot_3d_convergence(save_path=os.path.join(output_dir, 'convergence_3d.png'))
        self.plot_payoff_analysis(save_path=os.path.join(output_dir, 'payoff_analysis.png'))
        
        # Create analysis data
        convergence = self.simulator.analyze_convergence()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EGT Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .analysis {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>EGT Simulation Report</h1>
            
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Steps</td>
                    <td>{len(self.history['strategy_distribution'])}</td>
                </tr>
                <tr>
                    <td>Convergence Status</td>
                    <td>{"Converged" if convergence['converged'] else "Not Converged"}</td>
                </tr>
                <tr>
                    <td>Convergence Time</td>
                    <td>{convergence['convergence_time'] if convergence['converged'] else "N/A"}</td>
                </tr>
                <tr>
                    <td>Final Distribution Type</td>
                    <td>{convergence['distribution_type']}</td>
                </tr>
                <tr>
                    <td>Dominant Strategy</td>
                    <td>{convergence['dominant_strategy'] if convergence['dominant_strategy'] else "Mixed"}</td>
                </tr>
            </table>
            
            <h2>Strategy Evolution</h2>
            <div class="figure">
                <img src="strategy_evolution.png" alt="Strategy Evolution Over Time">
                <p>Evolution of strategy proportions over simulation steps</p>
            </div>
            
            <h2>Convergence Analysis</h2>
            <div class="figure">
                <img src="convergence_analysis.png" alt="Convergence Analysis">
                <p>Analysis of convergence rates and stability</p>
            </div>
            
            <h2>Strategy Space Trajectories</h2>
            <div class="figure">
                <img src="ternary_trajectories.png" alt="Ternary Trajectories">
                <p>Trajectory in ternary strategy space</p>
            </div>
            
            <div class="figure">
                <img src="convergence_3d.png" alt="3D Convergence">
                <p>3D visualization of strategy space trajectory</p>
            </div>
            
            <h2>Payoff Analysis</h2>
            <div class="figure">
                <img src="payoff_analysis.png" alt="Payoff Analysis">
                <p>Analysis of strategy payoffs over time</p>
            </div>
            
            <h2>Detailed Convergence Analysis</h2>
            <div class="analysis">
                <pre>{json.dumps(convergence, indent=4)}</pre>
            </div>
            
            <h2>Final Strategy Distribution</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Proportion</th>
                </tr>
        """
        
        # Add final distribution
        final_dist = self.simulator.get_current_distribution()
        for strategy, prop in final_dist.items():
            html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td>{prop:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Write HTML to file
        report_path = os.path.join(output_dir, 'report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
