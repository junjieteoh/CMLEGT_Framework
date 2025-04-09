#!/usr/bin/env python3
"""
Run experiments with multiple starting distributions for Collaborative ML with EGT Framework

This script:
1. Generates N evenly distributed starting points in the strategy simplex
2. Runs experiments for each starting point in parallel
3. Collects trajectory data from all experiments
4. Visualizes all trajectories in a 3D ternary plot
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Add parent directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Get parent directory

# When running from GitHub repo (where egt_framework contents are at root)
# we need to add the parent directory to the path
sys.path.insert(0, parent_dir)

# Local imports - these should work once the parent directory is in the Python path
import collaborative_ml_example as cml
from registry.experiment import ExperimentBuilder

def generate_simplex_points(num_points=50, method='even'):
    """
    Generate points evenly distributed in the 3D simplex (strategy space).
    
    Args:
        num_points: Number of points to generate
        method: Method for generating points ('even', 'random', 'grid')
        
    Returns:
        List of points [honest, withholding, adversarial] that sum to 1.0
    """
    points = []
    
    if method == 'random':
        # Random sampling approach
        for _ in range(num_points):
            # Generate random values and normalize
            p = np.random.random(3)
            p = p / np.sum(p)  # Normalize to sum to 1
            points.append(p.tolist())
            
    elif method == 'grid':
        # Grid-based approach (with step size based on num_points)
        step = 1.0 / np.ceil(np.sqrt(num_points / 2))
        for h in np.arange(0, 1 + step, step):
            for w in np.arange(0, 1 + step - h, step):
                a = 1 - h - w
                if a >= 0 and len(points) < num_points:
                    points.append([h, w, a])
    
    else:  # 'even' method - default
        # Generate points that are evenly spaced in the simplex
        # This includes vertices and points along edges
        
        # Always include vertices
        points.append([1.0, 0.0, 0.0])  # All honest
        points.append([0.0, 1.0, 0.0])  # All withholding
        points.append([0.0, 0.0, 1.0])  # All adversarial
        
        # If we want more than just vertices, add points along edges and interior
        if num_points > 3:
            # Determine how many more points we need
            remaining = num_points - 3
            
            # If just a few more, add points along edges
            if remaining <= 6:
                # Points along edges
                steps = min(3, remaining)
                step_size = 1.0 / (steps + 1)
                
                # Add points along honest-withholding edge
                for i in range(1, steps + 1):
                    ratio = i * step_size
                    points.append([1.0 - ratio, ratio, 0.0])
                    remaining -= 1
                    if remaining == 0:
                        break
                
                # Add points along withholding-adversarial edge
                if remaining > 0:
                    for i in range(1, steps + 1):
                        ratio = i * step_size
                        points.append([0.0, 1.0 - ratio, ratio])
                        remaining -= 1
                        if remaining == 0:
                            break
                
                # Add points along adversarial-honest edge
                if remaining > 0:
                    for i in range(1, steps + 1):
                        ratio = i * step_size
                        points.append([ratio, 0.0, 1.0 - ratio])
                        remaining -= 1
                        if remaining == 0:
                            break
            
            # If more points needed, add interior points in a grid pattern
            if remaining > 0:
                # Estimate step size to get approximately the right number of points
                grid_size = int(np.ceil(np.sqrt(remaining * 2)))
                step = 1.0 / (grid_size + 1)
                
                # Generate interior points
                for i in range(1, grid_size + 1):
                    h_ratio = i * step
                    for j in range(1, grid_size + 1):
                        w_ratio = j * step
                        a_ratio = 1.0 - h_ratio - w_ratio
                        if a_ratio > 0:  # Ensure valid simplex point
                            points.append([h_ratio, w_ratio, a_ratio])
                            if len(points) >= num_points:
                                break
                    if len(points) >= num_points:
                        break
    
    # Ensure all points are valid (sum to 1.0 and all values are positive)
    for i in range(len(points)):
        p = np.array(points[i])
        p = np.maximum(p, 0)  # Ensure all values are positive
        p = p / np.sum(p)  # Normalize to sum to 1
        points[i] = p.tolist()
    
    return points

def create_experiment_config(scenario_id, starting_point):
    """
    Create a custom experiment config based on a scenario with a modified starting distribution.
    
    Args:
        scenario_id: Base scenario ID to use
        starting_point: Strategy distribution to start from [honest, withholding, adversarial]
        
    Returns:
        Custom configuration dictionary
    """
    # Find config file in local configs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", f"{scenario_id}.json")
    
    # Try with _config suffix if not found
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir, "configs", f"{scenario_id}_config.json")
    
    # Load the configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            base_config = json.load(f)
    else:
        raise ValueError(f"Could not find config file for scenario: {scenario_id}")
        
    # Update the initial distribution
    base_config['initial_distribution'] = starting_point
    
    # Add strategy_distribution in the format the framework expects
    base_config['strategy_distribution'] = {
        'honest': starting_point[0],
        'withholding': starting_point[1],
        'adversarial': starting_point[2]
    }
    
    return base_config

def visualize_trajectories(results, output_dir, scenario_name, hide_grid=False, clean_background=True, 
                         line_width=1.0, arrow_count=5):
    """
    Create an improved visualization of all trajectories in a ternary plot.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save visualization
        scenario_name: Name of the scenario
        hide_grid: Whether to hide the grid lines (deprecated, grid always shown now)
        clean_background: Whether to aggressively clean the background
        line_width: Width of trajectory lines
        arrow_count: Approximate number of arrows per trajectory
    """
    try:
        import ternary
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection
        import numpy as np
        
        scale = 1.0 # Define scale for cartesian conversion needed for arrows
        
        # Get scenario description from config file
        scenario_desc = ""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "configs", f"{scenario_name}.json")
            if not os.path.exists(config_path):
                config_path = os.path.join(script_dir, "configs", f"{scenario_name}_config.json")
                
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    scenario_config = json.load(f)
                scenario_desc = scenario_config.get("name", scenario_name)
            else:
                print(f"Warning: Config file not found for scenario description: {config_path}")
                
        except Exception as e:
            print(f"Could not get scenario configuration: {e}")
        
        # Set figure parameters for better quality and text rendering
        plt.rcParams['text.antialiased'] = True
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
        
        # ---------- IMPROVED TERNARY PLOT ----------
        
        if clean_background:
            # Create a completely fresh figure with white background
            fig = plt.figure(figsize=(16, 14), facecolor='white')
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            
            # Create a fresh ternary plot on the clean axes
            tax = ternary.TernaryAxesSubplot(ax, scale=1.0)
            tax.boundary(linewidth=2.0)
            tax.set_title("")  # Remove default title (we'll set our own)
            tax.clear_matplotlib_ticks() # Clear default ticks before adding custom ones
            tax._redraw_labels() # Ensure labels are drawn cleanly
        else:
            # Use the standard ternary figure creation
            fig, tax = ternary.figure(scale=1.0)
            fig.set_size_inches(16, 14)
            tax.boundary(linewidth=2.0)
            tax.ax.set_facecolor('white')
        
        # Configure the plot
        # Add grid lines (always shown now)
        tax.gridlines(color="lightgray", multiple=0.1, linewidth=0.5, alpha=0.6)
        
        # Add numerical ticks
        tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset=0.02)
        
        # Set labels (commented out as requested)
        # tax.left_axis_label("Adversarial Strategy (100%)", fontsize=12, fontweight='bold')
        # tax.right_axis_label("Withholding Strategy (100%)", fontsize=12, fontweight='bold')
        # tax.bottom_axis_label("Honest Strategy (100%)", fontsize=12, fontweight='bold')
        
        # Add vertex labels to clearly mark corners with better positioning
        tax.annotate("Honest\n(1.0, 0.0, 0.0)", (1.0, 0.0, 0.0), fontsize=12, fontweight='bold', 
                   ha='center', va='bottom', xytext=(0,15), textcoords='offset points') # Increased vertical offset
        tax.annotate("Withholding\n(0.0, 1.0, 0.0)", (0.0, 1.0, 0.0), fontsize=12, fontweight='bold', 
                   ha='center', va='bottom', xytext=(0,5), textcoords='offset points')
        tax.annotate("Adversarial\n(0.0, 0.0, 1.0)", (0.0, 0.0, 1.0), fontsize=12, fontweight='bold', 
                   ha='left', va='top', xytext=(5,-5), textcoords='offset points')
        
        # Use Blues colormap for trajectory progression (light to dark blue)
        cmap = plt.cm.Blues
        norm = Normalize(vmin=0.0, vmax=1.0)
        
        # Plot each trajectory
        valid_trajectories = 0
        for result in results:
            if 'error' in result or 'trajectory' not in result:
                continue
                
            # Get trajectory data
            starting_point = result['starting_point']
            trajectory = result['trajectory']
            
            # Convert trajectory to the format expected by ternary
            try:
                if isinstance(trajectory[0], list):
                    # Ensure all points are valid (sum to ~1.0)
                    ternary_trajectory = []
                    for t in trajectory:
                        # Normalize point to ensure it sums to 1.0
                        t_sum = sum(t)
                        if t_sum > 0 and all(ti >= -1e-6 for ti in t):  # Avoid division by zero and small negatives
                            normalized_t = tuple(max(0, ti / t_sum) for ti in t)
                            # Final check for sum approx 1
                            if abs(sum(normalized_t) - 1.0) < 1e-6:
                                ternary_trajectory.append(normalized_t)
                            else:
                                print(f"Warning: Normalization failed for point {t} -> {normalized_t}")
                        else:
                            print(f"Warning: Invalid point {t} in trajectory (sum = {t_sum}, values={t})")
                    
                    # Skip if trajectory is too short or has invalid points
                    if len(ternary_trajectory) < 2:
                        print(f"Warning: Trajectory too short or invalid for run {result.get('run_id', 'unknown')}")
                        continue
                    
                    # Plot thin connecting line (mid-blue, semi-transparent)
                    tax.plot(ternary_trajectory, color=cmap(0.5), linewidth=line_width, alpha=0.6, zorder=3)
                    
                    # Plot small dots at each intermediate epoch point (fully opaque)
                    if len(ternary_trajectory) > 2: # Only if there are intermediate points
                        tax.scatter(ternary_trajectory[1:-1], marker='.', color='gray', s=10, zorder=4)

                    # Mark starting point with a black dot (less transparent)
                    tax.scatter([ternary_trajectory[0]], marker='o', color='#FF0000', s=50, 
                               edgecolor='black', linewidth=1.5, zorder=5, alpha=0.8) # Reduced from s=80 to s=50
                    
                    # Mark endpoint with a red square
                    tax.scatter([ternary_trajectory[-1]], marker='s', color='red', s=70, 
                               edgecolor='black', linewidth=1.5, zorder=5)
                    
                    # Add mini arrows along the trajectory with color gradient
                    traj_len = len(ternary_trajectory)
                    if arrow_count > 0 and traj_len > 1:
                        arrow_interval = max(1, traj_len // arrow_count)  # Evenly space arrows
                        for i in range(arrow_interval, traj_len - 1, arrow_interval):
                            if i + 1 < traj_len:
                                p1 = ternary_trajectory[i]
                                p2 = ternary_trajectory[i+1]
                                
                                # Calculate arrow direction in Cartesian coordinates for quiver
                                p1_cart = (scale * (0.5 * p1[1] + p1[0]), scale * (np.sqrt(3)/2 * p1[1]))
                                p2_cart = (scale * (0.5 * p2[1] + p2[0]), scale * (np.sqrt(3)/2 * p2[1]))
                                delta_cart = [p2_cart[0] - p1_cart[0], p2_cart[1] - p1_cart[1]]
                                
                                # Calculate position in trajectory (0 to 1)
                                position = i / (traj_len - 1)
                                
                                # Define gradient colors: light blue to dark blue
                                # Light blue for start: #ADD8E6, dark blue for end: #0000CD
                                start_color = np.array([173/255, 216/255, 230/255])  # #ADD8E6 (light blue)
                                end_color = np.array([0/255, 0/255, 205/255])    # #0000CD (medium blue)
                                arrow_color = tuple(start_color * (1-position) + end_color * position)
                                
                                # Only add arrow if there's significant movement
                                if np.sqrt(delta_cart[0]**2 + delta_cart[1]**2) > 0.005:
                                    tax.ax.quiver(
                                        p1_cart[0], p1_cart[1], delta_cart[0], delta_cart[1],
                                        angles='xy', scale_units='xy', scale=1,
                                        color=arrow_color, # Color based on position in trajectory
                                        width=0.0025, # Increased from 0.002 to 0.0025
                                        headwidth=6, # Slightly reduced head size
                                        headlength=6, # Slightly reduced head size
                                        headaxislength=5, # Slightly reduced head size
                                        alpha=0.9, # Increased from 0.7 to 0.9
                                        zorder=4
                                    )
                    
                    valid_trajectories += 1
                else:
                    # Handle simplified trajectories (start -> end only)
                    if 'final_distribution' in result and isinstance(result['final_distribution'], dict):
                        # Extract start and end points
                        start_point = starting_point
                        end_point = (
                            result['final_distribution'].get('honest', 0),
                            result['final_distribution'].get('withholding', 0),
                            result['final_distribution'].get('adversarial', 0)
                        )
                        
                        # Normalize end point to ensure it sums to 1.0
                        end_sum = sum(end_point)
                        if end_sum > 0 and all(ti >= -1e-6 for ti in end_point):
                            end_point = tuple(max(0, ti / end_sum) for ti in end_point)
                        else:
                            # If invalid endpoint, assume no change
                            end_point = start_point
                        
                        ternary_trajectory = [start_point, end_point]
                        
                        # Plot trajectory line
                        tax.plot(ternary_trajectory, color=cmap(0.5), linewidth=line_width, alpha=0.6, zorder=3) # Reduced from 0.75 to 0.6
                        
                        # Mark starting point with a black dot (less transparent)
                        tax.scatter([ternary_trajectory[0]], marker='o', color='#FF0000', s=50, 
                                  edgecolor='black', linewidth=1.5, zorder=5, alpha=0.8) # Reduced from s=80 to s=50
                        
                        # Mark endpoint with a red square
                        tax.scatter([ternary_trajectory[-1]], marker='s', color='red', s=70, 
                                  edgecolor='black', linewidth=1.5, zorder=5)
                        
                        # Add arrow with gradient color (simplified case)
                        if arrow_count > 0:
                           p1 = ternary_trajectory[0]
                           p2 = ternary_trajectory[1]
                           p1_cart = (scale * (0.5 * p1[1] + p1[0]), scale * (np.sqrt(3)/2 * p1[1]))
                           p2_cart = (scale * (0.5 * p2[1] + p2[0]), scale * (np.sqrt(3)/2 * p2[1]))
                           delta_cart = [p2_cart[0] - p1_cart[0], p2_cart[1] - p1_cart[1]]
                           
                           # Use midpoint color for simplified trajectory
                           midpoint_color = tuple([(173/255 + 0/255)/2, (216/255 + 0/255)/2, (230/255 + 205/255)/2])
                           
                           if np.sqrt(delta_cart[0]**2 + delta_cart[1]**2) > 0.005:
                               mid_cart = ((p1_cart[0] + p2_cart[0])/2, (p1_cart[1] + p2_cart[1])/2)
                               tax.ax.quiver(
                                   mid_cart[0]-delta_cart[0]/4, mid_cart[1]-delta_cart[1]/4, 
                                   delta_cart[0]/2, delta_cart[1]/2, 
                                   angles='xy', scale_units='xy', scale=1,
                                   color=midpoint_color, # Midpoint gradient color
                                   width=0.0025, # Increased from 0.002 to 0.0025
                                   headwidth=6, # Slightly reduced head size
                                   headlength=6, # Slightly reduced head size
                                   headaxislength=5, # Slightly reduced head size
                                   alpha=0.9, # Increased from 0.7 to 0.9
                                   zorder=4
                               )
                        
                        valid_trajectories += 1
                    else:
                        # Skip invalid trajectory data
                        continue
                
            except Exception as e:
                print(f"Error plotting trajectory for run {result.get('run_id', 'unknown')}: {str(e)}")
        
        # Add improved legend
        markers = [
            mpatches.Patch(color='none', label=f"{valid_trajectories} valid trajectories"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                     markeredgecolor='black', markersize=10, label='Starting Point', alpha=0.8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                     markeredgecolor='black', markersize=10, label='Ending Point'),
            # Arrow color gradient legend 
            plt.Line2D([0], [0], color='#ADD8E6', lw=2, label='Early Trajectory'),
            plt.Line2D([0], [0], color='#0000CD', lw=2, label='Late Trajectory')
        ]
        tax.ax.legend(handles=markers, loc='upper right', fontsize=12)
        
        # Set simplified title
        plt.title(f"{scenario_name}", fontsize=14, fontweight='bold', pad=20)
        
        # Save figure with improved resolution and updated filename
        plt.savefig(os.path.join(output_dir, 'ternary_trajectories_v2.png'), bbox_inches='tight', dpi=300, pad_inches=0.1)
        print(f"Plot saved to {os.path.join(output_dir, 'ternary_trajectories_v2.png')}")
        
        # Close the figure to free memory
        plt.close(fig)
        
        # The 3D plot code follows here...
        # [Keep existing 3D plot code from before]
        
    except ImportError as e:
        print(f"Error importing visualization libraries: {e}")
        print("Please install required packages: pip install python-ternary matplotlib numpy") 

def run_single_experiment(args):
    """
    Run a single experiment with a specific starting distribution.
    
    Args:
        args: Tuple containing (run_id, scenario, starting_point, num_clients, num_epochs)
        
    Returns:
        Dictionary with experiment results
    """
    run_id, scenario, starting_point, num_clients, num_epochs = args
    
    starting_dist_str = f"[{starting_point[0]:.2f}, {starting_point[1]:.2f}, {starting_point[2]:.2f}]"
    experiment_id = f"dist_{run_id:02d}"
    print(f"Starting experiment {experiment_id} with distribution {starting_dist_str}")
    
    # Create custom config with the specified starting distribution
    custom_config = create_experiment_config(scenario, starting_point)
    
    try:
        # Run the experiment
        simulator, history = cml.run_collaborative_ml_experiment(
            num_clients=num_clients,
            num_epochs=num_epochs,
            scenario=scenario,
            experiment_id=experiment_id,
            custom_config=custom_config
        )
        
        # Extract trajectory data
        trajectory = history.get('strategy_distribution', [])
        
        # Ensure trajectory is in the correct format
        if len(trajectory) > 0:
            # If trajectory data is empty or not formatted correctly, generate at least start and end points
            if not trajectory or not isinstance(trajectory[0], list):
                # Create a synthetic trajectory with at least start and end points
                starting_strat = [starting_point[0], starting_point[1], starting_point[2]]
                
                # Get current distribution from simulator
                final_dist = simulator.get_current_distribution()
                
                # Ensure the final distribution sums to 1.0
                final_honest = final_dist.get('honest', 0)
                final_withholding = final_dist.get('withholding', 0)
                final_adversarial = final_dist.get('adversarial', 0)
                final_sum = final_honest + final_withholding + final_adversarial
                
                if final_sum > 0:
                    final_strat = [
                        final_honest / final_sum,
                        final_withholding / final_sum,
                        final_adversarial / final_sum
                    ]
                else:
                    # If sum is zero, keep the starting distribution
                    final_strat = starting_strat
                
                # Create a simple trajectory with start and end points
                trajectory = [starting_strat, final_strat]
        else:
            # Create a synthetic trajectory if we have no data
            final_dist = simulator.get_current_distribution()
            
            # Ensure the final distribution sums to 1.0
            final_honest = final_dist.get('honest', 0)
            final_withholding = final_dist.get('withholding', 0)
            final_adversarial = final_dist.get('adversarial', 0)
            final_sum = final_honest + final_withholding + final_adversarial
            
            if final_sum > 0:
                final_strat = [
                    final_honest / final_sum,
                    final_withholding / final_sum,
                    final_adversarial / final_sum
                ]
            else:
                # If sum is zero, keep the starting distribution
                final_strat = [starting_point[0], starting_point[1], starting_point[2]]
            
            trajectory = [
                [starting_point[0], starting_point[1], starting_point[2]],
                final_strat
            ]
        
        # Convert to the format needed for visualization
        trajectory_data = {
            'run_id': run_id,
            'starting_point': starting_point,
            'trajectory': trajectory,
            'final_distribution': simulator.get_current_distribution()
        }
        
        print(f"Completed experiment {experiment_id}")
        return trajectory_data
        
    except Exception as e:
        print(f"Error in experiment {experiment_id}: {e}")
        return {
            'run_id': run_id,
            'starting_point': starting_point,
            'error': str(e)
        } 

def run_multi_distribution_experiments(
    scenario='balanced_incentives',
    num_points=50,
    num_clients=100,
    num_epochs=100,
    parallel=True,
    distribution_method='even',
    max_processes=None,
    hide_grid=False,
    clean_background=True,
    line_width=1.0,
    arrow_count=5
):
    """
    Run experiments with multiple starting distributions.
    
    Args:
        scenario: Scenario ID to use for experiments
        num_points: Number of starting distributions to generate
        num_clients: Number of clients in each experiment
        num_epochs: Number of epochs for each experiment
        parallel: Whether to run experiments in parallel
        distribution_method: Method for generating starting points ('even', 'random', 'grid')
        max_processes: Maximum number of processes to use (default: None = use all available cores)
        hide_grid: Whether to hide the grid lines in visualization
        clean_background: Whether to aggressively clean the background
        line_width: Width of trajectory lines
        arrow_count: Approximate number of arrows per trajectory
        
    Returns:
        List of experiment results
    """
    print(f"\nGenerating {num_points} starting distributions using '{distribution_method}' method")
    starting_points = generate_simplex_points(num_points, method=distribution_method)
    
    print(f"Running {len(starting_points)} experiments for scenario '{scenario}'")
    print(f"Epochs: {num_epochs}, Clients: {num_clients}, Parallel: {parallel}")
    
    # Create results directory within examples
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(script_dir, "results", f"multi_dist_{scenario}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save starting points
    with open(os.path.join(results_dir, "starting_points.json"), 'w') as f:
        json.dump(starting_points, f, indent=2)
    
    # Prepare arguments for each experiment
    all_experiment_args = [
        (i, scenario, point, num_clients, num_epochs) for i, point in enumerate(starting_points)
    ]
    
    # Run experiments
    start_time = time.time()
    if parallel:
        # Determine number of processes
        available_cpus = cpu_count()
        if max_processes is None:
            # Use all available CPUs by default
            num_processes = min(len(all_experiment_args), available_cpus)
        else:
            # Use specified maximum, but don't exceed available CPUs
            num_processes = min(max_processes, available_cpus, len(all_experiment_args))
        
        print(f"Running with {num_processes} parallel processes (out of {available_cpus} available CPUs)")
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_single_experiment, all_experiment_args)
    else:
        # Run sequentially
        print("Running sequentially")
        results = []
        for args in all_experiment_args:
            results.append(run_single_experiment(args))
    
    elapsed_time = time.time() - start_time
    print(f"All experiments completed in {elapsed_time:.1f} seconds!")
    
    # Save all results
    with open(os.path.join(results_dir, "experiment_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    try:
        print("Creating visualizations...")
        visualize_trajectories(
            results, 
            results_dir, 
            scenario,
            hide_grid=hide_grid,
            clean_background=clean_background,
            line_width=line_width,
            arrow_count=arrow_count
        )
        print("Visualizations created successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return results, results_dir

def main():
    """Main function to parse arguments and run experiments."""
    print("Starting main() function...")
    parser = argparse.ArgumentParser(description="Run multi-distribution EGT experiments")
    parser.add_argument("--scenario", type=str, default="balanced_incentives",
                       help="Scenario ID to use (e.g., ch5_s1_baseline_config)") # Updated help text
    parser.add_argument("--points", type=int, default=20,
                       help="Number of starting distributions (default: 20)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs (default: 20)")
    parser.add_argument("--clients", type=int, default=40,
                       help="Number of clients (default: 100)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run sequentially instead of in parallel")
    parser.add_argument("--method", type=str, default="even", choices=["even", "random", "grid"],
                       help="Method for generating starting points (default: even)")
    parser.add_argument("--processes", type=int, default=None,
                       help="Maximum number of parallel processes to use (default: use all available)")
    
    # Add new visualization parameters
    parser.add_argument("--hide-grid", action="store_true",
                       help="Hide grid lines in the visualization")
    parser.add_argument("--line-width", type=float, default=1.0,
                       help="Width of trajectory lines (default: 1.0)")
    parser.add_argument("--arrow-count", type=int, default=5,
                       help="Approximate number of arrows per trajectory (default: 5)")
    parser.add_argument("--no-clean-background", action="store_true",
                       help="Disable aggressive background cleaning")
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    # Run experiments
    try:
        print("Running experiments...")
        results, results_dir = run_multi_distribution_experiments(
            scenario=args.scenario.replace('_config',''), # Remove suffix if present
            num_points=args.points,
            num_clients=args.clients,
            num_epochs=args.epochs,
            parallel=not args.sequential,
            distribution_method=args.method,
            max_processes=args.processes,
            hide_grid=args.hide_grid,
            clean_background=not args.no_clean_background,
            line_width=args.line_width,
            arrow_count=args.arrow_count
        )
        
        print(f"\nExperiments completed successfully!")
        print(f"Results saved to: {results_dir}")
    except Exception as e:
        print(f"ERROR running experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Script starting...")
    main()
    print("Script completed.") 