#!/usr/bin/env python3
"""
Script to process CSV files and generate performance plots.
Takes input directory containing CSV files and output directory for plots.

Generates for each dataset:
1. Median performance curves across all permutations
2. Average performance curves with standard deviation across all permutations  
3. Comparison table showing difference from supervised baseline

COMPUTATION DETAILS:
===================

1. MEDIAN PERFORMANCE:
   - For each example count i and algorithm A: median_loss_A(i) = median over all permutations of loss_A(i)
   - Plots median_loss_A(i) vs example count i for each algorithm A

2. AVERAGE PERFORMANCE WITH STD:
   - For each example count i and algorithm A: 
     * avg_loss_A(i) = mean over all permutations of loss_A(i)
     * std_loss_A(i) = standard deviation over all permutations of loss_A(i)
   - Plots avg_loss_A(i) ± std_loss_A(i) vs example count i for each algorithm A

3. COMPARISON TABLE (Difference from Supervised):
   - For each algorithm A and dataset D:
     * final_loss_A_perm_j = loss_A at last example for permutation j
     * final_loss_Supervised_perm_j = loss_Supervised at last example for permutation j  
     * diff_A_perm_j = final_loss_A_perm_j - final_loss_Supervised_perm_j
     * mean_diff_from_supervised_A = mean over all permutations j of diff_A_perm_j
     * median_diff_from_supervised_A = median over all permutations j of diff_A_perm_j
   - Tables show both mean and median differences for each algorithm A and dataset D
   - Positive values mean algorithm performs worse than supervised
   - Negative values mean algorithm performs better than supervised
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

# Algorithm styling - reverted to original colors with improved visibility
algorithm_styles = {
    'Supervised': {'color': '#1f77b4', 'linestyle': '-', 'marker': '', 'label': 'Supervised', 'linewidth': 4, 'markersize': 0},
    'SquareCB': {'color': '#9467bd', 'linestyle': '--', 'marker': '', 'label': 'SquareCB', 'linewidth': 4, 'markersize': 0},
    'FastCB': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '', 'label': 'FastCB', 'linewidth': 4, 'markersize': 0},
    'ADACB': {'color': '#8B4513', 'linestyle': ':', 'marker': '', 'label': 'AdaCB', 'linewidth': 5, 'markersize': 0},
    'RegCB': {'color': '#17becf', 'linestyle': (0, (3, 1, 1, 1)), 'marker': '', 'label': 'RegCB', 'linewidth': 5, 'markersize': 0},
    'OPOCMAB': {'color': '#FF0000', 'linestyle': (0, (3, 1, 1, 1, 1, 1)), 'marker': '', 'label': 'OPO-CMAB', 'linewidth': 5, 'markersize': 0}
}

# Plot parameters
params = {
    'axes.labelsize': 32,
    'axes.titlesize': 32,
    'font.size': 32,
    'legend.fontsize': 20,  # Increased legend font size
    'figure.subplot.wspace': 0.02,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'text.usetex': False,
    'figure.figsize': [12, 9]  # Larger figure size for better visibility
}

plt.rcParams.update(params)
plt.rcParams['text.usetex'] = False
plt.rc('font', family='sans-serif')
plt.rcParams['axes.edgecolor'] = "0.15"
plt.rcParams['axes.linewidth'] = 1.25

# Disable scientific notation for x-axis
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False

def load_csv_data(input_dir):
    """
    Load all CSV files from input directory and return organized data.
    
    Returns:
        dict: {dataset: {algorithm: {permutation_id: DataFrame}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist!")
        return data
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    for csv_file in csv_files:
        try:
            # Parse filename to extract dataset and algorithm
            # Expected format: dataset_algorithm.csv
            filename_parts = csv_file.replace('.csv', '').split('_')
            if len(filename_parts) < 2:
                logger.warning(f"Skipping {csv_file} - cannot parse filename")
                continue
                
            # Handle dataset names that might contain underscores
            # Try to find the algorithm name at the end
            algorithm = None
            for algo in algorithm_styles.keys():
                if csv_file.endswith(f"_{algo}.csv"):
                    algorithm = algo
                    dataset = csv_file.replace(f"_{algo}.csv", "")
                    break
            
            if algorithm is None:
                logger.warning(f"Skipping {csv_file} - cannot identify algorithm")
                continue
            
            csv_path = os.path.join(input_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # Convert numeric columns to proper data types
            df['example_count'] = pd.to_numeric(df['example_count'], errors='coerce')
            df['progressive_loss'] = pd.to_numeric(df['progressive_loss'], errors='coerce')
            df['instantaneous_loss'] = pd.to_numeric(df['instantaneous_loss'], errors='coerce')
            df['permutation_id'] = pd.to_numeric(df['permutation_id'], errors='coerce')
            
            # Remove rows with NaN values in critical columns
            df = df.dropna(subset=['example_count', 'progressive_loss'])
            
            # Group by permutation_id
            for perm_id, group in df.groupby('permutation_id'):
                data[dataset][algorithm][perm_id] = group.sort_values('example_count')
                
            logger.info(f"Loaded {dataset}_{algorithm} with {len(df['permutation_id'].unique())} permutations")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue
    
    return data

def calculate_average_performance_with_std(data, dataset, algorithm):
    """
    Calculate average performance and standard deviation across all permutations for a dataset-algorithm pair.
    
    COMPUTATION:
    For each example count i:
    - avg_loss(i) = mean over all permutations of progressive_loss at example i
    - std_loss(i) = standard deviation over all permutations of progressive_loss at example i
    
    Returns:
        tuple: (example_counts, average_losses, std_losses)
    """
    if dataset not in data or algorithm not in data[dataset]:
        return None, None, None
    
    # Get all permutation data
    permutations = data[dataset][algorithm]
    if not permutations:
        return None, None, None
    
    # Find the maximum number of examples across all permutations
    max_examples = 0
    for perm_data in permutations.values():
        max_examples = max(max_examples, perm_data['example_count'].max())
    
    # Convert to integer to ensure range() works properly
    max_examples = int(max_examples)
    
    # Create x-axis from 1 to max_examples
    example_counts = list(range(1, max_examples + 1))
    
    # Calculate average and std loss for each example count
    average_losses = []
    std_losses = []
    for count in example_counts:
        losses_at_count = []
        for perm_data in permutations.values():
            # Find the closest example count if exact match doesn't exist
            perm_counts = perm_data['example_count'].values
            if count in perm_counts:
                loss = perm_data[perm_data['example_count'] == count]['progressive_loss'].iloc[0]
                losses_at_count.append(loss)
        
        if losses_at_count:
            average_losses.append(np.mean(losses_at_count))
            std_losses.append(np.std(losses_at_count))
        else:
            average_losses.append(np.nan)
            std_losses.append(np.nan)
    
    return example_counts, average_losses, std_losses

def calculate_median_performance(data, dataset, algorithm):
    """
    Calculate median performance across all permutations for a dataset-algorithm pair.
    
    COMPUTATION:
    For each example count i:
    - median_loss(i) = median over all permutations of progressive_loss at example i
    
    Returns:
        tuple: (example_counts, median_losses)
    """
    if dataset not in data or algorithm not in data[dataset]:
        return None, None
    
    # Get all permutation data
    permutations = data[dataset][algorithm]
    if not permutations:
        return None, None
    
    # Find the maximum number of examples across all permutations
    max_examples = 0
    for perm_data in permutations.values():
        max_examples = max(max_examples, perm_data['example_count'].max())
    
    # Convert to integer to ensure range() works properly
    max_examples = int(max_examples)
    
    # Create x-axis from 1 to max_examples
    example_counts = list(range(1, max_examples + 1))
    
    # Calculate median loss for each example count
    median_losses = []
    for count in example_counts:
        losses_at_count = []
        for perm_data in permutations.values():
            # Find the closest example count if exact match doesn't exist
            perm_counts = perm_data['example_count'].values
            if count in perm_counts:
                loss = perm_data[perm_data['example_count'] == count]['progressive_loss'].iloc[0]
                losses_at_count.append(loss)
        
        if losses_at_count:
            median_losses.append(np.median(losses_at_count))
        else:
            median_losses.append(np.nan)
    
    return example_counts, median_losses

def calculate_final_loss_differences(data):
    """
    Calculate the difference from supervised baseline for each algorithm and dataset.
    
    COMPUTATION:
    For each dataset D and algorithm A:
    1. final_loss_A_perm_j = progressive_loss of algorithm A at the last example for permutation j
    2. final_loss_Supervised_perm_j = progressive_loss of Supervised at the last example for permutation j
    3. diff_A_perm_j = final_loss_A_perm_j - final_loss_Supervised_perm_j
    4. mean_diff_from_supervised_A = mean over all permutations j of diff_A_perm_j
    5. median_diff_from_supervised_A = median over all permutations j of diff_A_perm_j
    
    Returns:
        dict: {dataset: {algorithm: {'mean_diff': mean_difference, 'median_diff': median_difference, 'all_diffs': [list_of_differences]}}}
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    for dataset in data:
            
        if 'Supervised' not in data[dataset]:
            logger.warning(f"No Supervised data found for dataset {dataset}")
            continue
        
        # Get supervised final losses for each permutation
        supervised_final_losses = {}
        for perm_id, perm_data in data[dataset]['Supervised'].items():
            if len(perm_data) > 0:
                final_loss = perm_data['progressive_loss'].iloc[-1]
                supervised_final_losses[perm_id] = final_loss
        
        if not supervised_final_losses:
            logger.warning(f"No valid Supervised data for dataset {dataset}")
            continue
        
        # Calculate differences for each algorithm
        for algorithm in data[dataset]:
            if algorithm == 'Supervised':
                results[dataset][algorithm] = {
                    'mean_diff': 0.0,
                    'median_diff': 0.0,
                    'all_diffs': [0.0] * len(supervised_final_losses)
                }
                continue
            
            differences = []
            for perm_id, perm_data in data[dataset][algorithm].items():
                if perm_id in supervised_final_losses and len(perm_data) > 0:
                    algo_final_loss = perm_data['progressive_loss'].iloc[-1]
                    supervised_final_loss = supervised_final_losses[perm_id]
                    diff = algo_final_loss - supervised_final_loss
                    differences.append(diff)
            
            if differences:
                mean_diff = np.mean(differences)
                median_diff = np.median(differences)
                results[dataset][algorithm] = {
                    'mean_diff': mean_diff,
                    'median_diff': median_diff,
                    'all_diffs': differences
                }
                logger.info(f"{dataset} - {algorithm}: mean diff = {mean_diff:.6f}, median diff = {median_diff:.6f} (over {len(differences)} permutations)")
            else:
                logger.warning(f"No valid comparison data for {dataset} - {algorithm}")
    
    return results

def create_comparison_table(comparison_data, output_dir):
    """
    Create a comparison table showing both mean and median differences from supervised baseline.
    Generates CSV files, text files, and visual table plots.
    Winners (smallest difference) are highlighted in gold in visual tables.
    """
    if not comparison_data:
        logger.warning("No comparison data to create table")
        return
    
    # Get all datasets and algorithms
    all_datasets = list(comparison_data.keys())
    all_algorithms = set()
    for dataset_data in comparison_data.values():
        all_algorithms.update(dataset_data.keys())
    all_algorithms = sorted([algo for algo in all_algorithms if algo != 'Supervised'])
    
    # Create DataFrame for mean differences
    mean_table_data = []
    median_table_data = []
    
    # Collect all values for mean calculation across datasets
    algorithm_mean_values = {algo: [] for algo in all_algorithms}
    algorithm_median_values = {algo: [] for algo in all_algorithms}
    
    for dataset in sorted(all_datasets):
        mean_row = {'Dataset': dataset}
        median_row = {'Dataset': dataset}
        
        # Collect values to find the winner (smallest difference)
        mean_values = {}
        median_values = {}
        
        for algorithm in all_algorithms:
            if algorithm in comparison_data[dataset]:
                mean_diff = comparison_data[dataset][algorithm]['mean_diff']
                median_diff = comparison_data[dataset][algorithm]['median_diff']
                mean_values[algorithm] = mean_diff
                median_values[algorithm] = median_diff
                mean_row[algorithm] = f"{mean_diff:.6f}"
                median_row[algorithm] = f"{median_diff:.6f}"
                
                # Collect for overall mean calculation
                algorithm_mean_values[algorithm].append(mean_diff)
                algorithm_median_values[algorithm].append(median_diff)
            else:
                mean_row[algorithm] = "N/A"
                median_row[algorithm] = "N/A"
        
        # Find winners (smallest difference - most negative or closest to zero)
        if mean_values:
            mean_winner = min(mean_values.keys(), key=lambda k: mean_values[k])
            
        if median_values:
            median_winner = min(median_values.keys(), key=lambda k: median_values[k])
                
        # Store winner information for visual tables
        mean_row['_winner'] = mean_winner if mean_values else None
        median_row['_winner'] = median_winner if median_values else None
        
        mean_table_data.append(mean_row)
        median_table_data.append(median_row)
    

    
    mean_df = pd.DataFrame(mean_table_data)
    median_df = pd.DataFrame(median_table_data)
    
    # Rename ADACB column to AdaCB for display in tables
    if 'ADACB' in mean_df.columns:
        mean_df = mean_df.rename(columns={'ADACB': 'AdaCB'})
        # Update _winner values to match the renamed column
        mean_df['_winner'] = mean_df['_winner'].replace('ADACB', 'AdaCB')
    if 'ADACB' in median_df.columns:
        median_df = median_df.rename(columns={'ADACB': 'AdaCB'})
        # Update _winner values to match the renamed column
        median_df['_winner'] = median_df['_winner'].replace('ADACB', 'AdaCB')
    
    # Save as CSV files
    mean_csv_path = os.path.join(output_dir, "comparison_table_mean_difference_from_supervised.csv")
    median_csv_path = os.path.join(output_dir, "comparison_table_median_difference_from_supervised.csv")
    
    mean_df.to_csv(mean_csv_path, index=False)
    median_df.to_csv(median_csv_path, index=False)
    
    logger.info(f"Saved mean comparison table: {mean_csv_path}")
    logger.info(f"Saved median comparison table: {median_csv_path}")
    
    # Create visual table plots
    create_visual_comparison_table(mean_df, "Mean Differences from Supervised", output_dir, "mean_differences_table.pdf")
    create_visual_comparison_table(median_df, "Median Differences from Supervised", output_dir, "median_differences_table.pdf")
    
    # Create a comprehensive formatted text table
    txt_path = os.path.join(output_dir, "comparison_table_difference_from_supervised.txt")
    with open(txt_path, 'w') as f:
        f.write("COMPARISON TABLE: Difference from Supervised Baseline\n")
        f.write("=" * 80 + "\n\n")
        f.write("COMPUTATION:\n")
        f.write("For each algorithm A and dataset D:\n")
        f.write("1. final_loss_A_perm_j = progressive_loss of A at last example for permutation j\n")
        f.write("2. final_loss_Supervised_perm_j = progressive_loss of Supervised at last example for permutation j\n")
        f.write("3. diff_A_perm_j = final_loss_A_perm_j - final_loss_Supervised_perm_j\n")
        f.write("4. mean_diff_from_supervised_A = mean over all permutations j of diff_A_perm_j\n")
        f.write("5. median_diff_from_supervised_A = median over all permutations j of diff_A_perm_j\n\n")
        f.write("INTERPRETATION:\n")
        f.write("- Positive values: Algorithm has higher loss than supervised\n")
        f.write("- Negative values: Algorithm has lower loss than supervised\n")
        f.write("- Zero: Same performance as supervised\n")
        f.write("- Winner (smallest difference) is highlighted in gold in visual tables\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write mean differences table
        f.write("MEAN DIFFERENCES FROM SUPERVISED:\n")
        f.write("-" * 50 + "\n")
        f.write(mean_df.to_string(index=False))
        f.write("\n\n")
        
        # Write median differences table
        f.write("MEDIAN DIFFERENCES FROM SUPERVISED:\n")
        f.write("-" * 50 + "\n")
        f.write(median_df.to_string(index=False))
        f.write("\n\n")
        
        # Add summary statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for algorithm in all_algorithms:
            mean_values = []
            median_values = []
            for dataset in all_datasets:
                if algorithm in comparison_data[dataset]:
                    mean_values.append(comparison_data[dataset][algorithm]['mean_diff'])
                    median_values.append(comparison_data[dataset][algorithm]['median_diff'])
            
            if mean_values:
                overall_mean_diff = np.mean(mean_values)
                overall_median_diff = np.median(median_values)
                std_mean_diff = np.std(mean_values)
                std_median_diff = np.std(median_values)
                
                f.write(f"{algorithm}:\n")
                f.write(f"  Overall mean difference: {overall_mean_diff:.6f} ± {std_mean_diff:.6f}\n")
                f.write(f"  Overall median difference: {overall_median_diff:.6f} ± {std_median_diff:.6f}\n")
                f.write(f"  Lower loss than supervised (mean) on {sum(1 for v in mean_values if v < 0)}/{len(mean_values)} datasets\n")
                f.write(f"  Lower loss than supervised (median) on {sum(1 for v in median_values if v < 0)}/{len(median_values)} datasets\n\n")
    
    logger.info(f"Saved comprehensive comparison table: {txt_path}")

def create_visual_comparison_table(df, title, output_dir, filename):
    """
    Create a visual table plot from a DataFrame.
    Winners (smallest difference) are highlighted in gold.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Remove the internal winner column before creating the table
    display_df = df.drop(columns=['_winner'], errors='ignore')
    
    # Create the table using the DataFrame data directly
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code cells based on winner status only
    for i in range(len(df)):
        winner_algorithm = df.iloc[i].get('_winner', None)  # Get winner for this row
        for j in range(1, len(display_df.columns)):  # Skip Dataset column, use display_df
            algorithm_name = display_df.columns[j]
            cell_value_str = display_df.iloc[i, j]
            
            if cell_value_str != 'N/A' and winner_algorithm == algorithm_name:
                # This is the winner algorithm for this dataset - gold color
                table[(i+1, j)].set_facecolor('#FFD700')  # Gold color for winners
                table[(i+1, j)].set_text_props(weight='bold')
    
    # Style header row
    for j in range(len(display_df.columns)):
        table[(0, j)].set_facecolor('#4A90E2')  # Nice blue header color
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style dataset column - keep it plain
    for i in range(1, len(display_df) + 1):
        dataset_name = display_df.iloc[i-1, 0]  # Get dataset name
        # All dataset rows get plain white background, just bold text
        table[(i, 0)].set_facecolor('white')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with winner color only
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FFD700', label='Winner (Best Performance)', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
    
    # Save the plot
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visual comparison table: {plot_path}")

def create_legend_plot(output_dir):
    """
    Create a separate legend plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create dummy lines for legend with proper styling
    lines = []
    for algorithm, style in algorithm_styles.items():
        line = ax.plot([], [], 
                color=style['color'], 
                linestyle=style['linestyle'], 
                marker=style['marker'], 
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                label=style['label'])[0]
        lines.append(line)
    
    # Remove the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create legend with larger font and proper positioning
    legend = ax.legend(handles=lines, 
                      bbox_to_anchor=(0.5, 0.5), 
                      loc='center', 
                      fontsize=24,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      ncol=2,
                      columnspacing=2,
                      handlelength=3,
                      handletextpad=1)
    
    # Make legend frame visible
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    
    # Save legend
    legend_path = os.path.join(output_dir, "legend.pdf")
    fig.savefig(legend_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved legend: {legend_path}")

def create_average_performance_plot(data, dataset, output_dir):
    """
    Create average performance plot with standard deviation for a dataset showing all algorithms.
    """
    if dataset not in data:
        logger.warning(f"No data found for dataset {dataset}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get all algorithms for this dataset
    algorithms = list(data[dataset].keys())
    logger.info(f"Creating average plot for dataset {dataset} with algorithms: {algorithms}")
    
    for algorithm in algorithms:
        if algorithm not in algorithm_styles:
            logger.warning(f"No style defined for algorithm {algorithm}")
            continue
        
        style = algorithm_styles[algorithm]
        
        example_counts, average_losses, std_losses = calculate_average_performance_with_std(data, dataset, algorithm)
        
        if example_counts is None or average_losses is None:
            logger.warning(f"No data for {dataset}_{algorithm}")
            continue
        
        # Remove NaN values
        valid_indices = ~np.isnan(average_losses)
        if not np.any(valid_indices):
            logger.warning(f"No valid data for {dataset}_{algorithm}")
            continue
        
        example_counts = np.array(example_counts)[valid_indices]
        average_losses = np.array(average_losses)[valid_indices]
        std_losses = np.array(std_losses)[valid_indices]
        
        # Plot line with enhanced visibility
        plt.plot(example_counts, average_losses, 
                color=style['color'], 
                linestyle=style['linestyle'], 
                marker=style['marker'], 
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                markeredgecolor='black',
                markeredgewidth=1,
                alpha=0.9,
                label=style['label'])
        
        # Plot standard deviation as shaded area
        plt.fill_between(example_counts, 
                        average_losses - std_losses, 
                        average_losses + std_losses,
                        color=style['color'], 
                        alpha=0.15)
    
    plt.xlabel('Number of Examples')
    plt.ylabel('PV Loss')
    plt.title(f'Average Performance with Std - Dataset {dataset}')
    plt.grid(True, alpha=0.3)
    
    # Add legend positioned to avoid covering curves
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=False, 
               framealpha=0.3, edgecolor='lightgray', facecolor='white')
    
    # Format x-axis to avoid scientific notation
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dataset}_average_performance.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved average plot: {plot_path}")

def create_median_performance_plot(data, dataset, output_dir):
    """
    Create median performance plot for a dataset showing all algorithms.
    """
    if dataset not in data:
        logger.warning(f"No data found for dataset {dataset}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get all algorithms for this dataset
    algorithms = list(data[dataset].keys())
    logger.info(f"Creating median plot for dataset {dataset} with algorithms: {algorithms}")
    
    for algorithm in algorithms:
        if algorithm not in algorithm_styles:
            logger.warning(f"No style defined for algorithm {algorithm}")
            continue
        
        style = algorithm_styles[algorithm]
        
        example_counts, median_losses = calculate_median_performance(data, dataset, algorithm)
        
        if example_counts is None or median_losses is None:
            logger.warning(f"No data for {dataset}_{algorithm}")
            continue
        
        # Remove NaN values
        valid_indices = ~np.isnan(median_losses)
        if not np.any(valid_indices):
            logger.warning(f"No valid data for {dataset}_{algorithm}")
            continue
        
        example_counts = np.array(example_counts)[valid_indices]
        median_losses = np.array(median_losses)[valid_indices]
        
        plt.plot(example_counts, median_losses, 
                color=style['color'], 
                linestyle=style['linestyle'], 
                marker=style['marker'], 
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                markeredgecolor='black',
                markeredgewidth=1,
                alpha=0.9,
                label=style['label'])
    
    plt.xlabel('Number of Examples')
    plt.ylabel('PV Loss')
    plt.title(f'Median Performance - Dataset {dataset}')
    plt.grid(True, alpha=0.3)
    
    # Add legend positioned to avoid covering curves
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=False, 
               framealpha=0.3, edgecolor='lightgray', facecolor='white')
    
    # Format x-axis to avoid scientific notation
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dataset}_median_performance.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved median plot: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Process CSV files and generate performance plots and comparison tables')
    parser.add_argument('input_dir', help='Directory containing CSV files')
    parser.add_argument('output_dir', help='Directory to save plots and tables')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Processing CSV files from: {args.input_dir}")
    logger.info(f"Saving plots to: {args.output_dir}")
    
    # Load data
    data = load_csv_data(args.input_dir)
    
    if not data:
        logger.error("No data loaded! Check input directory.")
        return
    
    # Get all datasets
    datasets = list(data.keys())
    logger.info(f"Found data for {len(datasets)} datasets: {datasets}")
    
    # Create legend plot
    create_legend_plot(args.output_dir)
    
    # Create plots for each dataset
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        
        # Create average performance plot (with std)
        create_average_performance_plot(data, dataset, args.output_dir)
        
        # Create median performance plot
        create_median_performance_plot(data, dataset, args.output_dir)
    
    # Create comparison table
    logger.info("Creating comparison table...")
    comparison_data = calculate_final_loss_differences(data)
    create_comparison_table(comparison_data, args.output_dir)
    
    logger.info(f"Completed! Generated {len(datasets) * 2 + 1} plots, 2 visual tables, 2 CSV files, and 1 text summary in {args.output_dir}")

if __name__ == '__main__':
    main() 