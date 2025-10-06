#!/usr/bin/env python3
"""
Statistical Significance Analysis Script

This script performs statistical significance analysis for pairwise algorithm comparisons.
For each pair of algorithms and each dataset, it computes statistical significance
of performance differences using the specified formula with normal CDF.

Usage:
    python3 statistical_significance_analysis.py input_csv_dir output_dir

Input: Directory containing CSV files from experiments
Output: Significance tables showing wins/losses between algorithm pairs
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from scipy.special import erf
import itertools
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def significance(p_a, p_b, n):
    """
    Compute statistical significance using the reference implementation.
    
    This matches the implementation in rank_algos.py:
    pval = 1 - erf(np.abs(diff / se))
    
    Args:
        p_a: PV-loss for algorithm a
        p_b: PV-loss for algorithm b  
        n: Number of examples
    
    Returns:
        p-value for two-tailed test (matching reference implementation)
    """
    diff = p_a - p_b
    se = 1e-6 + np.sqrt((p_a * (1 - p_a) / n) + (p_b * (1 - p_b) / n))
    pval = 1 - erf(np.abs(diff / se))
    return pval



def load_csv_data(input_dir):
    """
    Load all CSV files and organize by dataset and algorithm.
    
    Returns:
        dict: {dataset: {algorithm: {permutation_id: final_loss}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            # Parse filename: dataset_algorithm.csv
            filename_parts = csv_file.replace('.csv', '').split('_')
            if len(filename_parts) < 2:
                continue
                
            # Find algorithm name
            algorithm = None
            algorithms = ['Supervised', 'SquareCB', 'FastCB', 'ADACB', 'RegCB', 'OPOCMAB']
            for algo in algorithms:
                if csv_file.endswith(f"_{algo}.csv"):
                    algorithm = algo
                    dataset = csv_file.replace(f"_{algo}.csv", "")
                    break
            
            if algorithm is None:
                continue
            
            # Load CSV and get final losses for each permutation
            csv_path = os.path.join(input_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # Convert to numeric
            df['progressive_loss'] = pd.to_numeric(df['progressive_loss'], errors='coerce')
            df['permutation_id'] = pd.to_numeric(df['permutation_id'], errors='coerce')
            df['example_count'] = pd.to_numeric(df['example_count'], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna(subset=['progressive_loss', 'permutation_id', 'example_count'])
            
            # Get final loss for each permutation (last example)
            for perm_id, group in df.groupby('permutation_id'):
                if len(group) > 0:
                    final_loss = group['progressive_loss'].iloc[-1]  # Last progressive loss
                    max_examples = group['example_count'].max()  # Number of examples
                    data[dataset][algorithm][perm_id] = {
                        'final_loss': final_loss,
                        'num_examples': max_examples
                    }
            
            logger.info(f"Loaded {dataset}_{algorithm}: {len(data[dataset][algorithm])} permutations")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue
    
    return data

def compute_pairwise_significance(data, alpha=0.05):
    """
    Compute pairwise statistical significance for all algorithm pairs.
    
    Uses the same methodology as the reference implementation:
    - Two-tailed significance test with erf function
    - Directional winner determination based on mean losses
    
    Args:
        data: Loaded CSV data
        alpha: Significance level (default: 0.05)
    
    Returns:
        dict: Results for each dataset and algorithm pair
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    # Get all algorithms
    all_algorithms = set()
    for dataset_data in data.values():
        all_algorithms.update(dataset_data.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    logger.info(f"Analyzing {len(all_algorithms)} algorithms: {all_algorithms}")
    logger.info("Using two-tailed significance test (matches reference implementation)")
    

    
    for dataset in data.keys():
        dataset_data = data[dataset]
        
        # Get algorithms available for this dataset
        available_algorithms = [algo for algo in all_algorithms if algo in dataset_data]
        
        logger.info(f"Dataset {dataset}: {len(available_algorithms)} algorithms available")
        
        # Compare all pairs of algorithms
        for algo1, algo2 in itertools.combinations(available_algorithms, 2):
            # Get losses for each permutation
            losses1 = []
            losses2 = []
            num_examples = None
            
            # Get permutations available for both algorithms
            common_perms = set(dataset_data[algo1].keys()) & set(dataset_data[algo2].keys())
            
            for perm_id in common_perms:
                loss1 = dataset_data[algo1][perm_id]['final_loss']
                loss2 = dataset_data[algo2][perm_id]['final_loss']
                losses1.append(loss1)
                losses2.append(loss2)
                
                # Use number of examples (should be same for both algorithms)
                if num_examples is None:
                    num_examples = dataset_data[algo1][perm_id]['num_examples']
            
            if len(losses1) >= 3:  # Minimum samples for meaningful comparison
                # Compute means
                mean1 = np.mean(losses1)
                mean2 = np.mean(losses2)
                
                # Compute statistical significance (matches reference implementation)
                p_value = significance(mean1, mean2, num_examples)
                is_significant = p_value < alpha
                
                # Determine winner based on means (like reference)
                if is_significant:
                    if mean1 < mean2:
                        winner = algo1
                    elif mean2 < mean1:
                        winner = algo2
                    else:
                        winner = 'tie'
                else:
                    winner = 'no_significance'
                
                # Store one result (same p-value for both directions, like reference)
                results[dataset][f"{algo1}_vs_{algo2}"] = {
                    'algo1': algo1,
                    'algo2': algo2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'winner': winner,
                    'num_permutations': len(losses1),
                    'num_examples': num_examples
                }
                
                logger.info(f"{dataset}: {algo1} vs {algo2} - p={p_value:.6f}, winner={winner}")
            
            else:
                logger.warning(f"{dataset}: {algo1} vs {algo2} - insufficient data ({len(losses1)} permutations)")
    
    return results

def create_wins_losses_table(significance_results, all_algorithms):
    """
    Create a wins/losses table showing significant wins for each algorithm pair.
    Returns both per-dataset and summary tables.
    """
    # Initialize per-dataset and summary matrices
    per_dataset_wins = {}
    summary_wins_matrix = defaultdict(lambda: defaultdict(int))
    summary_total_comparisons = defaultdict(lambda: defaultdict(int))
    
    # Process each dataset separately
    for dataset, dataset_results in significance_results.items():
        # Initialize this dataset's matrix
        dataset_wins_matrix = defaultdict(lambda: defaultdict(int))
        dataset_total_comparisons = defaultdict(lambda: defaultdict(int))
        
        for comparison, result in dataset_results.items():
            algo1 = result['algo1']
            algo2 = result['algo2']
            winner = result['winner']
            
            # Count for this dataset
            dataset_total_comparisons[algo1][algo2] += 1
            dataset_total_comparisons[algo2][algo1] += 1
            
            # Count for summary across all datasets
            summary_total_comparisons[algo1][algo2] += 1
            summary_total_comparisons[algo2][algo1] += 1
            
            if winner == algo1:
                dataset_wins_matrix[algo1][algo2] += 1
                summary_wins_matrix[algo1][algo2] += 1
            elif winner == algo2:
                dataset_wins_matrix[algo2][algo1] += 1
                summary_wins_matrix[algo2][algo1] += 1
        
        # Create DataFrame for this dataset
        dataset_data = []
        for algo1 in all_algorithms:
            row = {'Algorithm': algo1}
            for algo2 in all_algorithms:
                if algo1 != algo2:
                    wins = dataset_wins_matrix[algo1][algo2]
                    total = dataset_total_comparisons[algo1][algo2]
                    if total > 0:
                        row[algo2] = f"{wins}/{total}"
                    else:
                        row[algo2] = "N/A"
                else:
                    row[algo2] = "-"
            dataset_data.append(row)
        
        per_dataset_wins[dataset] = pd.DataFrame(dataset_data)
    
    # Create summary DataFrame across all datasets
    summary_data = []
    for algo1 in all_algorithms:
        row = {'Algorithm': algo1}
        for algo2 in all_algorithms:
            if algo1 != algo2:
                wins = summary_wins_matrix[algo1][algo2]
                total = summary_total_comparisons[algo1][algo2]
                row[algo2] = f"{wins}/{total}"
            else:
                row[algo2] = "-"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    return per_dataset_wins, summary_df

def create_visual_wins_losses_table(wins_losses_df, output_dir):
    """
    Create a visual table plot for wins/losses matrix.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=wins_losses_df.values,
                    colLabels=wins_losses_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code cells based on win rates
    for i in range(len(wins_losses_df)):
        for j in range(1, len(wins_losses_df.columns)):  # Skip Algorithm column
            cell_value = wins_losses_df.iloc[i, j]
            
            if cell_value != "-":
                try:
                    # Parse "wins/total" format
                    wins, total = map(int, cell_value.split('/'))
                    win_rate = wins / total if total > 0 else 0
                    
                    # Color coding based on win rate
                    if win_rate >= 0.7:
                        # High win rate - dark green
                        table[(i+1, j)].set_facecolor('#228B22')
                        table[(i+1, j)].set_text_props(color='white', weight='bold')
                    elif win_rate >= 0.5:
                        # Good win rate - light green
                        table[(i+1, j)].set_facecolor('#90EE90')
                        table[(i+1, j)].set_text_props(weight='bold')
                    elif win_rate >= 0.3:
                        # Moderate win rate - yellow
                        table[(i+1, j)].set_facecolor('#FFFFE0')
                    else:
                        # Low win rate - light red
                        table[(i+1, j)].set_facecolor('#FFB6C1')
                except:
                    # Handle parsing errors
                    table[(i+1, j)].set_facecolor('#F0F0F0')
            else:
                # Diagonal cells
                table[(i+1, j)].set_facecolor('#D3D3D3')
    
    # Style header row
    for j in range(len(wins_losses_df.columns)):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style algorithm column
    for i in range(1, len(wins_losses_df) + 1):
        table[(i, 0)].set_facecolor('#E0E0E0')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Statistical Significance: Wins/Losses Matrix\n(p < 0.05)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#228B22', label='High Win Rate (≥70%)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#90EE90', label='Good Win Rate (50-69%)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFFFE0', label='Moderate Win Rate (30-49%)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Low Win Rate (<30%)', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "statistical_significance_wins_losses_table.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visual wins/losses table: {plot_path}")

def create_visual_p_values_table(p_values_df, output_dir):
    """
    Create a visual table plot for p-values matrix.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=p_values_df.values,
                    colLabels=p_values_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code cells based on p-values
    for i in range(len(p_values_df)):
        for j in range(1, len(p_values_df.columns)):  # Skip Algorithm column
            cell_value = p_values_df.iloc[i, j]
            
            if cell_value not in ["-", "N/A"]:
                try:
                    p_value = float(cell_value)
                    
                    # Color coding based on p-value
                    if p_value < 0.001:
                        # Highly significant - dark green
                        table[(i+1, j)].set_facecolor('#006400')
                        table[(i+1, j)].set_text_props(color='white', weight='bold')
                    elif p_value < 0.01:
                        # Very significant - medium green
                        table[(i+1, j)].set_facecolor('#228B22')
                        table[(i+1, j)].set_text_props(color='white', weight='bold')
                    elif p_value < 0.05:
                        # Significant - light green
                        table[(i+1, j)].set_facecolor('#90EE90')
                        table[(i+1, j)].set_text_props(weight='bold')
                    elif p_value < 0.1:
                        # Marginally significant - yellow
                        table[(i+1, j)].set_facecolor('#FFFFE0')
                    else:
                        # Not significant - light red
                        table[(i+1, j)].set_facecolor('#FFB6C1')
                except:
                    # Handle parsing errors
                    table[(i+1, j)].set_facecolor('#F0F0F0')
            else:
                # Diagonal cells or N/A
                table[(i+1, j)].set_facecolor('#D3D3D3')
    
    # Style header row
    for j in range(len(p_values_df.columns)):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style algorithm column
    for i in range(1, len(p_values_df) + 1):
        table[(i, 0)].set_facecolor('#E0E0E0')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Statistical Significance: Average P-Values Matrix\n(Across All Datasets)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#006400', label='Highly Sig. (p<0.001)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#228B22', label='Very Sig. (p<0.01)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#90EE90', label='Significant (p<0.05)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFFFE0', label='Marginal (p<0.1)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Not Sig. (p≥0.1)', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "statistical_significance_p_values_table.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visual p-values table: {plot_path}")

def create_visual_p_values_table_per_dataset(p_values_df, dataset, output_dir):
    """
    Create a visual table plot for p-values matrix for a specific dataset.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=p_values_df.values,
                    colLabels=p_values_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code cells based on p-values
    for i in range(len(p_values_df)):
        for j in range(1, len(p_values_df.columns)):  # Skip Algorithm column
            cell_value = p_values_df.iloc[i, j]
            
            if cell_value not in ["-", "N/A"]:
                try:
                    p_value = float(cell_value)
                    
                    # Color coding based on p-value
                    if p_value < 0.001:
                        color = '#8B0000'  # Dark red - very highly significant
                    elif p_value < 0.01:
                        color = '#FF4500'  # Orange red - highly significant  
                    elif p_value < 0.05:
                        color = '#FFD700'  # Gold - significant
                    elif p_value < 0.1:
                        color = '#FFFFE0'  # Light yellow - marginally significant
                    else:
                        color = '#F0F0F0'  # Light gray - not significant
                        
                    table[(i+1, j)].set_facecolor(color)
                    
                except ValueError:
                    table[(i+1, j)].set_facecolor('#FFFFFF')  # White for invalid values
            else:
                table[(i+1, j)].set_facecolor('#FFFFFF')  # White for "-" and "N/A"
    
    # Style header row
    for j in range(len(p_values_df.columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style algorithm column
    for i in range(1, len(p_values_df) + 1):
        table[(i, 0)].set_facecolor('#4A90E2')
        table[(i, 0)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Statistical Significance: P-Values Matrix for {dataset}\n(Two-tailed test)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#8B0000', label='p < 0.001 (Very Highly Significant)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FF4500', label='p < 0.01 (Highly Significant)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFD700', label='p < 0.05 (Significant)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#FFFFE0', label='p < 0.1 (Marginally Significant)', edgecolor='black'),
        Rectangle((0,0),1,1, facecolor='#F0F0F0', label='p ≥ 0.1 (Not Significant)', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"statistical_significance_p_values_table_{dataset}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visual p-values table for {dataset}: {plot_path}")

def create_p_values_table(significance_results, all_algorithms):
    """
    Create p-values tables for each dataset separately.
    """
    per_dataset_p_values = {}
    
    for dataset, dataset_results in significance_results.items():
        # Initialize p-values matrix for this dataset
        p_values_matrix = defaultdict(lambda: defaultdict(lambda: np.nan))
        
        for comparison, result in dataset_results.items():
            algo1 = result['algo1']
            algo2 = result['algo2']
            p_value = result['p_value']
            
            # Set symmetric p-values (two-tailed test)
            p_values_matrix[algo1][algo2] = p_value
            p_values_matrix[algo2][algo1] = p_value
        
        # Create DataFrame for this dataset
        dataset_data = []
        for algo1 in all_algorithms:
            row = {'Algorithm': algo1}
            for algo2 in all_algorithms:
                if algo1 != algo2:
                    p_val = p_values_matrix[algo1][algo2]
                    if not np.isnan(p_val):
                        row[algo2] = f"{p_val:.4f}"
                    else:
                        row[algo2] = "N/A"
                else:
                    row[algo2] = "-"
            dataset_data.append(row)
        
        per_dataset_p_values[dataset] = pd.DataFrame(dataset_data)
    
    return per_dataset_p_values

def get_short_algorithm_name(algo_name):
    """
    Convert algorithm names to shorter versions for table display.
    """
    name_mapping = {
        'ADACB': 'Ada',
        'AdaCB': 'Ada', 
        'SquareCB': 'Square',
        'FastCB': 'Fast',
        'RegCB': 'Reg',
        'OPO-CMAB': 'Opo',
        'OPOCMAB': 'Opo',
        'OPO_CMAB': 'Opo',
        'opo-cmab': 'Opo',
        'opocmab': 'Opo',
        'Supervised': 'Sup'
    }
    return name_mapping.get(algo_name, algo_name)

def create_comprehensive_p_values_table(significance_results, all_algorithms):
    """
    Create a comprehensive p-values table with datasets as rows and unique algorithm pairs as columns.
    Since p-values are symmetric (two-tailed test), we only include each pair once (A vs B, not B vs A).
    """
    # Generate unique algorithm pairs (combinations, not permutations)
    algorithm_pairs = list(itertools.combinations(all_algorithms, 2))
    
    # Create shorter pair labels
    pair_labels = []
    for pair in algorithm_pairs:
        short1 = get_short_algorithm_name(pair[0])
        short2 = get_short_algorithm_name(pair[1])
        pair_labels.append(f"{short1},{short2}")
    
    # Initialize data structure
    data = []
    datasets = sorted(significance_results.keys())
    
    for dataset in datasets:
        row = {'Dataset': dataset}
        dataset_results = significance_results[dataset]
        
        for i, pair in enumerate(algorithm_pairs):
            algo1, algo2 = pair
            pair_label = pair_labels[i]
            
            # Look for this comparison in either direction
            p_value = None
            for comparison, result in dataset_results.items():
                if ((result['algo1'] == algo1 and result['algo2'] == algo2) or
                    (result['algo1'] == algo2 and result['algo2'] == algo1)):
                    p_value = result['p_value']
                    break
            
            if p_value is not None:
                row[pair_label] = f"{p_value:.4f}"
            else:
                row[pair_label] = "N/A"
        
        data.append(row)
    
    return pd.DataFrame(data)

def create_comprehensive_wins_table(significance_results, all_algorithms):
    """
    Create a comprehensive wins table with datasets as rows and unique algorithm pairs as columns.
    Shows which algorithm wins significantly for each dataset and pair.
    """
    # Generate unique algorithm pairs
    algorithm_pairs = list(itertools.combinations(all_algorithms, 2))
    
    # Create shorter pair labels
    pair_labels = []
    for pair in algorithm_pairs:
        short1 = get_short_algorithm_name(pair[0])
        short2 = get_short_algorithm_name(pair[1])
        pair_labels.append(f"{short1},{short2}")
    
    # Initialize data structure
    data = []
    datasets = sorted(significance_results.keys())
    
    for dataset in datasets:
        row = {'Dataset': dataset}
        dataset_results = significance_results[dataset]
        
        for i, pair in enumerate(algorithm_pairs):
            algo1, algo2 = pair
            pair_label = pair_labels[i]
            
            # Look for this comparison in either direction
            winner = None
            is_significant = False
            for comparison, result in dataset_results.items():
                if ((result['algo1'] == algo1 and result['algo2'] == algo2) or
                    (result['algo1'] == algo2 and result['algo2'] == algo1)):
                    winner = result['winner']
                    is_significant = result['is_significant']
                    break
            
            if is_significant and winner in [algo1, algo2]:
                # Convert winner to short name
                short_winner = get_short_algorithm_name(winner)
                row[pair_label] = short_winner
            elif winner == 'tie':
                row[pair_label] = "Tie"
            elif winner == 'no_significance':
                row[pair_label] = "No sig."
            else:
                row[pair_label] = "N/A"
        
        data.append(row)
    
    return pd.DataFrame(data)

def create_visual_comprehensive_table(df, title, output_filename, output_dir, is_p_values=True):
    """
    Create a visual table for the comprehensive format (datasets × algorithm pairs).
    """
    fig, ax = plt.subplots(figsize=(20, max(8, len(df) * 0.8)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    
    if is_p_values:
        # Color code cells based on p-values
        for i in range(len(df)):
            for j in range(1, len(df.columns)):  # Skip Dataset column
                cell_value = df.iloc[i, j]
                
                if cell_value not in ["N/A"]:
                    try:
                        p_value = float(cell_value)
                        
                        # Color coding based on p-value
                        if p_value < 0.001:
                            color = '#8B0000'  # Dark red - very highly significant
                        elif p_value < 0.01:
                            color = '#FF4500'  # Orange red - highly significant  
                        elif p_value < 0.05:
                            color = '#FFD700'  # Gold - significant
                        elif p_value < 0.1:
                            color = '#FFFFE0'  # Light yellow - marginally significant
                        else:
                            color = '#F0F0F0'  # Light gray - not significant
                            
                        table[(i+1, j)].set_facecolor(color)
                        
                    except ValueError:
                        table[(i+1, j)].set_facecolor('#FFFFFF')  # White for invalid values
                else:
                    table[(i+1, j)].set_facecolor('#FFFFFF')  # White for N/A
        
        # Add legend for p-values
        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#8B0000', label='p < 0.001', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#FF4500', label='p < 0.01', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#FFD700', label='p < 0.05', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#FFFFE0', label='p < 0.1', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#F0F0F0', label='p ≥ 0.1', edgecolor='black')
        ]
    else:
        # Color code cells based on winners
        for i in range(len(df)):
            for j in range(1, len(df.columns)):  # Skip Dataset column
                cell_value = df.iloc[i, j]
                
                if cell_value == "No sig.":
                    color = '#F0F0F0'  # Light gray
                elif cell_value == "Tie":
                    color = '#FFFFE0'  # Light yellow
                elif cell_value == "N/A":
                    color = '#FFFFFF'  # White
                else:
                    color = '#90EE90'  # Light green for significant wins
                    
                table[(i+1, j)].set_facecolor(color)
        
        # Add legend for wins
        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#90EE90', label='Significant Win', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#FFFFE0', label='Tie', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#F0F0F0', label='No Significance', edgecolor='black'),
            Rectangle((0,0),1,1, facecolor='#FFFFFF', label='N/A', edgecolor='black')
        ]
    
    # Style header row
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#4A90E2')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style dataset column
    for i in range(1, len(df) + 1):
        table[(i, 0)].set_facecolor('#4A90E2')
        table[(i, 0)].set_text_props(weight='bold', color='white')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_elements))
    
    # Save the plot
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive table: {plot_path}")

def save_results(significance_results, output_dir, args):
    """
    Save all results to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all algorithms
    all_algorithms = set()
    for dataset_results in significance_results.values():
        for result in dataset_results.values():
            all_algorithms.add(result['algo1'])
            all_algorithms.add(result['algo2'])
    all_algorithms = sorted(list(all_algorithms))
    
    # Create wins/losses table
    per_dataset_wins, summary_wins_df = create_wins_losses_table(significance_results, all_algorithms)
    
    # Save per-dataset wins/losses tables
    for dataset, df in per_dataset_wins.items():
        dataset_path = os.path.join(output_dir, f"statistical_significance_wins_losses_{dataset}.csv")
        df.to_csv(dataset_path, index=False)
        logger.info(f"Saved per-dataset wins/losses table for {dataset}: {dataset_path}")

    # Create summary wins/losses table
    summary_wins_path = os.path.join(output_dir, "statistical_significance_wins_losses_summary.csv")
    summary_wins_df.to_csv(summary_wins_path, index=False)
    logger.info(f"Saved summary wins/losses table: {summary_wins_path}")
    
    # Create p-values tables
    per_dataset_p_values = create_p_values_table(significance_results, all_algorithms)
    
    # Save per-dataset p-values tables
    for dataset, df in per_dataset_p_values.items():
        dataset_path = os.path.join(output_dir, f"statistical_significance_p_values_{dataset}.csv")
        df.to_csv(dataset_path, index=False)
        logger.info(f"Saved per-dataset p-values table for {dataset}: {dataset_path}")

    # Create comprehensive p-values table
    comprehensive_p_values_df = create_comprehensive_p_values_table(significance_results, all_algorithms)
    comprehensive_p_values_path = os.path.join(output_dir, "statistical_significance_comprehensive_p_values.csv")
    comprehensive_p_values_df.to_csv(comprehensive_p_values_path, index=False)
    logger.info(f"Saved comprehensive p-values table: {comprehensive_p_values_path}")
    create_visual_comprehensive_table(comprehensive_p_values_df, "Statistical Significance: Comprehensive P-Values Matrix", "statistical_significance_comprehensive_p_values.pdf", output_dir, is_p_values=True)

    # Create comprehensive wins table
    comprehensive_wins_df = create_comprehensive_wins_table(significance_results, all_algorithms)
    comprehensive_wins_path = os.path.join(output_dir, "statistical_significance_comprehensive_wins.csv")
    comprehensive_wins_df.to_csv(comprehensive_wins_path, index=False)
    logger.info(f"Saved comprehensive wins table: {comprehensive_wins_path}")
    create_visual_comprehensive_table(comprehensive_wins_df, "Statistical Significance: Comprehensive Wins Matrix", "statistical_significance_comprehensive_wins.pdf", output_dir, is_p_values=False)

    # Create detailed results CSV
    detailed_data = []
    for dataset, dataset_results in significance_results.items():
        for comparison, result in dataset_results.items():
            detailed_data.append({
                'Dataset': dataset,
                'Algorithm_1': result['algo1'],
                'Algorithm_2': result['algo2'],
                'Mean_Loss_1': result['mean1'],
                'Mean_Loss_2': result['mean2'],
                'P_Value': result['p_value'],
                'Is_Significant': result['is_significant'],
                'Winner': result['winner'],
                'Num_Permutations': result['num_permutations'],
                'Num_Examples': result['num_examples']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = os.path.join(output_dir, "statistical_significance_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results: {detailed_path}")
    
    # Create visual wins/losses table
    create_visual_wins_losses_table(summary_wins_df, output_dir)

    # Create visual p-values tables for each dataset
    for dataset, df in per_dataset_p_values.items():
        create_visual_p_values_table_per_dataset(df, dataset, output_dir)

    # Create summary text file
    summary_path = os.path.join(output_dir, "statistical_significance_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("METHODOLOGY:\n")
        f.write("- Statistical significance test using error function (erf)\n")
        f.write(f"- Significance level: α = {args.alpha}\n")
        f.write("- Test type: Two-tailed (matches reference implementation)\n")
        f.write("- Formula: p-value = 1 - erf(|diff| / se)\n")
        f.write("- where se = 1e-6 + sqrt((p_a(1-p_a) + p_b(1-p_b))/n) and erf is the error function\n")
        f.write("- This exactly matches the reference implementation in rank_algos.py\n")
        
        f.write("WINS/LOSSES TABLE:\n")
        f.write("Each cell shows 'wins/total_comparisons' for row algorithm vs column algorithm\n")
        f.write("-" * 50 + "\n")
        f.write(summary_wins_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("AVERAGE P-VALUES TABLE:\n")
        f.write("Each cell shows average p-value across all datasets for row vs column comparison\n")
        f.write("-" * 50 + "\n")
        # The p-values are now per-dataset, so we need to aggregate them
        # This part of the summary needs to be updated to reflect the new structure
        # For now, we'll just show the summary table
        f.write("P-values are now per-dataset. See per-dataset p-values tables.\n")
        f.write("\n\n")
        
        # Summary statistics
        total_significant = sum(1 for dataset_results in significance_results.values() 
                              for result in dataset_results.values() 
                              if result['is_significant'])
        total_comparisons = sum(len(dataset_results) for dataset_results in significance_results.values())
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Total algorithm pairs compared: {len(list(itertools.combinations(all_algorithms, 2)))}\n")
        f.write(f"Total dataset comparisons: {total_comparisons}\n")
        f.write(f"Significant differences found: {total_significant}\n")
        f.write(f"Significance rate: {total_significant/total_comparisons*100:.1f}%\n")
        
        # Algorithm win rates
        f.write("\nINDIVIDUAL ALGORITHM PERFORMANCE:\n")
        for algo in all_algorithms:
            wins = sum(1 for dataset_results in significance_results.values()
                      for result in dataset_results.values()
                      if result['winner'] == algo)
            total = sum(1 for dataset_results in significance_results.values()
                       for result in dataset_results.values()
                       if result['algo1'] == algo or result['algo2'] == algo)
            if total > 0:
                win_rate = wins / total * 100
                f.write(f"{algo}: {wins}/{total} wins ({win_rate:.1f}%)\n")
    
    logger.info(f"Saved summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Statistical significance analysis for algorithm comparisons')
    parser.add_argument('input_dir', help='Directory containing CSV files')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')

    
    args = parser.parse_args()
    
    logger.info(f"Loading data from: {args.input_dir}")
    logger.info(f"Saving results to: {args.output_dir}")
    logger.info(f"Significance level: {args.alpha}")
    logger.info("Test type: Two-tailed (matches reference implementation)")
    
    # Load data
    data = load_csv_data(args.input_dir)
    
    if not data:
        logger.error("No data loaded! Check input directory.")
        return
    
    # Compute significance
    significance_results = compute_pairwise_significance(data, args.alpha)
    
    # Save results
    save_results(significance_results, args.output_dir, args)
    
    logger.info("Analysis completed!")

if __name__ == '__main__':
    main() 