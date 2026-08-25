"""
Plotting Module for Regime Representation Analysis

This module contains visualization functions for all regime representation analyses,
including conductance plots, embeddings, confusion matrices, and calibration curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')


def plot_conductance_by_layer(
    conductance_values: Dict[str, float],
    title: str = "Layer Conductance Analysis",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show_values: bool = True
) -> None:
    """
    Plot bar chart of conductance per layer.
    
    This visualization shows how much each transformer layer contributes to
    the final prediction, revealing the level of representation at different depths.
    
    Args:
        conductance_values: Dictionary mapping layer names to conductance values
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        show_values: Whether to show values on bars
    """
    print(f"Creating conductance by layer plot...")
    
    if not conductance_values:
        print("No conductance values to plot")
        return
    
    # Extract layer indices and values
    layer_names = list(conductance_values.keys())
    layer_indices = [int(name.split('_')[-1]) if '_' in name else i 
                    for i, name in enumerate(layer_names)]
    values = list(conductance_values.values())
    
    # Sort by layer index
    sorted_data = sorted(zip(layer_indices, layer_names, values))
    layer_indices, layer_names, values = zip(*sorted_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars with gradient coloring (deeper layers = darker)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Transformer Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Conductance Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-tick labels
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"Layer {i}" for i in layer_indices], rotation=45)
    
    # Add value labels on bars if requested
    if show_values:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add interpretation text
    interpretation = ("Higher conductance indicates greater contribution to final prediction.\n"
                     "Deeper layers with high conductance suggest regime-level encoding.")
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_latent_embeddings_2d(
    embeddings: Dict[str, np.ndarray],
    regime_labels: np.ndarray,
    label_mapping: Dict[int, str],
    method: str = 'tsne',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    max_layers: int = 6
) -> None:
    """
    Plot 2D scatter of latent embeddings colored by regime.
    
    This visualization shows how well different regimes are separated in the
    learned latent representations across different transformer layers.
    
    Args:
        embeddings: Dictionary mapping layer names to 2D embedding arrays
        regime_labels: Array of regime labels for each point
        label_mapping: Mapping from label indices to regime names
        method: Embedding method used (for title)
        figsize: Figure size
        save_path: Optional path to save the figure
        max_layers: Maximum number of layers to plot
    """
    print(f"Creating latent embeddings 2D plot...")
    
    if not embeddings:
        print("No embeddings to plot")
        return
    
    # Select layers to plot (limit to max_layers)
    layer_names = list(embeddings.keys())
    if len(layer_names) > max_layers:
        # Select evenly spaced layers
        indices = np.linspace(0, len(layer_names)-1, max_layers).astype(int)
        layer_names = [layer_names[i] for i in indices]
    
    # Set up the plot grid
    n_layers = len(layer_names)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Create color palette for regimes
    unique_regimes = np.unique(regime_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
    color_map = {regime: colors[i] for i, regime in enumerate(unique_regimes)}
    
    # Plot each layer
    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        embedding = embeddings[layer_name]
        
        # Create scatter plot
        for regime in unique_regimes:
            mask = regime_labels == regime
            if mask.sum() > 0:
                regime_name = label_mapping.get(regime, f'Regime {regime}')
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=[color_map[regime]], label=regime_name, 
                          alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # Customize subplot
        ax.set_title(f'{layer_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    # Main title
    main_title = f'Latent Space Embeddings by Regime ({method.upper()})'
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add interpretation text
    interpretation = ("Points of the same color should cluster together if regimes are well-separated.\n"
                     "Deeper layers should show better regime separation.")
    fig.text(0.02, 0.02, interpretation, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_linear_probe_confusion_matrix(
    probe_results: Dict[str, Dict[str, Any]],
    label_mapping: Dict[int, str],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    max_layers: int = 6
) -> None:
    """
    Plot confusion matrices of probe predictions.
    
    This visualization shows the classification accuracy of linear probes
    trained on different layer representations.
    
    Args:
        probe_results: Dictionary of probe results by layer
        label_mapping: Mapping from label indices to regime names
        figsize: Figure size
        save_path: Optional path to save the figure
        max_layers: Maximum number of layers to plot
    """
    print(f"Creating linear probe confusion matrices...")
    
    # Filter to layers with confusion matrices
    layers_with_cm = {name: results for name, results in probe_results.items() 
                     if 'confusion_matrix' in results}
    
    if not layers_with_cm:
        print("No confusion matrices to plot")
        return
    
    # Select layers to plot
    layer_names = list(layers_with_cm.keys())
    if len(layer_names) > max_layers:
        indices = np.linspace(0, len(layer_names)-1, max_layers).astype(int)
        layer_names = [layer_names[i] for i in indices]
    
    # Set up plot grid
    n_layers = len(layer_names)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Get regime names for labels
    regime_names = [label_mapping.get(i, f'Regime {i}') for i in sorted(label_mapping.keys())]
    
    # Plot confusion matrix for each layer
    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        cm = layers_with_cm[layer_name]['confusion_matrix']
        accuracy = layers_with_cm[layer_name].get('test_accuracy', 0)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=regime_names, yticklabels=regime_names,
                   ax=ax, cbar=True, square=True)
        
        # Customize subplot
        layer_title = layer_name.replace('_', ' ').title()
        ax.set_title(f'{layer_title}\nAccuracy: {accuracy:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Regime', fontsize=10)
        ax.set_ylabel('True Regime', fontsize=10)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    # Main title
    fig.suptitle('Linear Probe Classification Results', fontsize=16, fontweight='bold', y=0.98)
    
    # Add interpretation
    interpretation = ("Darker colors indicate higher accuracy. Better regime separation\n"
                     "should result in higher diagonal values (correct classifications).")
    fig.text(0.02, 0.02, interpretation, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_head_specialization_heatmap(
    specialization_results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    normalize_by_head: bool = True
) -> None:
    """
    Plot heatmap of conductance across layers × heads.
    
    This visualization shows functional specialization where different attention
    heads focus on different types of features (price-level, momentum, volatility).
    
    Args:
        specialization_results: Dictionary of specialization scores by head and category
        figsize: Figure size
        save_path: Optional path to save the figure
        normalize_by_head: Whether to normalize scores by head (row-wise normalization)
    """
    print(f"Creating head specialization heatmap...")
    
    if not specialization_results:
        print("No specialization results to plot")
        return
    
    # Extract heads and categories
    head_names = list(specialization_results.keys())
    if not head_names:
        return
        
    categories = list(specialization_results[head_names[0]].keys())
    
    # Create data matrix
    data_matrix = np.zeros((len(head_names), len(categories)))
    for i, head_name in enumerate(head_names):
        for j, category in enumerate(categories):
            data_matrix[i, j] = specialization_results[head_name].get(category, 0)
    
    # Normalize by head if requested
    if normalize_by_head:
        # Row-wise normalization (each head sums to 1)
        row_sums = data_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        data_matrix = data_matrix / row_sums
        cbar_label = "Normalized Specialization Score"
    else:
        cbar_label = "Raw Specialization Score"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(head_names)))
    ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                      rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([head.replace('_', ' ').title() for head in head_names], 
                      fontsize=10)
    
    # Add value annotations
    for i in range(len(head_names)):
        for j in range(len(categories)):
            value = data_matrix[i, j]
            color = 'white' if value > data_matrix.max()/2 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                   color=color, fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Feature Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Heads', fontsize=12, fontweight='bold')
    ax.set_title('Attention Head Specialization by Feature Category', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(categories)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(head_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add interpretation
    interpretation = ("Warmer colors indicate higher specialization. Each row shows how much\n"
                     "an attention head focuses on different feature categories.")
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_predictive_calibration_curve(
    calibration_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_confidence_bands: bool = True
) -> None:
    """
    Plot calibration curve of future-return deciles vs. mean outcome.
    
    This visualization shows whether the latent space geometry aligns with
    prospective outcomes through a monotonic calibration relationship.
    
    Args:
        calibration_results: Dictionary containing calibration metrics and curves
        figsize: Figure size
        save_path: Optional path to save the figure
        show_confidence_bands: Whether to show confidence bands around predictions
    """
    print(f"Creating predictive calibration curve...")
    
    if 'calibration_curve' not in calibration_results:
        print("No calibration curve data to plot")
        return
    
    calibration_curve = calibration_results['calibration_curve']
    bin_centers = calibration_results['bin_centers']
    bin_counts = calibration_results.get('bin_counts', np.ones_like(calibration_curve))
    bin_std = calibration_results.get('bin_std', np.zeros_like(calibration_curve))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main calibration plot
    ax1.plot(bin_centers, calibration_curve, 'o-', linewidth=3, markersize=8, 
            color='darkblue', label='Actual Outcomes', alpha=0.8)
    
    # Perfect calibration line (y = x)
    min_val, max_val = min(bin_centers.min(), calibration_curve.min()), max(bin_centers.max(), calibration_curve.max())
    ax1.plot([min_val, max_val], [min_val, max_val], '--', color='red', 
            linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    # Add confidence bands if requested
    if show_confidence_bands and bin_std is not None:
        # Use standard error of the mean
        stderr = bin_std / np.sqrt(bin_counts)
        ax1.fill_between(bin_centers, 
                        calibration_curve - 1.96*stderr,
                        calibration_curve + 1.96*stderr,
                        alpha=0.3, color='darkblue', label='95% Confidence')
    
    # Customize main plot
    ax1.set_xlabel('Expected Return Deciles', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Realized Mean Outcome', fontsize=12, fontweight='bold')
    ax1.set_title('Predictive Calibration in Latent Space', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add calibration metrics as text
    if 'mean_calibration_error' in calibration_results:
        mce = calibration_results['mean_calibration_error']
        monotonic = calibration_results.get('is_monotonic', False)
        
        metrics_text = f"Mean Calibration Error: {mce:.4f}\nMonotonic: {'Yes' if monotonic else 'No'}"
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Bottom plot: bin counts (sample distribution)
    ax2.bar(bin_centers, bin_counts, width=np.diff(bin_centers)[0]*0.8, 
           alpha=0.7, color='lightblue', edgecolor='darkblue')
    ax2.set_xlabel('Return Deciles', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=10, fontweight='bold')
    ax2.set_title('Sample Distribution Across Bins', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add sample count annotations
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax2.text(center, count + max(bin_counts)*0.01, f'{int(count)}', 
                    ha='center', va='bottom', fontsize=9)
    
    # Add interpretation
    interpretation = ("A monotonic calibration curve indicates good latent space geometry.\n"
                     "Points closer to the diagonal line indicate better calibration.")
    fig.text(0.02, 0.48, interpretation, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def create_comprehensive_regime_analysis_report(
    conductance_values: Dict[str, float],
    embeddings: Dict[str, np.ndarray],
    regime_labels: np.ndarray,
    label_mapping: Dict[int, str],
    probe_results: Dict[str, Dict[str, Any]],
    specialization_results: Dict[str, Dict[str, float]],
    calibration_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 24)
) -> None:
    """
    Create a comprehensive report with all regime analysis visualizations.
    
    This function combines all individual plots into a single comprehensive
    figure for regime representation analysis reporting.
    
    Args:
        conductance_values: Layer conductance values
        embeddings: Latent embeddings by layer
        regime_labels: Regime labels for observations
        label_mapping: Mapping from indices to regime names
        probe_results: Linear probe classification results
        specialization_results: Head specialization analysis results
        calibration_results: Predictive calibration results
        save_dir: Directory to save individual plots and combined report
        figsize: Figure size for the combined report
    """
    print("Creating comprehensive regime analysis report...")
    
    # Create combined figure
    fig = plt.figure(figsize=figsize)
    
    # Define subplot layout (4 rows, 2 columns with some spanning)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1, 1], hspace=0.4, wspace=0.3)
    
    # 1. Layer Conductance (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if conductance_values:
        layer_names = list(conductance_values.keys())
        values = list(conductance_values.values())
        layer_indices = [int(name.split('_')[-1]) if '_' in name else i for i, name in enumerate(layer_names)]
        sorted_data = sorted(zip(layer_indices, values))
        layer_indices, values = zip(*sorted_data)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
        ax1.bar(range(len(values)), values, color=colors, alpha=0.8)
        ax1.set_title('Layer Conductance Analysis', fontweight='bold')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Conductance')
        ax1.set_xticks(range(len(values)))
        ax1.set_xticklabels([f"L{i}" for i in layer_indices])
        ax1.grid(True, alpha=0.3)
    
    # 2. Latent Embeddings (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if embeddings and len(embeddings) > 0:
        # Plot the deepest layer embedding
        layer_name = list(embeddings.keys())[-1]
        embedding = embeddings[layer_name]
        
        unique_regimes = np.unique(regime_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            mask = regime_labels == regime
            if mask.sum() > 0:
                regime_name = label_mapping.get(regime, f'Regime {regime}')
                ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[colors[i]], label=regime_name, alpha=0.7, s=20)
        
        ax2.set_title(f'Latent Embeddings - {layer_name.replace("_", " ").title()}', fontweight='bold')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # 3. Linear Probe Accuracy Summary (middle row, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    if probe_results:
        layers = list(probe_results.keys())
        accuracies = [probe_results[layer].get('test_accuracy', 0) for layer in layers]
        precisions = [probe_results[layer].get('test_precision', 0) for layer in layers]
        recalls = [probe_results[layer].get('test_recall', 0) for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.25
        
        ax3.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        ax3.bar(x, precisions, width, label='Precision', alpha=0.8, color='lightgreen')
        ax3.bar(x + width, recalls, width, label='Recall', alpha=0.8, color='salmon')
        
        ax3.set_title('Linear Probe Classification Performance', fontweight='bold')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels([layer.replace('_', ' ').title() for layer in layers], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # 4. Head Specialization Summary (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if specialization_results:
        # Create simplified specialization visualization
        head_names = list(specialization_results.keys())[:8]  # Limit for readability
        categories = list(specialization_results[head_names[0]].keys()) if head_names else []
        
        if head_names and categories:
            data_matrix = np.zeros((len(head_names), len(categories)))
            for i, head_name in enumerate(head_names):
                for j, category in enumerate(categories):
                    data_matrix[i, j] = specialization_results[head_name].get(category, 0)
            
            # Normalize by head
            row_sums = data_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            data_matrix = data_matrix / row_sums
            
            im = ax4.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_title('Head Specialization', fontweight='bold')
            ax4.set_xticks(range(len(categories)))
            ax4.set_yticks(range(len(head_names)))
            ax4.set_xticklabels([cat[:8] for cat in categories], rotation=45, fontsize=8)
            ax4.set_yticklabels([head[:12] for head in head_names], fontsize=8)
    
    # 5. Calibration Curve (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    if 'calibration_curve' in calibration_results:
        calibration_curve = calibration_results['calibration_curve']
        bin_centers = calibration_results['bin_centers']
        
        ax5.plot(bin_centers, calibration_curve, 'o-', linewidth=2, markersize=6, color='darkblue')
        
        # Perfect calibration line
        min_val, max_val = bin_centers.min(), bin_centers.max()
        ax5.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.7)
        
        ax5.set_title('Predictive Calibration', fontweight='bold')
        ax5.set_xlabel('Expected Returns')
        ax5.set_ylabel('Realized Returns')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics (bottom row, spanning both columns)
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Create summary statistics table
    summary_text = "REGIME REPRESENTATION ANALYSIS SUMMARY\n" + "="*50 + "\n"
    
    if conductance_values:
        max_conductance_layer = max(conductance_values, key=conductance_values.get)
        summary_text += f"• Highest conductance: {max_conductance_layer} ({conductance_values[max_conductance_layer]:.4f})\n"
    
    if probe_results:
        best_layer = max(probe_results, key=lambda x: probe_results[x].get('test_accuracy', 0))
        best_accuracy = probe_results[best_layer].get('test_accuracy', 0)
        summary_text += f"• Best linear probe accuracy: {best_layer} ({best_accuracy:.3f})\n"
    
    if 'mean_calibration_error' in calibration_results:
        mce = calibration_results['mean_calibration_error']
        monotonic = calibration_results.get('is_monotonic', False)
        summary_text += f"• Calibration error: {mce:.4f}, Monotonic: {'Yes' if monotonic else 'No'}\n"
    
    if regime_labels is not None:
        regime_counts = np.bincount(regime_labels)
        summary_text += f"• Regime distribution: {dict(enumerate(regime_counts))}\n"
    
    summary_text += "\nINTERPRETATION:\n"
    summary_text += "• Higher layer conductance suggests regime-level encoding\n"
    summary_text += "• Higher probe accuracy indicates better regime separability\n"
    summary_text += "• Monotonic calibration shows aligned latent geometry\n"
    summary_text += "• Head specialization reveals functional organization"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    # Main title
    fig.suptitle('Comprehensive Regime Representation Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the comprehensive report
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save combined report
        combined_path = os.path.join(save_dir, 'regime_analysis_comprehensive_report.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive report saved to: {combined_path}")
        
        # Also save individual plots
        individual_plots = [
            (plot_conductance_by_layer, conductance_values, 'conductance_by_layer.png'),
            (plot_latent_embeddings_2d, (embeddings, regime_labels, label_mapping), 'latent_embeddings_2d.png'),
            (plot_linear_probe_confusion_matrix, (probe_results, label_mapping), 'linear_probe_confusion.png'),
            (plot_head_specialization_heatmap, specialization_results, 'head_specialization_heatmap.png'),
            (plot_predictive_calibration_curve, calibration_results, 'predictive_calibration_curve.png')
        ]
        
        for plot_func, plot_data, filename in individual_plots:
            try:
                if isinstance(plot_data, tuple):
                    plot_func(*plot_data, save_path=os.path.join(save_dir, filename))
                else:
                    plot_func(plot_data, save_path=os.path.join(save_dir, filename))
                plt.close()  # Close individual plots to free memory
            except Exception as e:
                print(f"Could not save {filename}: {e}")
    else:
        plt.show()
    
    print("Comprehensive regime analysis report completed!")
