import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Optional, Union
import torch


def load_metadata(metadata_path: str) -> Dict:
    """
    Load metadata from a JSON file.

    Args:
        metadata_path: Path to the metadata.json file

    Returns:
        Dictionary containing metadata information
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def create_feature_mapping(metadata: Dict) -> Dict[str, str]:
    """
    Create a mapping from feature indices to meaningful names based on metadata.

    Args:
        metadata: Dictionary containing metadata with 'env_state' information

    Returns:
        Dictionary mapping feature indices to feature names
    """
    feature_mapping = {}

    if 'env_state' not in metadata:
        print("Warning: No 'env_state' found in metadata")
        return feature_mapping

    env_state = metadata['env_state']

    # Create mapping for each feature
    for idx, feature_info in enumerate(env_state):
        if isinstance(feature_info, list) and len(feature_info) == 2:
            instrument, feature_type = feature_info
            # Create a meaningful name
            feature_name = f"{instrument}_{feature_type}"
            feature_mapping[str(idx)] = feature_name
        else:
            # Fallback for unexpected format
            feature_mapping[str(idx)] = f"feature_{idx}"

    return feature_mapping


def group_features_by_instrument(metadata: Dict) -> Dict[str, List[int]]:
    """
    Group feature indices by instrument for better organization.

    Args:
        metadata: Dictionary containing metadata with 'env_state' information

    Returns:
        Dictionary mapping instrument names to lists of feature indices
    """
    instrument_groups = {}

    if 'env_state' not in metadata:
        return instrument_groups

    env_state = metadata['env_state']

    for idx, feature_info in enumerate(env_state):
        if isinstance(feature_info, list) and len(feature_info) == 2:
            instrument, feature_type = feature_info
            if instrument not in instrument_groups:
                instrument_groups[instrument] = []
            instrument_groups[instrument].append(idx)

    return instrument_groups


def create_feature_importance_heatmap(
        shap_values: np.ndarray,
        feature_names: List[str],
        metadata: Dict,
        output_path: Optional[str] = None,
        title: str = "Feature Importance Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        max_features: int = 50
) -> plt.Figure:
    """
    Create a feature importance heatmap with meaningful feature names.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names (can be generic indices)
        metadata: Dictionary containing metadata for feature mapping
        output_path: Optional path to save the plot
        title: Title for the plot
        figsize: Figure size as (width, height)
        max_features: Maximum number of features to display

    Returns:
        matplotlib Figure object
    """
    # Create feature mapping
    feature_mapping = create_feature_mapping(metadata)

    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Create DataFrame with meaningful feature names
    feature_data = []
    for idx, (original_name, shap_val) in enumerate(zip(feature_names, mean_shap_values)):
        # Map to meaningful name if available
        meaningful_name = feature_mapping.get(str(idx), original_name)
        feature_data.append({
            'original_index': idx,
            'original_name': original_name,
            'meaningful_name': meaningful_name,
            'shap_value': shap_val
        })

    df_features = pd.DataFrame(feature_data)

    # Sort by SHAP value and take top features
    df_features = df_features.sort_values('shap_value', ascending=False).head(max_features)

    # Create heatmap data
    heatmap_data = df_features[['meaningful_name', 'shap_value']].set_index('meaningful_name')

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt='.4f',
        cmap='viridis',
        cbar_kws={'label': 'Mean |SHAP Value|'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_instrument_grouped_heatmap(
        shap_values: np.ndarray,
        metadata: Dict,
        output_path: Optional[str] = None,
        title: str = "Feature Importance by Instrument",
        figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a feature importance heatmap grouped by instrument.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        metadata: Dictionary containing metadata for feature mapping
        output_path: Optional path to save the plot
        title: Title for the plot
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """
    # Group features by instrument
    instrument_groups = group_features_by_instrument(metadata)
    feature_mapping = create_feature_mapping(metadata)

    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Create DataFrame grouped by instrument
    instrument_data = []
    for instrument, feature_indices in instrument_groups.items():
        for idx in feature_indices:
            if idx < len(mean_shap_values):
                meaningful_name = feature_mapping.get(str(idx), f"feature_{idx}")
                instrument_data.append({
                    'instrument': instrument,
                    'feature_name': meaningful_name,
                    'shap_value': mean_shap_values[idx]
                })

    df_instruments = pd.DataFrame(instrument_data)

    # Pivot data for heatmap
    heatmap_data = df_instruments.pivot_table(
        values='shap_value',
        index='instrument',
        columns='feature_name',
        aggfunc='mean'
    ).fillna(0)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.4f',
        cmap='viridis',
        cbar_kws={'label': 'Mean |SHAP Value|'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Instruments', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_feature_importance_summary(
        shap_values: np.ndarray,
        metadata: Dict,
        output_path: Optional[str] = None,
        title: str = "Feature Importance Summary",
        figsize: Tuple[int, int] = (12, 8),
        top_n: int = 20
) -> plt.Figure:
    """
    Create a comprehensive feature importance summary with multiple visualizations.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        metadata: Dictionary containing metadata for feature mapping
        output_path: Optional path to save the plot
        title: Title for the plot
        figsize: Figure size as (width, height)
        top_n: Number of top features to display

    Returns:
        matplotlib Figure object
    """
    feature_mapping = create_feature_mapping(metadata)

    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Create DataFrame with meaningful feature names
    feature_data = []
    for idx, shap_val in enumerate(mean_shap_values):
        meaningful_name = feature_mapping.get(str(idx), f"feature_{idx}")
        feature_data.append({
            'feature_name': meaningful_name,
            'shap_value': shap_val,
            'original_index': idx
        })

    df_features = pd.DataFrame(feature_data)
    df_features = df_features.sort_values('shap_value', ascending=False).head(top_n)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Bar plot of top features
    bars = ax1.barh(range(len(df_features)), df_features['shap_value'])
    ax1.set_yticks(range(len(df_features)))
    ax1.set_yticklabels(df_features['feature_name'])
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title(f'Top {top_n} Most Important Features')
    ax1.invert_yaxis()

    # Color bars by instrument (if possible)
    instrument_groups = group_features_by_instrument(metadata)
    colors = []
    for feature_name in df_features['feature_name']:
        color_found = False
        for instrument in instrument_groups.keys():
            if instrument in feature_name:
                colors.append(plt.cm.Set3(len(colors) % 12))
                color_found = True
                break
        if not color_found:
            colors.append('gray')

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Feature type distribution
    feature_types = []
    for feature_name in df_features['feature_name']:
        if '_' in feature_name:
            feature_type = feature_name.split('_')[-1]
        else:
            feature_type = 'Unknown'
        feature_types.append(feature_type)

    feature_type_counts = pd.Series(feature_types).value_counts()
    ax2.pie(feature_type_counts.values, labels=feature_type_counts.index, autopct='%1.1f%%')
    ax2.set_title('Distribution of Feature Types in Top Features')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def analyze_model_feature_importance(
        model_path: str,
        shap_values: np.ndarray,
        output_dir: str,
        model_name: str = "model"
) -> Dict:
    """
    Comprehensive feature importance analysis for a trained model.

    Args:
        model_path: Path to the model directory containing metadata.json
        shap_values: SHAP values array (n_samples, n_features)
        output_dir: Directory to save output plots
        model_name: Name for the model (used in file names)

    Returns:
        Dictionary containing analysis results
    """
    import os

    # Load metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    metadata = load_metadata(metadata_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate feature names (0, 1, 2, ...)
    feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    # Create different visualizations
    results = {}

    # 1. Standard feature importance heatmap
    fig1 = create_feature_importance_heatmap(
        shap_values, feature_names, metadata,
        output_path=os.path.join(output_dir, f'{model_name}_feature_importance_heatmap.png'),
        title=f"Feature Importance Heatmap - {model_name}"
    )
    results['heatmap_fig'] = fig1

    # 2. Instrument-grouped heatmap
    fig2 = create_instrument_grouped_heatmap(
        shap_values, metadata,
        output_path=os.path.join(output_dir, f'{model_name}_instrument_grouped_heatmap.png'),
        title=f"Feature Importance by Instrument - {model_name}"
    )
    results['instrument_heatmap_fig'] = fig2

    # 3. Comprehensive summary
    fig3 = create_feature_importance_summary(
        shap_values, metadata,
        output_path=os.path.join(output_dir, f'{model_name}_feature_importance_summary.png'),
        title=f"Feature Importance Summary - {model_name}"
    )
    results['summary_fig'] = fig3

    # 4. Save feature mapping for reference
    feature_mapping = create_feature_mapping(metadata)
    mapping_df = pd.DataFrame([
        {'original_index': k, 'meaningful_name': v}
        for k, v in feature_mapping.items()
    ])
    mapping_df.to_csv(os.path.join(output_dir, f'{model_name}_feature_mapping.csv'), index=False)

    # 5. Save top features analysis
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    feature_data = []
    for idx, shap_val in enumerate(mean_shap_values):
        meaningful_name = feature_mapping.get(str(idx), f"feature_{idx}")
        feature_data.append({
            'original_index': idx,
            'meaningful_name': meaningful_name,
            'shap_value': shap_val
        })

    features_df = pd.DataFrame(feature_data)
    features_df = features_df.sort_values('shap_value', ascending=False)
    features_df.to_csv(os.path.join(output_dir, f'{model_name}_feature_importance_ranking.csv'), index=False)

    results['feature_mapping'] = feature_mapping
    results['top_features'] = features_df.head(20)

    return results