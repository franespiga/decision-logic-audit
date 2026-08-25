"""
Assessments Module for Regime Representation Analysis

This module contains functions for evaluating the quality of regime representations,
including linear probe accuracy, head specialization metrics, and predictive calibration.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from captum.attr import LayerConductance

from ablation import compute_layer_specialization_with_rollout



def evaluate_linear_probe_accuracy(
    layer_representations: Dict[str, torch.Tensor],
    regime_labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    use_cross_validation: bool = True,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate regime classification accuracy using linear probes.
    
    This function trains linear classifiers on latent representations to assess
    how well regime information is encoded in different layers. High out-of-sample
    accuracy confirms that regimes are linearly separable in the learned representation.
    
    Args:
        layer_representations: Dictionary mapping layer names to representation tensors
        regime_labels: Array of regime labels for each observation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        use_cross_validation: Whether to use cross-validation for more robust estimates
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary mapping layer names to classification metrics
    """
    print(f"Evaluating linear probe accuracy for regime classification...")
    print(f"Regime distribution: {np.bincount(regime_labels)}")
    
    results = {}
    
    for layer_name, representations in layer_representations.items():
        print(f"\nAnalyzing {layer_name}...")
        
        # Convert to numpy if needed
        if isinstance(representations, torch.Tensor):
            X = representations.cpu().numpy()
        else:
            X = representations
            
        y = regime_labels
        
        # Ensure we have enough samples for each class
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        
        if min_samples < 5:
            print(f"  Warning: Minimum class has only {min_samples} samples, skipping {layer_name}")
            continue
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        layer_results = {}
        
        if use_cross_validation:
            # Cross-validation for robust estimates
            classifier = LogisticRegression(random_state=random_state, max_iter=1000)
            
            # Accuracy scores
            cv_scores = cross_val_score(classifier, X_scaled, y, cv=cv_folds, scoring='accuracy')
            layer_results['cv_accuracy_mean'] = cv_scores.mean()
            layer_results['cv_accuracy_std'] = cv_scores.std()
            
            # Precision scores
            cv_precision = cross_val_score(classifier, X_scaled, y, cv=cv_folds, scoring='precision_macro')
            layer_results['cv_precision_mean'] = cv_precision.mean()
            layer_results['cv_precision_std'] = cv_precision.std()
            
            # Recall scores  
            cv_recall = cross_val_score(classifier, X_scaled, y, cv=cv_folds, scoring='recall_macro')
            layer_results['cv_recall_mean'] = cv_recall.mean()
            layer_results['cv_recall_std'] = cv_recall.std()
            
            print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  CV Precision: {cv_precision.mean():.3f} ± {cv_precision.std():.3f}")
            print(f"  CV Recall: {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")
        
        # Train-test split for confusion matrix and detailed metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train classifier
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test)
        
        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        layer_results.update({
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'n_test_samples': len(y_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })
        
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Test Precision: {precision:.3f}")
        print(f"  Test Recall: {recall:.3f}")
        print(f"  Test F1: {f1:.3f}")
        
        results[layer_name] = layer_results
    
    return results

def analyze_head_specialization(
    attention_activations: Dict[str, torch.Tensor],
    feature_attributions: Dict[str, torch.Tensor],
    feature_mapping: Dict[int, str],
    categories: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze head-wise specialization using conductance decomposition.

    This function examines which attention heads specialize in different types
    of features (price-level, momentum, volatility) by analyzing their attribution
    patterns across feature categories, weighted by attention activations.

    Args:
        attention_activations: Dictionary of attention patterns by head
        feature_attributions: Dictionary of feature attributions by head
        feature_mapping: Mapping from feature indices to feature names
        categories: Feature categories for specialization analysis

    Returns:
        Dictionary mapping head names to specialization scores by category
    """
    print(f"Analyzing head-wise specialization...")

    if categories is None:
        # Default feature categories
        categories = {
            'price_level': ['Close', 'Open', 'High', 'Low', 'price'],
            'momentum': ['RSI', 'MACD', 'momentum', 'ROC', 'AROONOSC'],
            'volatility': ['NATR', 'volatility', 'std', 'VIX'],
            'volume': ['Volume', 'volume', 'OBV'],
            'technical': ['CCI', 'MFI', 'WILLR', 'STOCH']
        }

    specialization_results = dict()
    specialization_rolling_results = dict()

    specialization_scores =  compute_layer_specialization_with_rollout(
        attention_activations,
        feature_attributions,
        state_idx=1, action_idx=2
    )


    for idx, (head_name, attention_weights) in enumerate(attention_activations.items()):
        print(f"\nAnalyzing {head_name}...")

        # Get feature attributions for this head
        if head_name in feature_attributions:
            attributions = feature_attributions[head_name]
        else:
            # Skip if no attributions available for this head
            continue


        weighted_attributions = specialization_scores['local_scores'].squeeze(1)
        weighted_rolling_attributions = specialization_scores['rollout_scores'].squeeze(1)

        print(weighted_attributions.shape)

        # Compute specialized scores
        head_specialization = {}
        head_rolling_specialization = {}

        for category_name, patterns in categories.items():
            category_indices = []

            # Match feature names with categories
            for feat_idx, feat_name in feature_mapping.items():
                if feat_idx < weighted_attributions.shape[-1]:  # Ensure feature index is valid
                    for pattern in patterns:
                        if pattern.lower() in feat_name.lower():
                            category_indices.append(feat_idx)

            if category_indices:
                # Compute mean weighted attribution for this category
                category_score = weighted_attributions[idx, category_indices].sum().item()

                category_rolling_score = weighted_rolling_attributions[idx, category_indices].sum().item()

                head_specialization[category_name] = category_score
                head_rolling_specialization[category_name] = category_rolling_score

                print(f"  {category_name}: local-{category_score:.6f}, rollout-{category_rolling_score:.6f} ({len(category_indices)} features)")
                print(f"  {category_name} ({len(category_indices)} features)")
            else:
                head_specialization[category_name] = 0.0
                head_rolling_specialization[category_name] = 0.0
                print(f"  {category_name}: No features found")

        specialization_results[head_name] = head_specialization
        specialization_rolling_results[head_name] = head_rolling_specialization



    return {'local': specialization_results, 'rollout': specialization_rolling_results}


def assess_predictive_calibration(
    latent_representations: torch.Tensor,
    future_returns: np.ndarray,
    bin_assignments: np.ndarray,
    bin_centers: np.ndarray,
    method: str = 'distance_based'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Assess predictive calibration in latent space.

    This function evaluates whether the latent space geometry aligns with
    prospective outcomes by analyzing the relationship between latent
    embeddings and future returns.

    Args:
        latent_representations: Latent space embeddings
        future_returns: Array of future returns for each observation
        bin_assignments: Bin assignments for return deciles
        bin_centers: Centers of return bins
        method: Method for calibration assessment

    Returns:
        Dictionary containing calibration metrics and curves
    """
    print(f"Assessing predictive calibration in latent space...")
    print(f"Latent representations shape: {latent_representations.shape}")
    print(f"Future returns shape: {future_returns.shape}")

    # Convert to numpy if needed
    if isinstance(latent_representations, torch.Tensor):
        latent_data = latent_representations.cpu().numpy()
    else:
        latent_data = latent_representations

    calibration_results = {}

    # 1. Correlation analysis
    # Compute correlation between latent dimensions and future returns
    if latent_data.ndim == 2 and latent_data.shape[1] > 1:
        dimension_correlations = []
        for dim in range(latent_data.shape[1]):
            corr_pearson, p_pearson = pearsonr(latent_data[:, dim], future_returns)
            corr_spearman, p_spearman = spearmanr(latent_data[:, dim], future_returns)

            dimension_correlations.append({
                'dimension': dim,
                'pearson_correlation': corr_pearson,
                'pearson_p_value': p_pearson,
                'spearman_correlation': corr_spearman,
                'spearman_p_value': p_spearman
            })

        calibration_results['dimension_correlations'] = dimension_correlations

        # Overall correlation (using first principal component or mean)
        if latent_data.shape[1] > 1:
            # Use first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            latent_pc1 = pca.fit_transform(latent_data).flatten()
        else:
            latent_pc1 = latent_data.flatten()

        overall_corr_pearson, _ = pearsonr(latent_pc1, future_returns)
        overall_corr_spearman, _ = spearmanr(latent_pc1, future_returns)

        calibration_results['overall_pearson_correlation'] = overall_corr_pearson
        calibration_results['overall_spearman_correlation'] = overall_corr_spearman


    # 2. Calibration curve analysis
    # Group latent embeddings by return deciles and compute mean outcomes
    n_bins = len(bin_centers)
    calibration_curve = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_std = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        bin_mask = bin_assignments == bin_idx
        if bin_mask.sum() > 0:
            calibration_curve[bin_idx] = future_returns[bin_mask].mean()
            bin_counts[bin_idx] = bin_mask.sum()
            bin_std[bin_idx] = future_returns[bin_mask].std()

    calibration_results.update({
        'calibration_curve': calibration_curve,
        'bin_centers': bin_centers,
        'bin_counts': bin_counts,
        'bin_std': bin_std
    })

    # 3. Monotonicity test
    # Check if calibration curve is monotonic (indicates good calibration)
    diffs = np.diff(calibration_curve)
    monotonic_increasing = np.all(diffs >= 0)
    monotonic_decreasing = np.all(diffs <= 0)
    monotonic = monotonic_increasing or monotonic_decreasing

    calibration_results.update({
        'is_monotonic': monotonic,
        'monotonic_increasing': monotonic_increasing,
        'monotonic_decreasing': monotonic_decreasing,
        'monotonicity_violations': np.sum((diffs[:-1] * diffs[1:]) < 0)  # Sign changes
    })

    # 4. Calibration error metrics
    # Mean absolute calibration error
    expected_outcomes = bin_centers
    actual_outcomes = calibration_curve
    calibration_error = np.mean(np.abs(expected_outcomes - actual_outcomes))

    # Weighted calibration error (weighted by bin counts)
    weights = bin_counts / bin_counts.sum()
    weighted_calibration_error = np.sum(weights * np.abs(expected_outcomes - actual_outcomes))

    calibration_results.update({
        'mean_calibration_error': calibration_error,
        'weighted_calibration_error': weighted_calibration_error
    })

    print(f"Calibration metrics:")
    print(f"  Is monotonic: {monotonic}")
    print(f"  Mean calibration error: {calibration_error:.6f}")
    print(f"  Weighted calibration error: {weighted_calibration_error:.6f}")
    print(f"  Monotonicity violations: {calibration_results['monotonicity_violations']}")

    return calibration_results


def compute_regime_separability_metrics(
    layer_representations: Dict[str, torch.Tensor],
    regime_labels: np.ndarray,
    distance_metric: str = 'euclidean'
) -> Dict[str, Dict[str, float]]:
    """
    Compute regime separability metrics in latent space.
    
    This function quantifies how well different regimes are separated in
    the latent representations using various distance and clustering metrics.
    
    Args:
        layer_representations: Dictionary of layer representations
        regime_labels: Array of regime labels
        distance_metric: Distance metric for separability analysis
        
    Returns:
        Dictionary mapping layer names to separability metrics
    """
    print(f"Computing regime separability metrics...")
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.metrics.pairwise import pairwise_distances
    
    separability_results = {}
    
    for layer_name, representations in layer_representations.items():
        print(f"\nAnalyzing separability for {layer_name}...")
        
        # Convert to numpy
        if isinstance(representations, torch.Tensor):
            X = representations.cpu().numpy()
        else:
            X = representations
            
        layer_results = {}
        
        # Check if we have multiple regimes
        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < 2:
            print(f"  Skipping {layer_name}: Only one regime found")
            continue
            
        # 1. Silhouette score (higher is better, range [-1, 1])
        try:
            silhouette = silhouette_score(X, regime_labels, metric=distance_metric)
            layer_results['silhouette_score'] = silhouette
            print(f"  Silhouette score: {silhouette:.4f}")
        except Exception as e:
            print(f"  Could not compute silhouette score: {e}")
            layer_results['silhouette_score'] = np.nan
        
        # 2. Calinski-Harabasz score (higher is better)
        try:
            ch_score = calinski_harabasz_score(X, regime_labels)
            layer_results['calinski_harabasz_score'] = ch_score
            print(f"  Calinski-Harabasz score: {ch_score:.4f}")
        except Exception as e:
            print(f"  Could not compute Calinski-Harabasz score: {e}")
            layer_results['calinski_harabasz_score'] = np.nan
        
        # 3. Inter-cluster vs intra-cluster distance ratio
        try:
            distances = pairwise_distances(X, metric=distance_metric)
            
            # Compute intra-cluster distances
            intra_distances = []
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                if regime_mask.sum() > 1:
                    regime_distances = distances[regime_mask][:, regime_mask]
                    # Take upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(regime_distances, dtype=bool), k=1)
                    if mask.sum() > 0:
                        intra_distances.extend(regime_distances[mask])
            
            # Compute inter-cluster distances
            inter_distances = []
            for i, regime_i in enumerate(unique_regimes):
                for j, regime_j in enumerate(unique_regimes):
                    if i < j:  # Avoid double counting
                        mask_i = regime_labels == regime_i
                        mask_j = regime_labels == regime_j
                        if mask_i.sum() > 0 and mask_j.sum() > 0:
                            inter_dist = distances[mask_i][:, mask_j]
                            inter_distances.extend(inter_dist.flatten())
            
            if len(intra_distances) > 0 and len(inter_distances) > 0:
                mean_intra = np.mean(intra_distances)
                mean_inter = np.mean(inter_distances)
                separation_ratio = mean_inter / mean_intra if mean_intra > 0 else np.inf
                
                layer_results.update({
                    'mean_intra_distance': mean_intra,
                    'mean_inter_distance': mean_inter,
                    'separation_ratio': separation_ratio
                })
                
                print(f"  Mean intra-cluster distance: {mean_intra:.4f}")
                print(f"  Mean inter-cluster distance: {mean_inter:.4f}")
                print(f"  Separation ratio: {separation_ratio:.4f}")
            
        except Exception as e:
            print(f"  Could not compute distance-based metrics: {e}")
        
        # 4. Regime centroid analysis
        try:
            centroids = {}
            centroid_distances = {}
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                if regime_mask.sum() > 0:
                    centroid = X[regime_mask].mean(axis=0)
                    centroids[regime] = centroid
            
            # Compute pairwise centroid distances
            centroid_pairs = list(centroids.keys())
            for i, regime_i in enumerate(centroid_pairs):
                for j, regime_j in enumerate(centroid_pairs):
                    if i < j:
                        dist = np.linalg.norm(centroids[regime_i] - centroids[regime_j])
                        centroid_distances[f"{regime_i}_vs_{regime_j}"] = dist
            
            if centroid_distances:
                mean_centroid_distance = np.mean(list(centroid_distances.values()))
                layer_results.update({
                    'centroid_distances': centroid_distances,
                    'mean_centroid_distance': mean_centroid_distance
                })
                print(f"  Mean centroid distance: {mean_centroid_distance:.4f}")
                
        except Exception as e:
            print(f"  Could not compute centroid metrics: {e}")
        
        separability_results[layer_name] = layer_results
    
    return separability_results
