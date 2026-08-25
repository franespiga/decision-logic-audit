"""
Temporal Attribution Analysis Module

This module provides comprehensive temporal attribution analysis functions for Decision Transformer models,
including lag relevance, temporal half-life, volatility-elasticity, and stability assessments.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .. import _forward_action, temporal_attribution


def compute_temporal_occlusion_curves(
    model,
    states: torch.Tensor,
    reshaped_states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    regimes: Optional[np.ndarray] = None,
    target_action_idx: Optional[int] = None
) -> Dict:
    """
    Compute temporal occlusion curves by ablating each timestep and measuring impact on predictions.

    Args:
        model: Decision transformer model
        states: Input states tensor (batch_size, seq_length, state_dim)
        actions: Actions tensor (batch_size, seq_length, action_dim)
        returns_to_go: Returns-to-go tensor (batch_size, seq_length, 1)
        timesteps: Timesteps tensor (batch_size, seq_length)
        regimes: Optional regime labels for each sample (batch_size,)
        regime_names: Optional names for regimes
        target_action_idx: Optional specific action dimension to analyze

    Returns:
        Dictionary containing occlusion curves and regime-specific results
    """
    print("Computing temporal occlusion curves...")

    batch_size, seq_length = states.shape[0], states.shape[2]
    lags = reshaped_states.shape[-1]

    # Get baseline (non-ablated) predictions
    baseline_output = _forward_action(
        model, states, actions, returns_to_go, timesteps, target_action_idx=target_action_idx
    )

    if target_action_idx is not None:
        baseline_output = baseline_output.unsqueeze(-1)

    # Compute occlusion for each timestep
    occlusion_effects = torch.zeros(batch_size, baseline_output.shape[-1], lags)

    for t in range(lags):
        # Create ablated states by zeroing out timestep t
        ablated_states = reshaped_states.clone()
        ablated_states[:, :, :, t] = 0
        ablated_states = ablated_states.reshape((states.shape[0],1,-1))
        ablated_output = _forward_action(
            model, ablated_states, actions, returns_to_go, timesteps, target_action_idx=target_action_idx
        )

        if target_action_idx is not None:
            ablated_output = ablated_output.unsqueeze(-1)

        # Compute occlusion effect (baseline - ablated)
        delta = baseline_output - ablated_output
        occlusion_effects[:, :, t] = delta


    # Convert to relative lag positions (0 = most recent)
    lag_positions = torch.arange(lags - 1, -1, -1)

    # Organize results by regime if provided
    results = {
        'occlusion_curves': occlusion_effects,
        'lag_positions': lag_positions,
        'baseline_output': baseline_output,
        'seq_length': seq_length,
        'batch_size': batch_size
    }

    if regimes is not None:
        unique_regimes = np.unique(regimes)
        regime_results = {}
        for i, regime in enumerate(unique_regimes):
            regime_mask = regimes == regime
            regime_curves = occlusion_effects[regime_mask]

            regime_results[regime] = {
                'curves': regime_curves,
                'mean_curve': regime_curves.mean(dim=0),
                'std_curve': regime_curves.std(dim=0),
                'n_samples': regime_mask.sum().item()
            }

        results['regime_results'] = regime_results

    print(f"Computed occlusion curves for {batch_size} samples, {lags} timesteps")
    return results

from scipy.integrate import trapezoid
def compute_lag_relevance_auc_halflife(occlusion_results: Dict) -> Dict:
    """
    Compute Area Under Curve (AUC), Half-life and peak-lag index for temporal relevance analysis.

    Args:
        occlusion_results: Results from compute_temporal_occlusion_curves()

    Returns:
        Dictionary with AUC values, peak-lag indices, and summary statistics
    """
    print("Computing lag relevance AUC and peak-lag indices...")

    occlusion_curves = occlusion_results['occlusion_curves']
    lag_positions = occlusion_results['lag_positions']
    print(lag_positions)

    # Compute AUC for each sample and action dimension
    batch_size, action_dim, seq_length = occlusion_curves.shape

    # Take absolute values for AUC computation (total relevance)
    abs_curves = torch.abs(occlusion_curves)

    auc_values = torch.zeros(batch_size, action_dim)
    peak_lag_indices = torch.zeros(batch_size, action_dim, dtype=torch.long)
    half_life_values = torch.zeros(batch_size, action_dim)

    for batch_idx in range(batch_size):
        for action_idx in range(action_dim):
            curve = abs_curves[batch_idx, action_idx, :].detach().numpy()
            # Compute AUC using trapezoidal rule
            auc_raw = trapezoid(curve, -lag_positions.numpy())

            # --- Normalize by total attribution mass (optional but recommended)
            total_mass = np.sum(curve)
            auc_norm = auc_raw / total_mass if total_mass > 0 else 0.0

            auc_values[batch_idx, action_idx] = float(auc_norm)

            # Find peak lag (lag with maximum absolute attribution)
            peak_idx = torch.argmax(abs_curves[batch_idx, :, action_idx])
            peak_lag_indices[batch_idx, action_idx] = lag_positions[peak_idx]

            # Compute cumulative attribution mass from most recent (lag 0) to oldest
            # Note: curves are indexed by timestep, need to reverse for lag order
            reversed_curve = torch.flip(torch.Tensor(curve), dims=[0])
            cumulative_mass = torch.cumsum(reversed_curve, dim=0)
            total_mass = cumulative_mass[-1]

            if total_mass > 0:
                # Find where cumulative mass reaches 50% of total
                half_mass_threshold = total_mass / 2
                half_life_idx = torch.argmax((cumulative_mass >= half_mass_threshold).float())
                half_life_values[batch_idx, action_idx] = half_life_idx.float()
            else:
                # If no attribution, set to maximum lag
                half_life_values[batch_idx, action_idx] = seq_length - 1



    results = {
        'auc_values': auc_values,
        'peak_lag_indices': peak_lag_indices,
        'mean_auc': auc_values.mean(),
        'std_auc': auc_values.std(),
        'mean_peak_lag': peak_lag_indices.float().mean(),
        'std_peak_lag': peak_lag_indices.float().std(),
        'half_life_values': half_life_values,
        'mean_half_life': half_life_values.mean(),
        'std_half_life': half_life_values.std(),
        'median_half_life': half_life_values.median()
    }

    # Compute regime-specific results if available
    if 'regime_results' in occlusion_results:
        regime_stats = {}

        for regime_name, regime_data in occlusion_results['regime_results'].items():
            regime_curves = torch.abs(regime_data['curves'])
            n_samples, action_dim, seq_length = regime_curves.shape

            regime_auc_vals = torch.zeros(n_samples, action_dim)
            regime_peak_lags = torch.zeros(n_samples, action_dim, dtype=torch.long)
            regime_hl_vals = torch.zeros(n_samples, action_dim)

            for sample_idx in range(n_samples):
                for action_idx in range(action_dim):
                    curve = regime_curves[sample_idx, action_idx, :].detach().numpy()
                    # Compute AUC using trapezoidal rule
                    auc_raw = trapezoid(curve, -lag_positions.numpy())

                    # --- Normalize by total attribution mass (optional but recommended)
                    total_mass = np.sum(curve)
                    auc_norm = auc_raw / total_mass if total_mass > 0 else 0.0

                    regime_auc_vals[sample_idx, action_idx] = float(auc_norm)

                    peak_idx = torch.argmax(regime_curves[sample_idx, :, action_idx])
                    regime_peak_lags[sample_idx, action_idx] = lag_positions[peak_idx]

                    # Half-life
                    reversed_curve = torch.flip(torch.Tensor(curve), dims=[0])
                    cumulative_mass = torch.cumsum(reversed_curve, dim=0)
                    total_mass = cumulative_mass[-1]

                    if total_mass > 0:
                        half_mass_threshold = total_mass / 2
                        half_life_idx = torch.argmax((cumulative_mass >= half_mass_threshold).float())
                        regime_hl_vals[sample_idx, action_idx] = half_life_idx.float()
                    else:
                        regime_hl_vals[sample_idx, action_idx] = seq_length - 1

            regime_stats[regime_name] = {
                'auc_values': regime_auc_vals,
                'peak_lag_indices': regime_peak_lags,
                'mean_auc': regime_auc_vals.mean().item(),
                'std_auc': regime_auc_vals.std().item(),
                'mean_peak_lag': regime_peak_lags.float().mean().item(),
                'std_peak_lag': regime_peak_lags.float().std().item(),
                'half_life_values': regime_hl_vals,
                'mean_half_life': regime_hl_vals.mean().item(),
                'std_half_life': regime_hl_vals.std().item(),
                'median_half_life': regime_hl_vals.median().item()
            }

        results['regime_stats'] = regime_stats

    print(f"Computed AUC and peak-lag indices. Mean AUC: {results['mean_auc']:.4f}")
    print(f"Computed temporal half-lives. Mean L_1/2: {results['mean_half_life']:.2f}")
    return results



def compute_volatility_elasticity(
    occlusion_results: Dict,
    realized_volatility: np.ndarray,
    short_horizon_lags: List[int] = [0, 1, 2]
) -> Dict:
    """
    Assess volatility-elasticity of temporal focus by regressing volatility against short-term attribution.

    Args:
        occlusion_results: Results from compute_temporal_occlusion_curves()
        realized_volatility: Array of realized volatility for each sample
        short_horizon_lags: List of lag indices considered "short-term" (default: [0, 1, 2])

    Returns:
        Dictionary with regression results and elasticity measures
    """
    print(f"Computing volatility-elasticity analysis for lags {short_horizon_lags}...")

    occlusion_curves = occlusion_results['occlusion_curves'][:-1]
    seq_length = occlusion_curves.shape[-1]

    # Convert lag indices to timestep indices (reverse order)
    timestep_indices = [seq_length - 1 - lag for lag in short_horizon_lags if lag < seq_length]

    if not timestep_indices:
        raise ValueError(f"No valid timestep indices for lags {short_horizon_lags} in sequence of length {seq_length}")

    # Compute attribution mass in short horizons
    abs_curves = torch.abs(occlusion_curves)
    short_term_attribution = abs_curves[:, :, timestep_indices].sum(dim=2)  # Sum over selected timesteps
    total_attribution = abs_curves.sum(dim = 2)

    # Compute share of attribution in short horizons
    attribution_share = torch.zeros_like(short_term_attribution)
    valid_mask = total_attribution > 0
    attribution_share[valid_mask] = short_term_attribution[valid_mask] / total_attribution[valid_mask]

    batch_size, action_dim = attribution_share.shape

    # Prepare data for regression
    volatility_expanded = np.tile(realized_volatility.reshape(-1, 1), (1, action_dim))

    results = {}

    # Perform regression for each action dimension
    for action_idx in range(action_dim):
        y = attribution_share[:, action_idx].detach().numpy()
        x = volatility_expanded[:, action_idx].reshape(-1, 1)

        # Filter out invalid values
        valid_indices = ~(np.isnan(y) | np.isnan(x.flatten()) | np.isinf(y) | np.isinf(x.flatten()))

        if valid_indices.sum() < 10:  # Need minimum samples
            print(f"Warning: Insufficient valid samples for action {action_idx}")
            continue

        y_valid = y[valid_indices]
        x_valid = x[valid_indices]

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(x_valid, y_valid)

        # Compute predictions and statistics
        y_pred = reg.predict(x_valid)
        r2 = r2_score(y_valid, y_pred)

        # Compute confidence intervals (approximate)
        n = len(y_valid)
        mse = np.mean((y_valid - y_pred) ** 2)
        x_mean = np.mean(x_valid)
        sxx = np.sum((x_valid.flatten() - x_mean) ** 2)

        # Standard error of slope
        se_slope = np.sqrt(mse / sxx) if sxx > 0 else 0

        # 95% confidence interval
        t_val = stats.t.ppf(0.975, n - 2) if n > 2 else 1.96
        ci_lower = reg.coef_[0] - t_val * se_slope
        ci_upper = reg.coef_[0] + t_val * se_slope

        results[f'action_{action_idx}'] = {
            'coefficient': reg.coef_[0],
            'intercept': reg.intercept_,
            'r_squared': r2,
            'confidence_interval': (ci_lower, ci_upper),
            'standard_error': se_slope,
            'n_samples': n,
            'x_data': x_valid.flatten(),
            'y_data': y_valid,
            'y_pred': y_pred
        }

    # Aggregate results across actions
    coefficients = [results[key]['coefficient'] for key in results.keys()]
    r_squared_values = [results[key]['r_squared'] for key in results.keys()]

    aggregate_results = {
        'mean_coefficient': np.mean(coefficients),
        'std_coefficient': np.std(coefficients),
        'mean_r_squared': np.mean(r_squared_values),
        'short_horizon_lags': short_horizon_lags,
        'individual_results': results
    }

    # Compute aggregate regression (pooling all actions)
    all_y = []
    all_x = []
    for action_idx in range(action_dim):
        if f'action_{action_idx}' in results:
            all_y.extend(results[f'action_{action_idx}']['y_data'])
            all_x.extend(results[f'action_{action_idx}']['x_data'])

    if len(all_y) > 10:
        all_y = np.array(all_y)
        all_x = np.array(all_x).reshape(-1, 1)

        reg_agg = LinearRegression()
        reg_agg.fit(all_x, all_y)
        y_pred_agg = reg_agg.predict(all_x)

        aggregate_results['aggregate_regression'] = {
            'coefficient': reg_agg.coef_[0],
            'intercept': reg_agg.intercept_,
            'r_squared': r2_score(all_y, y_pred_agg),
            'n_samples': len(all_y),
            'x_data': all_x.flatten(),
            'y_data': all_y,
            'y_pred': y_pred_agg
        }

    print(f"Computed volatility-elasticity. Mean coefficient: {aggregate_results['mean_coefficient']:.4f}")
    return aggregate_results

def compute_rolling_stability_analysis(
    model,
    states: torch.Tensor,
    reshaped_states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    window_size: int = 50,
    stride: int = 25,
    min_window_size: int = 20
) -> Dict:
    """
    Perform rolling-window analysis to assess stability of temporal attribution patterns.

    Args:
        model: Decision transformer model
        states: Input states tensor
        actions: Actions tensor
        returns_to_go: Returns-to-go tensor
        timesteps: Timesteps tensor
        window_size: Size of rolling window
        stride: Stride between windows
        min_window_size: Minimum window size to consider
  a
    Returns:
        Dictionary with stability metrics and rolling results
    """
    print(f"Computing rolling stability analysis with window_size={window_size}, stride={stride}...")

    batch_size = states.shape[0]
    seq_length = reshaped_states.shape[-1]

    if batch_size < window_size:
        print(f"Warning: Batch size ({batch_size}) smaller than window size ({window_size}). Using batch size as window.")
        window_size = batch_size
        stride = batch_size

    # Compute rolling windows
    windows = []
    window_results = []

    for start_idx in range(0, batch_size - min_window_size + 1, stride):
        end_idx = min(start_idx + window_size, batch_size)

        if end_idx - start_idx < min_window_size:
            break

        window_states = states[start_idx:end_idx]
        window_reshaped_states = reshaped_states[start_idx:end_idx]
        window_actions = actions[start_idx:end_idx]
        window_returns = returns_to_go[start_idx:end_idx]
        window_timesteps = timesteps[start_idx:end_idx]

        # Compute temporal occlusion for this window
        #try:
        occlusion_results = compute_temporal_occlusion_curves(
            model,
            window_states,
            window_reshaped_states,
            window_actions,
            window_returns,
            window_timesteps
        )

        # Compute summary statistics for this window
        curves = torch.abs(occlusion_results['occlusion_curves'])
        window_mean_curve = curves.mean(dim=(0, 2))  # Average over batch and actions

        # Compute statistics: AUC, half-life for this window
        statistics = compute_lag_relevance_auc_halflife(occlusion_results)
        window_mean_half_life = statistics['mean_half_life'].item()
        window_mean_auc = statistics['mean_auc'].item()

        window_results.append({
            'window_start': start_idx,
            'window_end': end_idx,
            'window_size': end_idx - start_idx,
            'mean_curve': window_mean_curve,
            'mean_half_life': window_mean_half_life,
            'mean_auc': window_mean_auc
        })

        windows.append((start_idx, end_idx))

        #except Exception as e:
        #    print(f"Warning: Failed to process window {start_idx}-{end_idx}: {e}")
        #    continue

    if len(window_results) < 2:
        print("Warning: Insufficient windows for stability analysis")
        return {'windows': windows, 'window_results': window_results, 'stability_metrics': {}}

    # Compute stability metrics
    all_curves = torch.stack([wr['mean_curve'] for wr in window_results])
    all_half_lives = np.array([wr['mean_half_life'] for wr in window_results])
    all_aucs = np.array([wr['mean_auc'] for wr in window_results])

    # Curve stability: coefficient of variation across windows
    curve_means = all_curves.mean(dim=0)
    curve_stds = all_curves.std(dim=0)
    curve_cv = (curve_stds / (curve_means + 1e-8)).mean().item()

    # Half-life stability
    half_life_cv = np.std(all_half_lives) / (np.mean(all_half_lives) + 1e-8)

    # AUC stability
    auc_cv = np.std(all_aucs) / (np.mean(all_aucs) + 1e-8)

    # Correlation between consecutive windows (for temporal curves)
    correlations = []
    for i in range(len(all_curves) - 1):
        corr = torch.corrcoef(torch.stack([all_curves[i], all_curves[i + 1]]))[0, 1]
        if not torch.isnan(corr):
            correlations.append(corr.item())

    mean_correlation = np.mean(correlations) if correlations else 0.0

    stability_metrics = {
        'curve_coefficient_of_variation': curve_cv,
        'half_life_coefficient_of_variation': half_life_cv,
        'auc_coefficient_of_variation': auc_cv,
        'mean_window_correlation': mean_correlation,
        'n_windows': len(window_results),
        'mean_half_life_across_windows': np.mean(all_half_lives),
        'std_half_life_across_windows': np.std(all_half_lives),
        'mean_auc_across_windows': np.mean(all_aucs),
        'std_auc_across_windows': np.std(all_aucs)
    }

    results = {
        'windows': windows,
        'window_results': window_results,
        'stability_metrics': stability_metrics,
        'all_curves': all_curves,
        'all_half_lives': all_half_lives,
        'all_aucs': all_aucs
    }

    print(f"Computed stability analysis over {len(window_results)} windows")
    print(f"Curve CV: {curve_cv:.4f}, Half-life CV: {half_life_cv:.4f}, Mean correlation: {mean_correlation:.4f}")

    return results


def create_summary_tables(
    auc_results: Dict,
    elasticity_results: Dict,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create comprehensive summary tables for all temporal analyses.
    """
    tables = {}

    # Table 1: AUC and Peak-lag by regime & Table 2: Half-life by regime
    if 'regime_stats' in auc_results:
        auc_data = []
        hl_data = []
        for regime_name, regime_stats in auc_results['regime_stats'].items():

            auc_data.append({
                'Regime': regime_name,
                'Mean_AUC': regime_stats['mean_auc'],
                'Std_AUC': regime_stats['std_auc'],
                'Mean_Peak_Lag': regime_stats['mean_peak_lag'],
                'Std_Peak_Lag': regime_stats['std_peak_lag']
            })

            hl_data.append({
                'Regime': regime_name,
                'Mean_Half_Life': regime_stats['mean_half_life'],
                'Std_Half_Life': regime_stats['std_half_life'],
                'Median_Half_Life': regime_stats['median_half_life']
            })

        tables['auc_peak_lag_by_regime'] = pd.DataFrame(auc_data)
        tables['half_life_by_regime'] = pd.DataFrame(hl_data)

    # Table 3: Volatility-elasticity regression results
    if 'individual_results' in elasticity_results:
        reg_data = []
        for action_key, action_stats in elasticity_results['individual_results'].items():
            ci_lower, ci_upper = action_stats['confidence_interval']
            reg_data.append({
                'Action_Dimension': action_key,
                'Coefficient': action_stats['coefficient'],
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'R_Squared': action_stats['r_squared'],
                'N_Samples': action_stats['n_samples']
            })

        tables['volatility_elasticity_regression'] = pd.DataFrame(reg_data)

        # Add aggregate results
        if 'aggregate_regression' in elasticity_results:
            agg = elasticity_results['aggregate_regression']
            agg_row = pd.DataFrame({
                'Action_Dimension': ['Aggregate'],
                'Coefficient': [agg['coefficient']],
                'CI_Lower': [np.nan],  # Would need to compute separately
                'CI_Upper': [np.nan],
                'R_Squared': [agg['r_squared']],
                'N_Samples': [agg['n_samples']]
            })
            tables['volatility_elasticity_regression'] = pd.concat([
                tables['volatility_elasticity_regression'], agg_row
            ], ignore_index=True)

    # Save tables if requested
    if save_path:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for table_name, table_df in tables.items():
                table_df.to_excel(writer, sheet_name=table_name, index=False)
        print(f"Summary tables saved to {save_path}")

    return tables
