# Temporal Attribution Analysis Module

This module provides comprehensive temporal attribution analysis for Decision Transformer models, implementing four key assessments for understanding how models use temporal information in their decision-making process.

## Overview

The temporal analysis module examines how Decision Transformer models utilize information across different time horizons (lags) when making allocation decisions. This is crucial for understanding:

- **Temporal dependencies**: How much does the model rely on recent vs. historical information?
- **Market regime sensitivity**: Does temporal focus change across different market conditions?
- **Volatility adaptation**: Does the model adjust its temporal focus during volatile periods?
- **Stability**: Are temporal patterns consistent over time?

## Four Key Assessments

### 1. Lag Relevance and AUC Summary

**Purpose**: Analyze time-occlusion curves to understand the importance of different lags.

**Method**: 
- Ablate each timestep in the input sequence
- Measure impact on final predictions
- Compute Area Under Curve (AUC) and peak-lag indices

**Outputs**:
- Line plots of occlusion curves per regime with 95% CI
- Table of AUC and peak-lag index per regime

**Key Functions**:
```python
compute_temporal_occlusion_curves()
compute_lag_relevance_auc()
plot_occlusion_curves_by_regime()
```

### 2. Temporal Half-Life of Attribution

**Purpose**: Compute temporal half-life L₁/₂ - the minimum lag needed to accumulate half of the attribution mass.

**Method**:
- Calculate cumulative attribution mass from recent to historical lags
- Find point where 50% of total attribution is accumulated
- Shorter half-lives indicate reliance on recent information

**Outputs**:
- Table of L₁/₂ values per regime
- Bar chart comparing half-life across regimes

**Key Functions**:
```python
compute_temporal_half_life()
plot_half_life_comparison()
```

### 3. Volatility-Elasticity of Temporal Focus

**Purpose**: Assess whether temporal focus varies with market volatility.

**Method**:
- Regress realized volatility against attribution mass in recent lags (0-2)
- Positive elasticity confirms turbulent markets increase short-horizon reliance
- Compute regression coefficients with confidence intervals

**Outputs**:
- Scatter plot of realized volatility vs. short-term attribution share
- Table of regression coefficients with confidence intervals

**Key Functions**:
```python
compute_volatility_elasticity()
plot_volatility_elasticity_scatter()
```

### 4. Stability and Robustness of Attributions

**Purpose**: Confirm temporal attribution structure is stable across time.

**Method**:
- Rolling-window analysis of temporal patterns
- Compute stability metrics (coefficient of variation)
- Analyze consistency between different time periods

**Outputs**:
- Rolling window evolution plots
- Stability metrics summary
- Correlation analysis between windows

**Key Functions**:
```python
compute_rolling_stability_analysis()
plot_stability_analysis()
```

## Quick Start

### Comprehensive Analysis

Run all four assessments with a single function call:

```python
from src.ablation import run_comprehensive_temporal_analysis

# Prepare your data
states = torch.tensor(...)          # (batch_size, seq_length, state_dim)
actions = torch.tensor(...)         # (batch_size, seq_length, action_dim)  
returns_to_go = torch.tensor(...)   # (batch_size, seq_length, 1)
timesteps = torch.tensor(...)       # (batch_size, seq_length)
regimes = np.array(...)             # (batch_size,) - regime labels
realized_volatility = np.array(...) # (batch_size,) - volatility measure

# Run comprehensive analysis
results = run_comprehensive_temporal_analysis(
    model=your_model,
    states=states,
    actions=actions, 
    returns_to_go=returns_to_go,
    timesteps=timesteps,
    regimes=regimes,
    regime_names=['Bull', 'Bear', 'Sideways', 'Crisis'],
    realized_volatility=realized_volatility,
    save_figures=True,
    save_tables=True,
    output_dir="./results/temporal_analysis/"
)
```

### Individual Assessments

Run assessments separately for detailed analysis:

```python
from src.ablation import (
    compute_temporal_occlusion_curves,
    compute_lag_relevance_auc,
    compute_temporal_half_life,
    compute_volatility_elasticity
)

# 1. Compute temporal occlusion curves
occlusion_results = compute_temporal_occlusion_curves(
    model, states, actions, returns_to_go, timesteps, regimes, regime_names
)

# 2. Lag relevance analysis
auc_results = compute_lag_relevance_auc(occlusion_results)
print(f"Mean AUC: {auc_results['mean_auc']:.4f}")

# 3. Half-life analysis  
half_life_results = compute_temporal_half_life(occlusion_results)
print(f"Mean half-life: {half_life_results['mean_half_life']:.2f} lags")

# 4. Volatility-elasticity
elasticity_results = compute_volatility_elasticity(
    occlusion_results, realized_volatility, short_horizon_lags=[0, 1, 2]
)
print(f"Elasticity coefficient: {elasticity_results['mean_coefficient']:.4f}")
```

## Key Parameters

### Input Data Requirements

- **states**: Market state features (prices, indicators, etc.)
- **actions**: Portfolio allocation weights 
- **returns_to_go**: Target returns for conditioning
- **timesteps**: Sequence position indices
- **regimes**: Market regime labels (optional but recommended)
- **realized_volatility**: Volatility measure for elasticity analysis

### Configuration Parameters

- **short_horizon_lags**: Lags considered "short-term" (default: [0, 1, 2])
- **rolling_window_size**: Window size for stability analysis (default: 50)
- **rolling_stride**: Stride between windows (default: 25)
- **target_action_idx**: Specific action dimension to analyze (default: None = all)

## Output Structure

The comprehensive analysis returns a nested dictionary:

```python
results = {
    'occlusion': {
        'occlusion_curves': torch.Tensor,     # (batch, seq, actions)
        'regime_results': dict,               # Per-regime statistics
        'lag_positions': torch.Tensor         # Lag indices
    },
    'auc': {
        'auc_values': torch.Tensor,           # AUC per sample
        'peak_lag_indices': torch.Tensor,     # Peak lag per sample
        'regime_auc': dict                    # Per-regime AUC stats
    },
    'half_life': {
        'half_life_values': torch.Tensor,     # L_1/2 per sample
        'regime_half_lives': dict             # Per-regime half-life stats
    },
    'elasticity': {
        'individual_results': dict,           # Per-action regression
        'aggregate_regression': dict,         # Pooled regression
        'mean_coefficient': float             # Average elasticity
    },
    'stability': {
        'stability_metrics': dict,            # CV and correlation metrics
        'window_results': list,               # Per-window statistics
        'all_curves': torch.Tensor            # All window curves
    },
    'tables': {
        'auc_peak_lag_by_regime': pd.DataFrame,
        'half_life_by_regime': pd.DataFrame,
        'volatility_elasticity_regression': pd.DataFrame
    }
}
```

## Interpretation Guide

### Lag Relevance (AUC)
- **High AUC**: Model uses information from many timesteps
- **Low AUC**: Model focuses on specific lags only
- **Peak lag = 0**: Strong recency bias
- **Peak lag > 5**: Model values historical context

### Temporal Half-Life
- **L₁/₂ < 3**: Short-term focused (recent information)
- **L₁/₂ > 7**: Long-term focused (historical integration)
- **Regime differences**: Adaptive temporal focus

### Volatility-Elasticity  
- **Positive coefficient**: Higher volatility → more short-term focus
- **Negative coefficient**: Higher volatility → more long-term focus
- **R² > 0.1**: Meaningful volatility-temporal relationship

### Stability Metrics
- **CV < 0.2**: Stable temporal patterns
- **CV > 0.5**: Highly variable temporal focus
- **High correlation**: Consistent patterns over time

## Example Use Cases

### 1. Model Validation
```python
# Check if model has reasonable temporal dependencies
results = run_comprehensive_temporal_analysis(...)

# Validate expectations
assert results['half_life']['mean_half_life'] < 10, "Model too long-term focused"
assert results['auc']['mean_auc'] > 0.5, "Model not using temporal information"
```

### 2. Regime Analysis
```python
# Compare temporal patterns across market regimes
regime_half_lives = results['half_life']['regime_half_lives']
for regime, stats in regime_half_lives.items():
    print(f"{regime}: L_1/2 = {stats['mean_half_life']:.2f}")
```

### 3. Risk Assessment
```python
# Check volatility adaptation
elasticity = results['elasticity']['mean_coefficient'] 
if elasticity > 0:
    print("Model increases short-term focus during volatility - good risk adaptation")
else:
    print("Model doesn't adapt temporal focus to volatility - potential concern")
```

### 4. Model Comparison
```python
# Compare two models
results_model_a = run_comprehensive_temporal_analysis(model_a, ...)
results_model_b = run_comprehensive_temporal_analysis(model_b, ...)

print(f"Model A half-life: {results_model_a['half_life']['mean_half_life']:.2f}")
print(f"Model B half-life: {results_model_b['half_life']['mean_half_life']:.2f}")
```

## Advanced Usage

### Custom Lag Definitions
```python
# Analyze specific lag ranges
elasticity_results = compute_volatility_elasticity(
    occlusion_results, 
    realized_volatility,
    short_horizon_lags=[0, 1],        # Very short-term
    # short_horizon_lags=[0, 1, 2, 3], # Extended short-term
)
```

### Rolling Analysis Parameters
```python
# Fine-grained stability analysis
stability_results = compute_rolling_stability_analysis(
    model, states, actions, returns_to_go, timesteps,
    window_size=30,    # Smaller windows for finer analysis
    stride=5,          # More overlap between windows
    min_window_size=15 # Minimum viable window size
)
```

### Regime-Specific Analysis
```python
# Focus on specific regime
crisis_mask = regimes == 'Crisis'
crisis_results = run_comprehensive_temporal_analysis(
    model, 
    states[crisis_mask], 
    actions[crisis_mask],
    # ... other inputs
)
```

## Troubleshooting

### Common Issues

1. **Insufficient samples per regime**: Use larger datasets or combine similar regimes
2. **Volatility data missing**: Elasticity analysis will be skipped automatically  
3. **Very short sequences**: Some analyses require minimum sequence length (>5)
4. **Memory issues**: Use smaller batch sizes or rolling analysis

### Performance Tips

1. **Batch processing**: Process large datasets in chunks
2. **GPU acceleration**: Ensure tensors are on GPU for faster computation
3. **Selective analysis**: Run only needed assessments for faster execution
4. **Caching**: Save intermediate results for reuse

## Dependencies

Required packages:
- `torch` >= 1.9.0
- `numpy` >= 1.19.0  
- `pandas` >= 1.3.0
- `matplotlib` >= 3.3.0
- `seaborn` >= 0.11.0
- `scipy` >= 1.7.0
- `scikit-learn` >= 0.24.0

## References

This implementation is based on temporal attribution analysis techniques from:

1. **Time-series interpretability**: Methods for understanding temporal dependencies in sequential models
2. **Financial econometrics**: Volatility regime analysis and temporal aggregation
3. **Captum attribution**: Model interpretability for sequential decision making
4. **Portfolio optimization**: Temporal risk assessment and adaptive allocation strategies
