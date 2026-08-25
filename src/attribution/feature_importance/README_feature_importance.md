# Enhanced Feature Importance Analysis

This module provides utilities to create meaningful feature importance heatmaps by mapping agent state features to their original names from model metadata.

## Overview

The current feature importance analysis shows features as generic indices (0, 1, 2, ...). This module enhances the analysis by:

1. **Loading model metadata** to understand the feature structure
2. **Mapping feature indices** to meaningful names (e.g., `^GSPC_RSI`, `^FTSE_Close`)
3. **Creating organized visualizations** grouped by instrument
4. **Generating comprehensive summaries** with multiple analysis perspectives

## Key Files

- `feature_importance.py`: Comprehensive feature importance analysis functions
- `enhanced_shap_analysis.py`: Enhanced SHAP analysis utilities for integration with existing workflows

## Quick Start

### 1. Basic Usage

```python
from src.portfolio_optimization.ablation_studies.feature_importance import (
    load_metadata,
    create_feature_mapping,
    analyze_model_feature_importance
)

# Load metadata from your model
metadata = load_metadata("path/to/model/metadata.json")

# Create feature mapping
feature_mapping = create_feature_mapping(metadata)

# Run complete analysis
results = analyze_model_feature_importance(
    model_path="path/to/model",
    shap_values=your_shap_values,
    output_dir="./results",
    model_name="my_model"
)
```

### 2. Integration with Existing Analysis

```python
from src.portfolio_optimization.ablation_studies.enhanced_shap_analysis import (
    integrate_with_existing_analysis
)

# After your existing SHAP analysis
results = integrate_with_existing_analysis(
    model_path="path/to/model",
    shap_values=your_shap_values,
    X_data=your_X_data,
    output_dir="./enhanced_results"
)
```

### 3. Individual Functions

```python
from src.portfolio_optimization.ablation_studies.feature_importance import (
    create_feature_importance_heatmap,
    create_instrument_grouped_heatmap,
    create_feature_importance_summary
)

# Create specific visualizations
fig1 = create_feature_importance_heatmap(
    shap_values=shap_values,
    feature_names=feature_names,
    metadata=metadata,
    title="My Feature Importance Heatmap"
)

fig2 = create_instrument_grouped_heatmap(
    shap_values=shap_values,
    metadata=metadata,
    title="Feature Importance by Instrument"
)

fig3 = create_feature_importance_summary(
    shap_values=shap_values,
    metadata=metadata,
    title="Comprehensive Feature Importance Summary"
)
```

## Features

### 1. Meaningful Feature Names

Instead of generic indices, features are named meaningfully:
- `^GSPC_RSI` (S&P 500 RSI indicator)
- `^FTSE_Close` (FTSE 100 closing price)
- `^N225_NATR` (Nikkei 225 Normalized Average True Range)

### 2. Instrument Grouping

Features can be grouped by instrument for better organization:
- All S&P 500 features together
- All FTSE 100 features together
- All Nikkei 225 features together

### 3. Multiple Visualization Types

- **Standard heatmap**: Top features with importance values
- **Instrument-grouped heatmap**: Features organized by instrument
- **Comprehensive summary**: Bar charts and pie charts showing feature distribution

### 4. Comprehensive Analysis

The analysis includes:
- Feature importance rankings
- Instrument-wise analysis
- Feature type analysis (technical indicators vs. price data)
- Feature mapping reference files

## Example Output

### Feature Mapping
```
original_index,meaningful_name
0,^FTSE_RSI
1,^FTSE_AROONOSC
2,^FTSE_CCI
...
```

### Feature Importance Ranking
```
original_index,meaningful_name,instrument,feature_type,shap_value
45,^GSPC_RSI,^GSPC,RSI,0.0234
12,^FTSE_Close,^FTSE,Close,0.0198
78,^N225_NATR,^N225,NATR,0.0187
...
```

### Instrument Analysis
```
instrument,mean_importance,total_importance,feature_count
^GSPC,0.0123,0.2345,19
^FTSE,0.0101,0.1923,19
^N225,0.0098,0.1862,19
...
```

## Integration with Existing Notebooks

### Before (Generic Names)
```python
# Original analysis
shap.summary_plot(shap_values, X_test, show=False)
shap.plots.bar(shap_values, max_display=12, show=False)
```

### After (Meaningful Names)

```python
# Enhanced analysis
from src.portfolio_optimization.ablation_studies.enhanced_shap_analysis import (
    enhanced_shap_summary_plot,
    enhanced_shap_bar_plot
)

metadata = load_model_metadata("path/to/model")
enhanced_shap_summary_plot(shap_values, X_test, metadata)
enhanced_shap_bar_plot(shap_values, metadata)
```

## Benefits

1. **Interpretability**: Features are now named meaningfully instead of generic indices
2. **Organization**: Features can be grouped by instrument for better analysis
3. **Comprehensive**: Multiple visualization types provide different perspectives
4. **Integration**: Works with existing SHAP analysis workflows
5. **Reproducibility**: All analysis is saved to files for later reference

## Usage in Your Workflow

1. **Replace generic feature names** in your existing SHAP analysis
2. **Use the `analyze_model_feature_importance()` function** for complete analysis
3. **Integrate with your existing notebooks** by importing the utility functions
4. **Save results for publication** and further analysis

This approach makes your feature importance analysis much more meaningful and interpretable for understanding how different market instruments and technical indicators influence your portfolio optimization models. 