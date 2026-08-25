# Model Comparison Framework

A comprehensive framework for comparing multiple models across different analyses including ablation, feature importance, performance metrics, and risk analysis.

## 🎯 Overview

The Model Comparison Framework allows you to systematically compare multiple models across various analyses:

1. **Performance Metrics Comparison** - Compare annual returns, Sharpe ratios, volatility, etc.
2. **Feature Importance Comparison** - Compare which features are most important across models
3. **Ablation Analysis Comparison** - Compare how models respond to feature perturbations
4. **Risk Analysis Comparison** - Compare risk-averse and high-risk decision patterns
5. **Latent Space Comparison** - Compare internal representations across models

## 🚀 Quick Start

### Basic Usage

```python
from src.portfolio_optimization.ablation_studies.model_comparison import ModelComparisonFramework
from src.portfolio_optimization.ablation_studies.helpers.ablation_comparison import AblationComparison

# 1. Define your models
models_config = {
   'DT_16_10_TI_True': 'path/to/model1',
   'DT_28_10_TI_False': 'path/to/model2',
   'max_quadratic_utility_TI_True': 'path/to/model3'
}

# 2. Initialize framework
framework = ModelComparisonFramework(models_config)

# 3. Load models
framework.load_models()

# 4. Run comparisons
ablation_results = framework.run_ablation_comparison(backtest_data, ohlc_data)
feature_importance_results = framework.compare_feature_importance()
performance_results = framework.compare_performance_metrics(performance_data)

# 5. Generate report
report_path = framework.create_comprehensive_report()
```

### Advanced Ablation Comparison

```python
# Initialize ablation comparison
ablation_comp = AblationComparison(
    models=framework.models,
    metadata=framework.metadata
)

# Prepare ablation data
ablation_batches = ablation_comp.prepare_ablation_data(backtest_data, ohlc_data)

# Compare different aspects
feature_importance_results = ablation_comp.compare_feature_importance()
decision_drivers_results = ablation_comp.compare_decision_drivers()
risk_analysis_results = ablation_comp.compare_risk_analysis()

# Create visualizations
plot_paths = ablation_comp.create_comparison_visualizations()
```

## 📊 Analysis Types

### 1. Performance Metrics Comparison

Compare financial performance metrics across models:

```python
performance_data = {
    'DT_16_10_TI_True': pd.DataFrame({
        'Annual return': [0.15, 0.12, 0.18],
        'Sharpe ratio': [1.2, 1.1, 1.3],
        'Annual volatility': [0.12, 0.11, 0.13]
    }),
    'DT_28_10_TI_False': pd.DataFrame({
        'Annual return': [0.14, 0.13, 0.16],
        'Sharpe ratio': [1.1, 1.0, 1.2],
        'Annual volatility': [0.13, 0.12, 0.14]
    })
}

performance_results = framework.compare_performance_metrics(performance_data)
```

**Outputs:**
- Performance comparison CSV files
- Box plots comparing metrics across models
- Correlation heatmaps
- Ranking tables

### 2. Feature Importance Comparison

Compare which features are most important across models:

```python
# Run feature importance analysis
feature_importance_results = framework.compare_feature_importance()

# For advanced ablation comparison
ablation_comp.compare_feature_importance(target_action_idx=0)
```

**Outputs:**
- Feature importance heatmaps
- Feature mapping CSV files
- Comparative feature importance plots

### 3. Ablation Analysis Comparison

Compare how models respond to feature perturbations:

```python
# Run ablation comparison
ablation_results = framework.run_ablation_comparison(
    backtest_data=backtest_data,
    ohlc_data=ohlc_data,
    ref_date='2016-01-01',
    ablation_params={
        'lookback': 16,
        'rebalance_every': 1,
        'annualized_return': 0.15
    }
)
```

**Outputs:**
- Ablation results JSON files
- Feature perturbation analysis
- Model sensitivity comparisons

### 4. Risk Analysis Comparison

Compare risk-averse and high-risk decision patterns:

```python
# Compare risk analysis
risk_analysis_results = ablation_comp.compare_risk_analysis(
    risk_threshold=0.05,
    risk_averse_threshold=0.05
)
```

**Outputs:**
- Risk analysis comparison plots
- High-risk vs risk-averse sample ratios
- Risk metric comparisons

### 5. Decision Drivers Comparison

Compare what drives decisions across models:

```python
# Compare decision drivers
decision_drivers_results = ablation_comp.compare_decision_drivers(
    target_action_idx=0  # Analyze specific action dimension
)
```

**Outputs:**
- Decision drivers heatmaps
- Attribution analysis results
- Convergence metrics

## 🔧 Configuration

### Model Configuration

```python
models_config = {
    'model_id_1': 'path/to/model1',
    'model_id_2': 'path/to/model2',
    'model_id_3': 'path/to/model3'
}
```

### Custom Model Loader

```python
def custom_model_loader(model_path):
    """Load your specific model type"""
    from your_model_module import YourModel
    return YourModel.from_pretrained(model_path)

framework.load_models(model_loader_func=custom_model_loader)
```

### Ablation Parameters

```python
ablation_params = {
    'lookback': 16,                    # Historical window size
    'rebalance_every': 1,              # Rebalancing frequency
    'annualized_return': 0.15,         # Target return
    'use_trajectory_return_objective': False,
    'use_incremental_timesteps': True,
    'ppy': 252                         # Periods per year
}
```

## 📈 Output Structure

```
model_comparison_results/
├── performance_comparison.csv          # Performance metrics
├── performance_summary_stats.csv       # Statistical summaries
├── performance_comparison_boxplots.png # Performance plots
├── performance_correlation_heatmap.png # Correlation analysis
├── feature_importance_comparison.csv   # Feature importance data
├── ablation_results.json              # Ablation analysis results
├── model_comparison_report.md         # Comprehensive report
└── advanced_ablation_comparison/
    ├── feature_importance_comparison.png
    ├── decision_drivers_comparison.png
    ├── risk_analysis_comparison.png
    └── latent_space_comparison.png
```

## 📋 Report Contents

The comprehensive report includes:

1. **Summary** - Overview of models and analysis
2. **Model Details** - Configuration and metadata for each model
3. **Ablation Analysis Results** - Feature importance and perturbation analysis
4. **Feature Importance Comparison** - Cross-model feature importance analysis
5. **Performance Comparison** - Financial metrics and rankings
6. **Risk Analysis** - Risk-averse and high-risk pattern comparisons

## 🎨 Visualization Examples

### Performance Comparison
- Box plots comparing metrics across models
- Correlation heatmaps of performance metrics
- Ranking tables for each metric

### Feature Importance Comparison
- Heatmaps showing feature importance across models
- Bar charts of top features per model
- Comparative feature importance plots

### Risk Analysis Comparison
- High-risk vs risk-averse sample ratios
- Risk metric comparisons across models
- Risk pattern visualizations

## 🔍 Advanced Usage

### Comparing Specific Model Families

```python
# Focus on specific model families
families = ['DT_16_10_TI_True', 'DT_28_10_TI_False']
family_models = {k: v for k, v in models_config.items() if any(f in k for f in families)}

framework = ModelComparisonFramework(family_models)
```

### Custom Analysis Parameters

```python
# Custom ablation parameters
ablation_params = {
    'lookback': 28,
    'rebalance_every': 10,
    'annualized_return': 0.25,
    'use_trajectory_return_objective': True
}

# Custom risk thresholds
risk_analysis_results = ablation_comp.compare_risk_analysis(
    risk_threshold=0.10,        # Higher risk threshold
    risk_averse_threshold=0.02  # Lower risk-averse threshold
)
```

### Export Specific Formats

```python
# Export only CSV files
export_paths = framework.export_results(format='csv')

# Export only JSON files
export_paths = framework.export_results(format='json')

# Export all formats
export_paths = framework.export_results(format='all')
```

## 🛠️ Integration with Existing Workflows

### With Real Data

```python
# Load your actual data
data = pd.read_hdf('data/portfolio_instruments/data.h5', 'technical_indicators')
ohlc_data = pd.read_hdf('data/portfolio_instruments/data.h5', 'instruments')

# Run comparison
framework = ModelComparisonFramework(models_config)
framework.load_models()
ablation_results = framework.run_ablation_comparison(data, ohlc_data)
```

### With Your Model Type

```python
def load_decision_transformer_model(model_path):
    """Load your specific Decision Transformer model"""
    from src.portfolio_optimization.agents.decision_transformer.model import DecisionTransformerModel
    return DecisionTransformerModel.from_pretrained(model_path)

framework.load_models(model_loader_func=load_decision_transformer_model)
```

## 📊 Example Results

### Performance Rankings
```
Annual Return Rankings:
1. max_quadratic_utility_TI_True: 0.1600
2. DT_16_10_TI_True: 0.1500
3. DT_28_10_TI_False: 0.1400

Sharpe Ratio Rankings:
1. max_quadratic_utility_TI_True: 1.3000
2. DT_16_10_TI_True: 1.2000
3. DT_28_10_TI_False: 1.1000
```

### Feature Importance Comparison
- Heatmaps showing which features are most important across models
- Comparative analysis of feature sensitivity
- Cross-model feature importance patterns

### Risk Analysis Results
- High-risk sample ratios for each model
- Risk-averse decision pattern comparisons
- Risk metric distributions

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```python
   # Ensure your model loader function is correct
   def custom_loader(path):
       # Your model loading code here
       return model
   ```

2. **Data Format Issues**
   ```python
   # Ensure backtest_data has meaningful column names
   backtest_data.columns = [('instrument', 'feature') for ...]
   ```

3. **Memory Issues**
   ```python
   # Use smaller batch sizes for large models
   ablation_params = {'lookback': 8, 'rebalance_every': 5}
   ```

### Debug Mode

```python
# Enable verbose output
framework = ModelComparisonFramework(models_config, verbose=True)
```

## 📚 Dependencies

- `torch` - PyTorch for model operations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `tqdm` - Progress bars
- `captum` - Attribution analysis (for ablation)

## 🤝 Contributing

To extend the framework:

1. Add new analysis types to `ModelComparisonFramework`
2. Create new visualization methods
3. Add export formats
4. Extend the report generation

## 📄 License

This framework is part of the portfolio optimization project and follows the same license terms. 