# Regime Representation Analysis Module

This module provides comprehensive analysis of latent dimensions and regime representation in decision transformer models. It implements the four key analyses described in the research paper for understanding how transformer models encode market regime information.

## Overview

The module conducts four main types of analysis:

1. **Level of Representation**: Layer conductance analysis showing that deeper layers encode regime-level abstractions while shallow layers capture short-term fluctuations
2. **Linear Probe Accuracy**: Evaluation of regime information embedded in latent states through linear classification
3. **Head-wise Specialization**: Analysis of functional specialization across attention heads
4. **Predictive Calibration**: Assessment of latent space geometry alignment with prospective outcomes

## Module Structure

```
src/regime_representation/
├── __init__.py                 # Module interface and exports
├── transformations.py          # Data transformation and extraction functions
├── assessments.py             # Analysis and evaluation functions  
├── plotting.py                # Visualization functions
├── run_regime_analysis.py     # Main orchestrator script
├── example_usage.py           # Comprehensive examples and demonstrations
└── README.md                  # This documentation
```

## Core Components

### 1. Transformations (`transformations.py`)

Functions for extracting and transforming model representations:

- `extract_layer_representations()`: Extract hidden states from all transformer layers
- `extract_attention_head_activations()`: Extract attention patterns from all heads
- `prepare_regime_labels()`: Prepare regime labels for classification analysis
- `create_latent_embeddings()`: Create 2D embeddings using t-SNE/UMAP for visualization
- `compute_predictive_bins()`: Group latent embeddings by future return deciles
- `extract_feature_importance_by_category()`: Categorize features for specialization analysis

### 2. Assessments (`assessments.py`)

Functions for evaluating representation quality:

- `compute_layer_conductance()`: Measure layer contributions using Captum LayerConductance
- `evaluate_linear_probe_accuracy()`: Train linear classifiers on latent representations
- `analyze_head_specialization()`: Analyze functional specialization across attention heads
- `assess_predictive_calibration()`: Evaluate latent space geometry alignment
- `compute_regime_separability_metrics()`: Quantify regime separability using clustering metrics

### 3. Plotting (`plotting.py`)

Visualization functions for all analyses:

- `plot_conductance_by_layer()`: Bar chart of conductance per layer
- `plot_latent_embeddings_2d()`: 2D scatter of latent embeddings colored by regime
- `plot_linear_probe_confusion_matrix()`: Confusion matrices of probe predictions
- `plot_head_specialization_heatmap()`: Heatmap of conductance across layers × heads
- `plot_predictive_calibration_curve()`: Calibration curve of future-return deciles vs mean outcome
- `create_comprehensive_regime_analysis_report()`: Combined visualization report

## Quick Start

### Basic Usage

```python
from src.regime_representation import RegimeRepresentationAnalyzer
from src.models.model import TrainableDT
from src.models.config import DecisionTransformerConfig

# Initialize model
config = DecisionTransformerConfig()
model = TrainableDT(config)

# Initialize analyzer
analyzer = RegimeRepresentationAnalyzer(model=model)

# Run complete analysis
results = analyzer.run_complete_analysis(
    states=states,
    actions=actions,
    returns_to_go=returns_to_go,
    timesteps=timesteps,
    regime_labels=regime_labels,
    label_mapping=label_mapping,
    feature_mapping=feature_mapping,
    future_returns=future_returns
)

# Create visualizations
analyzer.create_visualizations(save_dir='output/')
```

### Command Line Usage

```bash
python src/regime_representation/run_regime_analysis.py \
    --model_path /path/to/model \
    --data_path /path/to/data \
    --output_dir regime_analysis_results
```

## Analysis Details

### 1. Level of Representation Analysis

**Purpose**: Determine whether deeper layers encode regime-level abstractions while shallow layers capture short-term fluctuations.

**Method**: Uses Captum's LayerConductance to measure how much each transformer layer contributes to the final prediction.

**Output**: 
- Bar chart showing conductance values per layer
- Higher conductance in deeper layers suggests regime-level encoding

**Key Findings Expected**:
- Deeper layers should show higher conductance values
- Gradual increase in conductance with layer depth indicates hierarchical representation learning

### 2. Linear Probe Accuracy Analysis

**Purpose**: Evaluate regime information embedded in latent states by training linear classifiers.

**Method**: 
- Extracts hidden representations from each layer
- Trains logistic regression classifiers to predict regime labels
- Uses cross-validation for robust performance estimates

**Output**:
- Classification accuracy, precision, and recall by layer
- Confusion matrices showing classification performance
- Regime separability metrics (silhouette score, Calinski-Harabasz score)

**Key Findings Expected**:
- Higher accuracy in deeper layers indicates better regime encoding
- High out-of-sample accuracy confirms regimes are linearly separable

### 3. Head-wise Specialization Analysis

**Purpose**: Reveal functional specialization where different attention heads focus on different feature types.

**Method**:
- Extracts attention patterns from all heads across all layers
- Computes feature attributions for each head
- Categorizes features (price-level, momentum, volatility, technical indicators)
- Analyzes specialization patterns

**Output**:
- Heatmap showing specialization scores across heads and feature categories
- Ranking of heads by specialization in each category

**Key Findings Expected**:
- Early heads attend to price-level context
- Mid-layer heads focus on momentum oscillators  
- Deeper heads process volatility-adjusted signals

### 4. Predictive Calibration Analysis

**Purpose**: Assess whether latent space geometry aligns with prospective outcomes.

**Method**:
- Groups latent embeddings into bins by deciles of future returns
- Computes calibration curve showing relationship between expected and realized outcomes
- Tests for monotonicity and calibration error

**Output**:
- Calibration curve plot with confidence bands
- Calibration metrics (mean calibration error, monotonicity)
- Correlation analysis between latent dimensions and future returns

**Key Findings Expected**:
- Monotonic calibration curve indicates good latent space geometry
- Low calibration error shows alignment between embeddings and outcomes

## Data Requirements

### Input Data Format

The analysis requires the following inputs:

1. **Model Tensors**:
   - `states`: (batch_size, seq_length, state_dim) - Input state sequences
   - `actions`: (batch_size, seq_length, action_dim) - Action sequences
   - `returns_to_go`: (batch_size, seq_length, 1) - Target return sequences
   - `timesteps`: (batch_size, seq_length) - Timestep sequences

2. **Regime Labels**:
   - `regime_labels`: (batch_size,) - Integer regime labels for each observation
   - `label_mapping`: Dict mapping regime integers to regime names

3. **Feature Information**:
   - `feature_mapping`: Dict mapping feature indices to descriptive names
   - `future_returns`: (batch_size,) - Future returns for calibration analysis

### Regime Definition Options

The module supports multiple approaches for defining market regimes:

1. **Time-based Regimes**:
```python
regime_config = {
    'method': 'time_based',
    'regimes': {
        'bull_market': ['2020-01-01', '2021-12-31'],
        'sideways_uncertain': ['2022-01-01', '2022-12-31'],
        'recovery_growth': ['2023-01-01', '2024-12-31']
    }
}
```

2. **Custom Regime Mapping**:
```python
custom_regime_mapping = {
    'COVID_Crash': [('2020-02-20', '2020-04-30')],
    'Recovery_Bull': [('2020-05-01', '2021-12-31')],
    'Inflation_Uncertainty': [('2022-01-01', '2022-12-31')]
}
```

3. **Volatility-based Regimes** (future implementation):
- Automatic regime detection based on rolling volatility statistics
- Threshold-based regime identification

## Advanced Usage

### Individual Analysis Components

You can run each analysis component individually:

```python
# Level of representation
level_results = analyzer.run_level_of_representation_analysis(
    states, actions, returns_to_go, timesteps
)

# Linear probe accuracy
probe_results = analyzer.run_linear_probe_analysis(
    layer_representations, regime_labels, label_mapping
)

# Head specialization
head_results = analyzer.run_head_specialization_analysis(
    states, actions, returns_to_go, timesteps, feature_mapping
)

# Predictive calibration
calibration_results = analyzer.run_predictive_calibration_analysis(
    layer_representations, future_returns
)
```

### Custom Feature Categories

Define custom feature categories for specialization analysis:

```python
custom_categories = {
    'price_momentum': ['Close', 'RSI', 'MACD'],
    'volatility_risk': ['ATR', 'VIX', 'volatility'],
    'volume_flow': ['Volume', 'OBV', 'CMF'],
    'sentiment': ['Fear_Greed', 'VIX', 'Put_Call_Ratio']
}

specialization_results = analyze_head_specialization(
    attention_activations,
    feature_attributions, 
    feature_mapping,
    categories=custom_categories
)
```

### Custom Visualization Settings

```python
# Individual plots with custom settings
plot_conductance_by_layer(
    conductance_values,
    title="Layer Conductance Analysis - Bull vs Bear Markets",
    figsize=(15, 8),
    show_values=True
)

plot_latent_embeddings_2d(
    embeddings,
    regime_labels,
    label_mapping,
    method='umap',
    max_layers=4
)
```

## Interpretation Guidelines

### Layer Conductance Interpretation

- **High conductance in deeper layers**: Indicates regime-level abstraction encoding
- **Gradual increase with depth**: Shows hierarchical representation learning
- **Sharp increases**: May indicate layer specialization for regime detection

### Linear Probe Interpretation

- **Accuracy > 0.7**: Strong regime separability in latent space
- **Increasing accuracy with depth**: Progressive regime encoding
- **High precision/recall**: Balanced regime discrimination

### Head Specialization Interpretation

- **Clear category specialization**: Functional organization of attention heads
- **Layer-wise progression**: Early layers → price context, deeper layers → complex signals
- **Balanced specialization**: Good coverage of different feature types

### Calibration Interpretation

- **Monotonic curve**: Good latent space geometry
- **Low calibration error (< 0.05)**: Strong alignment with outcomes
- **High correlation (> 0.3)**: Predictive latent representations

## Technical Requirements

### Dependencies

```
torch >= 1.9.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
captum >= 0.4.0
umap-learn >= 0.5.0
```

### Hardware Requirements

- **Memory**: 8GB+ RAM recommended for large datasets
- **GPU**: Optional, but recommended for large models
- **Storage**: 1GB+ for output visualizations and results

### Performance Considerations

- **Batch size**: Reduce for memory constraints
- **Number of layers**: More layers increase analysis depth but computation time
- **Cross-validation folds**: More folds improve probe accuracy estimates but increase runtime

## Example Output

The analysis generates comprehensive visualizations and metrics:

### Generated Files

1. **conductance_by_layer.png**: Bar chart of layer conductance values
2. **latent_embeddings_2d.png**: 2D embeddings colored by regime  
3. **linear_probe_confusion.png**: Confusion matrices for linear probes
4. **head_specialization_heatmap.png**: Specialization heatmap
5. **predictive_calibration_curve.png**: Calibration curve with confidence bands
6. **regime_analysis_comprehensive_report.png**: Combined analysis report

### Metrics Output

```
REGIME REPRESENTATION ANALYSIS SUMMARY
=====================================
• Highest conductance: layer_5 (0.2847)
• Best linear probe accuracy: layer_4 (0.823)
• Calibration error: 0.0234, Monotonic: Yes
• Regime distribution: {0: 67, 1: 64, 2: 69}

INTERPRETATION:
• Higher layer conductance suggests regime-level encoding
• Higher probe accuracy indicates better regime separability  
• Monotonic calibration shows aligned latent geometry
• Head specialization reveals functional organization
```

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce batch size
   - Use CPU instead of GPU for analysis
   - Process data in smaller chunks

2. **Low Linear Probe Accuracy**:
   - Check regime label quality
   - Ensure sufficient samples per regime
   - Verify model is properly trained

3. **No Attention Weights**:
   - Ensure model supports `output_attentions=True`
   - Check transformer architecture compatibility

4. **Poor Calibration**:
   - Verify future returns calculation
   - Check for data leakage
   - Ensure temporal consistency

### Model Compatibility

The module is designed for decision transformer models but can be adapted for:
- Standard transformers with minor modifications
- Other sequence models with attention mechanisms
- Custom transformer architectures

### Extension Points

The module can be extended with:
- Additional regime detection methods
- Custom attribution techniques
- Alternative embedding methods
- Domain-specific feature categories

## Citation

If you use this module in your research, please cite:

```bibtex
@article{regime_representation_analysis,
  title={Latent Dimensions and Regime Representation in Decision Transformers},
  author={Your Name},
  journal={Mathematics Special Issue},
  year={2024}
}
```

## Support

For questions, issues, or contributions:
- Open an issue in the repository
- Contact the development team
- Check the example usage script for common patterns
