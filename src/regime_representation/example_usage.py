"""
Example Usage of Regime Representation Analysis

This script demonstrates how to use the regime representation module
to conduct comprehensive analysis of latent dimensions and regime representation
in decision transformer models.
"""

import torch
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append("../../")
sys.path.append("../")

# Import the regime representation analyzer
from .run_regime_analysis import RegimeRepresentationAnalyzer

# Import model utilities
from ..models.model import TrainableDT
from ..models.config import DecisionTransformerConfig
from ..ablation.preprocessing import prepare_ablation_inputs


def create_demo_data(config: DecisionTransformerConfig, n_samples: int = 200):
    """
    Create demonstration data for regime analysis.
    
    This function creates synthetic data that simulates different market regimes
    for demonstrating the regime representation analysis capabilities.
    """
    print(f"Creating demo data with {n_samples} samples...")
    
    # Create synthetic market data with distinct regimes
    regime_periods = {
        0: (0, n_samples // 3),          # Bull market (first third)
        1: (n_samples // 3, 2 * n_samples // 3),  # Sideways/uncertain (middle third)
        2: (2 * n_samples // 3, n_samples)       # Recovery/growth (last third)
    }
    
    # Initialize data structures
    states = torch.zeros(n_samples, 16, config.state_dim)  # 16 timesteps lookback
    actions = torch.zeros(n_samples, 16, config.act_dim)
    returns_to_go = torch.zeros(n_samples, 16, 1)
    timesteps = torch.arange(16).repeat(n_samples, 1)
    
    regime_labels = np.zeros(n_samples, dtype=int)
    future_returns = np.zeros(n_samples)
    
    # Generate regime-specific patterns
    for regime_id, (start_idx, end_idx) in regime_periods.items():
        regime_size = end_idx - start_idx
        
        if regime_id == 0:  # Bull market
            # Higher returns, lower volatility, momentum indicators
            base_return = 0.08
            volatility = 0.15
            momentum_factor = 1.2
        elif regime_id == 1:  # Sideways/uncertain
            # Lower returns, higher volatility, mixed signals
            base_return = 0.02
            volatility = 0.25
            momentum_factor = 0.8
        else:  # Recovery/growth
            # Moderate returns, moderate volatility, strong signals
            base_return = 0.12
            volatility = 0.20
            momentum_factor = 1.1
        
        # Generate regime-specific features
        for i in range(start_idx, end_idx):
            # Price-level features (first quarter of features)
            price_features = config.state_dim // 4
            states[i, :, :price_features] = torch.randn(16, price_features) * volatility + base_return
            
            # Momentum features (second quarter)
            momentum_start = price_features
            momentum_end = price_features * 2
            momentum_signals = torch.randn(16, price_features) * momentum_factor
            states[i, :, momentum_start:momentum_end] = momentum_signals
            
            # Volatility features (third quarter)
            vol_start = momentum_end
            vol_end = price_features * 3
            vol_signals = torch.abs(torch.randn(16, price_features)) * volatility
            states[i, :, vol_start:vol_end] = vol_signals
            
            # Technical indicators (last quarter)
            tech_start = vol_end
            tech_signals = torch.randn(16, config.state_dim - tech_start) * 0.5
            states[i, :, tech_start:] = tech_signals
            
            # Generate actions (portfolio weights)
            if regime_id == 0:  # Bull market - more aggressive
                actions[i, :, :] = torch.softmax(torch.randn(16, config.act_dim) + 1.0, dim=-1)
            elif regime_id == 1:  # Sideways - more conservative
                actions[i, :, :] = torch.softmax(torch.randn(16, config.act_dim) - 0.5, dim=-1)
            else:  # Recovery - balanced
                actions[i, :, :] = torch.softmax(torch.randn(16, config.act_dim), dim=-1)
            
            # Returns to go
            returns_to_go[i, :, 0] = base_return + torch.randn(16) * 0.02
            
            # Regime labels and future returns
            regime_labels[i] = regime_id
            future_returns[i] = base_return + np.random.randn() * volatility
    
    # Create feature mapping
    feature_mapping = {}
    feature_idx = 0
    
    # Price features
    for i in range(config.state_dim // 4):
        feature_mapping[feature_idx] = f"Price_Level_{i}"
        feature_idx += 1
    
    # Momentum features
    for i in range(config.state_dim // 4):
        feature_mapping[feature_idx] = f"Momentum_RSI_{i}"
        feature_idx += 1
    
    # Volatility features
    for i in range(config.state_dim // 4):
        feature_mapping[feature_idx] = f"Volatility_ATR_{i}"
        feature_idx += 1
    
    # Technical indicators
    remaining_features = config.state_dim - feature_idx
    for i in range(remaining_features):
        feature_mapping[feature_idx] = f"Technical_CCI_{i}"
        feature_idx += 1
    
    # Label mapping
    label_mapping = {
        0: 'Bull Market',
        1: 'Sideways Uncertain', 
        2: 'Recovery Growth'
    }
    
    # Create dates
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    demo_data = {
        'states': states,
        'actions': actions,
        'returns_to_go': returns_to_go,
        'timesteps': timesteps,
        'regime_labels': regime_labels,
        'label_mapping': label_mapping,
        'feature_mapping': feature_mapping,
        'future_returns': future_returns,
        'dates': dates
    }
    
    print(f"Demo data created:")
    print(f"  States shape: {states.shape}")
    print(f"  Regime distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    return demo_data


def demonstrate_individual_analyses():
    """
    Demonstrate each analysis component individually.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING INDIVIDUAL ANALYSES")
    print("="*80)
    
    # Setup
    config = DecisionTransformerConfig(
        state_dim=64,  # Larger state for better demonstration
        act_dim=4,
        hidden_size=128,
        n_layer=6,     # More layers for better analysis
        n_head=4       # Multiple heads for specialization analysis
    )
    
    model = TrainableDT(config)
    model.eval()
    
    # Create demo data
    demo_data = create_demo_data(config, n_samples=150)
    
    # Initialize analyzer
    analyzer = RegimeRepresentationAnalyzer(
        model=model,
        device=torch.device('cpu'),
        random_state=42
    )
    
    # 1. Level of Representation Analysis
    print("\n1. LEVEL OF REPRESENTATION ANALYSIS")
    print("-" * 50)
    level_results = analyzer.run_level_of_representation_analysis(
        demo_data['states'],
        demo_data['actions'], 
        demo_data['returns_to_go'],
        demo_data['timesteps']
    )
    
    print("\nLayer conductance values:")
    for layer, conductance in level_results['conductance_values'].items():
        print(f"  {layer}: {conductance:.6f}")
    
    # 2. Linear Probe Analysis
    print("\n2. LINEAR PROBE ACCURACY ANALYSIS")
    print("-" * 50)
    probe_results = analyzer.run_linear_probe_analysis(
        level_results['layer_representations'],
        demo_data['regime_labels'],
        demo_data['label_mapping']
    )
    
    print("\nLinear probe accuracy by layer:")
    for layer, metrics in probe_results['probe_results'].items():
        accuracy = metrics.get('test_accuracy', 0)
        print(f"  {layer}: {accuracy:.3f}")
    
    # 3. Head Specialization Analysis
    print("\n3. HEAD-WISE SPECIALIZATION ANALYSIS")
    print("-" * 50)
    head_results = analyzer.run_head_specialization_analysis(
        demo_data['states'],
        demo_data['actions'],
        demo_data['returns_to_go'],
        demo_data['timesteps'],
        demo_data['feature_mapping']
    )
    
    print("\nTop specialized heads by category:")
    specialization = head_results['specialization_results']
    if specialization:
        categories = list(list(specialization.values())[0].keys())
        for category in categories:
            scores = [(head, scores.get(category, 0)) for head, scores in specialization.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"  {category}: {scores[0][0]} ({scores[0][1]:.4f})")
    
    # 4. Predictive Calibration Analysis
    print("\n4. PREDICTIVE CALIBRATION ANALYSIS")
    print("-" * 50)
    calibration_results = analyzer.run_predictive_calibration_analysis(
        level_results['layer_representations'],
        demo_data['future_returns'],
        demo_data['dates']
    )
    
    cal_metrics = calibration_results['calibration_results']
    print(f"\nCalibration metrics:")
    print(f"  Mean calibration error: {cal_metrics.get('mean_calibration_error', 0):.4f}")
    print(f"  Is monotonic: {cal_metrics.get('is_monotonic', False)}")
    print(f"  Overall correlation: {cal_metrics.get('overall_spearman_correlation', 0):.4f}")
    
    return analyzer


def demonstrate_complete_analysis():
    """
    Demonstrate the complete analysis pipeline.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING COMPLETE ANALYSIS PIPELINE")
    print("="*80)
    
    # Setup
    config = DecisionTransformerConfig(
        state_dim=48,
        act_dim=4,
        hidden_size=128,
        n_layer=4,
        n_head=4
    )
    
    model = TrainableDT(config)
    model.eval()
    
    # Create demo data
    demo_data = create_demo_data(config, n_samples=200)
    
    # Initialize analyzer
    analyzer = RegimeRepresentationAnalyzer(
        model=model,
        device=torch.device('cpu'),
        random_state=42
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        states=demo_data['states'],
        actions=demo_data['actions'],
        returns_to_go=demo_data['returns_to_go'],
        timesteps=demo_data['timesteps'],
        regime_labels=demo_data['regime_labels'],
        label_mapping=demo_data['label_mapping'],
        feature_mapping=demo_data['feature_mapping'],
        future_returns=demo_data['future_returns'],
        dates=demo_data['dates']
    )
    
    # Print summary
    print("\nANALYSIS SUMMARY:")
    print("-" * 40)
    
    metadata = results['metadata']
    print(f"Samples analyzed: {metadata['n_samples']}")
    print(f"Regimes identified: {metadata['n_regimes']}")
    print(f"Regime distribution: {metadata['regime_distribution']}")
    
    # Best performing layers
    level_data = results['level_of_representation']
    best_conductance_layer = max(level_data['conductance_values'], 
                               key=level_data['conductance_values'].get)
    print(f"Highest conductance layer: {best_conductance_layer}")
    
    probe_data = results['linear_probe_accuracy']
    best_probe_layer = max(probe_data['probe_results'], 
                          key=lambda x: probe_data['probe_results'][x].get('test_accuracy', 0))
    best_accuracy = probe_data['probe_results'][best_probe_layer]['test_accuracy']
    print(f"Best linear probe: {best_probe_layer} ({best_accuracy:.3f} accuracy)")
    
    cal_data = results['predictive_calibration']
    cal_error = cal_data['calibration_results']['mean_calibration_error']
    is_monotonic = cal_data['calibration_results']['is_monotonic']
    print(f"Calibration error: {cal_error:.4f}, Monotonic: {is_monotonic}")
    
    return analyzer, results


def demonstrate_visualizations():
    """
    Demonstrate the visualization capabilities.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING VISUALIZATION CAPABILITIES")
    print("="*80)
    
    # Run complete analysis first
    analyzer, results = demonstrate_complete_analysis()
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations(
        save_dir="regime_analysis_demo_output",
        create_comprehensive_report=True
    )
    
    print("\nVisualization files created in 'regime_analysis_demo_output/' directory:")
    import os
    if os.path.exists("regime_analysis_demo_output"):
        files = os.listdir("regime_analysis_demo_output")
        for file in sorted(files):
            print(f"  - {file}")


def demonstrate_custom_regime_definition():
    """
    Demonstrate how to use custom regime definitions.
    """
    print("\n" + "="*80)
    print("DEMONSTRATING CUSTOM REGIME DEFINITIONS")
    print("="*80)
    
    # Create sample dates
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Define custom regime mapping
    custom_regime_mapping = {
        'COVID_Crash': [('2020-02-20', '2020-04-30')],
        'Recovery_Bull': [('2020-05-01', '2021-12-31')], 
        'Inflation_Uncertainty': [('2022-01-01', '2022-12-31')],
        'Stabilization': [('2023-01-01', '2023-12-31')]
    }
    
    # Import the function
    from .transformations import prepare_regime_labels
    
    # Prepare regime labels
    regime_labels, label_mapping = prepare_regime_labels(
        dates=dates,
        custom_regime_mapping=custom_regime_mapping
    )
    
    print(f"\nCustom regime analysis:")
    print(f"Total periods: {len(dates)}")
    print(f"Regime distribution:")
    for regime_id, regime_name in label_mapping.items():
        count = (regime_labels == regime_id).sum()
        percentage = 100 * count / len(regime_labels)
        print(f"  {regime_name}: {count} days ({percentage:.1f}%)")


def main():
    """
    Main demonstration function.
    """
    print("REGIME REPRESENTATION ANALYSIS - EXAMPLE USAGE")
    print("=" * 80)
    print("This script demonstrates the comprehensive regime representation")
    print("analysis capabilities for decision transformer models.")
    print("=" * 80)
    
    try:
        # 1. Demonstrate individual analyses
        analyzer = demonstrate_individual_analyses()
        
        # 2. Demonstrate complete analysis pipeline
        demonstrate_complete_analysis()
        
        # 3. Demonstrate visualizations
        demonstrate_visualizations()
        
        # 4. Demonstrate custom regime definitions
        demonstrate_custom_regime_definition()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey capabilities demonstrated:")
        print("✓ Layer conductance analysis (level of representation)")
        print("✓ Linear probe accuracy evaluation")
        print("✓ Head-wise specialization analysis")
        print("✓ Predictive calibration assessment")
        print("✓ Comprehensive visualization generation")
        print("✓ Custom regime definition support")
        print("\nCheck the 'regime_analysis_demo_output/' directory for generated plots!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
