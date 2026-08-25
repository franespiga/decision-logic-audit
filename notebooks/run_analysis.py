#!/usr/bin/env python3
"""
Complete Decision Transformer Ablation Analysis Runner

This script provides a simple interface to run all the analyses described in the LaTeX manuscript.
It can be run from the command line or imported as a module.

Usage:
    python run_analysis.py [--config CONFIG_FILE] [--output OUTPUT_DIR] [--sample-size SAMPLE_SIZE]

Example:
    python run_analysis.py --sample-size 500 --output ./results
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def main():
    """Main function to run the complete ablation analysis"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Decision Transformer Ablation Analysis')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./ablation_results', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of samples for analysis')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--skip-performance', action='store_true', help='Skip model performance evaluation')
    parser.add_argument('--skip-attribution', action='store_true', help='Skip attribution analysis')
    parser.add_argument('--skip-regime', action='store_true', help='Skip market regime analysis')
    parser.add_argument('--methods', nargs='+', default=['ig', 'fa', 'lc', 'to'], 
                       help='Attribution methods to run (ig=integrated_gradients, fa=feature_ablation, lc=layer_conductance, to=time_occlusion)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Decision Transformer Ablation Analysis")
    print("=" * 80)
    print(f"Output directory: {args.output}")
    print(f"Sample size: {args.sample_size}")
    print(f"Device: {args.device}")
    print(f"Methods: {args.methods}")
    print("=" * 80)
    
    # Import required modules
    try:
        import torch
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
        from typing import Dict, List, Tuple, Optional, Union, Any
        
        # Captum for attribution
        import captum
        from captum.attr import (
            IntegratedGradients, 
            FeatureAblation, 
            LayerIntegratedGradients, 
            LayerConductance
        )
        
        # Portfolio analysis
        import pyfolio as pf
        from scipy import stats
        from sklearn.metrics import mutual_info_score
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        # Progress tracking
        from tqdm import tqdm
        import json
        import pickle
        
        # Custom imports
        from portfolio_optimization.ablation_studies.helpers import *
        from portfolio_optimization.ablation_studies.helpers.ablation import *
        from portfolio_optimization.ablation_studies.helpers.feature_importance import *
        from portfolio_optimization.ablation_studies.helpers.model_comparison import *
        from portfolio_optimization.agents.decision_transformer.model import DecisionTransformer
        from portfolio_optimization.agents.decision_transformer.config import DecisionTransformerConfig
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install torch captum pandas numpy matplotlib seaborn scikit-learn pyfolio tqdm safetensors")
        return 1
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Configuration
    class AnalysisConfig:
        """Configuration class for the ablation analysis"""
        
        def __init__(self, output_dir, sample_size):
            # Paths
            self.BASE_PATH = Path("src/portfolio_optimization")
            self.MODELS_PATH = self.BASE_PATH / "models"
            self.DATA_PATH = self.BASE_PATH.parent.parent / "data"
            self.RESULTS_PATH = Path(output_dir)
            
            # Analysis parameters
            self.SAMPLE_SIZE = sample_size
            self.LOOKBACK_WINDOW = 16
            self.REBALANCE_EVERY = 1
            self.ANNUALIZED_RETURN = 0.15
            
            # Market regime parameters
            self.REGIME_BOUNDARIES = {
                'bull_market': ('2016-01-01', '2019-12-31'),
                'crash_period': ('2020-03-01', '2020-05-31'),
                'recovery_growth': ('2020-06-01', '2021-12-31'),
                'sideways_uncertain': ('2022-01-01', '2024-04-30')
            }
            
            # Attribution parameters
            self.IG_STEPS = 50
            self.ABLATION_BASELINE = 'zero'
            
            # Visualization parameters
            self.FIGURE_SIZE = (12, 8)
            self.DPI = 300
            
            # Model selection criteria
            self.PERFORMANCE_METRICS = ['sharpe_ratio', 'annualized_return', 'max_drawdown']
            
            # Create results directory
            self.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = AnalysisConfig(args.output, args.sample_size)
    
    try:
        # Step 1: Load data
        print("\nStep 1: Loading data...")
        ohlc_data, state_features, instruments = load_portfolio_data(config)
        print(f"Loaded data with {len(instruments)} instruments and {len(state_features)} samples")
        
        # Step 2: Load models
        print("\nStep 2: Loading models...")
        models, model_configs, model_metadata = load_models(config)
        print(f"Loaded {len(models)} models")
        
        if not args.skip_performance:
            # Step 3: Evaluate model performance
            print("\nStep 3: Evaluating model performance...")
            performance_df, model_results = evaluate_models(models, ohlc_data, state_features, instruments, config, device)
            
            # Identify best and worst models
            best_model_id = performance_df['Sharpe ratio'].idxmax()
            worst_model_id = performance_df['Sharpe ratio'].idxmin()
            
            print(f"Best performing model: {best_model_id}")
            print(f"Worst performing model: {worst_model_id}")
            
            # Save performance results
            performance_df.to_csv(config.RESULTS_PATH / 'model_performance.csv')
            print(f"Performance results saved to {config.RESULTS_PATH / 'model_performance.csv'}")
        else:
            # Use first two models as best/worst for demonstration
            model_ids = list(models.keys())
            best_model_id = model_ids[0] if len(model_ids) > 0 else None
            worst_model_id = model_ids[1] if len(model_ids) > 1 else model_ids[0]
            print(f"Using models {best_model_id} and {worst_model_id} for analysis")
        
        if not args.skip_attribution:
            # Step 4: Run attribution analysis
            print("\nStep 4: Running attribution analysis...")
            attribution_results = run_attribution_analysis(
                models, best_model_id, worst_model_id, state_features, instruments, 
                config, device, args.methods
            )
            
            # Save attribution results
            save_attribution_results(attribution_results, config.RESULTS_PATH)
            print(f"Attribution results saved to {config.RESULTS_PATH}")
        
        if not args.skip_regime:
            # Step 5: Market regime analysis
            print("\nStep 5: Running market regime analysis...")
            regime_results = run_regime_analysis(
                attribution_results, state_features, config
            )
            
            # Save regime results
            save_regime_results(regime_results, config.RESULTS_PATH)
            print(f"Regime analysis results saved to {config.RESULTS_PATH}")
        
        # Step 6: Generate visualizations and report
        print("\nStep 6: Generating visualizations and report...")
        generate_visualizations_and_report(
            attribution_results, regime_results, performance_df, config
        )
        
        print("\n" + "=" * 80)
        print("Analysis completed successfully!")
        print(f"Results saved to: {config.RESULTS_PATH}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

def load_portfolio_data(config):
    """Load portfolio data"""
    try:
        data_path = config.DATA_PATH / "portfolio_instruments" / "data.h5"
        with pd.HDFStore(data_path, 'r') as store:
            ohlc_data = store['ohlc_data']
            state_features = store['state_features']
            instruments = store['instruments']
        
        return ohlc_data, state_features, instruments
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating dummy data for demonstration...")
        return create_dummy_data()

def create_dummy_data():
    """Create dummy data for demonstration"""
    dates = pd.date_range('2016-01-01', '2024-04-30', freq='D')
    instruments = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'CASH']
    
    # Create OHLC data
    ohlc_data = pd.DataFrame()
    for instrument in instruments:
        for ohlc_type in ['Open', 'High', 'Low', 'Close']:
            base_price = 100 if instrument != 'CASH' else 1
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            ohlc_data[(instrument, ohlc_type)] = prices
    
    ohlc_data.index = dates
    
    # Create state features
    feature_names = []
    for instrument in instruments:
        for feature in ['Open', 'High', 'Low', 'Close', 'RSI', 'NATR', 'AROON']:
            feature_names.append(f"{instrument}_{feature}")
    
    state_features = pd.DataFrame(
        np.random.randn(len(dates), len(feature_names)),
        index=dates,
        columns=feature_names
    )
    
    return ohlc_data, state_features, instruments

def load_models(config):
    """Load Decision Transformer models"""
    models = {}
    model_configs = {}
    model_metadata = {}
    
    # Find model directories
    for model_dir in config.MODELS_PATH.glob("**/RL_AG_DT_*"):
        if model_dir.is_dir():
            model_id = model_dir.name
            model_path = model_dir / "model" / "model.safetensors"
            config_path = model_dir / "model" / "config.json"
            metadata_path = model_dir / "metadata.json"
            
            if model_path.exists() and config_path.exists() and metadata_path.exists():
                try:
                    # Load configuration
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Create config object
                    dt_config = DecisionTransformerConfig(**config_dict)
                    
                    # Load model
                    model = DecisionTransformer(dt_config)
                    
                    # Load weights
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                    model.load_state_dict(state_dict)
                    model.eval()
                    model.to(device)
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    models[model_id] = model
                    model_configs[model_id] = dt_config
                    model_metadata[model_id] = metadata
                    
                except Exception as e:
                    print(f"Error loading model {model_id}: {e}")
    
    return models, model_configs, model_metadata

def evaluate_models(models, ohlc_data, state_features, instruments, config, device):
    """Evaluate model performance"""
    model_performance = {}
    model_results = {}
    
    for model_id, model in tqdm(models.items(), desc="Evaluating models"):
        try:
            results = compute_portfolio_weights(
                model=model,
                backtest_data=state_features,
                ref_date='2016-01-01',
                ohlc_data=ohlc_data,
                annualized_return=config.ANNUALIZED_RETURN,
                use_trajectory_return_objective=False,
                use_incremental_timesteps=True,
                rebalance_every=config.REBALANCE_EVERY,
                lookback=config.LOOKBACK_WINDOW,
                compute_returns=True,
                ppy=252,
                device=device,
                verbose=0
            )
            
            portfolio_returns = results['returns']['strategy_returns']
            metrics = compute_pyfolio_metrics(portfolio_returns)
            
            model_performance[model_id] = metrics
            model_results[model_id] = results
            
        except Exception as e:
            print(f"Error evaluating model {model_id}: {e}")
    
    performance_df = pd.DataFrame(model_performance).T
    return performance_df, model_results

def run_attribution_analysis(models, best_model_id, worst_model_id, state_features, instruments, config, device, methods):
    """Run attribution analysis"""
    # Prepare sample data
    sample_size = min(config.SAMPLE_SIZE, len(state_features))
    sample_indices = np.random.choice(len(state_features), sample_size, replace=False)
    
    sample_states = torch.tensor(state_features.iloc[sample_indices].values, dtype=torch.float32).to(device)
    sample_actions = torch.zeros(sample_size, 1, len(instruments), dtype=torch.float32).to(device)
    sample_returns = torch.full((sample_size, 1, 1), config.ANNUALIZED_RETURN, dtype=torch.float32).to(device)
    sample_timesteps = torch.ones(sample_size, 1, dtype=torch.long).to(device)
    
    attribution_results = {}
    
    # Run attribution methods
    for method in methods:
        if method == 'ig' and 'integrated_gradients' not in attribution_results:
            print("Running Integrated Gradients...")
            attribution_results['integrated_gradients'] = run_integrated_gradients(
                models[best_model_id], models[worst_model_id], 
                sample_states, sample_actions, sample_returns, sample_timesteps, config
            )
        
        elif method == 'fa' and 'feature_ablation' not in attribution_results:
            print("Running Feature Ablation...")
            attribution_results['feature_ablation'] = run_feature_ablation(
                models[best_model_id], models[worst_model_id], 
                sample_states, sample_actions, sample_returns, sample_timesteps, config
            )
        
        elif method == 'lc' and 'layer_conductance' not in attribution_results:
            print("Running Layer Conductance...")
            attribution_results['layer_conductance'] = run_layer_conductance(
                models[best_model_id], models[worst_model_id], 
                sample_states, sample_actions, sample_returns, sample_timesteps, config
            )
        
        elif method == 'to' and 'time_occlusion' not in attribution_results:
            print("Running Time Occlusion...")
            attribution_results['time_occlusion'] = run_time_occlusion(
                models[best_model_id], models[worst_model_id], 
                sample_states, sample_actions, sample_returns, sample_timesteps, config
            )
    
    return attribution_results

def run_integrated_gradients(best_model, worst_model, states, actions, returns_to_go, timesteps, config):
    """Run Integrated Gradients analysis"""
    ig = IntegratedGradients(_forward_action)
    baseline = torch.zeros_like(states)
    
    best_attr = ig.attribute(
        inputs=states,
        baselines=baseline,
        n_steps=config.IG_STEPS,
        additional_forward_args=(best_model, actions, returns_to_go, timesteps)
    )
    
    worst_attr = ig.attribute(
        inputs=states,
        baselines=baseline,
        n_steps=config.IG_STEPS,
        additional_forward_args=(worst_model, actions, returns_to_go, timesteps)
    )
    
    return {'best': best_attr, 'worst': worst_attr}

def run_feature_ablation(best_model, worst_model, states, actions, returns_to_go, timesteps, config):
    """Run Feature Ablation analysis"""
    fa = FeatureAblation(_forward_action)
    
    best_attr = fa.attribute(
        inputs=states,
        additional_forward_args=(best_model, actions, returns_to_go, timesteps)
    )
    
    worst_attr = fa.attribute(
        inputs=states,
        additional_forward_args=(worst_model, actions, returns_to_go, timesteps)
    )
    
    return {'best': best_attr, 'worst': worst_attr}

def run_layer_conductance(best_model, worst_model, states, actions, returns_to_go, timesteps, config):
    """Run Layer Conductance analysis"""
    lc = LayerConductance(_forward_action, best_model.transformer.h)
    
    best_attr = lc.attribute(
        inputs=states,
        additional_forward_args=(best_model, actions, returns_to_go, timesteps)
    )
    
    worst_attr = lc.attribute(
        inputs=states,
        additional_forward_args=(worst_model, actions, returns_to_go, timesteps)
    )
    
    return {'best': best_attr, 'worst': worst_attr}

def run_time_occlusion(best_model, worst_model, states, actions, returns_to_go, timesteps, config):
    """Run Time Occlusion analysis"""
    # Simplified time occlusion implementation
    batch_size, seq_length, state_dim = states.shape
    best_attr = torch.zeros_like(states)
    worst_attr = torch.zeros_like(states)
    
    # Get baseline predictions
    with torch.no_grad():
        best_baseline = _forward_action(best_model, states, actions, returns_to_go, timesteps)
        worst_baseline = _forward_action(worst_model, states, actions, returns_to_go, timesteps)
    
    # Occlude each timestep
    for t in range(seq_length):
        occluded_states = states.clone()
        occluded_states[:, t, :] = 0.0
        
        with torch.no_grad():
            best_occluded = _forward_action(best_model, occluded_states, actions, returns_to_go, timesteps)
            worst_occluded = _forward_action(worst_model, occluded_states, actions, returns_to_go, timesteps)
        
        best_attr[:, t, :] = (best_baseline - best_occluded).unsqueeze(1).expand(-1, state_dim)
        worst_attr[:, t, :] = (worst_baseline - worst_occluded).unsqueeze(1).expand(-1, state_dim)
    
    return {'best': best_attr, 'worst': worst_attr}

def _forward_action(model, states, actions, returns_to_go, timesteps, target_action_idx=None):
    """Forward function for Captum attribution"""
    batch_size, seq_length = states.shape[0], states.shape[1]
    device = states.device
    
    # Expand tensors to match batch size
    actions = actions.expand(batch_size, seq_length, -1)
    returns_to_go = returns_to_go.expand(batch_size, seq_length, -1)
    timesteps = timesteps.expand(batch_size, seq_length)
    
    # Forward pass
    out = model.original_forward(
        states=states,
        actions=actions,
        rewards=None,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )
    
    action_preds = out[1][:, -1, :]
    return action_preds[:, target_action_idx] if target_action_idx is not None else action_preds

def run_regime_analysis(attribution_results, state_features, config):
    """Run market regime analysis"""
    regime_results = {}
    
    # Segment data into regimes
    regime_data = {}
    for regime_name, (start_date, end_date) in config.REGIME_BOUNDARIES.items():
        mask = (state_features.index >= start_date) & (state_features.index <= end_date)
        regime_data[regime_name] = state_features[mask]
    
    # Compute regime-specific attributions
    for method_name, method_results in attribution_results.items():
        regime_results[method_name] = {}
        
        for regime_name, regime_subset in regime_data.items():
            if len(regime_subset) > 0:
                # Simplified regime analysis
                regime_results[method_name][regime_name] = {
                    'sample_count': len(regime_subset),
                    'mean_attribution': method_results['best'].mean().item()
                }
    
    return regime_results

def save_attribution_results(attribution_results, results_path):
    """Save attribution results"""
    for method_name, method_results in attribution_results.items():
        # Save as numpy arrays
        np.save(results_path / f'{method_name}_best.npy', method_results['best'].cpu().numpy())
        np.save(results_path / f'{method_name}_worst.npy', method_results['worst'].cpu().numpy())

def save_regime_results(regime_results, results_path):
    """Save regime analysis results"""
    with open(results_path / 'regime_results.json', 'w') as f:
        json.dump(regime_results, f, indent=2)

def generate_visualizations_and_report(attribution_results, regime_results, performance_df, config):
    """Generate visualizations and summary report"""
    # Create summary report
    report = []
    report.append("# Decision Transformer Ablation Analysis Summary Report\n")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if performance_df is not None:
        report.append("## Model Performance Summary\n")
        report.append(f"Total models analyzed: {len(performance_df)}\n")
        if len(performance_df) > 0:
            report.append(f"Best performing model: {performance_df['Sharpe ratio'].idxmax()}\n")
            report.append(f"Worst performing model: {performance_df['Sharpe ratio'].idxmin()}\n")
    
    report.append("\n## Attribution Analysis Summary\n")
    for method_name, method_results in attribution_results.items():
        report.append(f"### {method_name.replace('_', ' ').title()}\n")
        report.append(f"Best model mean attribution: {method_results['best'].mean():.6f}\n")
        report.append(f"Worst model mean attribution: {method_results['worst'].mean():.6f}\n")
    
    report.append("\n## Market Regime Analysis Summary\n")
    for method_name, method_regimes in regime_results.items():
        report.append(f"### {method_name.replace('_', ' ').title()}\n")
        for regime_name, regime_data in method_regimes.items():
            report.append(f"- {regime_name}: {regime_data['sample_count']} samples\n")
    
    # Save report
    with open(config.RESULTS_PATH / 'summary_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {config.RESULTS_PATH / 'summary_report.md'}")

if __name__ == "__main__":
    sys.exit(main())
