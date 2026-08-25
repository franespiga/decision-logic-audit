"""
Main Orchestrator Script for Regime Representation Analysis

This script runs the complete regime representation analysis pipeline,
including all four main components:
1. Level of representation (layer conductance)
2. Linear probe accuracy
3. Head-wise specialization  
4. Predictive calibration in latent space

Usage:
    python run_regime_analysis.py --model_path /path/to/model --data_path /path/to/data
"""



def main():
    """
    Main function for running regime representation analysis from command line.
    """
    parser = argparse.ArgumentParser(description='Run Regime Representation Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared ablation data')
    parser.add_argument('--output_dir', type=str, default='regime_analysis_results',
                       help='Output directory for results and plots')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run analysis on (cpu/cuda)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    
    # Load model (implementation depends on your model saving format)
    print(f"Loading model from: {args.model_path}")
    # model = torch.load(args.model_path, map_location=device)
    # For now, create a dummy model for demonstration
    config = DecisionTransformerConfig()
    model = TrainableDT(config)
    model.to(device)
    model.eval()
    
    # Load prepared data (implementation depends on your data format)
    print(f"Loading data from: {args.data_path}")
    # This would be your actual data loading logic
    # For demonstration, create dummy data
    batch_size = 100
    seq_length = 16
    
    dummy_data = {
        'states': torch.randn(batch_size, seq_length, config.state_dim),
        'actions': torch.randn(batch_size, seq_length, config.act_dim),
        'returns_to_go': torch.randn(batch_size, seq_length, 1),
        'timesteps': torch.arange(seq_length).repeat(batch_size, 1),
        'regime_labels': np.random.randint(0, 3, batch_size),
        'future_returns': np.random.randn(batch_size) * 0.1
    }
    
    # Create label mapping
    label_mapping = {0: 'Bull Market', 1: 'Sideways Uncertain', 2: 'Recovery Growth'}
    
    # Create feature mapping
    feature_mapping = {i: f'feature_{i}' for i in range(config.state_dim)}
    
    # Initialize analyzer
    analyzer = RegimeRepresentationAnalyzer(
        model=model,
        device=device,
        random_state=args.random_state
    )
    
    # Run complete analysis
    print("\nRunning complete regime representation analysis...")
    results = analyzer.run_complete_analysis(
        states=dummy_data['states'],
        actions=dummy_data['actions'],
        returns_to_go=dummy_data['returns_to_go'],
        timesteps=dummy_data['timesteps'],
        regime_labels=dummy_data['regime_labels'],
        label_mapping=label_mapping,
        feature_mapping=feature_mapping,
        future_returns=dummy_data['future_returns']
    )
    
    # Create visualizations
    analyzer.create_visualizations(save_dir=args.output_dir)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'regime_analysis_results.pkl')
    analyzer.save_results(results_path)
    
    print(f"\nAnalysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
