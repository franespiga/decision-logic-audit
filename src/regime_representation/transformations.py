"""
Transformations Module for Regime Representation Analysis

This module contains functions for extracting and transforming model representations
for regime analysis, including layer activations, attention patterns, and latent embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from captum.attr import LayerConductance
import sys
sys.path.append("../../")



def _forward(model,
             states,
             actions,
             returns_to_go,
             timesteps,
             attention_mask=None, **kwargs):
    """
    Forward function for Captum attribution. Returns the predicted action(s) for the last timestep.

    Args:
        model: The decision transformer model
        states: Tensor of states (batch_size, seq_length, state_dim)
        actions: Tensor of actions (batch_size or 1, seq_length, action_dim)
        returns_to_go: Tensor of returns-to-go (batch_size or 1, seq_length, 1)
        timesteps: Tensor of timesteps (batch_size or 1, seq_length)
        attention_mask: Optional attention mask (batch_size or 1, seq_length)
        target_action_idx: Optional specific action index to return

    Returns:
        Predicted actions for the last timestep, either for all dimensions or specific index
    """
    batch_size, seq_length = states.shape[0], states.shape[1]
    device = states.device

    # Ensure all inputs are properly shaped tensors
    if not torch.is_tensor(states):
        states = torch.tensor(states, device=device)
    if not torch.is_tensor(actions):
        actions = torch.tensor(actions, device=device)
    if not torch.is_tensor(returns_to_go):
        returns_to_go = torch.tensor(returns_to_go, device=device)
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device)


    def _expand_tensor(tensor, target_batch_size, seq_len, name):
        """Helper function to expand tensors to match batch size"""
        if tensor.shape[0] != target_batch_size:
            if tensor.shape[0] == 1:
                return tensor.expand(target_batch_size, -1, -1) if tensor.dim() == 3 else tensor.expand(
                    target_batch_size, -1)
            else:
                repeat_factor = target_batch_size // tensor.shape[0]
                remainder = target_batch_size % tensor.shape[0]

                if remainder == 0:
                    return tensor.repeat(repeat_factor, 1, 1) if tensor.dim() == 3 else tensor.repeat(repeat_factor, 1)

                repeated = tensor.repeat(repeat_factor, 1, 1) if tensor.dim() == 3 else tensor.repeat(repeat_factor, 1)
                remainder_slice = tensor[:remainder]
                return torch.cat([repeated, remainder_slice], dim=0)

        if tensor.shape[1] != seq_len:
            return tensor.expand(-1, seq_len, -1) if tensor.dim() == 3 else tensor.expand(-1, seq_len)

        return tensor


    # Expand all tensors to match batch size and sequence length
    actions = _expand_tensor(actions, batch_size, seq_length, 'actions')
    returns_to_go = _expand_tensor(returns_to_go, batch_size, seq_length, 'returns_to_go')
    timesteps = _expand_tensor(timesteps, batch_size, seq_length, 'timesteps')


    # Reshape tensors to ensure consistent dimensions
    if len(actions.shape) == 2:
        actions = actions.unsqueeze(1)  # Add sequence dimension
    if len(returns_to_go.shape) == 2:
        returns_to_go = returns_to_go.unsqueeze(1)  # Add sequence dimension

    # Expand other tensors to match states if needed
    actions = actions.expand(batch_size, seq_length, -1)
    returns_to_go = returns_to_go.expand(batch_size, seq_length, -1)

    # Handle NaN values
    states = torch.nan_to_num(states, nan=0.0)
    actions = torch.nan_to_num(actions, nan=0.0)
    returns_to_go = torch.nan_to_num(returns_to_go, nan=0.0)

    # Handle attention mask separately due to its optional nature
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
    else:
        attention_mask = _expand_tensor(attention_mask, batch_size, seq_length, 'attention_mask')

    if len(timesteps.shape) == 1:
        timesteps = timesteps.unsqueeze(0)  # Add batch dimension
    timesteps = timesteps.expand(batch_size, seq_length)

    # Remove torch.no_grad() to allow gradients for attribution
    out = model.original_forward(
        states=states,
        actions=actions,
        rewards=None,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        output_hidden_states=kwargs.get('output_hidden_states', True),  # This is crucial to get hidden states
        return_dict=kwargs.get('return_dict', True)
    )

    return out



def extract_layer_representations(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    target_layers: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract hidden representations from all transformer layers.

    This function runs the model with output_hidden_states=True and extracts
    the hidden states from each transformer layer for regime analysis.

    Args:
        model: Decision transformer model
        states: Input states tensor (batch_size, seq_length, state_dim)
        actions: Actions tensor (batch_size, seq_length, action_dim)
        returns_to_go: Returns-to-go tensor (batch_size, seq_length, 1)
        timesteps: Timesteps tensor (batch_size, seq_length)
        attention_mask: Optional attention mask
        target_layers: Optional list of specific layer names to extract

    Returns:
        Dictionary mapping layer names to their hidden state tensors
    """
    print(f"Extracting layer representations...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")

    # Ensure consistent tensor shapes
    batch_size, seq_length = states.shape[0], states.shape[-1]

    device = states.device
    layer_representations = dict()

    with torch.no_grad():
        # Forward pass with hidden states extraction
        output = _forward(
            model = model,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            # Extract hidden states from each layer
            for layer_idx, hidden_state in enumerate(output.hidden_states):
                layer_name = f"layer_{layer_idx}"

                if target_layers is None or layer_name in target_layers:
                    # Take the last timestep representation
                    layer_representations[layer_name] = hidden_state[:, -1, :].detach()

            print(f"Extracted representations from {len(layer_representations)} layers")
        else:
            print("Warning: No hidden states found in model output")
            # Fallback: extract from encoder layers directly
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'h'):
                print("Attempting manual layer extraction...")
                # This requires a custom forward pass through each layer
                # Implementation depends on model architecture
                pass

    return layer_representations


# Alias used by regime-sensitivity notebook cells
extract_layer_reps = extract_layer_representations


def prepare_regime_labels(
    dates: Union[List, pd.DatetimeIndex],
    regime_config: Optional[Dict] = None,
    custom_regime_mapping: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Prepare regime labels for classification analysis.
    
    This function assigns regime labels to time periods based on market conditions
    or custom mappings for linear probe analysis.
    
    Args:
        dates: List or DatetimeIndex of observation dates
        regime_config: Configuration for automatic regime detection
        custom_regime_mapping: Manual mapping of date ranges to regime labels
        
    Returns:
        Tuple of (regime_labels, label_mapping) where:
        - regime_labels: Array of integer regime labels
        - label_mapping: Dictionary mapping integers to regime names
    """
    print(f"Preparing regime labels for {len(dates)} observations...")
    
    if custom_regime_mapping is not None:
        # Use custom mapping
        regime_labels = np.zeros(len(dates), dtype=int)
        label_mapping = {}
        
        for regime_id, (regime_name, date_ranges) in enumerate(custom_regime_mapping.items()):
            label_mapping[regime_id] = regime_name
            
            for start_date, end_date in date_ranges:
                mask = (pd.to_datetime(dates) >= pd.to_datetime(start_date)) & \
                       (pd.to_datetime(dates) <= pd.to_datetime(end_date))
                regime_labels[mask] = regime_id
                
    else:
        # Default regime detection based on time periods or volatility
        regime_config = regime_config or {
            'method': 'time_based',
            'regimes': {
                'bull_market': ['2020-01-01', '2021-12-31'],
                'sideways_uncertain': ['2022-01-01', '2022-12-31'], 
                'recovery_growth': ['2023-01-01', '2024-12-31']
            }
        }
        
        if regime_config['method'] == 'time_based':
            regime_labels = np.zeros(len(dates), dtype=int)
            label_mapping = {}
            
            for regime_id, (regime_name, date_range) in enumerate(regime_config['regimes'].items()):
                label_mapping[regime_id] = regime_name
                start_date, end_date = date_range
                
                mask = (pd.to_datetime(dates) >= pd.to_datetime(start_date)) & \
                       (pd.to_datetime(dates) <= pd.to_datetime(end_date))
                regime_labels[mask] = regime_id
                
        else:
            # Could implement volatility-based or other automatic regime detection
            raise NotImplementedError(f"Regime detection method '{regime_config['method']}' not implemented")
    
    # Handle any unlabeled periods
    unlabeled_mask = regime_labels == 0
    if unlabeled_mask.sum() > 0:
        print(f"Warning: {unlabeled_mask.sum()} observations have no regime label (assigned to 'Unknown')")
        if 'Unknown' not in label_mapping.values():
            unknown_id = max(label_mapping.keys()) + 1 if label_mapping else 0
            label_mapping[unknown_id] = 'Unknown'
            regime_labels[unlabeled_mask] = unknown_id
    
    print(f"Regime distribution:")
    for regime_id, regime_name in label_mapping.items():
        count = (regime_labels == regime_id).sum()
        print(f"  {regime_name}: {count} observations ({100*count/len(regime_labels):.1f}%)")
    
    return regime_labels, label_mapping


def create_latent_embeddings(
    layer_representations: Dict[str, torch.Tensor],
    method: str = 'tsne',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Create 2D embeddings of latent representations for visualization.
    
    This function applies dimensionality reduction techniques to high-dimensional
    layer representations for visualization and regime separability analysis.
    
    Args:
        layer_representations: Dictionary of layer representations
        method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        n_components: Number of output dimensions (typically 2 for visualization)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the chosen method
        
    Returns:
        Dictionary mapping layer names to 2D embedding arrays
    """
    print(f"Creating {method.upper()} embeddings with {n_components} components...")
    
    embeddings = {}
    
    for layer_name, representations in layer_representations.items():
        print(f"Processing {layer_name} with shape {representations.shape}")
        
        # Convert to numpy if needed
        if isinstance(representations, torch.Tensor):
            repr_data = representations.cpu().numpy()
        else:
            repr_data = representations
            
        # Handle different methods
        if method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000)
            )
        elif method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1)
            )
        elif method.lower() == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        # Fit and transform
        embedding = reducer.fit_transform(repr_data)
        embeddings[layer_name] = embedding
        
        print(f"  {layer_name}: {repr_data.shape} -> {embedding.shape}")
    
    return embeddings


def compute_predictive_bins(
    future_returns: np.ndarray,
    latent_representations: torch.Tensor,
    n_bins: int = 10,
    method: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predictive bins for calibration analysis.
    
    This function groups latent embeddings into bins based on future returns
    for predictive calibration assessment.
    
    Args:
        future_returns: Array of future returns for each observation
        latent_representations: Latent space representations 
        n_bins: Number of bins to create
        method: Binning method ('quantile', 'uniform')
        
    Returns:
        Tuple of (bin_assignments, bin_edges, bin_centers)
    """
    print(f"Computing predictive bins using {method} method with {n_bins} bins...")
    
    if method == 'quantile':
        # Create quantile-based bins
        bin_edges = np.quantile(future_returns, np.linspace(0, 1, n_bins + 1))
    elif method == 'uniform':
        # Create uniform bins
        bin_edges = np.linspace(future_returns.min(), future_returns.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")
    
    # Assign observations to bins
    bin_assignments = np.digitize(future_returns, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"Bin distribution:")
    for i in range(n_bins):
        count = (bin_assignments == i).sum()
        print(f"  Bin {i}: {count} observations ({100*count/len(bin_assignments):.1f}%)")
    
    return bin_assignments, bin_edges, bin_centers


def extract_feature_importance_by_category(
    attributions: torch.Tensor,
    feature_mapping: Dict[int, str],
    categories: Dict[str, List[str]] = None
) -> Dict[str, float]:
    """
    Group feature attributions by category for head specialization analysis.
    
    This function categorizes features (e.g., price-level, momentum, volatility)
    and computes aggregate attribution scores for head specialization analysis.
    
    Args:
        attributions: Attribution tensor from conductance analysis
        feature_mapping: Mapping from feature indices to feature names
        categories: Dictionary mapping category names to feature patterns
        
    Returns:
        Dictionary mapping category names to aggregated attribution scores
    """
    if categories is None:
        # Default categorization
        categories = {
            'price_level': ['Close', 'Open', 'High', 'Low', 'price'],
            'momentum': ['RSI', 'MACD', 'momentum', 'ROC'],
            'volatility': ['ATR', 'volatility', 'std', 'VIX'],
            'volume': ['Volume', 'volume', 'OBV']
        }
    
    print(f"Categorizing features into {len(categories)} categories...")
    
    # Convert attributions to numpy
    if isinstance(attributions, torch.Tensor):
        attr_data = attributions.detach().cpu().numpy()
    else:
        attr_data = attributions
    
    # Flatten if needed
    if attr_data.ndim > 1:
        attr_data = np.abs(attr_data).mean(axis=0).flatten()
    
    category_scores = {}
    
    for category_name, patterns in categories.items():
        category_indices = []
        
        # Find features matching category patterns
        for feat_idx, feat_name in feature_mapping.items():
            if feat_idx < len(attr_data):
                for pattern in patterns:
                    if pattern.lower() in feat_name.lower():
                        category_indices.append(feat_idx)
                        break
        
        if category_indices:
            # Aggregate attributions for this category
            category_score = np.mean(np.abs(attr_data[category_indices]))
            category_scores[category_name] = category_score
            print(f"  {category_name}: {len(category_indices)} features, score = {category_score:.6f}")
        else:
            category_scores[category_name] = 0.0
            print(f"  {category_name}: No matching features found")
    
    return category_scores

from captum.attr import LayerConductance
import torch
import torch.nn as nn
from typing import Dict, Optional


def compute_layer_conductance(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns_to_go: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    normalize_by_depth: bool = True,
    target_class: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute layer conductance to assess level of representation.

    This function measures how much each transformer layer contributes to the
    final prediction, revealing whether deeper layers encode regime-level
    abstractions while shallow layers capture short-term fluctuations.

    Args:
        model: Decision transformer model
        states: Input states tensor (batch_size, seq_length, state_dim)
        actions: Actions tensor
        returns_to_go: Returns-to-go tensor
        timesteps: Timesteps tensor
        attention_mask: Optional attention mask
        normalize_by_depth: Whether to normalize conductance by layer depth
        target_class: Specific output dimension to target (optional)

    Returns:
        Dictionary mapping layer indices/names to conductance values
    """
    print(f"Computing layer conductance for regime analysis...")
    print(f"Input shapes - States: {states.shape}")

    # Set model to evaluation mode
    model.eval()

    # Ensure inputs don't require gradients (LayerConductance handles this internally)
    states = states.clone().detach()
    actions = actions.clone().detach()
    returns_to_go = returns_to_go.clone().detach()
    timesteps = timesteps.clone().detach()

    if attention_mask is not None:
        attention_mask = attention_mask.clone().detach()


    def forward_wrapper(states, actions, returns_to_go, timesteps,
                        attention_mask = None):
        """
        Forward function for conductance analysis.

        Args:
            input_states: States tensor that LayerConductance will modify

        Returns:
            Scalar output for conductance computation
        """
        # Forward pass through the model
        """
        output = model(
          states=input_states,
          actions=actions,
          returns_to_go=returns_to_go,
          timesteps=timesteps,
          attention_mask=attention_mask,
          output_hidden_states=False,
          return_dict=True
        )
        """
        #states, actions, returns_to_go, timesteps = input_tuple
        batch_size, seq_length = states.shape[0], states.shape[1]

        device = states.device
        print(f"Samples:{batch_size}, Sequence length:{seq_length}")

        # Ensure all inputs are properly shaped tensors
        if not torch.is_tensor(states):
            states = torch.tensor(states, device=device)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, device=device)
        if not torch.is_tensor(returns_to_go):
            returns_to_go = torch.tensor(returns_to_go, device=device)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype =  torch.long,  device=device)

        def _expand_tensor(tensor, target_batch_size, seq_len, name):
            """Helper function to expand tensors to match batch size"""
            if tensor.shape[0] != target_batch_size:
                if tensor.shape[0] == 1:
                    return tensor.expand(target_batch_size, -1, -1) if tensor.dim() == 3 else tensor.expand(
                        target_batch_size, -1)
                else:
                    repeat_factor = target_batch_size // tensor.shape[0]
                    remainder = target_batch_size % tensor.shape[0]

                    if remainder == 0:
                        return tensor.repeat(repeat_factor, 1, 1) if tensor.dim() == 3 else tensor.repeat(repeat_factor, 1)

                    repeated = tensor.repeat(repeat_factor, 1, 1) if tensor.dim() == 3 else tensor.repeat(repeat_factor, 1)
                    remainder_slice = tensor[:remainder]
                    return torch.cat([repeated, remainder_slice], dim=0)

            if tensor.shape[1] != seq_len:
                return tensor.expand(-1, seq_len, -1) if tensor.dim() == 3 else tensor.expand(-1, seq_len)

            return tensor

        # Expand all tensors to match batch size and sequence length
        actions = _expand_tensor(actions, batch_size, seq_length, 'actions')
        returns_to_go = _expand_tensor(returns_to_go, batch_size, seq_length, 'returns_to_go')
        timesteps = _expand_tensor(timesteps, batch_size, seq_length, 'timesteps')

        # Reshape tensors to ensure consistent dimensions
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)  # Add sequence dimension
        if len(returns_to_go.shape) == 2:
            returns_to_go = returns_to_go.unsqueeze(1)  # Add sequence dimension

        # Expand other tensors to match states if needed
        actions = actions.expand(batch_size, seq_length, -1)
        returns_to_go = returns_to_go.expand(batch_size, seq_length, -1)

        # Handle NaN values
        states = torch.nan_to_num(states, nan=0.0)
        actions = torch.nan_to_num(actions, nan=0.0)
        returns_to_go = torch.nan_to_num(returns_to_go, nan=0.0)

        # Handle attention mask separately due to its optional nature
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=states.device)
        else:
            attention_mask = _expand_tensor(attention_mask, batch_size, seq_length, 'attention_mask')

        if len(timesteps.shape) == 1:
            timesteps = timesteps.unsqueeze(0)  # Add batch dimension
        timesteps = torch.tensor(timesteps.expand(batch_size, seq_length), dtype=torch.long, device=states.device)

        output = model.original_forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )

        # Extract action predictions
        action_preds = output.action_preds  # Shape: [batch_size, seq_length, act_dim]

        # For conductance, we need a scalar output
        # Option 1: Sum all predictions
        if target_class is None:
            return action_preds.sum().view(-1)
            # Option 2: Target specific action dimension
        else:
            return action_preds[:, :, target_class].sum().view(-1)



    conductance_results = {}

    # Extract transformer layers from the Decision Transformer
    transformer_layers = None

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'h'):
        transformer_layers = model.encoder.h
        print(f"Found {len(transformer_layers)} transformer layers")
    else:
        print("Warning: Could not access transformer layers")
        return conductance_results


    # Compute conductance for each layer
    for layer_idx, layer in enumerate(transformer_layers):
        try:
            print(f"Computing conductance for layer {layer_idx}...")

            # Create LayerConductance for this specific layer
            layer_conductance = LayerConductance(forward_wrapper, layer)

            input_tuple = (
                states, actions, returns_to_go, timesteps
            )



            # Try multiple baselines to see which gives better signal
            baselines_to_try = [
                ("zero", torch.zeros_like(states)),
                ("mean", torch.full_like(states, states.mean().item())),
                ("random", torch.randn_like(states) * states.std().item() * 0.1)
            ]

            for baseline_name, baseline in baselines_to_try:
                print(f"  Trying {baseline_name} baseline...")

                bl = (
                    baseline,
                    torch.ones_like(actions)/actions.shape[-1],
                    returns_to_go, #torch.zeros_like(returns_to_go),
                    timesteps
                )



                conductance = layer_conductance.attribute(
                    inputs=input_tuple,
                    baselines=bl,
                    target=None,
                    n_steps=50,
                    method='gausslegendre',
                    internal_batch_size=None,
                    attribute_to_layer_input=True  # Attribute to layer output
                )

                # Try different aggregation methods
                conductance_mean = torch.abs(conductance).mean().item()
                conductance_sum = torch.abs(conductance).sum().item()
                conductance_norm = torch.norm(conductance).item()

                print(f"    Mean: {conductance_mean:.2e}")
                print(f"    Sum: {conductance_sum:.2e}")
                print(f"    Norm: {conductance_norm:.2e}")

                # Use the sum or norm if they're larger
                if conductance_sum > conductance_mean:
                    conductance_value = conductance_sum
                else:
                    conductance_value = conductance_norm

            baseline = (
                torch.zeros_like(states),
                torch.ones_like(actions)/actions.shape[-1],
                returns_to_go,
                timesteps
            )

            conductance = layer_conductance.attribute(
                inputs=input_tuple,
                baselines=baseline,
                target=None,  # We handle targeting in forward_wrapper
                n_steps=50,   # Number of integration steps
                method='gausslegendre',
                internal_batch_size=None,
                attribute_to_layer_input=True  # Attribute to layer output
            )

            # Aggregate conductance value
            # Take mean absolute value across all dimensions
            conductance_value = torch.abs(conductance).sum().item()

            # Normalize by layer depth if requested
            if normalize_by_depth:
                # Deeper layers get higher weight in normalization
                depth_weight = (layer_idx + 1) / len(transformer_layers)
                conductance_value = conductance_value / depth_weight

            conductance_results[f"layer_{layer_idx}"] = conductance_value
            print(f"  Layer {layer_idx}: conductance = {conductance_value:.6f}")

        except Exception as e:

            print(f"  Warning: Could not compute conductance for layer {layer_idx}: {e}")
            conductance_results[f"layer_{layer_idx}"] = 0.0
            raise ValueError("Stop")

    print(f"Computed conductance for {len(conductance_results)} layers")
    return conductance_results

