import torch
from captum.attr import IntegratedGradients, FeatureAblation, LayerIntegratedGradients, LayerConductance, Saliency, DeepLift, GradientShap, Lime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from scipy import stats
import pandas as pd

# Helper: Forward function for Captum (returns action_preds for a specific action index)
def _forward_action(model, states, actions, returns_to_go, timesteps, attention_mask=None, target_action_idx=None):
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
        output_hidden_states=True,  # This is crucial to get hidden states
        return_dict=True,
    )

    # Return either specific action index or all actions for the last timestep
    action_preds = out[1][:, -1, :]
    return action_preds[:, target_action_idx] if target_action_idx is not None else action_preds

# 1. Feature importance per technical indicator/instrument
def feature_importance_per_indicator(model, states, actions, returns_to_go, timesteps, baseline=None, target_action_idx=None):
    """
    Computes feature importance for each technical indicator per instrument using FeatureAblation.
    Returns attributions per feature for the last action.
    """
    print(f"Starting feature importance computation...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")
    print(f"Returns to go: {returns_to_go.shape}, Timesteps: {timesteps.shape}")

    # Ensure states requires gradients for attribution
    states = states.clone().detach().requires_grad_(True)

    def forward_wrapper(input_states):
        return _forward_action(model, input_states, actions, returns_to_go, timesteps, target_action_idx=target_action_idx)

    ablator = FeatureAblation(forward_wrapper)
    if baseline is None:
        baseline = torch.zeros_like(states)

    print(f"Running feature ablation with baseline shape: {baseline.shape}")
    attributions = ablator.attribute(states, baselines=baseline)
    print(f"Attribution computation completed. Output shape: {attributions.shape}")
    return attributions

# 2. Market regime in latent space
def latent_space_market_regime(model, states, actions, returns_to_go, timesteps, market_labels=None):
    """
    Extracts hidden states from the transformer and returns them for visualization.
    """
    print(f"Extracting latent representations...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")
    print(f"Returns to go: {returns_to_go.shape}, Timesteps: {timesteps.shape}")

    # Ensure all tensors have consistent shapes
    batch_size = states.shape[0]
    seq_length = states.shape[1]

    # Reshape tensors to ensure consistent dimensions
    if len(actions.shape) == 2:
        actions = actions.unsqueeze(1)  # Add sequence dimension
    if len(returns_to_go.shape) == 2:
        returns_to_go = returns_to_go.unsqueeze(1)  # Add sequence dimension

    # Expand other tensors to match states if needed
    actions = actions.expand(batch_size, seq_length, -1)
    returns_to_go = returns_to_go.expand(batch_size, seq_length, -1)

    if len(timesteps.shape) == 1:
        timesteps = timesteps.unsqueeze(0)  # Add batch dimension
    timesteps = timesteps.expand(batch_size, seq_length)

    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=states.device)


    with torch.no_grad():
        out = model.original_forward(
            states=states,
            actions=actions,
            rewards=None,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            output_hidden_states=True,  # This is crucial to get hidden states
            return_dict=True,
        )

        # Check if hidden_states are available
        if 'hidden_states' in out:
            # out['hidden_states'] is a tuple of (layer, batch, seq, hidden)
            # We'll use the last hidden state for the last timestep
            last_hidden = out['hidden_states'][-1][:, -1, :].detach().numpy()
            print(f"Extracted hidden states shape: {last_hidden.shape}")
        else:
            # Fallback: try to get hidden states from the transformer output
            print("Warning: 'hidden_states' not found in output. Trying alternative extraction...")
            # Get the transformer's last hidden state directly
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # Run through the model manually to extract hidden states
                # This is a fallback method
                hidden_states = model.get_hidden_states(
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    attention_mask=attention_mask
                )
                last_hidden = hidden_states[:, -1, :].detach().numpy()
                print(f"Extracted hidden states via fallback: {last_hidden.shape}")
            else:
                # Last resort: use the state embeddings
                print("Using state embeddings as proxy for hidden states...")
                state_embeddings = model.embed_state(states)
                last_hidden = state_embeddings[:, -1, :].detach().numpy()
                print(f"Using state embeddings shape: {last_hidden.shape}")

    return {'hidden': last_hidden, 'labels': market_labels}


# 3. Decision drivers (input/latent to allocation)
def decision_drivers(model, states, actions, returns_to_go, timesteps, target_action_idx=None):
    """
    Uses IntegratedGradients to attribute the last action to the input features.
    If target_action_idx is None, computes attribution for the sum of all actions.
    """
    print(f"Computing decision drivers attribution...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")
    print(f"Returns to go: {returns_to_go.shape}, Timesteps: {timesteps.shape}")

    # Ensure states requires gradients for attribution
    states = states.clone().detach().requires_grad_(True)

    if target_action_idx is not None:
        # Attribution for a specific action dimension
        def forward_wrapper(input_states):
            return _forward_action(model, input_states, actions, returns_to_go, timesteps, target_action_idx=target_action_idx)
        print(f"Computing attribution for action index {target_action_idx}")
    else:
        # Attribution for sum of all actions (scalar output)
        def forward_wrapper(input_states):
            output = _forward_action(model, input_states, actions, returns_to_go, timesteps, target_action_idx=None)
            return output.sum(dim=-1)  # Sum across action dimensions to get scalar per sample
        print("Computing attribution for sum of all actions")


    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(states)

    print(f"Running IntegratedGradients with baseline shape: {baseline.shape}")

    attributions, delta = ig.attribute(states, baselines=baseline, return_convergence_delta=True)


    print(f"Attribution computation completed. Output shape: {attributions.shape}")
    print(f"Convergence delta: {delta.mean().item():.6f}")

    return {'attributions': attributions, 'delta': delta}

# 4. High-risk (high exposure) situation analysis
def high_risk_situation_attribution(model, states, actions, returns_to_go, timesteps, risk_threshold=0.2):
    """
    Finds samples where the predicted action std is above risk_threshold, then attributes those using IntegratedGradients.
    """
    print(f"Analyzing high-risk situations with threshold {risk_threshold}")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")

    with torch.no_grad():
        action_preds = _forward_action(model, states, actions, returns_to_go, timesteps)
        stds = action_preds.std(dim=-1)
        mask = stds > risk_threshold

    if mask.sum() == 0:
        print(f"No high-risk samples found with std > {risk_threshold}")
        return None

    print(f"Found {mask.sum().item()} high-risk samples out of {len(mask)}")

    # Attribution for sum of all actions (to get scalar output)
    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions[mask], returns_to_go[mask], timesteps[mask])
        return output.sum(dim=-1)

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(states[mask])
    attributions = ig.attribute(states[mask], baselines=baseline)

    print(f"High-risk attribution completed. Shape: {attributions.shape}")
    return {'attributions': attributions, 'indices': mask.nonzero(as_tuple=True)[0]}

# 5. Risk-averse situation analysis
def risk_averse_situation_attribution(model, states, actions, returns_to_go, timesteps, risk_threshold=0.05):
    """
    Finds samples where the predicted action std is below risk_threshold, then attributes those using IntegratedGradients.
    """
    print(f"Analyzing risk-averse situations with threshold {risk_threshold}")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")

    with torch.no_grad():
        action_preds = _forward_action(model, states, actions, returns_to_go, timesteps)
        stds = action_preds.std(dim=-1)
        mask = stds < risk_threshold

    if mask.sum() == 0:
        print(f"No risk-averse samples found with std < {risk_threshold}")
        return None

    print(f"Found {mask.sum().item()} risk-averse samples out of {len(mask)}")

    # Attribution for sum of all actions (to get scalar output)
    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions[mask], returns_to_go[mask], timesteps[mask])
        return output.sum(dim=-1)

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(states[mask])
    attributions = ig.attribute(states[mask], baselines=baseline)

    print(f"Risk-averse attribution completed. Shape: {attributions.shape}")
    return {'attributions': attributions, 'indices': mask.nonzero(as_tuple=True)[0]}

# 6a. Temporal attribution
def temporal_attribution(model, states, actions, returns_to_go, timesteps, target_action_idx=None):
    """
    Uses FeatureAblation to ablate each timestep in the input sequence and measure its effect on the last action.
    """
    def ablate_timestep(input_states):
        # input_states: (batch, seq, features)
        # We'll ablate each timestep by zeroing it out
        results = []
        for t in range(input_states.shape[1]):
            ablated = input_states.clone()
            ablated[:, t, :] = 0
            out = _forward_action(model, ablated, actions, returns_to_go, timesteps, target_action_idx=target_action_idx)
            results.append(out)
        return torch.stack(results, dim=1)  # (batch, seq, ...)
    # For simplicity, just return the ablation function for user to call per batch
    return ablate_timestep(states)

# 6b. Diversification/concentration drivers
def diversification_attribution(model, states, actions, returns_to_go, timesteps, instrument_slices=None):
    """
    Attributes model output to input features grouped by instrument. instrument_slices is a list of (start, end) indices for each instrument in the state vector.
    """
    print(f"Computing diversification attribution...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")

    if instrument_slices is None:
        # Assume equal split if not provided
        n_instruments = model.config.act_dim
        feat_per_inst = states.shape[-1] // n_instruments
        instrument_slices = [(i*feat_per_inst, (i+1)*feat_per_inst) for i in range(n_instruments)]
        print(f"Using default instrument slices: {len(instrument_slices)} instruments with {feat_per_inst} features each")
    else:
        print(f"Using provided instrument slices: {len(instrument_slices)} instruments")

    # Attribution for sum of all actions (to get scalar output)
    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(states)
    attributions, _ = ig.attribute(states, baselines=baseline, return_convergence_delta=True)

    # Sum attributions per instrument
    inst_attr = [attributions[..., sl[0]:sl[1]].sum(dim=-1) for sl in instrument_slices]

    print(f"Diversification attribution completed. Overall shape: {attributions.shape}")
    print(f"Instrument attributions computed for {len(inst_attr)} instruments")

    return {'attributions': attributions, 'instrument_attributions': inst_attr, 'instrument_slices': instrument_slices}

# 6c. Reward/return-to-go reliance
def reward_signal_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Uses IntegratedGradients to attribute the last action to the returns_to_go input.
    """
    print(f"Computing reward signal attribution...")
    print(f"Input shapes - States: {states.shape}, Returns to go: {returns_to_go.shape}")

    # Attribution for sum of all actions (to get scalar output)
    def forward_wrapper(input_returns):
        output = _forward_action(model, states, actions, input_returns, timesteps)
        return output.sum(dim=-1)

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(returns_to_go)
    attributions, delta = ig.attribute(returns_to_go, baselines=baseline, return_convergence_delta=True)

    print(f"Reward signal attribution completed. Shape: {attributions.shape}")
    print(f"Convergence delta: {delta.mean().item():.6f}")

    return {'attributions': attributions, 'delta': delta}

# Attention head analysis

def extract_attention_weights(model, states, actions, returns_to_go, timesteps, attention_mask=None):
    """
    Runs the model with output_attentions=True and returns the attention weights for all layers and heads.
    Returns a list of attention tensors (one per layer), each of shape (batch, n_heads, seq, seq).
    """
    print(f"Extracting attention weights...")
    print(f"Input shapes - States: {states.shape}, Actions: {actions.shape}")
    print(f"Returns to go: {returns_to_go.shape}, Timesteps: {timesteps.shape}")

    # Ensure all tensors have consistent shapes
    batch_size = states.shape[0]
    seq_length = states.shape[1]

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

    # Handle attention mask separately due to its optional nature
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    else:
        attention_mask = _expand_tensor(attention_mask, batch_size, seq_length, 'attention_mask')

    if len(timesteps.shape) == 1:
        timesteps = timesteps.unsqueeze(0)  # Add batch dimension
    timesteps = timesteps.expand(batch_size, seq_length)

    with torch.no_grad():
        out = model.original_forward(
            states=states,
            actions=actions,
            rewards=None,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

        if 'attentions' in out:
            print(f"Extracted attention weights for {len(out['attentions'])} layers")
            return out['attentions']
        else:
            print("Warning: 'attentions' not found in output. Model may not support attention extraction.")
            return None


def plot_attention_heatmap_over_time(attentions, input_series=None, reward_series=None, layer=0, head=0, time_labels=None, input_labels=None, figsize=(16,6)):
    """
    Plots a heatmap of attention weights over time for a given layer and head.
    Optionally overlays input features and a reward time series.
    - attentions: tuple/list of (layer, batch, n_heads, seq, seq)
    - input_series: (seq,) or (seq, n_features) array to plot below the heatmap
    - reward_series: (seq,) array to plot as a line below the heatmap
    - layer, head: which attention layer/head to plot
    - time_labels: optional list of time labels for x/y axes
    - input_labels: optional list of input feature names
    """
    print(f"Plotting attention heatmap for layer {layer}, head {head}")
    print(f"Attention tensor shape: {attentions[layer].shape}")

    attn = attentions[layer][0, head].detach().numpy()  # (seq, seq)
    print(f"Attention matrix shape: {attn.shape}")

    plt.figure(figsize=figsize)

    # Handle None labels by setting them to False (no labels)
    x_labels = time_labels if time_labels is not None else False
    y_labels = time_labels if time_labels is not None else False

    sns.heatmap(attn, cmap='viridis', cbar=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.title(f'Attention Heatmap (Layer {layer}, Head {head})')
    plt.xlabel('Key/Time')
    plt.ylabel('Query/Time')
    plt.tight_layout()
    plt.show()

    if input_series is not None:
        plt.figure(figsize=(figsize[0], 2))
        if input_series.ndim == 1:
            plt.plot(input_series, label='Input', alpha=0.7)
        else:
            for i in range(min(input_series.shape[1], 5)):  # Limit to 5 features max
                label = input_labels[i] if input_labels and i < len(input_labels) else f'Input {i}'
                plt.plot(input_series[:, i], label=label, alpha=0.7)
        plt.title('Input Features Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if reward_series is not None:
        plt.figure(figsize=(figsize[0], 2))
        plt.plot(reward_series, color='red', label='Reward', alpha=0.7)
        plt.title('Reward Over Time')
        plt.xlabel('Time')
        plt.ylabel('Reward Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # =============================================================================
# TIME-SERIES & TEMPORAL DYNAMICS ANALYSES
# =============================================================================

def rolling_window_attribution(model, states, actions, returns_to_go, timesteps, window_size=50, stride=25):
    """
    Computes attribution over rolling windows to understand how attribution patterns change over time.
    """
    print(f"Computing rolling window attribution with window_size={window_size}, stride={stride}")

    num_samples = states.shape[0]
    attributions_list = []
    indices_list = []

    for start_idx in range(0, num_samples - window_size + 1, stride):
        end_idx = start_idx + window_size

        window_states = states[start_idx:end_idx]
        window_actions = actions[start_idx:end_idx]
        window_returns = returns_to_go[start_idx:end_idx]
        window_timesteps = timesteps[start_idx:end_idx]

        # Compute attribution for this window
        attr_result = decision_drivers(model, window_states, window_actions, window_returns, window_timesteps)
        attributions_list.append(attr_result['attributions'].mean(dim=0))  # Average over batch
        indices_list.append((start_idx, end_idx))

    return {'attributions': torch.stack(attributions_list), 'windows': indices_list}

def lag_effect_analysis(model, states, actions, returns_to_go, timesteps, max_lag=5):
    """
    Analyzes how features from different lags (t-1, t-2, etc.) influence current decisions.
    """
    print(f"Analyzing lag effects up to {max_lag} timesteps")

    def forward_with_lag_ablation(input_states, lag_t):
        # Ablate features at timestep t-lag_t
        ablated_states = input_states.clone()
        if lag_t < input_states.shape[1]:
            ablated_states[:, -(lag_t+1), :] = 0
        output = _forward_action(model, ablated_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    lag_attributions = {}
    baseline_output = _forward_action(model, states, actions, returns_to_go, timesteps).sum(dim=-1)

    for lag in range(max_lag + 1):
        ablated_output = forward_with_lag_ablation(states, lag)
        attribution = baseline_output - ablated_output
        lag_attributions[f'lag_{lag}'] = attribution.detach().numpy()

    return lag_attributions

def momentum_vs_mean_reversion_attribution(model, states, actions, returns_to_go, timesteps, price_columns):
    """
    Analyzes attribution for momentum vs mean reversion signals.
    """
    print("Analyzing momentum vs mean reversion attribution patterns")

    # Calculate momentum and mean reversion signals
    momentum_features = []
    mean_rev_features = []

    for col in price_columns:
        # Simple momentum: return over last N periods
        momentum = states[:, :, col] - states[:, :, col].roll(5, dims=1)
        momentum_features.append(momentum.unsqueeze(-1))

        # Simple mean reversion: distance from moving average
        ma = states[:, :, col].rolling(window=10, dim=1).mean()
        mean_rev = states[:, :, col] - ma
        mean_rev_features.append(mean_rev.unsqueeze(-1))

    momentum_tensor = torch.cat(momentum_features, dim=-1)
    mean_rev_tensor = torch.cat(mean_rev_features, dim=-1)

    # Compute attributions for each signal type
    def momentum_forward(mom_states):
        # Use momentum features instead of original
        modified_states = states.clone()
        modified_states[:, :, price_columns] = mom_states
        output = _forward_action(model, modified_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    ig_momentum = IntegratedGradients(momentum_forward)
    momentum_attr = ig_momentum.attribute(momentum_tensor, baselines=torch.zeros_like(momentum_tensor))

    return {
        'momentum_attribution': momentum_attr,
        'mean_reversion_attribution': mean_rev_tensor,  # Placeholder for now
        'momentum_features': momentum_tensor,
        'mean_reversion_features': mean_rev_tensor
    }

def volatility_regime_attribution(model, states, actions, returns_to_go, instrument_returns, timesteps, return_window=20, vol_threshold=2.0):
    """
    Analyzes how model behavior changes during high vs low volatility periods.
    
    Args:
        model: The decision transformer model
        states: Tensor of states (batch_size, seq_length, state_dim)
        actions: Tensor of actions (batch_size, seq_length, action_dim)
        returns_to_go: Tensor of returns-to-go (batch_size, seq_length, 1)
        instrument_returns: DataFrame of instrument returns with shape (n_timesteps, n_instruments)
        timesteps: Tensor of timesteps (batch_size, seq_length)
        return_window: Window size for rolling volatility calculation (default: 20)
        vol_threshold: Z-score threshold for identifying volatile periods (default: 2.0)
    
    Returns:
        Dictionary with attribution results for high and low volatility regimes
    """
    print(f"Analyzing volatility regime attribution with return_window={return_window}, vol_threshold={vol_threshold}")
    
    import pandas as pd
    import numpy as np
    
    # Convert instrument_returns to numpy if it's a DataFrame
    if hasattr(instrument_returns, 'values'):
        returns_data = instrument_returns.values
    else:
        returns_data = instrument_returns
    
    # Ensure returns_data is 2D: (timesteps, instruments)
    if returns_data.ndim == 1:
        returns_data = returns_data.reshape(-1, 1)
    
    n_timesteps, n_instruments = returns_data.shape
    
    # Calculate rolling statistics for each instrument
    rolling_vol = np.zeros_like(returns_data)
    rolling_vol_std = np.zeros_like(returns_data)
    
    for instrument in range(n_instruments):
        instrument_returns_series = returns_data[:, instrument]
        
        # Calculate rolling mean and standard deviation
        rolling_mean = pd.Series(instrument_returns_series).rolling(window=return_window, min_periods=1).mean().values
        rolling_std = pd.Series(instrument_returns_series).rolling(window=return_window, min_periods=1).std().values
        
        # Calculate rolling standard deviation of the standard deviation
        rolling_vol_std[:, instrument] = pd.Series(rolling_std).rolling(window=return_window, min_periods=1).std().values
        
        # Normalize rolling standard deviation as Z-score
        rolling_vol_mean = pd.Series(rolling_std).rolling(window=return_window, min_periods=1).mean().values
        rolling_vol_std_of_std = pd.Series(rolling_std).rolling(window=return_window, min_periods=1).std().values
        
        # Avoid division by zero
        rolling_vol_std_of_std = np.where(rolling_vol_std_of_std == 0, 1e-8, rolling_vol_std_of_std)
        
        # Z-score normalization
        rolling_vol[:, instrument] = (rolling_std - rolling_vol_mean) / rolling_vol_std_of_std
    
    # Aggregate volatility across instruments (mean of absolute Z-scores)
    aggregated_vol = np.mean(np.abs(rolling_vol), axis=1)
    
    # Identify volatile periods (absolute Z-score above threshold)
    volatile_mask = np.abs(aggregated_vol) > vol_threshold
    stable_mask = np.abs(aggregated_vol) <= vol_threshold
    
    print(f"Volatility analysis: {volatile_mask.sum()} volatile periods, {stable_mask.sum()} stable periods")
    print(f"Volatility Z-score range: [{aggregated_vol.min():.3f}, {aggregated_vol.max():.3f}]")
    
    # Convert to tensor masks for indexing
    volatile_indices = torch.tensor(np.where(volatile_mask)[0], dtype=torch.long)
    stable_indices = torch.tensor(np.where(stable_mask)[0], dtype=torch.long)
    
    results = {}
    
    # Process volatile regime
    if len(volatile_indices) > 0:
        volatile_states = states[volatile_indices]
        volatile_actions = actions[volatile_indices]
        volatile_returns = returns_to_go[volatile_indices]
        volatile_timesteps = timesteps[volatile_indices]
        
        try:
            attr_result = decision_drivers(model, volatile_states, volatile_actions, volatile_returns, volatile_timesteps)
            results['high_vol'] = {
                'attributions': attr_result['attributions'],
                'indices': volatile_indices,
                'count': len(volatile_indices),
                'volatility_zscore': aggregated_vol[volatile_mask]
            }
        except Exception as e:
            print(f"Warning: Could not compute attribution for volatile regime: {e}")
            results['high_vol'] = {
                'attributions': None,
                'indices': volatile_indices,
                'count': len(volatile_indices),
                'volatility_zscore': aggregated_vol[volatile_mask]
            }
    
    # Process stable regime
    if len(stable_indices) > 0:
        stable_states = states[stable_indices]
        stable_actions = actions[stable_indices]
        stable_returns = returns_to_go[stable_indices]
        stable_timesteps = timesteps[stable_indices]
        
        try:
            attr_result = decision_drivers(model, stable_states, stable_actions, stable_returns, stable_timesteps)
            results['low_vol'] = {
                'attributions': attr_result['attributions'],
                'indices': stable_indices,
                'count': len(stable_indices),
                'volatility_zscore': aggregated_vol[stable_mask]
            }
        except Exception as e:
            print(f"Warning: Could not compute attribution for stable regime: {e}")
            results['low_vol'] = {
                'attributions': None,
                'indices': stable_indices,
                'count': len(stable_indices),
                'volatility_zscore': aggregated_vol[stable_mask]
            }
    
    return results

def seasonal_attribution(model, states, actions, returns_to_go, timesteps, dates):
    """
    Analyzes seasonal patterns in attribution (monthly/quarterly).
    """
    print("Analyzing seasonal attribution patterns")

    # Extract month and quarter from dates
    months = pd.to_datetime(dates).month
    quarters = pd.to_datetime(dates).quarter

    seasonal_results = {}

    # Monthly analysis
    for month in range(1, 13):
        month_mask = months == month
        if month_mask.sum() > 10:  # Need sufficient samples
            month_indices = torch.tensor(np.where(month_mask)[0])
            month_states = states[month_indices]
            month_actions = actions[month_indices]
            month_returns = returns_to_go[month_indices]
            month_timesteps = timesteps[month_indices]

            attr_result = decision_drivers(model, month_states, month_actions, month_returns, month_timesteps)
            seasonal_results[f'month_{month}'] = {
                'attributions': attr_result['attributions'].mean(dim=0),
                'count': len(month_indices)
            }

    return seasonal_results

# =============================================================================
# PORTFOLIO CONSTRUCTION & RISK ANALYSIS
# =============================================================================

def sharpe_ratio_attribution(model, states, actions, returns_to_go, timesteps, returns_data):
    """
    Analyzes what inputs most influence risk-adjusted return optimization (Sharpe ratio).
    """
    print("Computing Sharpe ratio attribution")

    def sharpe_forward(input_states):
        action_preds = _forward_action(model, input_states, actions, returns_to_go, timesteps)

        # Calculate portfolio returns using predicted weights
        portfolio_returns = (action_preds.unsqueeze(1) * returns_data).sum(dim=-1)

        # Calculate Sharpe ratio (simplified)
        mean_return = portfolio_returns.mean(dim=1)
        std_return = portfolio_returns.std(dim=1)
        sharpe = mean_return / (std_return + 1e-8)

        return sharpe

    ig = IntegratedGradients(sharpe_forward)
    baseline = torch.zeros_like(states)
    attributions = ig.attribute(states, baselines=baseline)

    return {'attributions': attributions}

def drawdown_attribution(model, states, actions, returns_to_go, timesteps, returns_data):
    """
    Analyzes what drives maximum drawdown decisions.
    """
    print("Computing drawdown attribution")

    def drawdown_forward(input_states):
        action_preds = _forward_action(model, input_states, actions, returns_to_go, timesteps)

        # Calculate portfolio returns
        portfolio_returns = (action_preds.unsqueeze(1) * returns_data).sum(dim=-1)

        # Calculate cumulative returns
        cumulative_returns = torch.cumprod(1 + portfolio_returns, dim=1)

        # Calculate running maximum
        running_max = torch.cummax(cumulative_returns, dim=1)[0]

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min(dim=1)[0]

        return -max_drawdown  # Negative because we want to minimize drawdown

    ig = IntegratedGradients(drawdown_forward)
    baseline = torch.zeros_like(states)
    attributions = ig.attribute(states, baselines=baseline)

    return {'attributions': attributions}

def correlation_based_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Analyzes how inter-asset correlations influence allocation decisions.
    """
    print("Computing correlation-based attribution")

    # Extract price data and compute correlations
    seq_len = states.shape[1]
    n_assets = actions.shape[-1]

    correlations = []
    for i in range(states.shape[0]):
        # Extract asset prices for this sample
        asset_prices = states[i, :, :n_assets]  # Assuming first n_assets features are prices

        # Compute correlation matrix
        corr_matrix = torch.corrcoef(asset_prices.T)
        correlations.append(corr_matrix.flatten())

    correlation_features = torch.stack(correlations)

    def correlation_forward(corr_features):
        # This is a simplified version - in practice, you'd need to
        # reconstruct states from correlation features
        output = _forward_action(model, states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    ig = IntegratedGradients(correlation_forward)
    baseline = torch.zeros_like(correlation_features)
    attributions = ig.attribute(correlation_features, baselines=baseline)

    return {'attributions': attributions, 'correlations': correlation_features}

def rebalancing_trigger_attribution(model, states, actions, returns_to_go, timesteps, threshold=0.05):
    """
    Analyzes what features trigger significant portfolio changes.
    """
    print(f"Analyzing rebalancing triggers with threshold {threshold}")

    # Calculate portfolio weight changes
    action_preds = _forward_action(model, states, actions, returns_to_go, timesteps)
    prev_actions = actions[:, 0, :].squeeze()

    weight_changes = torch.abs(action_preds - prev_actions).sum(dim=-1)
    rebalance_mask = weight_changes > threshold

    if rebalance_mask.sum() == 0:
        print("No significant rebalancing events found")
        return None

    print(f"Found {rebalance_mask.sum()} significant rebalancing events")

    # Compute attribution for rebalancing events
    rebalance_states = states[rebalance_mask]
    rebalance_actions = actions[rebalance_mask]
    rebalance_returns = returns_to_go[rebalance_mask]
    rebalance_timesteps = timesteps[rebalance_mask]

    attr_result = decision_drivers(model, rebalance_states, rebalance_actions, rebalance_returns, rebalance_timesteps)

    return {
        'attributions': attr_result['attributions'],
        'rebalance_indices': rebalance_mask.nonzero(as_tuple=True)[0],
        'weight_changes': weight_changes[rebalance_mask]
    }

# =============================================================================
# MARKET REGIME & CONDITION ANALYSIS
# =============================================================================

def crisis_period_attribution(model, states, actions, returns_to_go, timesteps, dates, crisis_periods):
    """
    Analyzes model behavior during market crashes and crisis periods.
    """
    print("Analyzing crisis period attribution")

    date_index = pd.to_datetime(dates)
    crisis_results = {}

    for crisis_name, (start_date, end_date) in crisis_periods.items():
        crisis_mask = (date_index >= start_date) & (date_index <= end_date)

        if crisis_mask.sum() > 0:
            crisis_indices = torch.tensor(np.where(crisis_mask)[0])
            crisis_states = states[crisis_indices]
            crisis_actions = actions[crisis_indices]
            crisis_returns = returns_to_go[crisis_indices]
            crisis_timesteps = timesteps[crisis_indices]

            attr_result = decision_drivers(model, crisis_states, crisis_actions, crisis_returns, crisis_timesteps)
            crisis_results[crisis_name] = {
                'attributions': attr_result['attributions'],
                'period': (start_date, end_date),
                'count': len(crisis_indices)
            }

    return crisis_results

def interest_rate_environment_attribution(model, states, actions, returns_to_go, timesteps, interest_rates):
    """
    Analyzes how different interest rate environments affect model decisions.
    """
    print("Analyzing interest rate environment attribution")

    # Define rate regimes
    low_rate_threshold = np.percentile(interest_rates, 33)
    high_rate_threshold = np.percentile(interest_rates, 67)

    rate_regimes = {
        'low_rates': interest_rates <= low_rate_threshold,
        'medium_rates': (interest_rates > low_rate_threshold) & (interest_rates <= high_rate_threshold),
        'high_rates': interest_rates > high_rate_threshold
    }

    regime_results = {}

    for regime_name, regime_mask in rate_regimes.items():
        if regime_mask.sum() > 0:
            regime_indices = torch.tensor(np.where(regime_mask)[0])
            regime_states = states[regime_indices]
            regime_actions = actions[regime_indices]
            regime_returns = returns_to_go[regime_indices]
            regime_timesteps = timesteps[regime_indices]

            attr_result = decision_drivers(model, regime_states, regime_actions, regime_returns, regime_timesteps)
            regime_results[regime_name] = {
                'attributions': attr_result['attributions'],
                'rate_range': (interest_rates[regime_mask].min(), interest_rates[regime_mask].max()),
                'count': len(regime_indices)
            }

    return regime_results

# =============================================================================
# MULTI-STEP & SEQUENTIAL ANALYSIS
# =============================================================================

def trajectory_attribution(model, states, actions, returns_to_go, timesteps, sequence_length=10):
    """
    Analyzes how early decisions in a sequence influence later ones.
    """
    print(f"Analyzing trajectory attribution over {sequence_length} steps")

    def trajectory_forward(input_states):
        # Compute multi-step trajectory
        current_states = input_states.clone()
        trajectory_actions = []

        for step in range(sequence_length):
            step_actions = _forward_action(model, current_states, actions, returns_to_go, timesteps)
            trajectory_actions.append(step_actions)

            # Update states for next step (simplified)
            # In practice, you'd need proper environment dynamics
            if step < sequence_length - 1:
                current_states = current_states.roll(-1, dims=1)

        # Return final trajectory outcome
        return trajectory_actions[-1].sum(dim=-1)

    ig = IntegratedGradients(trajectory_forward)
    baseline = torch.zeros_like(states)
    attributions = ig.attribute(states, baselines=baseline)

    return {'attributions': attributions}

def return_to_go_sensitivity_analysis(model, states, actions, returns_to_go, timesteps, perturbation_range=0.1):
    """
    Analyzes how changing target returns affects feature importance.
    """
    print(f"Analyzing return-to-go sensitivity with perturbation range {perturbation_range}")

    sensitivity_results = {}
    original_returns = returns_to_go.clone()

    # Test different return-to-go values
    perturbations = [-perturbation_range, -perturbation_range/2, 0, perturbation_range/2, perturbation_range]

    for i, perturbation in enumerate(perturbations):
        perturbed_returns = original_returns * (1 + perturbation)

        attr_result = decision_drivers(model, states, actions, perturbed_returns, timesteps)
        sensitivity_results[f'perturbation_{perturbation:.2f}'] = {
            'attributions': attr_result['attributions'],
            'return_multiplier': 1 + perturbation
        }

    return sensitivity_results

def context_length_attribution(model, states, actions, returns_to_go, timesteps, context_lengths=[5, 10, 16, 20]):
    """
    Analyzes optimal lookback window through attribution analysis.
    """
    print(f"Analyzing context length attribution for lengths: {context_lengths}")

    context_results = {}
    max_length = max(context_lengths)

    for length in context_lengths:
        if length <= states.shape[1]:
            # Truncate to desired context length
            truncated_states = states[:, -length:, :]
            truncated_actions = actions[:, -1:, :]  # Keep only last action
            truncated_timesteps = timesteps[:, -1:]  # Keep only last timestep

            attr_result = decision_drivers(model, truncated_states, truncated_actions, returns_to_go, truncated_timesteps)
            context_results[f'length_{length}'] = {
                'attributions': attr_result['attributions'],
                'context_length': length
            }

    return context_results

# =============================================================================
# ADVANCED CAPTUM TECHNIQUES
# =============================================================================

def shapley_value_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Computes Shapley values for fair attribution allocation using GradientShap.
    """
    print("Computing Shapley value attribution")

    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    # Use GradientShap which approximates Shapley values
    gradient_shap = GradientShap(forward_wrapper)

    # Create random baseline distribution
    baseline_dist = torch.randn_like(states) * 0.1

    attributions = gradient_shap.attribute(
        states,
        baselines=baseline_dist,
        n_samples=50,
        stdevs=0.02
    )

    return {'attributions': attributions}

def deeplift_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Uses DeepLIFT to understand how features differ from reference values.
    """
    print("Computing DeepLIFT attribution")

    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    deeplift = DeepLift(forward_wrapper)
    baseline = torch.zeros_like(states)

    attributions = deeplift.attribute(states, baselines=baseline)

    return {'attributions': attributions}

def saliency_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Computes simple gradient-based saliency maps.
    """
    print("Computing saliency attribution")

    def forward_wrapper(input_states):
        output = _forward_action(model, input_states, actions, returns_to_go, timesteps)
        return output.sum(dim=-1)

    saliency = Saliency(forward_wrapper)
    attributions = saliency.attribute(states, abs=True)

    return {'attributions': attributions}

# =============================================================================
# CAUSAL & COUNTERFACTUAL ANALYSIS
# =============================================================================

def counterfactual_attribution(model, states, actions, returns_to_go, timesteps, feature_idx, counterfactual_values):
    """
    Analyzes "what if" scenarios by replacing specific features with counterfactual values.
    """
    print(f"Computing counterfactual attribution for feature {feature_idx}")

    results = {}
    original_output = _forward_action(model, states, actions, returns_to_go, timesteps).sum(dim=-1)

    for cf_name, cf_value in counterfactual_values.items():
        # Create counterfactual states
        cf_states = states.clone()
        cf_states[:, :, feature_idx] = cf_value

        cf_output = _forward_action(model, cf_states, actions, returns_to_go, timesteps).sum(dim=-1)
        effect = cf_output - original_output

        results[cf_name] = {
            'effect': effect,
            'counterfactual_value': cf_value,
            'mean_effect': effect.mean().item()
        }

    return results

def intervention_attribution(model, states, actions, returns_to_go, timesteps, intervention_mask, intervention_value):
    """
    Analyzes effects of hypothetical market interventions.
    """
    print("Computing intervention attribution")

    # Apply intervention to specified features
    intervened_states = states.clone()
    intervened_states[:, :, intervention_mask] = intervention_value

    # Compare original vs intervened outcomes
    original_output = _forward_action(model, states, actions, returns_to_go, timesteps).sum(dim=-1)
    intervened_output = _forward_action(model, intervened_states, actions, returns_to_go, timesteps).sum(dim=-1)

    intervention_effect = intervened_output - original_output

    return {
        'intervention_effect': intervention_effect,
        'original_output': original_output,
        'intervened_output': intervened_output,
        'mean_effect': intervention_effect.mean().item()
    }

def causal_discovery_attribution(model, states, actions, returns_to_go, timesteps):
    """
    Uses attribution to understand potential causal relationships between features.
    """
    print("Computing causal discovery attribution")

    n_features = states.shape[-1]
    causal_matrix = torch.zeros(n_features, n_features)

    # For each feature, compute how much it influences the attribution of other features
    for target_feature in range(n_features):
        # Ablate target feature
        ablated_states = states.clone()
        ablated_states[:, :, target_feature] = 0

        # Compute attribution with and without target feature
        original_attr = decision_drivers(model, states, actions, returns_to_go, timesteps)['attributions']
        ablated_attr = decision_drivers(model, ablated_states, actions, returns_to_go, timesteps)['attributions']

        # Measure difference in attribution for each feature
        attr_diff = torch.abs(original_attr - ablated_attr).mean(dim=0).squeeze()
        causal_matrix[target_feature, :] = attr_diff

    return {'causal_matrix': causal_matrix}


# =============================================================================
# VISUALIZATION AND UTILITY FUNCTIONS
# =============================================================================

def plot_attribution_heatmap(attributions, title="Attribution Heatmap", feature_names=None, figsize=(12, 8)):
    """
    Plots a heatmap of attribution values.
    """
    plt.figure(figsize=figsize)

    attr_data = attributions.detach().numpy()
    if attr_data.ndim > 2:
        attr_data = attr_data.squeeze()

    if attr_data.ndim == 1:
        attr_data = attr_data.reshape(1, -1)

    sns.heatmap(
        attr_data.T,
        cmap='RdBu_r',
        center=0,
        cbar=True,
        xticklabels=False,
        yticklabels=feature_names if feature_names else False
    )

    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def plot_temporal_attribution(attributions, title="Temporal Attribution", figsize=(12, 6)):
    """
    Plots attribution over time.
    """
    plt.figure(figsize=figsize)

    attr_data = attributions.detach().numpy()
    if attr_data.ndim > 1:
        attr_data = attr_data.mean(axis=tuple(range(1, attr_data.ndim)))

    plt.plot(attr_data, alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Attribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance_ranking(attributions, feature_names=None, top_k=20, figsize=(10, 8)):
    """
    Plots top-k most important features.
    """
    plt.figure(figsize=figsize)

    attr_data = attributions.detach().numpy()
    if attr_data.ndim > 1:
        attr_data = np.abs(attr_data).mean(axis=0).squeeze()

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(attr_data))]

    # Get top-k features
    top_indices = np.argsort(np.abs(attr_data))[-top_k:]
    top_values = attr_data[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    # Plot horizontal bar chart
    plt.barh(range(len(top_values)), top_values)
    plt.yticks(range(len(top_values)), top_names)
    plt.xlabel('Attribution')
    plt.title(f'Top {top_k} Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_attributions(attr_dict, title="Attribution Comparison", figsize=(15, 10)):
    """
    Compares multiple attribution results in a single plot.
    """
    fig, axes = plt.subplots(len(attr_dict), 1, figsize=figsize, sharex=True)
    if len(attr_dict) == 1:
        axes = [axes]

    for i, (name, attributions) in enumerate(attr_dict.items()):
        attr_data = attributions.detach().numpy()
        if attr_data.ndim > 1:
            attr_data = attr_data.mean(axis=0).squeeze()

        axes[i].plot(attr_data, alpha=0.7, label=name)
        axes[i].set_ylabel('Attribution')
        axes[i].set_title(f'{name} Attribution')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel('Features/Time')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def create_attribution_summary(attr_results, model_name="Decision Transformer"):
    """
    Creates a comprehensive summary of all attribution results.
    """
    summary = {
        'model': model_name,
        'timestamp': pd.Timestamp.now(),
        'analyses_performed': list(attr_results.keys()),
        'summary_statistics': {}
    }

    for analysis_name, results in attr_results.items():
        if 'attributions' in results:
            attr = results['attributions']
            summary['summary_statistics'][analysis_name] = {
                'mean_abs_attribution': torch.abs(attr).mean().item(),
                'max_attribution': attr.max().item(),
                'min_attribution': attr.min().item(),
                'std_attribution': attr.std().item(),
                'shape': list(attr.shape)
            }

    return summary


import torch

def compute_local_specialization(attn_layer, attributions, state_idx=1, action_idx=2, topk=20):
    """
    Compute local specialization for a single layer.

    Parameters
    ----------
    attn_layer : torch.Tensor
        Attention weights for one layer, shape (N, H, S, S).
    attributions : torch.Tensor
        Feature attribution tensor, shape (N, seq_len, F) or (N, F).
    state_idx : int
        Index of the state token in the sequence.
    action_idx : int
        Index of the action token in the sequence.
    topk : int
        Number of top features to return.

    Returns
    -------
    dict with:
        - scores: (H, F) tensor of feature dominance per head
        - topk: list of (feature_idx, score) pairs for the average across heads
    """
    if attributions.ndim == 3:
        A = attributions.squeeze(1).abs()  # (N,F)
    else:
        A = attributions.abs()  # (N,F)

    N, H, S, _ = attn_layer.shape
    F = A.shape[-1]

    scores = torch.zeros(H, F, device=A.device)
    alpha = attn_layer[:, :, action_idx, state_idx]  # (N,H)

    for h in range(H):
        scores[h] = (alpha[:,h].unsqueeze(1) * A).mean(dim=0)

    scores = scores / (scores.sum(-1, keepdim=True) + 1e-12)

    avg = scores.mean(0)
    vals, idxs = torch.topk(avg, k=topk)

    return {
        "scores": scores,  # (H,F)
        "topk": list(zip(idxs.tolist(), vals.tolist()))
    }


def compute_attention_rollout(attentions, start_token=1, end_token=2):
    """
    Compute attention rollout from start_token -> end_token across layers.

    Parameters
    ----------
    attentions : list of tensors
        Each of shape (N, H, S, S), one per layer.
    start_token : int
        Index of the source token (default=state token=1).
    end_token : int
        Index of the target token (default=action token=2).

    Returns
    -------
    rollout : Tensor
        Shape (N, H, L) with cumulative attention per layer/head.
    """
    N, H, S, _ = attentions[0].shape
    L = len(attentions)

    print(f"Shapes: {N}, {H}, {S}, {L}")

    # Initialize rollout matrix as identity
    eye = torch.eye(S, device=attentions[0].device).unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
    rollout = eye.repeat(N, H, 1, 1)

    outputs = torch.zeros(N, H, L, device=attentions[0].device)
    for l in range(L):
        attn = attentions[l] + eye  # residual connection trick
        attn = attn / attn.sum(-1, keepdim=True)
        rollout = torch.matmul(attn, rollout)  # cumulative multiplication
        outputs[:, :, l] = rollout[:, :, end_token, start_token]

    return outputs  # (N,H,L)



def compute_layer_specialization_with_rollout(attentions, attributions, state_idx=1, action_idx=2, topk=20):
    """
    Combine local and rollout attention with feature attributions.
    """
    layer_ids = list(attentions.keys())

    N = attributions[layer_ids[0]].shape[0]
    F = attributions[layer_ids[0]].shape[-1]
    L = len(attentions)
    H = 1

    print(f"N: {N}, F: {F}, L: {L}, H: {H}")


    # collapse seq length if =1

    reshaped_attributions = dict()
    attribution_tensor = []
    for k, A in attributions.items():
        if A.shape[1]==1:
            reshaped_attributions[k]  = A.squeeze(1).abs()
        else:
            reshaped_attributions[k]  = A.abs()

        attribution_tensor.append(reshaped_attributions[k])

    A = torch.stack(attribution_tensor)
    #print("Shape A:", A.shape)

    # --- Local specialization ---
    local_scores = torch.zeros(L, H, F, device=A.device)

    for l, (k,attn) in enumerate(attentions.items()):
        attn = attn.unsqueeze(1)
        alpha = attn[:, :, action_idx, state_idx]  # (N,H)
        #print("Shape alpha:", alpha.shape)
        for h in range(H):
            _ = (alpha[:,h].unsqueeze(1) * A)
            _ = _.mean(dim = 1)

            #print("Shape _:", _.shape)
            local_scores[l] = _[l,:]
    local_scores = local_scores / (local_scores.sum(-1, keepdim=True) + 1e-12)

    print("local scores computed")

    # --- Rollout specialization ---
    atts = torch.stack([v for k,v  in attentions.items()]).unsqueeze(1)
    rollouts = compute_attention_rollout(atts, start_token=state_idx, end_token=action_idx)  # (N,H,L)
    rollout_scores = torch.zeros(L, H, F, device=A.device)


    for l in range(L):
        for h in range(H):
            _ = (rollouts[:,h,l].unsqueeze(1) * A).mean(dim=0)
            print(f"Shape _: {_.shape}")

            rollout_scores[l] = _[l,:]
    rollout_scores = rollout_scores / (rollout_scores.sum(-1, keepdim=True) + 1e-12)

    # --- Extract top-k features per layer ---
    local_avg = local_scores.mean(1)   # (L,F)
    rollout_avg = rollout_scores.mean(1)  # (L,F)

    local_topk, rollout_topk = [], []
    for l in range(L):
        lv, li = torch.topk(local_avg[l], k=topk)
        rv, ri = torch.topk(rollout_avg[l], k=topk)
        local_topk.append(list(zip(li.tolist(), lv.tolist())))
        rollout_topk.append(list(zip(ri.tolist(), rv.tolist())))

    return {
        "local_scores": local_scores,    # (L,H,F)
        "rollout_scores": rollout_scores, # (L,H,F)
        "local_topk": local_topk,
        "rollout_topk": rollout_topk
    }
