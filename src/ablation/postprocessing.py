import torch
import numpy as np


def postprocess_attributions(attributions,
                             feature_names=None,
                             lookback=None,
                             num_indicators=4,
                             num_instruments=6):
    """
    Postprocess attributions to make them interpretable

    Parameters:
    -----------
    attributions : torch.Tensor
        Attribution tensor of shape (samples, 1, features*instruments*lookback)
    feature_names : list, optional
        List of feature names. If None, will be generated based on aggregation strategy
    top_k : int
        Number of top features to display
    lookback : int
        Number of lag periods (required for proper reshaping)
    num_indicators : int
        Number of features/indicators (default: 4)
    num_instruments : int
        Number of instruments (default: 6)
    """

    attr_data = attributions.detach().cpu().numpy()

    # Handle different input dimensions
    if attr_data.ndim == 3:
        # Shape: (samples, 1, features*instruments*lookback)
        attr_data = attr_data.squeeze(axis=1)  # Remove middle dimension
    elif attr_data.ndim > 2:
        attr_data = attr_data.squeeze()

    # If lookback is not provided, try to infer it
    if lookback is None:
        total_features = attr_data.shape[-1]
        lookback = total_features // (num_indicators * num_instruments)
        print(f"Inferred lookback: {lookback}")

    # Reshape to (samples, features, instruments, lookback)
    try:
        attr_reshaped = attr_data.reshape(attr_data.shape[0], num_indicators, num_instruments, lookback)
    except ValueError as e:
        print(f"Reshape error: {e}")
        print(f"Expected shape: ({attr_data.shape[0]}, {num_indicators}, {num_instruments}, {lookback})")
        print(f"Total elements needed: {num_indicators * num_instruments * lookback}")
        print(f"Actual last dimension: {attr_data.shape[-1]}")
        # Fallback to original behavior
        attr_data = np.abs(attr_data).mean(axis=0) if attr_data.ndim > 1 else attr_data
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(attr_data))]
        raise

    # Reshape feature_names if provided
    if feature_names and len(feature_names) == num_indicators * num_instruments:
        # Create base 2D tensor
        feature_names = ['_'.join(i) for i in feature_names]
        feature_names_2d = np.array(feature_names).reshape(num_indicators, num_instruments)

        # Repeat across lookback dimension and add lag suffixes
        feature_names_tensor = np.empty((num_indicators, num_instruments, lookback), dtype=object)

        for lag in range(lookback):
            lag_suffix = lookback - 1 - lag  # From lookback-1 to 0


            for i in range(num_indicators):
                for j in range(num_instruments):
                    base_name = feature_names_2d[i, j]
                    feature_names_tensor[i, j, lag] = f"{base_name}_{lag_suffix}"
    else:
        # Generate default feature names as 3D tensor
        feature_names_tensor = np.empty((num_indicators, num_instruments, lookback), dtype=object)

        for lag in range(lookback):
            lag_suffix = lookback - 1 - lag  # From lookback-1 to 0
            for i in range(num_indicators):
                for j in range(num_instruments):
                    feature_names_tensor[i, j, lag] = f"feature_{i}_instrument_{j}_{lag_suffix}"

    return attr_reshaped, feature_names_tensor


def aggregate_attributions(attr_tensor, feature_names_tensor, top_k,
                           lookback, num_indicators, num_instruments,
                           aggregation_strategy="lag_wise", specific_lag=None, specific_indicator=None,
                           specific_instrument=None, aggregation_method="max"):

    """
      Aggregate attributions based on aggregation strategy

      Parameters:
      -----------
      attr_tensor : torch.Tensor
        Attribution tensor of shape (samples, 1, features*instruments*lookback)
      feature_names_tensor : torch.Tensor
          Tensor of indexed feature names.
      top_k : int
          Number of top features to display
      lookback : int
          Number of lag periods (required for proper reshaping)
      num_indicators : int
          Number of features/indicators (default: 4)
      num_instruments : int
          Number of instruments (default: 6)
          aggregation_strategy : str
          One of "lag_wise", "instrument_wise", "feature_wise"
      specific_lag : int, optional
          Specific lag to analyze (for lag_wise strategy)
      specific_indicator : int, optional
          Specific indicator to analyze (for indicator_wise strategy)
      specific_instrument : int, optional
          Specific instrument to analyze (for instrument_wise strategy)
      aggregation_method : str
          Method for aggregation: "max", "mean", "median"
    """

    # Handle different aggregation strategies
    if aggregation_strategy == "lag_wise":
        # Lag analysis per feature and instrument
        if specific_lag is not None and specific_lag < lookback:
            specific_lag_idx = (lookback-1)-specific_lag
            # Extract specific lag across all features and instruments
            attr_for_lag = attr_tensor[:, :, :, specific_lag_idx]  # (samples, features, instruments)
            feature_names_tensor = feature_names_tensor[:, :, specific_lag_idx]

            # Aggregate across samples
            if aggregation_method == "max":
                attr_aggregated = np.abs(attr_for_lag).max(axis=0)
            elif aggregation_method == "mean":
                attr_aggregated = np.abs(attr_for_lag).mean(axis=0)
            elif aggregation_method == "median":
                attr_aggregated = np.median(np.abs(attr_for_lag), axis=0)

            # Flatten to get feature importance per (feature, instrument) combination
            attr_data = attr_aggregated.flatten()
            feature_names_tensor = feature_names_tensor.flatten()

        else:
            # Aggregate across all lags when specific_lag is None
            # Aggregate across lags first, then across samples
            if aggregation_method == "max":
                attr_across_lags = np.abs(attr_tensor).max(axis=3)  # Max across lags
                attr_aggregated = attr_across_lags.max(axis=0)  # Max across samples
            elif aggregation_method == "mean":
                attr_across_lags = np.abs(attr_tensor).mean(axis=3)  # Mean across lags
                attr_aggregated = attr_across_lags.mean(axis=0)  # Mean across samples
            elif aggregation_method == "median":
                attr_across_lags = np.median(np.abs(attr_tensor), axis=3)  # Median across lags
                attr_aggregated = np.median(attr_across_lags, axis=0)  # Median across samples

            # Flatten to get feature importance per (feature, instrument) combination
            attr_data = attr_aggregated.flatten()
            feature_names_tensor = feature_names_tensor[:, :, 0].flatten()


    elif aggregation_strategy == "instrument_wise":
        # Instrument analysis per indicator and lag
        if specific_instrument is not None and specific_instrument < num_instruments:
            # Extract specific instrument across all indicators and lags
            attr_for_instrument = attr_tensor[:, :, specific_instrument, :]  # (samples, indicators, lags)
            feature_names_tensor = feature_names_tensor[:, specific_instrument, :]
            # Aggregate across samples
            if aggregation_method == "max":
                attr_aggregated = np.abs(attr_for_instrument).max(axis=0)
            elif aggregation_method == "mean":
                attr_aggregated = np.abs(attr_for_instrument).mean(axis=0)
            elif aggregation_method == "median":
                attr_aggregated = np.median(np.abs(attr_for_instrument), axis=0)

            # Flatten to get feature importance per (indicator, lag) combination
            attr_data = attr_aggregated.flatten()
            feature_names_tensor = feature_names_tensor.flatten()

        else:
            # Aggregate across all instruments when specific_instrument is None
            # Aggregate across instruments first, then across samples
            if aggregation_method == "max":
                attr_across_instruments = np.abs(attr_tensor).max(axis=2)  # Max across instruments
                attr_aggregated = attr_across_instruments.max(axis=0)  # Max across samples
            elif aggregation_method == "mean":
                attr_across_instruments = np.abs(attr_tensor).mean(axis=2)  # Mean across instruments
                attr_aggregated = attr_across_instruments.mean(axis=0)  # Mean across samples
            elif aggregation_method == "median":
                attr_across_instruments = np.median(np.abs(attr_tensor), axis=2)  # Median across instruments
                attr_aggregated = np.median(attr_across_instruments, axis=0)  # Median across samples

            # Flatten to get feature importance per (indicator, lag) combination
            attr_data = attr_aggregated.flatten()
            feature_names_tensor = feature_names_tensor[:, 0, :].flatten()

    elif aggregation_strategy == "indicator_wise":
        # Instrument analysis per indicator and lag
        if specific_indicator is not None and specific_indicator < num_indicators:
            # Extract specific instrument across all indicators and lags
            attr_for_indicator = attr_tensor[:, specific_indicator, :, :]  # (samples, indicators, lags)
            feature_names_tensor = feature_names_tensor[specific_indicator, :, :]
            # Aggregate across samples
            if aggregation_method == "max":
                attr_aggregated = np.abs(attr_for_indicator).max(axis=0)
            elif aggregation_method == "mean":
                attr_aggregated = np.abs(attr_for_indicator).mean(axis=0)
            elif aggregation_method == "median":
                attr_aggregated = np.median(np.abs(attr_for_indicator), axis=0)

            # Flatten to get feature importance per (indicator, lag) combination
            attr_data = attr_aggregated.flatten()

        else:
            # Aggregate across all instruments when specific_instrument is None
            # Aggregate across instruments first, then across samples
            if aggregation_method == "max":
                attr_across_indicators = np.abs(attr_reshaped).max(axis=2)  # Max across instruments
                attr_aggregated = attr_across_indicators.max(axis=0)  # Max across samples
            elif aggregation_method == "mean":
                attr_across_indicators = np.abs(attr_reshaped).mean(axis=2)  # Mean across instruments
                attr_aggregated = attr_across_indicators.mean(axis=0)  # Mean across samples
            elif aggregation_method == "median":
                attr_across_indicators = np.median(np.abs(attr_reshaped), axis=2)  # Median across instruments
                attr_aggregated = np.median(attr_across_indicators, axis=0)  # Median across samples

            # Flatten to get feature importance per (indicator, lag) combination
            attr_data = attr_aggregated.flatten()
            feature_names_tensor = feature_names_tensor[0, :, :].flatten()


    else:
        raise ValueError("aggregation_strategy must be one of 'lag_wise', 'instrument_wise', 'feature_wise'")

    top_indices = np.argsort(np.abs(attr_data))[-top_k:]
    top_values = attr_data[top_indices]
    top_names = feature_names_tensor[top_indices]

    # Print summary statistics
    print(f"Top {len(top_values)} features:")
    print(f"  Max attribution: {np.max(np.abs(top_values)):.6f}")
    print(f"  Min attribution: {np.min(np.abs(top_values)):.6f}")
    print(f"  Mean attribution: {np.mean(np.abs(top_values)):.6f}")

    return top_indices, top_values, top_names
