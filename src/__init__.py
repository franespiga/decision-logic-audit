import torch 
import pandas as pd 
import numpy as np 
import re
import pyfolio

# ANCILLARY MODEL LIBRARIES ----
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os

# PYPFOPT FUNCTIONS ----
from pypfopt.expected_returns import mean_historical_return, returns_from_prices
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import EfficientFrontier, CLA


def compute_portfolio_weights(
        model,
        backtest_data:pd.DataFrame,
        ref_date:str,
        ohlc_data: pd.DataFrame = None,
        annualized_return:float = 0.15,
        use_trajectory_return_objective:bool = False,
        use_incremental_timesteps:bool=True,
        rebalance_every:int = 1,
        lookback:int = 16,
        compute_returns:bool = True,
        ppy: int = 252,
        device = torch.device('cpu'),
        verbose:int = 0,
        writer = None
):

    model = model.to(device)
    state_dim = model.config.state_dim
    act_dim = model.config.act_dim


    if use_trajectory_return_objective:
        total_cumulative_return = np.log(
            equivalent_returns(
                annual_return=annualized_return,
                periods=backtest_data[ref_date:].shape[0],
                periods_per_year=ppy)
        )
    else:
        total_cumulative_return = annualized_return

    target_return = torch.tensor(total_cumulative_return, device=device, dtype=torch.float32).reshape(1, 1)
    trajectory_tuple = None
    ts = 1
    """
    NOTE to anyone:
    I set ts=0 to force the timesteps to be 0 and for the next "rolling period" of 1 year (my episode len) to get the annualized return target.
    This is far more flexible and potentially will condition the rest of the embeddings once they are encoded, but need to give it a more thorough thought.

    Option A-current: no ts is taken so every time it feels "it has a 252-period investment episode ahead and is not impacted.
    Option B-desirable: full total cumulatiDeve return throughout the trajectory available is taken into account (8+ years), however, as the model max encoded timestep is 252, it will not work. A potential workaround might be to reset the cycle as well, drawing some inspiration from the Longformer, although this is not even 20% correct.
    Option C-overkill: consider every "rebalancing timestep" as 0...252 point in the trajectory and see the effect. This will mean also changing the return to go (remaining reward) to condition the agent to be more aggressive / risk averse when needed.
    """
    re = rebalance_every-1

    decision_series = {}
    return_series = {}


    # Define return matrices from the OHLC data
    if ohlc_data is not None:
        R_h, R_b, R_s = create_return_matrices(ohlc_data)
        R_h.index = pd.to_datetime(R_h.index)
        R_b.index = pd.to_datetime(R_b.index)
        R_s.index = pd.to_datetime(R_s.index)
    elif compute_returns:
        raise ValueError("ohlc_data is None. pd.DataFrame should be provided")

    error_count = 0
    for idx, row in backtest_data[pd.to_datetime(ref_date):].iterrows():
        # Prepare observation state ---
        observation_frame = backtest_data.loc[:idx, :].tail(lookback+1)
        if idx!= backtest_data.index[-1]:
            next_trading_day = backtest_data[idx:].head(2).tail(1).index[0]
        else:
            break

        obs = torch.from_numpy(observation_frame.values).reshape(1, 1, -1)
        if obs.size()[-1]<state_dim:
            # pad observation
            padded_obs = torch.zeros(1,1,state_dim)
            padded_obs[:, :, :obs.size()[-1]] = obs
            obs = padded_obs
        elif obs.size()[-1]<state_dim:
            #truncate observation
            obs = obs[:,:,:state_dim]
        else:
            pass
        states = obs.to(device=device, dtype=torch.float32)

        # Store previous action and update reward
        if trajectory_tuple is not None:
            # Unravel the trajectory returned by the agent
            action = trajectory_tuple['action_preds']
            action/=action.sum()
            if use_trajectory_return_objective:
                target_return -= torch.sum(rewards)

        else:
            # Obtain first agent action and reward
            action = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
            action[:,:,-1] = 1.0
            rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)


        # Update timestep & rebalancing counters ---
        ts+=1*use_incremental_timesteps
        re+=1
        decision_series[idx] = action.cpu().numpy().reshape(-1)

        # Check if enough history depth is available for observations, break otherwise ---
        if observation_frame.shape[0] < lookback:
            break

        if re == rebalance_every:
            re = 0 # reset counter
            # Compute rewards and update the return to go
            # Rewards is the expected reward between rebalancing periods,
            # it does not seem very accurate.
            # We should use the actual trajectory

            # Prepare timesteps and mask ---
            timesteps = torch.tensor(ts, device=device, dtype=torch.long).reshape(1, 1)
            attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Get new action ---
                trajectory_tuple = model.original_forward(
                    states=states,
                    actions=action,
                    rewards=rewards,
                    returns_to_go=target_return,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            if writer is not None:
                writer.add_graph(model, X)
                writer = None
                make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

            current_weights = decision_series[idx]
            new_weights = trajectory_tuple['action_preds']
            new_weights /=new_weights.sum()

            if torch.isnan(new_weights).any():
                new_weights = current_weights

            if compute_returns:
                #try:
                w_h, w_b, w_s = decompose_weights_tensor(
                    torch.Tensor(new_weights).view(-1),
                    torch.Tensor(current_weights).view(-1))

                r_b = R_b[next_trading_day:].head(1)
                r_s = R_s[next_trading_day:].head(1)
                r_h = R_h[next_trading_day:].head(rebalance_every)

                if any(np.array([r_h.shape[0], r_s.shape[0], r_b.shape[0]])==0):
                    break
                #r_H = r_h
                r_h = torch.cumprod(1+torch.Tensor(r_h.values), dim=0)[-1,:]-1
                #rr_H = 1+torch.Tensor(r_H.values)
                r_b = torch.Tensor(r_b.values)
                r_s = torch.Tensor(r_s.values)

                pf_h = torch.dot(w_h, r_h.view(-1))
                pf_b = torch.dot(w_b, r_b.view(-1))
                pf_s = torch.dot(w_s, r_s.view(-1))

                pf_rets = pf_h + pf_b + pf_s

                return_series[next_trading_day] = pf_rets.detach().numpy().tolist()

                rewards = trajectory_tuple['return_preds']
                if use_trajectory_return_objective:
                    target_return-=pf_rets

                #except Exception as e:
                #    print(f"Exception {e} computing portfolio returns and weights")

    df_weights = pd.DataFrame.from_dict(decision_series, orient='index', columns=instruments)
    df_weights = df_weights.div(df_weights.sum(axis=1), axis=0)
    assert np.abs(df_weights.sum(axis = 1).max()-1) < 1e-5
    df_weights.index = pd.to_datetime(df_weights.index)


    if compute_returns:
        df_strategy = pd.DataFrame.from_dict(return_series,
                                             orient = 'index',
                                             columns = ['strategy_returns'])
        df_strategy.index = pd.to_datetime(df_strategy.index)


        df_buy_and_hold = ohlc_data.loc[:,[(i, 'Close') for i in instruments]].pct_change(fill_method=None).dropna()
        df_buy_and_hold.columns = [i[0] for i in df_buy_and_hold.columns]
        df_buy_and_hold['equal_weights'] = df_buy_and_hold.mean(axis = 1)
        df_buy_and_hold.index = pd.to_datetime(df_buy_and_hold.index)
        #ToDo: ensure that merging is done outer to avoid missing things, fill with zeros

        df_comparison = df_buy_and_hold.merge(df_strategy, right_index = True, left_index = True)

    if verbose>0:
        missing_indices = set(df_weights.index)-set(data.index)
        print(f"Missing indices: {list(missing_indices)}")
        missing_indices = set(data.index)-set(df_weights.index)
        print(f"Missing indices: {list(missing_indices)}")

    return {'weights':df_weights,
            'returns': df_strategy,
            'comparison': df_comparison}


def compute_portfolio_returns(weights: pd.DataFrame,
                              ohlc_prices: pd.DataFrame,
                              strategy_name: str = "portfolio_returns") -> pd.DataFrame:
    """
    Compute portfolio daily returns based on weights and OHLC prices.

    Parameters:
    -----------
    weights : pd.DataFrame
        A DataFrame where each column is an instrument and rows are portfolio weights for that instrument.
        The index should be dates.
    ohlc_prices : pd.DataFrame
        A MultiIndex DataFrame containing OHLC prices with level 0 as instrument names and level 1 as OHLC types.
        The column ('Close') should be used for the computation.
    strategy_name : str, optional
        The name of the resulting portfolio return series, by default "portfolio_returns".

    Returns:
    --------
    pd.DataFrame
        A DataFrame with dates as the index and a single column containing the portfolio's daily returns.
    """

    # Select the 'Close' prices and forward fill missing values
    close_prices = ohlc_prices.loc[:, (slice(None), 'Close')].droplevel(level=1, axis=1).ffill()

    # Compute daily returns
    daily_returns = close_prices.pct_change().dropna()

    # Align weights and daily returns to have the same dates and instrument columns
    common_dates = weights.index.intersection(daily_returns.index)
    common_instruments = weights.columns.intersection(daily_returns.columns)
    #print(f"OHLC prices:{daily_returns.shape[0]}, Weights:{weights.shape[0]}")
    aligned_weights = weights.loc[common_dates, common_instruments]
    aligned_daily_returns = daily_returns.loc[common_dates, common_instruments]
    #print(f"OHLC prices:{aligned_daily_returns.shape[0]}, Weights:{aligned_weights.shape[0]}")
    # Compute portfolio returns (dot product of weights and returns)
    portfolio_returns = (aligned_weights.values * aligned_daily_returns.values).sum(axis=1)

    # Create the output DataFrame with the strategy name as the column
    return pd.DataFrame(data={strategy_name: portfolio_returns}, index=common_dates)


def compute_pyfolio_metrics(portfolio_returns: pd.Series):
    """
    Compute performance metrics for portfolio returns using Pyfolio.

    Parameters:
    -----------
    portfolio_returns : pd.Series
        Series containing portfolio returns with a DateTime index.

    Returns:
    --------
    dict
        A dictionary containing performance metrics computed by Pyfolio.
    """
    # Ensure the portfolio returns are in the correct format
    portfolio_returns = portfolio_returns.dropna()

    # Use Pyfolio's tear sheet to compute performance metrics
    # This will print the tear sheet as well
    performance_metrics = pyfolio.timeseries.perf_stats(portfolio_returns)

    # Optionally, generate a complete tear sheet
    #pyfolio.create_full_tear_sheet(portfolio_returns)

    return performance_metrics.to_dict()


def parse_experiment_label(label: str):
    """
    Parses an experiment label and extracts key parameters into a dictionary.

    Parameters:
    -----------
    label : str
        The experiment label string.

    Returns:
    --------
    dict
        A dictionary with extracted parameters.
    """
    pattern = r"^(.*?)_ar_([-+]?\d*\.\d+|\d+)_re_(\d+)_it_(True|False)_tro_(True|False)_rd_(\d{4}-\d{2}-\d{2})"

    match = re.search(pattern, label)

    if not match:
        raise ValueError("Label format is incorrect or missing expected parameters.")

    return {
        "model_type": match.group(1),  # Everything before "ar_"
        "annualized_return_target": float(match.group(2)),  # Extract float
        "rebalance_every": int(match.group(3)),  # Extract integer
        "time_embeddings": match.group(4) == "True",  # Convert to boolean
        "use_trajectory_return_objective": match.group(5) == "True",  # Convert to boolean
        "reference_date": match.group(6)  # Extract date string
    }





def find_directories_by_patterns(directory, patterns):
    """
    Finds all directories in a directory tree that match any of the patterns.

    Parameters:
    -----------
    directory : str
        The path to the directory to search in.
    patterns : list
        A list of patterns (regular expressions) to match directories against.

    Returns:
    --------
    list
        A list of directory paths matching any of the given patterns.
    """
    matching_dirs = []
    compiled_patterns = [re.compile(pattern) for pattern in patterns]  # Compile patterns for efficiency

    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            # Check if any pattern matches the directory's full path
            if all(pattern.search(full_path) for pattern in compiled_patterns):
                matching_dirs.append(full_path)

    return matching_dirs