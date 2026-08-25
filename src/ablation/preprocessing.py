import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
sys.path.append("../../")


def compute_OHLC_returns(df_ohlc: pd.DataFrame) -> tuple:
    """
    This function takes an OHLC matrix of prices and computes the daily returns for each of the available actions.
    (Buy) action has a daily return of Close_t/Open_t
    (Sell) action has a daily return of the overnight change Open_t/Close_t-1
    (Hold) action has a daily return of Close_t/Close_t-1
    :param df_ohlc:
    :return:
    """
    df_rets = df_ohlc.copy(deep = True)
    df_rets['r_b'] = (df_rets['Close'] / df_rets['Open']) - 1
    df_rets['r_s'] = (df_rets['Open'] / df_rets['Close'].shift(1)) - 1
    df_rets['r_h'] = (df_rets['Close'] / df_rets['Close'].shift(1)) - 1

    return df_rets.loc[:, ['r_h', 'r_b', 'r_s']]

def create_return_matrices(df_instruments: pd.DataFrame) -> tuple:
    """
    This function takes a dataframe of OHLC instruments and computes the return matrices Rh, Rb and Rs for each action
    (Buy) action has a daily return of Close_t/Open_t
    (Sell) action has a daily return of the overnight change Open_t/Close_t-1
    (Hold) action has a daily return of Close_t/Close_t-1
    :param df_instruments: dataframe with OHLC of the data and column index [i,j] where i is the instrument and j the OHLC prices
    :return:
    """
    r_h = dict()
    r_b = dict()
    r_s = dict()
    ordered_instruments = list()
    for c in df_instruments.columns:
        if not c[0] in ordered_instruments:
            ordered_instruments.append(c[0])
    for i in ordered_instruments:
        _ = compute_OHLC_returns(df_instruments[i])
        r_h[i] = _['r_h'].values
        r_b[i] = _['r_b'].values
        r_s[i] = _['r_s'].values

    r_h = pd.DataFrame(r_h, index=df_instruments.index).dropna()
    r_b = pd.DataFrame(r_b, index=df_instruments.index).dropna()
    r_s = pd.DataFrame(r_s, index=df_instruments.index).dropna()

    return r_h, r_b, r_s

def equivalent_returns(annual_return: float, periods: int, periods_per_year:int = 252) -> float:
    """
    Calculates the equivalent return for a given number of periods based on the annual return.

    Parameters:
    -----------
    annual_return : float
        The annual return as a decimal (e.g., 0.10 for 10% annual return).
    periods : int
        The number of periods (e.g., days) for which to calculate the equivalent return.

    Returns:
    --------
    float
        The equivalent return for the given number of periods.
    """
    if periods > periods_per_year:
        return (1 + annual_return) ** (periods / periods_per_year) - 1
    else:
        return  (1 + annual_return) ** (periods / periods_per_year) - 1

import math

def prepare_ablation_inputs(
        model,
        backtest_data: pd.DataFrame,
        ref_date: str,
        ohlc_data: pd.DataFrame,
        lookback: int = 16,
        rebalance_every: int = 1,
        annualized_return: float = 0.15,
        use_trajectory_return_objective: bool = False,
        use_incremental_timesteps: bool = True,
        ppy: int = 252,
        device: torch.device = torch.device('cpu'),
):
    """
    Walks through backtest_data from ref_date, replicating compute_portfolio_weights'
    input construction, and returns stacked tensors suitable for ablation:
      - states:       (N, 1, lookback+1, state_dim)
      - actions:      (N, 1, act_dim)
      - returns_to_go:(N, 1, 1)
      - timesteps:    (N, 1)
      - attention_mask:(N, 1)
    where N = number of times we actually call the model (every rebalance step).
    """
    model = model.to(device)
    state_dim = model.config.state_dim
    act_dim   = model.config.act_dim


    for idx, row in backtest_data[pd.to_datetime(ref_date):].iterrows():
        # build the lookback window
        window = backtest_data.loc[:idx].tail(lookback+1)
        if window.shape[0] < lookback+1:
            break

        # prepare state
        obs = torch.from_numpy(window.values).float().reshape(1, 1, -1)

    feature_mapping = create_feature_mapping_from_labels(backtest_data.columns, lookback+1)
    # prepare state
    obs = torch.from_numpy(window.values).float().reshape(1, 1, -1)

    # compute fixed target return
    if use_trajectory_return_objective:
        total_cumulative_return = np.log(
            equivalent_returns(
                annual_return=annualized_return,
                periods=backtest_data[ref_date:].shape[0],
                periods_per_year=ppy)
        )
    else:
        total_cumulative_return = annualized_return
    target_return = torch.tensor(
        total_cumulative_return,
        device=device,
        dtype=torch.float32
    ).reshape(1, 1)

    # returns matrices for reward calc
    R_h, R_b, R_s = create_return_matrices(ohlc_data)
    for R in (R_h, R_b, R_s):
        R.index = pd.to_datetime(R.index)

    # collectors
    states_list = []
    actions_list = []
    returns_list = []
    timesteps_list = []
    mask_list = []
    rewards_list = []

    ts = 0
    re = rebalance_every - 1
    trajectory_tuple = None
    rewards = torch.zeros(1,1,device=device)

    # iterate
    for idx, row in backtest_data[pd.to_datetime(ref_date):].iterrows():
        # build the lookback window
        window = backtest_data.loc[:idx].tail(lookback+1)
        if window.shape[0] < lookback+1:
            break

        # prepare state
        obs = torch.from_numpy(window.values).float().reshape(1, 1, -1)
        feature_str = ', '.join(list(feature_mapping.values()))
        assert len(feature_mapping)==state_dim, f"expected {state_dim} features, got {len(feature_mapping)}: {feature_str}"

        L = obs.size(-1)
        if L < state_dim:
            pad = torch.zeros(1,1,state_dim-L)
            obs = torch.cat([pad, obs], dim=-1)
        else:
            obs = obs[..., -state_dim:]
        states = obs.to(device)

        # action / rewards / return_to_go / timestep / mask
        if trajectory_tuple is None:
            action = torch.zeros(1,1,act_dim,device=device)
            action[:,:, -1] = 1.0
            rewards = torch.zeros(1,1,device=device)
        else:
            action = trajectory_tuple['action_preds']
            rewards = trajectory_tuple['return_preds'] if 'return_preds' in trajectory_tuple else rewards


        
        # timestep & mask
        ts = ts + (1 if use_incremental_timesteps else 0)
        re += 1
        timesteps = torch.tensor(ts, device=device).reshape(1,1)
        attention_mask = torch.zeros(1,1,device=device)

        # store
        states_list.append(states)
        actions_list.append(action)
        returns_list.append(target_return)
        timesteps_list.append(timesteps)
        mask_list.append(attention_mask)
        rewards_list.append(rewards.reshape(-1))

        # only run model on rebalance steps
        if re == rebalance_every:
            re = 0
            with torch.no_grad():
                trajectory_tuple = model.original_forward(
                    states=states,
                    actions=action,
                    rewards=rewards,
                    returns_to_go=target_return,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            # update target_return if trajectory objective
            if use_trajectory_return_objective:
                target_return = target_return - rewards


    # stack into tensors
    batch = {
        'states':       torch.cat(states_list,    dim=0),
        'actions':      torch.cat(actions_list,   dim=0),
        'returns_to_go':torch.cat(returns_list,   dim=0),
        'timesteps':    torch.cat(timesteps_list, dim=0),
        'attention_mask':torch.cat(mask_list,     dim=0),
        'rewards': torch.cat(rewards_list, dim=0), 
        'instrument_returns': R_h[ref_date:],
        'labels' : feature_mapping, 
        'dates': backtest_data[pd.to_datetime(ref_date):].index
    }

    return batch


def create_feature_mapping_from_labels(feature_labels: List[str], lookback: int) -> Dict[int, str]:
    """
    Create a mapping from feature indices to meaningful names from feature labels.

    Args:
        feature_labels: List of feature names in the order they appear in the state
        lookback: Lookback window size

    Returns:
        Dictionary mapping feature indices to feature names
    """
    feature_mapping = {}

    for lb in range(lookback-1, -1, -1):
        for idx, label in enumerate(feature_labels):

            # Handle different label formats
            if isinstance(label, tuple) and len(label) == 2:
                # Format: (instrument, feature_type)
                instrument, feature_type = label
                feature_name = f"{instrument}_{feature_type}_lb_{lb}"
            elif isinstance(label, str):
                # Format: "instrument_feature_type" or just the feature name
                feature_name = f"{label}_lb_{lb}"
            else:
                # Fallback for unexpected format
                feature_name = f"feature_{idx}_lb_{lb}"

            feature_mapping[len(feature_mapping)] = feature_name



    return feature_mapping
