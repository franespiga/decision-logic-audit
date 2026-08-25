# DATASET LOADING ---
import pandas as pd
import h5py

def load_datasets(data_path:str, cfg:dict):

    try:
        data = pd.read_hdf(data_path, 'instruments')
        data.index = pd.to_datetime(data.index)
        data = data.loc[cfg.get('start_date'):,:]
        observations = data.loc[:, [i for i in data.columns if i[1]=='Close']].dropna()
    except Exception as e:
        print(e)
        with h5py.File(data_path, 'r') as h5f:
            print(','.join(list(h5f.keys())))
        raise ValueError("No instruments dataset in HDF5 file")

    try:
        indicators = pd.read_hdf(data_path, 'technical_indicators')
        indicators.index = pd.to_datetime(indicators.index)#.strftime('%Y-%m-%d')
    except:
        with h5py.File(data_path, 'r') as h5f:
            print(','.join(list(h5f.keys())))
        raise ValueError("No technical_indicators dataset in HDF5 file")

    return data, indicators, observations


