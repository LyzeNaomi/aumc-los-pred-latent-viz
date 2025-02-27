#Dependencies
import joblib
import pandas as pd
# from ray import tune
import torch.nn as nn
from metrics import MSLELoss

from sklearn.preprocessing import MinMaxScaler

def get_config(model, mask):
    #we used this only to find the hyps of the sequential deep learning models. ML baselines hyps are found using the grid search in get_test_config below
    path = ''
    df_ts = pd.read_csv(path + 'train/timeseries.csv', nrows=1)
    if mask == False:
        df_ts_cols = df_ts.filter(like = '_mask')
        df_ts = df_ts.drop(columns=df_ts_cols.columns)
    df_static = pd.read_csv(path + 'train/static_data.csv', nrows=1)
    
    #Define the i/o dims
    input_dim_ts = len(df_ts.columns)-1
    input_dim_static = len(df_static.columns)-1
    output_dim = 1
    input_dim = input_dim_ts + input_dim_static
    
    config = {} #defined as in get_test_config below
    return config

def get_test_config(model, mask, re_run=None, config_path=None, ml_baseline=None):
    '''
    put params here to test the code
    re_run: re-running our best model multiple times
    config_path: path to the config dump where hps are
    '''
    path = ''
    if not ml_baseline:
        df_ts = pd.read_csv(path + 'train/timeseries.csv', nrows=1)
        if mask == False:
            df_ts_cols = df_ts.filter(like = '_mask')
            df_ts = df_ts.drop(columns=df_ts_cols.columns)
        df_static = pd.read_csv(path + 'train/static_data.csv', nrows=1) 
        #Define the i/o dims for ts  & static data
        input_dim_ts = len(df_ts.columns)-1
        input_dim_static = len(df_static.columns)-1 
        input_dim = input_dim_ts + input_dim_static
        print(f"input dim from config file: {input_dim}")
    else: #if ml_baseline
        df = pd.read_csv(path + "train/static_inputs.csv", nrows=1)
        input_dim = len(df.columns)-1

    #common pararms
    common_parms = {
    'path': path,
    'mask':mask,
    "input_dim": input_dim,
    'merge_meds': False,
    'meds_scaling': False,
    'epochs': 50,
    'output_dim': 1,
    'epoch_number': 0,
    "fusion": "early",
    'loss_fn': MSLELoss(),
    'tune': False,
    'shift': True,
    'save_path': path,
    'api_ke': '',
    'get_embeddings': False,
    'copy_tr_preds': False,
    'copy_val_preds':False,
    'copy_ts_preds':False,
    "device": None,
    "optimizer": "adamw", 
    'drop_last_sampler': False,
    'early_stop' : True,
    'patience' : 4,
    "delta": 0.005, #default value
    'case' : 'any_past_epoch'
    }
    
    #model params

    #### mean/median
    if model == 'mean' or model == 'median':
        config = {
            "api_key": '8d579c56c0eda8929353aac11d399b96e5aa9e71',
            "model_type": model,
            'time_window': ,
            'bsize': ,
        }
        config.update(common_parms)

    
    ##### Transformer 
    elif model == 'Transformer':
        config = { 
            "ml_baseline": ml_baseline,
            "hidden_dim": ,
            "num_layers": ,
            "dropout_prob":, 
            'apply_drp': False,
            "learning_rate": ,
            "bsize": ,
            "weight_decay":,
            'num_heads': ,
            'feedfw_size': ,  
            "model_type": model,
            'apply_pe': True, 
            'norm_first': False,
            'embed_inputs': '1dconv',
            'time_window': ,
        }
        config.update(common_parms)

    #### TCN 
    elif model == 'TCN' or model == 'TCN_att':   
        config = {
            "ml_baseline": ml_baseline,
            "kernel_size": ,
            "dropout_prob_tcn": , 
            'dropout_prob_fcn': , 
            "learning_rate": ,
            "bsize":,
	        "num_heads":,
            "num_layers":,
            "num_channels":[], 
            "weight_decay": , 
            'feedfw_size': ,
            "is_sequence":False,
            "model_type": model,
            'feat_extract_only': True,
            'time_window': }
        config.update(common_parms)
            

    # GRU/LSTM
    elif model == 'GRU' or model == 'LSTM':
        config = {
        "ml_baseline": ml_baseline,
        "hidden_dim": ,
        "num_layers":,
        "num_heads":, 
        "dropout_prob":, 
        "learning_rate": , 
        "bsize": 4,
        "weight_decay": , 
        "model_type": model,       
        'time_window': ,
            }
        config.update(common_parms)

    #LR
    elif model == 'LR':
        config = {
        "ml_baseline": ml_baseline,
        "mask": mask, 
        "fusion": "early-ml_baseline",
        "input_dim": input_dim,
        "bsize": ,
        "weight_decay": ,
        "learning_rate": ,
        "model_type": model,
        'shift': False, 
        'time_window': 5,
        }
        config.update(common_parms)

    #FFN
    else: #model == 'FFN':
        config = {
        "ml_baseline": ml_baseline,
        "mask": mask, #mask is always False and early fusion NO when ml_baseline = True
        "fusion": "early-ml_baseline",
        "input_dim": input_dim, 
        "bsize": 32,
        "weight_decay": ,
       "model_type": model,
        'shift': False, 
        'time_window': 5,
        }
        config.update(common_parms)

    # if re_run: #we need to re-write some items of the config because otherwise they have been overwritten during training (like epoch number) or difficult to handle (like loss funx)
    #     dump = joblib.load(config_path)
    #     config = dump['config']
    #     config['loss_fn'] = MSLELoss()
    #     config['epoch_number'] = ''
    #     config['path'] = ''
    #     config['drop_last_sampler'] =  ''
    #     config['time_window'] = ''
    return config

