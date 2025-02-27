#import dependencies

import os
import torch
import joblib
import numpy as np
import pandas as pd

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, BatchSampler

from utils import RandomSampler
from utils import collate_fn, data_init_test
# from utils_ml import data_init_test

from metrics import Metrics
from models import GRUModel, TransformerEncoder, TCN, LSTMModel, LinearRegression, FFN, TCN_att

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running test on: {device}')

def create_directory(parent_dir, folder):
    path = os.path.join(parent_dir, folder)
    return path

def run_on_test(config_path = None, config = None, best_model_dict_path = None, copy_ts_preds=None, dump_path=None):

    if not config_path is None:
        config = joblib.load(config_path+"config.joblib")
        if config['get_embeddings']:
            config['bsize'] = 1
    else:
        config = config   

    embeddings_path = create_directory(config['path'], config['model_type']+'/embeddings/')

    test_hadmid, test_set = data_init_test(config)

    if not config['ml_baseline']:
        test_sampler = BatchSampler(RandomSampler(test_hadmid), batch_size=config['bsize'], drop_last=False)
        test_loader = DataLoader(test_set, sampler=test_sampler, collate_fn=collate_fn)
    else:
        test_loader = DataLoader(test_set, batch_size=config['bsize'], shuffle=True, drop_last=False)

    # define and import the model here 
    print(f"model type for Test: {config['model_type']}")
    if config['model_type'] == 'mean':
        if config['data'] == 'MIMIC':
            mean = 3.53
        elif config['data'] == 'AUMC':
            mean = 4.59
    elif config['model_type'] == 'median':
        if config['data'] == 'MIMIC':
            median = 1.97
        elif config['data'] == 'AUMC':
            median = 1.08

    elif config['model_type'] == 'GRU': 
        model = GRUModel(config['input_dim'], config['hidden_dim'], config['num_layers'], config['output_dim'], config['dropout_prob'], device=config['device'])
        model.to(device)
      
    elif config['model_type'] == 'LSTM':
        model = LSTMModel(config['input_dim'], config['hidden_dim'], config['num_layers'], config['output_dim'], config['dropout_prob'], device=config['device'])
        model.to(device)
 
    elif config['model_type'] == 'TCN':
        model = TCN(config['input_dim'], config['output_dim'], config['num_channels'], config['num_layers'], config['kernel_size'],
                         config['dropout_prob_tcn'], config['is_sequence'], config['feedfw_size'], config['feat_extract_only'], 
		                    config['dropout_prob_fcn'])
        model.to(device)

    elif config['model_type'] == 'TCN_att':
        model = TCN_att(config['input_dim'], config['output_dim'], config['num_channels'], config['num_layers'], config['kernel_size'],
                         config['dropout_prob_tcn'], config['is_sequence'], config['feedfw_size'], config['feat_extract_only'], 
		                    config['dropout_prob_fcn'], config['num_heads'])
        model.to(device)

    elif config['model_type'] == 'Transformer':
        model = TransformerEncoder(config['input_dim'], config['hidden_dim'], config['dropout_prob'], config['apply_drp'], config['num_heads'], config['feedfw_size'], 
                    config['num_layers'], config['output_dim'], config['apply_pe'], config['norm_first'], config['embed_inputs'])
        model.to(device)
    
    elif config['model_type'] == 'LR':
        model = LinearRegression(config['input_dim'], config['output_dim'])
        model.to(device)
    
    elif config['model_type'] == 'FFN':
        model = FFN(config['input_dim'], config['output_dim'], config['hidden_dim'], config['dropout_prob'])
        model.to(device)   

    else:
        raise Exception("Improper model specification")
    
    if not config['model_type'] == 'mean' and not config['model_type'] == 'median':
        model.load_state_dict(torch.load(best_model_dict_path))

    with torch.no_grad():
        if not config['model_type'] == 'mean' and not config['model_type'] == 'median':
            model.eval()       

        running_metrics = [0., 0., 0.] ; header=True ; total_samples = 0.
        
        for i, data in enumerate(test_loader):
            if not config['ml_baseline']:
                inputs, target, seq_l, b_size, indices = data
                pack_inputs = pack_padded_sequence(inputs, seq_l.cpu().numpy(), batch_first=True, enforce_sorted=True)#
                #Place tensors on device
                inputs, pack_inputs = inputs.to(device), pack_inputs.to(device)
            else: #if config['ml_baseline']
                static_inputs, target = data
                static_inputs = static_inputs.to(device)
                        
            target = target.view(-1,1) ; target = target.to(device)

            if not config['ml_baseline']:
                if config['model_type'] == 'Transformer':
                    if config['get_embeddings']:
                        out, out_embeds = model(inputs.float(), seq_l[0], seq_l, config['get_embeddings'])
                    else:
                        out = model(inputs.float(), seq_l[0], seq_l)
                
                elif config['model_type'] == 'TCN' or config['model_type'] == 'TCN_att':
                    inputs = inputs.permute(0,2,1) ; 
                    if config['get_embeddings']:
                        out, out_embeds = model(inputs.float(), seq_l, config['get_embeddings'])
                    else:
                        out = model(inputs.float(), seq_l)
                            
                elif config['model_type'] == 'mean':
                    out = torch.full(target.size(), mean)
                
                elif config['model_type'] == 'median':
                    out = torch.full(target.size(), median)
                
                else: #LSTM/GRU
                    if config['get_embeddings']:
                        out, out_embeds = model(pack_inputs.float(), b_size, config['get_embeddings']) 
                    else:
                        out = model(pack_inputs.float(), b_size)
            else: #if config['ml_baseline']
                out = model(static_inputs.float())

            if config['get_embeddings']:
                joblib.dump(indices, embeddings_path + config['model_type'] + "_test_hadmids_batch_" + str(i) + ".joblib", compress=1)    
                joblib.dump(out_embeds, embeddings_path + config['model_type'] + "_test_embeddings_batch_" + str(i) + ".joblib", compress=1)
                joblib.dump(target, embeddings_path + config['model_type'] + "_test_target_batch_" + str(i) + ".joblib", compress=1)
                
            if config['copy_ts_preds']:
                check_preds = pd.DataFrame(np.column_stack((target.cpu().detach().numpy(), out.cpu().detach().numpy())), columns=['target', 'predictions']); 
                check_preds.to_csv(config['save_path'] + config['model_type'] +"_ts_preds_"+ str(i) +".csv", index=False) 

            all_metrics = Metrics.diff_metrics(target.float(), out.float())  ; total_samples += b_size
            running_metrics = [r + m.item() for r, m in zip(running_metrics, all_metrics)]   
                     
            preds = pd.DataFrame(np.column_stack((target.cpu().detach().numpy(), out.cpu().detach().numpy())), columns=['true','preds'])
            index_df = pd.DataFrame(indices, columns=['index']) ; index_df.to_csv(dump_path+"indices.csv", index=False, mode='a', header=header)
            preds.to_csv(dump_path +"test_preds.csv", index=False, mode='a', header=header) 
            header = False
        avg_metrics = [metric/total_samples for metric in running_metrics] 

    return avg_metrics, embeddings_path
