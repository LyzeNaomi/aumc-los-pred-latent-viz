import os
import gc
import time
import torch
import wandb
import joblib
import numpy as np
import pandas as pd
from ray import tune
from random import sample
from datetime import datetime

from torchviz import make_dot
from torchsummary import summary

from utils import RandomSampler, set_api_key

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, BatchSampler, random_split

from metrics import Metrics

from utils import collate_fn, data_init_train
# from utils_ml import data_init_train

from models import GRUModel, TransformerEncoder,  LSTMModel, TCN, LinearRegression, FFN, TCN_att

wandb_on = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")


def train_one_epoch(model_type, model, loss_fn, optimizer, data_loader, save_path, copy_tr_preds=None, ml_baseline=None):

    running_loss = 0. ; running_metrics = [0., 0., 0.] ; j=0 ; total_samples = 0. ; total_ids=0.

    for i, batch in enumerate(data_loader):
        if not ml_baseline:
            padded_inputs, target, seq_length, b_size, indices = batch
            packed_inputs = pack_padded_sequence(padded_inputs, seq_length.cpu().numpy(), batch_first=True, enforce_sorted=True)
            #Place everything on device
            padded_inputs, packed_inputs = padded_inputs.to(device), packed_inputs.to(device) 
        else: #if ml_baseline
            static_inputs, target = batch #or batch.inp, batch.tgt
            static_inputs = static_inputs.to(device) 
        

        target = target.view(-1,1) ; target = target.to(device)
        model.zero_grad()

        if not ml_baseline: #here we are running DL models
            if model_type == 'Transformer':
                outputs = model(padded_inputs.float(), seq_length[0], seq_length)

            elif model_type == 'TCN' or model_type == 'TCN_att': #pass the padded inputs to TCN after permutation to fit (N, C_in, L_in)
                padded_inputs = padded_inputs.permute(0,2,1)
                outputs = model(padded_inputs.float(), seq_length)
                
            else: #for both GRU & LSTM                  
                outputs = model(packed_inputs.float(), b_size) 
        
        else: #if ml_baseline
            outputs = model(static_inputs.float())
        
        if copy_tr_preds:
            check_preds = pd.DataFrame(np.column_stack((target.cpu().detach().numpy(), outputs.cpu().detach().numpy())), columns=['target', 'predictions']); check_preds.to_csv(save_path + model_type +"_tr_preds.csv", index=False)

        outputs = outputs.to(device)
        loss = loss_fn(outputs.float(), target.float())

        all_metrics = Metrics.diff_metrics(target.float(), outputs.float())
        total_samples += b_size # Accumulate number of samples processed
        total_ids += len(indices)
        

        #backpropagation & gradient calculation
        loss.backward()

        #Adjust model weights
        optimizer.step()

        #Gather & report
        running_loss += loss.item()
        running_metrics = [r + m.item() for r, m in zip(running_metrics, all_metrics)]
        j+=1

        #Report loss averaged over batches
        # last_loss = running_loss/j
        print(' batch {} loss: {}'.format(i + 1, loss.item())); print(f"total number of ids in train set: {total_ids}")
    
    avg_loss = running_loss/j
    avg_metrics = [metric/total_samples for metric in running_metrics]

    return avg_loss, avg_metrics


def validate_one_epoch(data_loader, model_type, model, loss_fn, save_path=None, copy_val_preds=None, ml_baseline=None):
    
    running_loss = 0. ; running_metrics = [0., 0., 0.] ; j=0 ; total_samples=0. ; total_ids=0.

    for i, vdata in enumerate(data_loader):
        if not ml_baseline:
            inputs, target, seq_l, b_size, indices = vdata
            pack_inputs = pack_padded_sequence(inputs, seq_l.cpu().numpy(), batch_first=True, enforce_sorted=True)
            #Place tensors on device
            inputs, pack_inputs= inputs.to(device), pack_inputs.to(device)
        else: #if ml_baseline
           static_inputs, target = vdata 
           static_inputs = static_inputs.to(device)
        
        target = target.view(-1, 1) ;  target = target.to(device)

        if not ml_baseline:
            if model_type == 'Transformer':
                outputs = model(inputs.float(), seq_l[0], seq_l)
            elif model_type == 'TCN' or model_type == 'TCN_att': #pass the padded inputs to TCN after permutation to fit (N, C_in, L_in)
                inputs = inputs.permute(0,2,1)
                outputs = model(inputs.float(), seq_l)
            else:
                outputs = model(pack_inputs.float(), b_size)
        else: #if ml_baseline
            outputs = model(static_inputs.float())

        if copy_val_preds:
            check_preds = pd.DataFrame(np.column_stack((target.cpu().detach().numpy(), outputs.cpu().detach().numpy())), columns=['target', 'predictions']); check_preds.to_csv(save_path + model_type +"_val_preds.csv", index=False)

        loss = loss_fn(outputs.float(), target.float())
        running_loss += loss.item()

        #validation metrics
        all_metrics = Metrics.diff_metrics(target.float(), outputs.float())
        total_samples += b_size
        total_ids += len(indices)

        running_metrics = [r + m.item() for r, m in zip(running_metrics, all_metrics)]
        j+=1

        # last_loss = running_loss/j
        print(' batch {} vloss: {}'.format(j, loss.item())) ; print(f"total number of ids in validation set: {total_ids}")
    
    avg_loss = running_loss/j
    avg_metrics = [metric/total_samples for metric in running_metrics]
    return avg_loss, avg_metrics

        

def train(config, checkpoint_dir=None):
    
    # set global params to track training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    stamp = '{}_{}'.format(config['model_type'], timestamp)
    if not os.path.exists(config['path'] + stamp):
        dir_path = os.path.join(config['path'], stamp)
        os.mkdir(dir_path)  
        joblib.dump(config, dir_path+"/config.joblib")
    
    #init the data - same function is used for ML baselines !!!
    valid_hadmid, train_hadmid, valid_set, train_set = data_init_train(config)
    
    #get the tuning setting
    tuning = config['tune']

    valid_loss = [] ; best_vloss = 100000000. ; header=True

    #get the model type
    print("model type: {}".format(config['model_type']))

    if config['model_type'] == 'GRU': 
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
    

    #define optimizers.
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], 
        weight_decay = config["weight_decay"]) 
    elif config["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"], 
        weight_decay = config["weight_decay"])
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], 
        weight_decay = config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], 
        weight_decay = config["weight_decay"])
    else:
        raise Exception("Improper optimizer specification")
    
    #define loss function
    loss_fn = config["loss_fn"]

    #start training:
    training_start_time = time.time()

    #loop over epochs
    for epoch in range(config['epochs']):
        if not config['ml_baseline']:
            train_sampler = BatchSampler(RandomSampler(train_hadmid), batch_size=config['bsize'], drop_last=False)
            valid_sampler = BatchSampler(RandomSampler(valid_hadmid), batch_size=config['bsize'], drop_last=False) 
            
            train_loader = DataLoader(train_set, sampler=train_sampler, collate_fn=collate_fn)
            validation_loader = DataLoader(valid_set, sampler=valid_sampler, collate_fn=collate_fn)

        else: #if config['ml_baseline']
            print(f"we are running ml baseline dataloaders")
            train_loader = DataLoader(train_set, batch_size=config['bsize'], shuffle=True)
            validation_loader = DataLoader(valid_set, batch_size=config['bsize'], shuffle=True)

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_loss, avg_metrics = train_one_epoch(config['model_type'], model, loss_fn, optimizer, train_loader, config["path"], config['copy_tr_preds'], config['ml_baseline'])       
        print(f'========= Training results at Epoch: {epoch}, with average loss: {avg_loss} and metrics: {avg_metrics} =========')
        
       
        # We don't need gradients on to do reporting
        model.train(False)
        avg_vloss, avg_vmetrics = validate_one_epoch(validation_loader, config['model_type'], model, loss_fn, config["path"], config["copy_val_preds"], config['ml_baseline'])        
        valid_loss.append(avg_vloss) #using the total valid loss
        print(f'========= Validation results at Epoch: {epoch}, with average loss: {avg_vloss} and metrics: {avg_vmetrics} =========')

        if tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
                # Save state to checkpoint file.
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

            # print(f"checkpoint path: {checkpoint_path}")
            tune.report(loss = avg_loss, val_loss=avg_vloss)
        
        train_metrics_info = pd.DataFrame([avg_metrics], columns = ['MAE', 'MSE', 'MSLE']) 
        val_metrics_info = pd.DataFrame([avg_vmetrics], columns =  ['MAE', 'MSE', 'MSLE'])
        
        #copy metrics out in a loop 
        train_metrics_info.to_csv(dir_path + '/train_metrics.csv', index=False, header=header, mode='a')
        val_metrics_info.to_csv(dir_path + '/val_metrics.csv', index=False, header=header, mode='a')
        header = False 

        if not tuning:
            model_p = 'model_{}_{}'.format(timestamp, epoch)
            state_p = 'state_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), dir_path +'/'+ model_p)

        # Track best performance
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        
        if wandb_on:
            if not tuning:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'valid_loss': avg_vloss,
                    'best_valid_loss': best_vloss,

                })
        #custom made Early stopping criteria
        # first scenario: val loss decreases but step (delta is not big)
        # second scenario: val loss starts increasing continuously
        if config['early_stop']:
            vloss_curr_epoch = valid_loss[-1] #get val loss at current epoch
            T_big = [vloss_curr_epoch > val for val in valid_loss]#check if possible to compare a digit with a whole list and return a list
            T_small = [vloss_curr_epoch < (val + config['delta']) for val in valid_loss]#v_end is less than prev losses but not enough. So, if not this happens then we break        
            if config['case'] == "any_past_epoch":
                T = sum(T_small) ; L = len(valid_loss) ; F = L-T
                if sum(T_big) >= config['patience']:
                    print(f"Current val loss is greater than {config['patience']} others")
                    break
                elif F >= config['patience']:
                    print(f"Current val loss is not small enough compared to {config['patience']} others")
                    break
            if config['case'] == "consec_epochs":
                T = sum(T_small[-config['patience']:]) ; L = len(valid_loss) ; F = L-T
                if sum(T_big[-config['patience']:]) == config['patience']: #slicing here to get the last p entries before last validation loss
                    break
                elif F == config['patience']:
                    break
        config['epoch_number'] += 1
        print('Training this epoch, took {:.2f}s'.format((time.time() - training_start_time)/config['epoch_number']))

        torch.cuda.empty_cache()
        gc.collect()
    #End time
    total_time = (time.time() - training_start_time)
    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

    if not tuning:
        return model_p, total_time, dir_path, timestamp
    
    elif tuning: 
        return  ...

    
