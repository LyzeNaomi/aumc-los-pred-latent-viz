# Import dependencies
import os
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

from train import train
from test import run_on_test, create_directory
from hyperparameters import hyper_params
from config import get_config, get_test_config
from embeddings import plot_tsne_decomp, _transform_non_padded_emb

def run_hyper_param(config, wandb):
    # Init hyper param search
    hyper_init = hyper_params(config)

    # Run hyper parameter search
    hyper_init.main_hyperparam(num_samples=5, max_num_epochs=10, tune_scheduler = 'ASHA', wandb=wandb)
    print(f'Hyper params init results {hyper_init.best_trial.config}')

    #Get only the hidden units
    hidden_units = hyper_init.best_trial.config['hidden_dim']
    number_layers = hyper_init.best_trial.config['num_layers']

    #Fix the hidden units based on asha search
    config['hidden_dim'] = hidden_units 
    config['num_layers'] = number_layers 

    # Init final hyper_params
    hyper_p = hyper_params(config)

    # Run hyper parameter search PBT
    hyper_p.main_hyperparam(num_samples=5, max_num_epochs=10, tune_scheduler = 'PBT', wandb=wandb)
    
    print(f'Hyper params results {hyper_p.best_trial.config}')
    return hyper_p.best_trial.config

def plot_results(config, dir_p):
    train_loss = pd.read_csv(dir_p + "/train_metrics.csv") #avg_loss
    val_loss = pd.read_csv(dir_p + "/val_metrics.csv")

    plt.plot(train_loss.MSLE)
    plt.plot(val_loss.MSLE)
    plt.title(f"Model = {config['model_type']} loss after {train_loss.shape[0]} epochs")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(dir_p + "/losses")
    plt.close()


def plot_embeddings(embeddings_path, exp_model_name, emb_save_path, color_group, config, figsize, timestep, manifold_reducer):
    emb_df = _transform_non_padded_emb(path_model = embeddings_path, model=exp_model_name)
    save_path = emb_save_path
    plot_path = create_directory(save_path + exp_model_name +"/"+color_group+"/", 'embeddings plot/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    labels = pd.read_csv(config['path'] + 'test/labels.csv')
    adm_test_data = pd.read_csv(config['path'] + 'test/test_admissions.csv')
    plot_tsne_decomp(emb_df=emb_df, T=24*15, figsize=figsize, LoS=labels, model_name=exp_model_name, t=timestep, save_path=plot_path, 
                     color_group=color_group, adm_test_data=adm_test_data, scale_emb=True, manifold_reducer=manifold_reducer)

def run_experiment(exp_model_name, test=False, wandb=True, plot=False, mask=True, test_only = False, model_path = None, dump_path = None,
                    re_run=None, embed_only=None, plot_embed=None, config_path = None,
                   timestep=None, color_group=None, manifold_reducer=None, embeddings_path=None, figsize=None, emb_save_path=None, ml_baseline=None):
    
    if not embed_only:
        if test:
            optim_config = get_test_config(exp_model_name, mask, re_run, config_path, ml_baseline=ml_baseline)
        elif not test:
            print('Running hyper parameter tuning')
            
            # Parse the config
            config = get_config(exp_model_name, mask)
            print(f'Initial config for: {exp_model_name}, {config}')
            optim_config = run_hyper_param(config, wandb) #This will run hyper param tuning and return the best params

            # Train model + timing, save model
            # first set Tune in the config to false
            optim_config['tune'] = False
            print(f'Optim config {optim_config}')

        if test_only:
            test_metrics, embeddings_path = run_on_test(config_path=dump_path, best_model_dict_path= model_path, dump_path=dump_path) #config = optim_config
            print(f'Performance against test is: {test_metrics}')
            test_metrics = pd.DataFrame(test_metrics).T
            test_metrics.columns = ['MAE', 'MSE', 'MSLE']
            pd.DataFrame(test_metrics).to_csv(dump_path+'/test_metrics.csv')

        else:
            #Train the model from scratch
            model_p, total_time, dir_p, stamp = train(optim_config) 

            result_dict = {
                'model_id': model_p,
                'train_time': total_time,
                'config': optim_config
            }

            dump_path = dir_p + '/' + exp_model_name + '_results_dump'
            joblib.dump(result_dict, dump_path)

            print('trained model')

            # Get model path
            valid_loss = pd.read_csv(dir_p + '/val_metrics.csv') 
            best_model_at_epoch = np.argmin(valid_loss.MSLE)
            model_p = 'model_{}_{}'.format(stamp, best_model_at_epoch)
            path_best_model_dict = dir_p + '/' + model_p
            
            print(path_best_model_dict)

            print(dump_path, path_best_model_dict)

            test_metrics, embeddings_path = run_on_test(config_path = dir_p+'/', best_model_dict_path = path_best_model_dict, dump_path=dir_p+'/')
            
            print(f'Performance against test is: {test_metrics}')
            
            test_metrics = pd.DataFrame(test_metrics).T
            test_metrics.columns = ['MAE', 'MSE', 'MSLE']
            pd.DataFrame(test_metrics).to_csv(dir_p + '/test_metrics.csv')
            
            # plot resuts by calling funx below.
            if plot:
                plot_results(optim_config, dir_p)
        
        if plot_embed:
            plot_embeddings(embeddings_path, exp_model_name, emb_save_path, color_group, config, figsize, timestep, manifold_reducer)
    
    else: #here we are running only embeddings
        if plot_embed:
            plot_embeddings(embeddings_path, exp_model_name, emb_save_path, color_group, config, figsize, timestep, manifold_reducer)


if __name__ == '__main__':
    run_experiment('AUMC', 'TCN_att', test=True, wandb=True, plot=True, mask=True, test_only=False,  #TODO: Impose by some means that mask=False when ml_baseline=True
                    model_path ='',
		    embed_only=False, plot_embed=False, timestep=1, color_group='bins',
                    manifold_reducer='tsne', embeddings_path='', figsize = (12, 10), emb_save_path='', re_run=False, ml_baseline=False)
        