#Import dependencies
import os
import torch
import wandb
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler

#1. We will start with the consideration that baselines will be done only when we are doing stacking
class var_extract():
    def first_entry(arr):
        return arr.iloc[0]
    def last_entry(arr):
        return arr.iloc[-1]
    def _range(arr):
        return arr.max() - arr.min()
    def lfdiff(arr): #calculates the difference btw first and last entry
        return arr.iloc[-1] - arr.iloc[0]
    def freq(arr):
        arr_df = pd.DataFrame(arr)
        return (arr_df== 1).sum(axis=0)

def get_stats(df, list_of_stats, original_cols, index):
    df = df.agg(list_of_stats) ; merged_df = pd.DataFrame()
    for id in df.index:
        cols_names = [cols+"_"+id for cols in original_cols]
        id_cols = pd.DataFrame(df.loc[id]).T.reset_index(drop=True) ; id_cols = id_cols.rename(columns = dict(zip(id_cols.columns, cols_names)))
        merged_df = pd.concat([merged_df, id_cols], axis=1) ; 
    merged_df.index = [index]
    return merged_df

def apply_stats(df_data, id, inputs, mask_blocks, static_data=None, outputs=None): 
    original_cols = df_data.columns.tolist()
    index = str(id)+"_"+str(len(mask_blocks)) ; 
    freq_df =  pd.DataFrame(var_extract.freq(mask_blocks)).T.reset_index(drop=True)
    freq_names = [cols + "_freq" for cols in  original_cols] ; freq_df = freq_df.rename(columns=dict(zip(freq_df.columns, freq_names))) ; freq_df.index = [index]
    inputs_df = get_stats(pd.DataFrame(inputs), ['mean'], original_cols, index=index)
    #then I can merge static after merging all extracted   
    inputs_df = pd.concat([inputs_df, freq_df], axis=1) ; static_data.index = [index]
    inputs_df = pd.concat([inputs_df, static_data], axis=1) 
    return inputs_df.values 

def seq_gen_at_xhrs_1h_shift(timeseries_data_los_hadmid, timeseries_data_los_hadmid_copy, i, inputs, outputs, j=None, _1h_shift_window=None,
                             ml_baseline=None, mask_df=None, mask_df_copy=None, id=None, window=None, static_data=None):
    if _1h_shift_window:
        timeseries_data_los_hadmid_nolos = timeseries_data_los_hadmid.iloc[j:i].drop(columns = ['hadmid','lengthofstay','mortality', 'rem_los']) #we do not use time for ML baseline
        inputs.append(timeseries_data_los_hadmid_nolos.values)
        indices = set(timeseries_data_los_hadmid_nolos.index).intersection(set(timeseries_data_los_hadmid_copy.index))
    else: #seq_gen_over5hrs condition - 1h window stacking
        timeseries_data_los_hadmid_nolos = timeseries_data_los_hadmid.iloc[:i].drop(columns = ['hadmid','lengthofstay','mortality', 'rem_los'])
        if not ml_baseline:
            inputs.append(timeseries_data_los_hadmid_nolos.values)
            indices = set(timeseries_data_los_hadmid_nolos.index).intersection(set(timeseries_data_los_hadmid_copy.index))
        else: #if ml_baseline
            mask_blocks = mask_df.iloc[:i].values
            inputs.append(apply_stats(timeseries_data_los_hadmid_nolos, id, timeseries_data_los_hadmid_nolos.values,
                                      mask_blocks, static_data))
            indices = set(timeseries_data_los_hadmid_nolos.index).intersection(set(timeseries_data_los_hadmid_copy.index))
            mask_df_copy.drop(index=indices, inplace=True)              
    rem_los = timeseries_data_los_hadmid.iloc[:i+1]['rem_los']    
    outputs.append(rem_los.values[-1])
    timeseries_data_los_hadmid_copy.drop(index = indices, inplace=True)    
    if ml_baseline:
        return inputs, outputs, rem_los, timeseries_data_los_hadmid_copy, mask_df_copy
    else: #if not ml_baseline
        return inputs, outputs, rem_los, timeseries_data_los_hadmid_copy

def data_days(timeseries_data_los):
    #convert here the outcome from hours to days
    timeseries_data_los['rem_los'] = timeseries_data_los['lengthofstay'] - timeseries_data_los['time']
    timeseries_data_los['rem_los'] = timeseries_data_los['rem_los']/24
    return timeseries_data_los

def collate_fn(batch): #helps perform some further pre-processing on the tensors
    patient_input_array = batch[0][0]
    patient_output_array = batch[0][1]
    patient_indices = batch[0][2]
    ml_baseline = batch[0][3]
    b_size = len(patient_input_array)
    output_tensor = torch.Tensor(patient_output_array)

    if ml_baseline:
        input_tensor = torch.Tensor(np.array(patient_input_array)) ;
        return torch.squeeze(input_tensor), torch.squeeze(output_tensor)
    
    else: #if not ml_baseline     
        seq_lengths = torch.LongTensor(list(map(len, patient_input_array)))
        padded_tensor = torch.zeros(len(patient_input_array), seq_lengths.max(), patient_input_array[0].shape[1], dtype=torch.float)
        for idx, (seq, seqlen) in enumerate(zip(patient_input_array, seq_lengths)):
            padded_tensor[idx, :seqlen] = torch.Tensor(seq)   
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)       
        padded_tensor = padded_tensor[perm_idx]        
        output_tensor = output_tensor[perm_idx]
        return padded_tensor, output_tensor, seq_lengths, b_size, patient_indices

class RandomSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.shuffle()

    def shuffle(self):
        self.seed = random.randint(0, 2**32 - 1)

    def __iter__(self):
        i = self.data
        indexes = list(i)
        random.Random(self.seed).shuffle(indexes)
        return iter(indexes)
    
    def __len__(self):
        return len(self.data)
    

class UCILoader(Dataset):
    '''
    Custom dataloader to load samples form the timeseries data
    in the desired format
    '''
    def __init__(self, data_path, labels_path, static_path, data, mask=True, fusion="early", time_window=5, _1h_shift_window=False, 
                 ml_baseline=False):
        '''
        time_window=5, default value to perform 1h stacking
        '''
        self.df_data = pd.read_csv(data_path)
        self.df_labels = pd.read_csv(labels_path)
        self.df_static = pd.read_csv(static_path)
        self.fusion = fusion
        self.index = None
        self.time_window = time_window
        self._1h_shift_window = _1h_shift_window
        self.ml_baseline = ml_baseline
        
        if mask == False:
            self.mask_cols = self.df_data.filter(like='_mask')
            self.df_data = self.df_data.drop(columns = self.mask_cols.columns)
        
        self.timeseries_data_los = self.df_data.merge(self.df_labels[['hadmid','lengthofstay','mortality']], on=['hadmid'], how='inner')
        self.timeseries_data_los = data_days(timeseries_data_los = self.timeseries_data_los)
                
        self.timeseries_data_los.set_index(['hadmid'], inplace=True)
        self.static_data = self.df_static.set_index(['hadmid'])


    def __len__(self):
        return len(self.df_data.hadmid.unique())
        
    def __getitem__(self, index):
        self.index = index
        inputs, outputs = [], []

        for id in index:
            timeseries_data_los_hadmid = self.timeseries_data_los.loc[[id]]
            if self.fusion == "early": #for ml_baseline this is always not the case
                static_data_los_hadmid = self.static_data.loc[[id]]
                timeseries_data_los_hadmid = timeseries_data_los_hadmid.merge(static_data_los_hadmid, left_index=True, right_index=True)
            
            timeseries_data_los_hadmid.reset_index(inplace=True)

            if timeseries_data_los_hadmid.time.max() <= self.time_window:
                if not self.ml_baseline:
                    inputs.append(timeseries_data_los_hadmid.drop(columns = ['hadmid','lengthofstay','mortality', 'rem_los']).values)
                else: #if self.ml_baseline
                    static_data_los_hadmid = self.static_data.loc[[id]]
                    temp_inputs = timeseries_data_los_hadmid.drop(columns = ['time','hadmid','lengthofstay','mortality', 'rem_los']).values
                    mask_blocks = self.mask_cols.values
                    inputs.append(apply_stats(self.df_data, id, temp_inputs, mask_blocks, static_data_los_hadmid)) #have to append static data here
                interm_output = max(timeseries_data_los_hadmid.rem_los.values[-1]-(1/24), 0)
                outputs.append(interm_output)
            
            elif timeseries_data_los_hadmid.shape[0] < (self.time_window+1) and timeseries_data_los_hadmid.rem_los.values[-1] == 0:
                iloc = timeseries_data_los_hadmid.shape[0]
                if not self.ml_baseline:
                    inputs.append(timeseries_data_los_hadmid.iloc[:iloc-1].drop(columns = ['hadmid','lengthofstay','mortality', 'rem_los']).values)
                else: #if self.ml_baseline
                    static_data_los_hadmid = self.static_data.loc[[id]]
                    temp_inputs = timeseries_data_los_hadmid.iloc[:iloc-1].drop(columns = ['time','hadmid','lengthofstay','mortality', 'rem_los']).values
                    mask_blocks = self.mask_cols.values
                    inputs.append(apply_stats(self.df_data, id, temp_inputs, mask_blocks, static_data_los_hadmid))                 
                interm_output = max(timeseries_data_los_hadmid.rem_los.values[-1]-(1/24), 0) 
                outputs.append(interm_output)

            else:
                i = self.time_window ; j=0; timeseries_data_los_hadmid_copy = timeseries_data_los_hadmid.copy() ; rem_los = timeseries_data_los_hadmid.iloc[:i]['rem_los']                
                if self.ml_baseline:
                    mask_df_copy = self.mask_cols.copy()

                if timeseries_data_los_hadmid.rem_los.values[-1] > 0:
                    while not timeseries_data_los_hadmid_copy.empty:
                        if not self.ml_baseline: #we are generating sequences
                            inputs, outputs, rem_los, timeseries_data_los_hadmid_copy = seq_gen_at_xhrs_1h_shift(timeseries_data_los_hadmid, timeseries_data_los_hadmid_copy, i, inputs, outputs, j, self._1h_shift_window) ; i+=1 ; j+=1
                        else: #if self.ml_baseline
                            static_data_los_hadmid = self.static_data.loc[[id]]
                            inputs, outputs, rem_los, timeseries_data_los_hadmid_copy, mask_df_copy = seq_gen_at_xhrs_1h_shift(timeseries_data_los_hadmid, timeseries_data_los_hadmid_copy, i, inputs, outputs, j, self._1h_shift_window,
                                                                                                                                        self.ml_baseline, self.mask_cols, mask_df_copy, id, self.time_window, static_data_los_hadmid) ; i+=1 ; j+=1
                        outputs[-1] = max(outputs[-1]-(1/24), 0)
                else:
                    while timeseries_data_los_hadmid_copy.shape[0]!=1:
                        if not self.ml_baseline:
                            inputs, outputs, rem_los, timeseries_data_los_hadmid_copy = seq_gen_at_xhrs_1h_shift(timeseries_data_los_hadmid, timeseries_data_los_hadmid_copy, i, inputs, outputs, j, self._1h_shift_window) ; i+=1 ; j+=1
                        else: #if self.ml_baseline
                            static_data_los_hadmid = self.static_data.loc[[id]]
                            inputs, outputs, rem_los, timeseries_data_los_hadmid_copy, mask_df_copy = seq_gen_at_xhrs_1h_shift(timeseries_data_los_hadmid, timeseries_data_los_hadmid_copy, i, inputs, outputs, j, self._1h_shift_window,
                                                                                                                                        self.ml_baseline, self.mask_cols, mask_df_copy, id, self.time_window, static_data_los_hadmid) ; i+=1 ; j+=1         
        #print(f"inputs: {inputs}") #make sure to transform inputs to tensors for the pytorch models
        return inputs, outputs, index, self.ml_baseline                 


def data_init_train(config):
    valid_hadmid = pd.read_csv(config['path'] + 'val/labels.csv', usecols=['hadmid'])
    valid_hadmid = valid_hadmid.hadmid.unique()

    train_hadmid = pd.read_csv(config['path'] + 'train/labels.csv', usecols=['hadmid']) 
    train_hadmid = train_hadmid.hadmid.unique()

    train_set = UCILoader(data_path=config['path'] + 'train/timeseries.csv', labels_path=config['path'] + 'train/labels.csv', 
                          static_path=config['path'] + 'train/static_data.csv', data=config['data'], mask=config['mask'], 
                          fusion=config["fusion"], time_window=config['time_window'], _1h_shift_window=config['shift'])
    valid_set = UCILoader(data_path=config['path'] + 'val/timeseries.csv', labels_path=config['path'] + 'val/labels.csv', 
                          static_path=config['path'] + 'val/static_data.csv',data=config['data'], mask=config['mask'], 
                          fusion=config["fusion"], time_window=config['time_window'], _1h_shift_window=config['shift'])
    return valid_hadmid, train_hadmid, valid_set, train_set
        

def data_init_test(config):
    test_hadmid = pd.read_csv(config['path'] + 'test/labels.csv', usecols=['hadmid'])
    test_hadmid = test_hadmid.hadmid.unique()
    test_set = UCILoader(data_path=config['path'] + 'test/timeseries.csv', labels_path=config['path'] + 'test/labels.csv', 
                         static_path=config['path'] + 'test/static_data.csv', data=config['data'], mask=config['mask'], 
                         fusion=config["fusion"], time_window=config['time_window'], _1h_shift_window=config['shift'])
    return test_hadmid, test_set

#wandb config
def set_api_key(api_key = None):
    WANDB_ENV_VAR = "WANDB_API_KEY"
    if api_key:
        os.environ[WANDB_ENV_VAR] = api_key
    elif not os.environ.get(WANDB_ENV_VAR):
        try:
            # Check if user is already logged into wandb.
            wandb.ensure_configured()
            if wandb.api.api_key:
                print("Already logged into W&B.")
                return
        except AttributeError:
            pass
        raise ValueError(
            "No WandB API key found. Either set the {} environment "
            "variable, pass `api_key` or `api_key_file` to the"
            "`WandbLoggerCallback` class as arguments, "
            "or run `wandb login` from the command line".format(WANDB_ENV_VAR)
        )