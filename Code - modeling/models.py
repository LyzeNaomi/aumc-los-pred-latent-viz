# Import dependencies
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

#Linear class
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, stats_inputs):
        outputs =  self.linear(stats_inputs) 
        return outputs 
    
class FFN(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_proba):
        super(FFN, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p = dropout_proba)

    def forward(self, stats_inputs):
        hidden = self.dropout(self.relu(self.layer1(stats_inputs)))
        outputs = self.relu(self.layer2(hidden)) 
        return outputs

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, device=None):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob       
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)       
        self.gru_dropout = nn.Dropout(p = dropout_prob)
        self.relu = nn.ReLU()
        
    def init_hidden(self, b_size):
        return torch.nn.init.orthogonal_(torch.randn(self.num_layers, b_size, self.hidden_dim)).to(device)
    
    def forward(self, packed_inputs, b_size, get_embeddings=None):
        self.hidden = self.init_hidden(b_size)
        outputs, hidden = self.gru(packed_inputs, self.hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        if get_embeddings:
            embeddings = outputs[0, :, :]

        hidden = self.gru_dropout(hidden)
        hidden = self.fc(hidden[-1, :, :])                            
        outputs = self.relu(hidden)
        
        del(hidden) ; torch.cuda.empty_cache()

        if not get_embeddings:
            return outputs
        else:
            return outputs, embeddings
        
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, device=None):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.lstm_dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
    
    def init_hidden(self, b_size):
        return torch.nn.init.orthogonal_(torch.randn(self.num_layers, b_size, self.hidden_dim)).to(device)
    
    def forward(self, packed_inputs, b_size, get_embeddings=None):
        self.hidden = self.init_hidden(b_size) ; self.cell_state = self.init_hidden(b_size)
        outputs, (hidden, cn) = self.lstm(packed_inputs, (self.hidden, self.cell_state))  
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        if get_embeddings:
            embeddings = outputs[0, :, :]

        hidden = self.lstm_dropout(hidden)
        hidden = self.fc(hidden[-1, :, :])               
               
        outputs = self.relu(hidden)

        del(hidden) ; torch.cuda.empty_cache()

        if not get_embeddings:
            return outputs
        else:
            return outputs, embeddings
    

class DeterministicPositionalEncoding(nn.Module):
    '''
    d_model: changed to hidden_dim cuz I think this corresponds or should correspond to the number of hidden units
    max_len: corresponds to time points or hours within ICU. These are 
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)) OR div_term = 1/(10000.0 ** ((torch.arange(0, hidden_dim, 2) / hidden_dim)))
    buffers: params registers as buffers in the model and should be saved and restored in the state_dict, but are not trained by the optimizer.
    d_model = hidden_dim
    '''
    def __init__(self, hidden_dim, dropout_prob, apply_drp: bool, max_len=24*14):
        super().__init__()
        self.apply_drp = apply_drp
        self.dropout_prob = nn.Dropout(p = dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        dpe = torch.zeros(max_len, 1, hidden_dim)
        dpe[:, 0, 0::2] = torch.sin(position * div_term)
        dpe[:, 0, 1::2] = torch.cos(position * div_term) #shape here: T, 1, H
        dpe = dpe.permute(1, 0, 2)
        self.register_buffer('dpe', dpe)

    def forward(self, inputs):
        '''
        apply_drop: whether to apply dropout on inputs like on pytorch page or not.
        size of inputs: [batch , seq len, hidden_dim|input_feats]
        size of pe: [seq len, batch, hidden_dim] --> need to reshape pe to size [batch, seq len, hidden dim]
        also, when adding, we need to make sure that seq lengths of each patient within the patch corresponds
        chech to tune the dropout prob in the inputs
        '''   
        inputs = inputs + self.dpe[:,:inputs.size(1) ,:]
        if self.apply_drp:
            inputs = self.dropout_prob(inputs)
        else:
            inputs = inputs
        return inputs 
    
class TransformerEncoder(nn.Module):
    '''
    batch_seq_lengths are the sequence lengths of the original inputs which are padded later on.
    '''
    def __init__(self, input_size, hidden_dim, dropout_prob, apply_drp, num_heads, feedforward_size, num_layers, output_dim, 
                apply_pe:bool, norm_first:bool,embed_inputs=None, device=None):
        super().__init__()

        self.device = device
        self.apply_pe = apply_pe
        self.norm_first = norm_first
        self.output_dim = output_dim
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.embed_inputs = embed_inputs

        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.input_embedding = nn.Conv1d(in_channels=input_size, out_channels=self.hidden_dim, kernel_size=1)
        self.det_pos_encoder = DeterministicPositionalEncoding(self.hidden_dim, dropout_prob, apply_drp)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=feedforward_size, dropout=dropout_prob, 
                                                              activation='relu', batch_first=True, norm_first=self.norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=num_layers)

    def source_mask(self, max_seq_length):
        '''
        fux to include a source mask to prevent the use of further time points when computing the representations (attention values)
        of others.
                '''
        mask = (torch.triu(torch.ones(max_seq_length, max_seq_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, inputs, max_seq_length, batch_seq_lengths, get_embeddings=None):
        '''
        batch_seq_lengths = list of seq length per block per batch
        size of inputs: [batch size, seqlength, num_feats]
        we use batch_first = True both on inputs and positional encodings
        we decide to apply an input embedding to inputs or not using Conv1D 

        In this funx I hope that the returned dimensions of input_embeddings from applying the 1d conv or pos_encoder.
        src_key_padding_mask = [N, S]
        '''
        if self.embed_inputs == '1dconv':
            inputs = inputs.permute(0,2,1)
            inputs = self.input_embedding(inputs) * math.sqrt(self.hidden_dim)
            inputs = inputs.permute(0,2,1) 
        elif self.embed_inputs == 'linear':
            '''
            This here comes from Sec 3.1, page4, paper by Zerveas et al. on Transformer based framework for MTS series rep learning.
            they do not rescale the weights with factor = math.sqrt(self.hidden_dim) 
            '''
            inputs = nn.Linear(inputs, self.hidden_dim) 
        else:
            inputs = inputs

        if self.apply_pe:
            inputs = self.det_pos_encoder(inputs)
        
        src_key_padding_mask = torch.zeros(inputs.shape[0], inputs.shape[1]).to(device)
        num_blocks_max_length = batch_seq_lengths.tolist().count(max_seq_length)
        for i in range(num_blocks_max_length, len(batch_seq_lengths)):
            src_key_padding_mask[i][-(src_key_padding_mask.shape[1]-batch_seq_lengths[i].item()):] = 1
        outputs = self.transformer_encoder(src = inputs, mask = self.source_mask(max_seq_length), src_key_padding_mask = src_key_padding_mask.bool())
        
        idx = torch.tensor([s-1 for s in batch_seq_lengths])

        embeddings = {}
        if get_embeddings:
            max_seq_l = batch_seq_lengths[0] #cuz bsize=1 when getting embeddings
            embeddings[max_seq_l] = outputs[0, :max_seq_l, :]

        outputs = outputs[torch.arange(outputs.size(0)), idx]
        outputs = self.fc(outputs)
        outputs = self.relu(outputs)
        print(f'output after final linear layer and slicing: {outputs.shape}')
        
        if not get_embeddings:
            return outputs
        else:
            return outputs, embeddings

#Code from https://github.com/locuslab/TCN
       
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): 
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module): 
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation)) 
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1) 
        self.conv2.weight.data.normal_(0, 0.1) 
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01) 
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) #residual connection
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_layers, kernel_size=2, dropout=0.2): #
        super(TemporalConvNet, self).__init__()
        layers = [] ; num_channels = num_channels*num_layers
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)] 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, num_layers, kernel_size, dropout_prob, is_sequence, feedfw_size, 
                        feat_extract_only, dropout_prob_fcn):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, num_layers, kernel_size, dropout_prob)
        self.fcn = nn.Linear(num_channels[-1], output_dim)
        self.fcn_1 = nn.Linear(num_channels[-1], feedfw_size)
        self.fcn_2 = nn.Linear(feedfw_size, output_dim)
        
        self.is_sequence = is_sequence #if we only want to predict the last time step set is sequence to False
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob_fcn)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.feat_extract_only = feat_extract_only

    def forward(self, inputs, seq_length, get_embeddings=None):
        """Inputs need to have dimension (N, C_in, L_in) batch_size, input size, seq_length""" 
        y1 = self.tcn(inputs) ; del(inputs) ; torch.cuda.empty_cache()
        if self.is_sequence:
            output = self.fcn(y1.transpose(1, 2))

        else:
            interm_output = torch.zeros(y1[:,:,-1].shape).to(device) ; embeddings = {}
            for pos, seq in enumerate(seq_length):
                interm_output[pos] = y1[pos,:,seq-1] 

            if get_embeddings:
                interm_embedding = y1[pos, :, seq_length[0]]
                embeddings[seq_length[0]] = interm_embedding.permute(1, 0)
            
            del(y1) ; del(pos) ; del(seq) ; del(seq_length) ;torch.cuda.empty_cache()

            if self.feat_extract_only:
                output = self.relu(self.fcn(interm_output)) 
            else:
                output = self.relu(self.fcn_2(self.dropout(self.tanh(self.fcn_1(interm_output)))))
            
            del(interm_output); torch.cuda.empty_cache()
            
        if not get_embeddings:
            return output
        else:
            return output, embeddings
        

class TCN_att(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, num_layers, kernel_size, dropout_prob, is_sequence, feedfw_size, 
                        feat_extract_only, dropout_prob_fcn, num_heads):
        super(TCN_att, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, num_layers, kernel_size, dropout_prob)
        self.fcn = nn.Linear(num_channels[-1], output_dim)
        self.fcn_1 = nn.Linear(num_channels[-1], feedfw_size)
        self.fcn_2 = nn.Linear(feedfw_size, output_dim)
        
        self.is_sequence = is_sequence #if we only want to predict the last time step set is sequence to False
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob_fcn)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.multi_head_attn = nn.MultiheadAttention(num_channels[-1], num_heads, batch_first=True)

        self.feat_extract_only = feat_extract_only

    def forward(self, inputs, seq_length, get_embeddings=None):
        """Inputs need to have dimension (N, C_in, L_in) batch_size, input size, seq_length""" 
        y1 = self.tcn(inputs) ; del(inputs) ; torch.cuda.empty_cache()
        print(f"output: {y1.shape}")
        if self.is_sequence:
            output = self.fcn(y1.transpose(1, 2))

        else:
            interm_output = torch.zeros(y1[:,:,-1].shape).to(device) ; embeddings = {}
            
            for pos, seq in enumerate(seq_length):
                interm_output[pos] = y1[pos,:,seq-1] 
            
            print(f'interm_out: {interm_output.shape}')

            
            #add attention
            self_attn, _ = self.multi_head_attn(interm_output, interm_output, interm_output) #get an interaction between the output channels
            
            if get_embeddings:
                interm_embedding = y1[0, :, :seq_length[0]] ; key = str(seq_length[0].item())
                embeddings[key] = interm_embedding.permute(1, 0).clone()               
                att_embs = interm_output + self_attn
                print(f"self_attention: {self_attn}")
                print(f"attention embds: {att_embs} and shape: {att_embs.shape}")
                pred_length = att_embs.shape[0]
                embeddings[key][-pred_length:] = att_embs
            
            del(y1) ; del(pos) ; del(seq) ; del(seq_length) ; torch.cuda.empty_cache()

            if self.feat_extract_only:
                output = self.relu(self.fcn(interm_output + self_attn)) 
            else:
                output = self.relu(self.fcn_2(self.dropout(self.tanh(self.fcn_1(interm_output + self_attn)))))
            
            del(self_attn) ; torch.cuda.empty_cache()
            
        if not get_embeddings:
            return output
        else:
            return output, embeddings

