import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import itertools

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from pytorch_forecasting.metrics import MAE, RMSE, MAPE

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
    
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
      
        attention_weight:
            att_w : size (N, T, 1)
    
        return:
            utter_rep: size (N, H)
        """
        
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


# A custom attention layer
class SelfAttention(nn.Module):
    def __init__(self, attention_size, att_hops, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.ut_dense =  nn.Sequential(
                nn.Linear(hidden_size, attention_size),
                nn.Tanh()
         )
        
        self.et_dense = nn.Linear(attention_size, att_hops)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        ut = self.ut_dense(inputs)

        # et shape: [batch_size, seq_len, att_hops]
        et = self.et_dense(ut)

        att_scores = self.softmax(torch.permute(et, (0, 2, 1)))

        # # re-normalize the masked scores
        # _sums = scores.sum(-1, keepdim=True)  # sums per row
        # att_scores = scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        
        # print("att_scores.shape: ", att_scores.shape, "inputs.shape", inputs.shape)

        # multiply each hidden state with the attention weights
        output = torch.bmm(att_scores, inputs)

        return output, att_scores



class HierLstmat(nn.Module):

    def __init__(self, num_corridor, hidden_size, num_layers, natt_unit, natt_hops, nfc):
        super(HierLstmat, self).__init__()

        self.input_size = num_corridor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_len = 0
    
        # Initialize LSTM Cell for the first layer
        self.lstm_cell_layer_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        # Initialize LSTM Cell for the second layer
        self.lstm_cell_layer_2 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # default maximum upgrade length
        self.up_len = 80      

        # flattern
        self.handle_hops = nn.Sequential(
            nn.Flatten()
            )

        # output layer
        self.output_layer =  nn.Sequential(
            nn.Linear(self.hidden_size * natt_hops, nfc),
            nn.ReLU(),
            nn.Linear(nfc, num_corridor)
        )

        # attention layer
        self.att_encoder = SelfAttention(natt_unit, natt_hops)      
        
        # attention pooling layer
        self.att_pooling = SelfAttentionPooling(self.hidden_size)

    def forward(self, x):

      self.sequence_len = x.shape[1]
      batch_Size = x.shape[0]
      sequence_input = x.transpose(0, 1)

      # print("sequence_input.device: ", sequence_input.device)
      
      # batch_size x hidden_size
      hidden_state = torch.zeros(batch_Size, self.hidden_size).to(device)
      cell_state = torch.zeros(batch_Size, self.hidden_size).to(device)
      hidden_state_2 = torch.zeros(batch_Size, self.hidden_size).to(device)
      cell_state_2 = torch.zeros(batch_Size, self.hidden_size).to(device)

      # print("hidden_state.device: ", hidden_state.device)
      
      # weights initialization
      torch.nn.init.xavier_normal_(hidden_state)
      torch.nn.init.xavier_normal_(cell_state)
      torch.nn.init.xavier_normal_(hidden_state_2)
      torch.nn.init.xavier_normal_(cell_state_2)

      # set upgrade length
      up_len = min(self.up_len, math.floor(math.sqrt(self.sequence_len)))
      # evenly spaced index
      idx = np.linspace(up_len - 1, math.pow(up_len, 2) - 1, num = up_len)
      # print("sequence index: ", idx)

      # initiate pooling hidden_sates an cell states
      interverl_hidden_states = torch.empty(batch_Size, self.hidden_size, device = device)
      interverl_cell_states = torch.empty(batch_Size, self.hidden_size, device = device)
      outer_hidden_states = torch.empty(batch_Size, self.hidden_size, device = device)
      outer_cell_states = torch.empty(batch_Size, self.hidden_size, device = device)
      
      # Unfolding LSTM
      # Last hidden_state will be used to feed the fully connected neural net
      for i in range(self.sequence_len):
        
        hidden_state, cell_state = self.lstm_cell_layer_1(sequence_input[i], (hidden_state, cell_state))

        if  torch.isnan(interverl_hidden_states).sum() == 0 :
            interverl_hidden_states = hidden_state[None, :]
            interverl_cell_states = cell_state[None, :]

        else:
            interverl_hidden_states = torch.cat((interverl_hidden_states, hidden_state[None, :]), 0) # TimeSteps * Batch  * Feature
            interverl_cell_states = torch.cat((interverl_cell_states, cell_state[None, :]), 0)       # TimeSteps * Batch  * Feature
            # print("interverl_hidden_states: ", interverl_hidden_states.shape)

        if i in idx or (i == self.sequence_len - 1):

            interverl_hidden_states = interverl_hidden_states.transpose(0,1)
            interverl_cell_states = interverl_cell_states.transpose(0,1)

            # print("interverl_hidden_states: ", interverl_hidden_states.shape)
            # print("interverl_cell_states: ", interverl_cell_states.shape)

            layer1_cell_states = torch.cat((interverl_cell_states, cell_state_2[None, :].transpose(0,1)), 1)       # Batch * (TimeSteps + 1) * Feature

            layer2_input = self.att_pooling(interverl_hidden_states)   # Batch * Feature
            cell_state_2 = self.att_pooling(layer1_cell_states)       # Batch * Feature
            
            hidden_state_2, cell_state_2 = self.lstm_cell_layer_2(layer2_input, (hidden_state_2, cell_state_2))
            interverl_hidden_states = torch.empty(batch_Size, self.hidden_size, device = device)
            interverl_cell_states = torch.empty(batch_Size, self.hidden_size, device = device)

            if torch.isnan(outer_hidden_states).sum() == 0:
              outer_hidden_states = hidden_state_2[None, :]
            else:
              outer_hidden_states = torch.cat((outer_hidden_states, hidden_state_2[None, :]))  # # Sequence * Batch  * hiddensize

      up_x = torch.transpose(outer_hidden_states, 1, 0)   # size: (batch, Sequence, hiddensize)
      # print("up_x.shape: ", up_x.shape)   # up_x.shape:  torch.Size([250, 120])
      # print("reshaped up_x: ", up_x.view(batch_Size, -1, self.hidden_size).shape)   # reshaped up_x:  torch.Size([batch_size, sample_Size, hidden_size])

      att_output, att_scores = self.att_encoder(up_x.view(batch_Size, -1, self.hidden_size))
      # att_output shape [batch_size, att_hops, LSTM_nhidden]

      att_output_flattern = self.handle_hops(att_output)  # [batch_size, att_hops * LSTM_nhidden]
      # print("att_output_flattern: ", att_output_flattern.shape)

      # Last hidden state is passed through a fully connected neural net
      output = self.output_layer(att_output_flattern)
      
      return output

if __name__ == '__main__':
    num_corridor =  num_corridor       # number of corridors also the channel in NTC
    hidden_size = 120        # lstm hidden_dim
    num_layers = 1          # lstm layers
    attention_size = 300     # the hidden_units of attention layer
    natt_hops = 2
    nfc = 512               # fully connected layer
    drop_prob = 0.5         # fully connected layer
    batch_size = 50
    output = HierLstmat(num_corridor, hidden_size, num_layers, attention_size, natt_hops, nfc)