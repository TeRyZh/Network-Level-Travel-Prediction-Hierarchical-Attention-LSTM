import torch
import torch.nn as nn
import math
import numpy as np
from logging import getLogger

import sys
sys.path.insert(0, 'D:\\Dropbox\\0_Project_Ongoing\\Big Data Application for Travel Time prediction\\Bigscity-LibCity')

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel



class HierAttnLstm(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()

        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.natt_unit = self.hidden_size
        self.natt_hops = config.get('natt_hops', 4)
        self.nfc = config.get('nfc', 256)
        self.max_up_len = config.get('max_up_len', 80)

        self.input_size = self.num_nodes * self.feature_dim

        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(self.input_size, self.hidden_size)
        ] + [
            nn.LSTMCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)
        ])
        self.hidden_state_pooling = nn.ModuleList([
            SelfAttentionPooling(self.hidden_size) for _ in range(self.num_layers - 1)
        ])
        self.cell_state_pooling = nn.ModuleList([
            SelfAttentionPooling(self.hidden_size) for _ in range(self.num_layers - 1)
        ])
        self.self_attention = SelfAttention(self.natt_unit, self.natt_hops)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size * self.natt_hops, self.nfc),
            nn.ReLU(),
            nn.Linear(self.nfc, self.num_nodes * self.output_dim)
        )

    def forward(self, batch):
        src = batch['X'].clone()  # [batch_size, input_window, num_nodes, feature_dim]
        src = src.permute(1, 0, 2, 3)  # [input_window, batch_size, num_nodes, feature_dim]
        # print("src shape: ", src.shape)
        batch_size = src.shape[1]
        src = src.reshape(self.input_window, batch_size, self.num_nodes * self.feature_dim)

        outputs = []
        for i in range(self.output_window):
            hidden_states = [torch.zeros(batch_size, self.hidden_size).to(self.device) for _ in range(self.num_layers)]
            cell_states = [torch.zeros(batch_size, self.hidden_size).to(self.device) for _ in range(self.num_layers)]

            bottom_layer_outputs = []
            cell_states_history = [[] for _ in range(self.num_layers)]
            for t in range(self.input_window):
                hidden_states[0], cell_states[0] = self.lstm_cells[0](src[t], (hidden_states[0], cell_states[0]))
                bottom_layer_outputs.append(hidden_states[0])
                cell_states_history[0].append(cell_states[0])

            bottom_layer_outputs = torch.stack(bottom_layer_outputs, dim=1)
            cell_states_history[0] = torch.stack(cell_states_history[0], dim=1)

            for layer in range(1, self.num_layers):
                layer_inputs = bottom_layer_outputs if layer == 1 else layer_outputs
                layer_outputs = []
                cell_states_history[layer] = []
                layer_strides = self.calculate_stride(layer_inputs.size(1))

                for start, end in layer_strides:
                    segment = layer_inputs[:, start:end, :]
                    cell_segment = cell_states_history[layer-1][:, start:end, :]

                    pooled_hidden = self.hidden_state_pooling[layer-1](segment)
                    pooled_cell = self.cell_state_pooling[layer-1](torch.cat([cell_segment, cell_states[layer].unsqueeze(1)], dim=1))
                    hidden_states[layer], cell_states[layer] = self.lstm_cells[layer](pooled_hidden, (hidden_states[layer], pooled_cell))
                    layer_outputs.append(hidden_states[layer])
                    cell_states_history[layer].append(cell_states[layer])

                layer_outputs = torch.stack(layer_outputs, dim=1)
                cell_states_history[layer] = torch.stack(cell_states_history[layer], dim=1)

            # print("layer_outputs shape: ", layer_outputs.shape) # [batch, sequence, hidden_size]

            attended_features, _ = self.self_attention(layer_outputs)
            flattened = attended_features.view(batch_size, -1)
            out = self.fc_layer(flattened)
            out = out.view(batch_size, self.num_nodes, self.output_dim)
            outputs.append(out.clone())
            
            if i < self.output_window - 1:
                src = torch.cat((src[1:, :, :], out.reshape(batch_size, self.num_nodes * self.feature_dim).unsqueeze(0)), dim=0)

        outputs = torch.stack(outputs)
        # outputs = [output_window, batch_size, num_nodes, output_dim]
        return outputs.permute(1, 0, 2, 3)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_stride(self, sequence_len):
        up_len = min(self.max_up_len, math.ceil(math.sqrt(sequence_len)))
        idx = np.linspace(0, sequence_len - 1, num=up_len + 3).astype(int)
        if idx[-1] != sequence_len - 1:
            idx = np.append(idx, sequence_len - 1)
        strides = list(zip(idx[:-1], idx[1:]))
        return strides

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep

class SelfAttention(nn.Module):
    def __init__(self, attention_size, att_hops):
        super(SelfAttention, self).__init__()
        self.ut_dense = nn.Sequential(
            nn.Linear(attention_size, attention_size),
            nn.Tanh()
        )
        self.et_dense = nn.Linear(attention_size, att_hops)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        ut = self.ut_dense(inputs)
        # et shape: [batch_size, seq_len, att_hops]
        et = self.et_dense(ut)
        att_scores = self.softmax(torch.permute(et, (0, 2, 1)))
        output = torch.bmm(att_scores, inputs)
        return output, att_scores


def generate_pseudo_inputs(batch_size=500, input_window=48, output_window=6, num_nodes=325, feature_dim=1):
    """Generates pseudo inputs for the HierAttnLstm model."""
    X = torch.rand(batch_size, input_window, num_nodes, feature_dim)  
    y = torch.rand(batch_size, output_window, num_nodes, feature_dim)
    return {'X': X, 'y': y}


if __name__ == '__main__':

# python libcity/model/traffic_flow_prediction/HierAttnLstm.py

    config = {
        'input_window': 48,
        'output_window': 6,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'hidden_size': 64,  # You can adjust these parameters
        'num_layers': 2,
        'natt_unit': 64,
        'natt_hops': 4,
        'nfc': 256,
        'max_up_len': 80,
    }

    data_feature = {
        'num_nodes': 325,
        'feature_dim': 1,
        'output_dim': 1,  # Assuming you want to predict 1 output feature
        'scaler': None  # For simplicity, we'll ignore scaling here
    }

    model = HierAttnLstm(config, data_feature)
    batch = generate_pseudo_inputs()

    # Forward Pass
    output = model(batch)
    print("Output shape:", output.shape)  # Should be (5206, 6, 325, 1)