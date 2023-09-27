import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn.utils.rnn import pad_packed_sequence

import numpy
import copy
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RNN_layer(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, batch_first = True, bidirectional = True)
        # self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first = True, bidirectional = True)
        
        # self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : [sen_len, batch_size]
        # embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(src)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return outputs, hidden, cell

class Feat_Merger(nn.Module):

    def __init__(self):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(768, 768)
        self.fc_text = nn.Linear(768, 768)

        # input_feat_size, output_feat_size, num_layers
        self.rnn_layer = nn.RNN(
            768, 768, 1,
            nonlinearity='relu', 
            dropout = 0, 
            bidirectional=False, 
            batch_first = True
        )  
        

    def forward(self, speech = None, text = None):
        t_feat, s_feat = None, None

        if text is not None:
            t_feat = self.fc_text(text)
            t_feat = F.normalize(t_feat, p=2, dim=-1)

        if speech is not None:
            
            _, h_n  = self.rnn_layer(speech)
            s_feat = self.fc_speech(h_n[0])
            s_feat = F.normalize(s_feat, p=2, dim=-1)

        return s_feat, t_feat