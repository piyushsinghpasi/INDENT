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


class Feat_Merger(nn.Module):

    def __init__(self):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(768, 768)
        self.fc_text = nn.Linear(768, 768)
        self.norm_speech = nn.LayerNorm(768)
        self.norm_text = nn.LayerNorm(768)

        # input_feat_size, output_feat_size, num_layers
        self.rnn_layer = nn.RNN(
            768, 768, 1,
            nonlinearity='tanh', 
            dropout = 0, 
            bidirectional=False, 
            batch_first = True
        )  
        

    def forward(self, speech = None, text = None):
        t_feat, s_feat = None, None

        if text is not None:
            t_feat = self.fc_text(text)
            t_feat = self.norm_text(t_feat)

        if speech is not None:
            
            _, h_n  = self.rnn_layer(speech)
            s_feat = self.fc_speech(h_n[0])
            s_feat = self.norm_speech(s_feat)

        return s_feat, t_feat