import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.espnet_attn import MultiHeadedAttention
# from torch.nn.utils.rnn import pad_packed_sequence
import scipy.stats as stats
import numpy
import copy
import math

class SimpleConv1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dropout = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, stride = stride, groups=in_channel)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Input Given: (B, L, K)
        Input Needed to conv should be (B, K, L) --> Batch size B, Num channels K, Input length L
        Hence permute
        
        Also,
        (B=(C1+C2+C3+C4+C5), T_max, D) shape, B batch size, T_max padded to max timesteps, D dimensional feat
        '''
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        return x

class Mask_and_Mean(nn.Module):
    def __init__(self, hidden_dim, kernel_size, stride):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        
    
    def forward(self, convd_speech, chunk_timestep_len, max_pad):
        """Take mean

        Args:
            convd_speech (torch.Tensor): B', T', D
            chunk_timestep_len (torch.Tensor): for each chunk len of the timesteps 
            max_pad (int): _description_

        Returns:
            _type_: _description_
        """
        curr_device = convd_speech.device
        mask_seq_len = convd_speech.shape[1]
        B = convd_speech.shape[0]

        convd_chunk_timestep_len = chunk_timestep_len #torch.floor( (chunk_timestep_len - self.kernel_size / self.stride) + 1) 
        mask = torch.ones((B, mask_seq_len, self.hidden_dim), dtype = torch.bool).to(curr_device)
        row_idx = torch.arange(mask_seq_len).to(curr_device)
        mask[row_idx >= convd_chunk_timestep_len] = False
        mask = ~mask

        unpadded_convd_speech = convd_speech.masked_fill(mask, 0.)
        agg_speech = (unpadded_convd_speech).sum(dim=1) / convd_chunk_timestep_len

        return agg_speech
        
class Gaussian_cross_attn(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        
    def generate_weights(self, num_chunks, max_chunk, curr_device):

        frame = torch.zeros((max_chunk, 5)).float().to(curr_device)
        # frame.requires_grad = True
        if num_chunks == 1:
            frame[:num_chunks] = 1.0
            return frame
        
        output = []
        for i in range(num_chunks):
            mu = (i*4)/(num_chunks-1)
            l = stats.norm.pdf(range(5), mu, self.std)
            m1 = min(l)
            m2 = max(l)
            l = [(x-m1)/(m2-m1) for x in l]
            output += [l]

        
        output = torch.Tensor(output).float()
        output[0] = torch.Tensor([1., 0., 0., 0., 0.]).float()
        output[-1] = torch.Tensor([0., 0., 0., 0., 1.]).float()
        frame[:num_chunks] = output
        
        return frame

    def forward(self, segment_len, max_chunk, Q_emb):
        batch_wts = []
        for num_chunk in segment_len:
            wts = self.generate_weights(num_chunk.item(), max_chunk, Q_emb.device).unsqueeze(0)
            batch_wts.append(wts)

        #  B x max_chunk x 5
        batch_wts = torch.cat(batch_wts, dim=0).float()
        
        # B x M x 5 @ B x 5 x D
        return batch_wts @ Q_emb
            
        
class Feat_Merger(nn.Module):

    def __init__(self, input_dim = 768, hidden_dim = 768, dropout = 0.3, kernel_size = 20, stride = 2, std = 2.5):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(input_dim, input_dim)
        self.fc_speech2 = nn.Linear(input_dim, hidden_dim)
        # text is 768
        self.fc_text = nn.Linear(hidden_dim, hidden_dim)
        
        self.aggregation_layer = SimpleConv1D(input_dim, input_dim, kernel_size = kernel_size, stride = stride)
    
        self.mask_and_mean_layer = Mask_and_Mean(hidden_dim = input_dim, kernel_size = kernel_size, stride = stride)
        self.self_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type='self')

        self.gaussian_cross_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type='cross')
        # self.gaussian_cross_attn_layer = Gaussian_cross_attn(std)
        
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.speech_norm = nn.LayerNorm(input_dim)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        self.text_dropout = nn.Dropout(dropout)
        self.speech_dropout = nn.Dropout(dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, speech, text, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask = None, padding_cross_attn_mask = None):
        '''
        Batch B, Timesteps T, Hidden Dim D, No. of Qs per Segment Q
        speech: (B, T, D)
        text: (B, Q, D)
        '''

        t_feat = text
        t_feat = t_feat +  self.text_norm(self.text_dropout(self.fc_text(t_feat)))
        if speech is None: return t_feat
        
        convd_speech  = self.aggregation_layer(speech)
        agg_speech = self.mask_and_mean_layer(convd_speech, chunk_timestep_len, max_pad)
        
        agg_speech = self.norm1(agg_speech)
        s_feat = agg_speech + self.fc_speech(agg_speech)
        s_feat = self.speech_norm(self.speech_dropout(s_feat))
        s_feat = self.norm2(self.dropout2(self.fc_speech2(s_feat)))

        # s_feat = self.norm_speech(s_feat)

        # B x max_chunk x D
        s_feat = self.sequence_to_padding(s_feat, segment_len)#.to(DEVICE)
        # print(s_feat.size())
        # print(speech_attn_mask)
        src1 = self.self_attn_layer(s_feat, s_feat, s_feat, mask = speech_attn_mask)
        
        src1 = s_feat + self.dropout1(src1)
        
        # src2 = self.linear2(self.dropout2(F.relu(self.linear1(src1))))
        # self_attd_chunk = src1 + self.dropout2(src2)
        self_attd_chunk = self.self_attn_norm(src1)

        cross_attd_chunk = self.gaussian_cross_attn_layer(s_feat, t_feat, t_feat, mask = padding_cross_attn_mask) 
        # cross_attd_chunk = self.gaussian_cross_attn_layer(segment_len, self_attd_chunk.size()[1], t_feat)
        cross_attd_chunk = self.cross_attn_norm(cross_attd_chunk)

        return self_attd_chunk, cross_attd_chunk

    def sequence_to_padding(self, x, length): 
        # declare the shape, it can work for x of any shape.
        ret_tensor = torch.zeros((length.shape[0], torch.max(length)) + tuple(x.shape[1:])).to(x.device) 
        cum_len = 0  
        for i, l in enumerate(length): 
            ret_tensor[i, :l] = x[cum_len: cum_len+l] 
            cum_len += l 
        return ret_tensor