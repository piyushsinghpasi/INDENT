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
    def __init__(self, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
    
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
        mask = torch.ones((B, mask_seq_len, self.hidden_dim), dtype = torch.bool).to(curr_device)
        row_idx = torch.arange(mask_seq_len).to(curr_device)
        mask[row_idx >= chunk_timestep_len] = False
        mask = ~mask

        unpadded_convd_speech = convd_speech.masked_fill(mask, 0.)
        agg_speech = (unpadded_convd_speech).sum(dim=1) / chunk_timestep_len

        return agg_speech
        
class Gaussian_cross_attn(nn.Module):
    def __init__(self):
        super().__init__()
        
    def generate_weights(self, num_chunks, max_chunk, curr_device):
        std = 2.5

        frame = torch.zeros((max_chunk, 5)).float().to(curr_device)
        # frame.requires_grad = True
        if num_chunks == 1:
            frame[:num_chunks] = 1.0
            return frame
        
        output = []
        for i in range(num_chunks):
            mu = (i*4)/(num_chunks-1)
            l = stats.norm.pdf(range(5), mu, std)
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
                
        return batch_wts @ Q_emb
            
        
class Feat_Merger(nn.Module):

    def __init__(self, input_dim = 768, hidden_dim = 768, dropout = 0.3, PyramidalConv = False, kernel_size = 20):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(hidden_dim, hidden_dim)
        self.fc_text = nn.Linear(input_dim, hidden_dim)
        
        self.aggregation_layer = SimpleConv1D(input_dim, hidden_dim, kernel_size = kernel_size, stride = 2)
    
        self.mask_and_mean_layer = Mask_and_Mean(hidden_dim = hidden_dim, kernel_size = kernel_size)
        self.self_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type='self')

        # self.gaussian_cross_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type='cross')
        self.gaussian_cross_attn_layer = Gaussian_cross_attn()
        
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.speech_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        self.text_dropout = nn.Dropout(dropout)
        self.speech_dropout = nn.Dropout(dropout)
    
    def forward(self, speech, text, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask = None, padding_cross_attn_mask = None):
        '''
        Batch B, Timesteps T, Hidden Dim D, No. of Qs per Segment Q
        speech: (B, T, D)
        text: (B, Q, D)
        '''

        
        t_feat = self.fc_text(text)
        t_feat = self.text_norm(self.text_dropout(text))
        # t_feat = F.normalize(t_feat, p=2, dim=-1)
        
        if speech is None: return t_feat
        
        convd_speech  = self.aggregation_layer(speech)
        agg_speech = self.mask_and_mean_layer(convd_speech, chunk_timestep_len, max_pad)
        
        s_feat = self.fc_speech(agg_speech)
        s_feat = self.speech_norm(self.speech_dropout(s_feat))

        # B x max_chunk x D
        s_feat = self.sequence_to_padding(s_feat, segment_len)#.to(DEVICE)

        self_attd_chunk = self.self_attn_layer(s_feat, s_feat, s_feat, mask = speech_attn_mask)
        # self_attd_chunk = self.norm_self(self_attd_chunk)

        # cross_attd_chunk = self.gaussian_cross_attn_layer(s_feat, t_feat, t_feat, mask = padding_cross_attn_mask)
        cross_attd_chunk = self.gaussian_cross_attn_layer(segment_len, self_attd_chunk.size()[1], t_feat)
        
        print(self_attd_chunk @ t_feat.permute(0, 2, 1))
        cross_attd_chunk = self_attd_chunk

        print(0, 0)
        print(cross_attd_chunk[0][0] @ t_feat[0].T)
        print(cross_attd_chunk[0][0] @ text[0].T)

        print(1, 0)
        print(cross_attd_chunk[0][1] @ t_feat[0].T)
        print(cross_attd_chunk[0][1] @ text[0].T)
        
        print(0, 1)
        print(cross_attd_chunk[0][0] @ t_feat[1].T)
        print(cross_attd_chunk[0][0] @ text[1].T)

        print(1, 1)
        print(cross_attd_chunk[0][1] @ t_feat[1].T)
        print(cross_attd_chunk[0][1] @ text[1].T)

        t_feat = F.normalize(t_feat, p=2, dim=-1)
        text = F.normalize(text, p=2, dim=-1)
        cross_attd_chunk = F.normalize(cross_attd_chunk, p=2, dim=-1)
        
        print("\nnorm\n")
        print(0, 0)
        print(cross_attd_chunk[0][0] @ t_feat[0].T)
        print(cross_attd_chunk[0][0] @ text[0].T)

        print(1, 0)
        print(cross_attd_chunk[0][1] @ t_feat[0].T)
        print(cross_attd_chunk[0][1] @ text[0].T)
        
        print(0, 1)
        print(cross_attd_chunk[0][0] @ t_feat[1].T)
        print(cross_attd_chunk[0][0] @ text[1].T)

        print(1, 1)
        print(cross_attd_chunk[0][1] @ t_feat[1].T)
        print(cross_attd_chunk[0][1] @ text[1].T)
        # M x D, 5, D
        # tfm = t_feat @ t_feat.permute(0, 2, 1)
        
        # tm = text @ text.permute(0, 2, 1)
        
        # # tfm = torch.softmax(tfm, dim=-1)
        # # tm = torch.softmax(tm, dim=-1)
        # for b in range(text.size(0)):
        #     for q in range(text.size(1)):
        #         print(tfm[b, q, :].detach().cpu().numpy(), "\t", tm[b, q, :].detach().cpu().numpy(), tm[b, q, :].detach().cpu().numpy().sum())
        #     print()

        # text = text.view(-1, 768)
        # print(text.size())
        # torch.set_printoptions(profile="full")
        # tm = text @ text.T
        # print(torch.round(tm, decimals=1))
        # torch.set_printoptions(profile="default")
        # print((tm > 0.9).sum(-1))
        
        # tm = text @ text.permute(0, 2, 1)
        
        # # tfm = torch.softmax(tfm, dim=-1)
        # # tm = torch.softmax(tm, dim=-1)
        # for b in range(text.size(0)):
        #     for q in range(text.size(1)):
        #         print(tfm[b, q, :].detach().cpu().numpy(), "\t", tm[b, q, :].detach().cpu().numpy(), tm[b, q, :].detach().cpu().numpy().sum())
        #     print()
        self_attd_chunk = self.self_attn_norm(self_attd_chunk)
        cross_attd_chunk = self.cross_attn_norm(cross_attd_chunk)
        
        
        # self_attd_chunk = F.normalize(self_attd_chunk, p=2, dim=-1)
        # cross_attd_chunk = F.normalize(cross_attd_chunk, p=2, dim=-1)

        return self_attd_chunk, cross_attd_chunk

    def sequence_to_padding(self, x, length): 
        # declare the shape, it can work for x of any shape.
        ret_tensor = torch.zeros((length.shape[0], torch.max(length)) + tuple(x.shape[1:])).to(x.device) 
        cum_len = 0  
        for i, l in enumerate(length): 
            ret_tensor[i, :l] = x[cum_len: cum_len+l] 
            cum_len += l 
        return ret_tensor