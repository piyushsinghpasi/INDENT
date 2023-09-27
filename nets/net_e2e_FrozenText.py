import torch
import torch.nn as nn
import torch.nn.functional as F
# import nets.espnet_attn as espnet_attn
# from torch.nn.utils.rnn import pad_packed_sequence
import scipy.stats as stats
import numpy
import copy
import math

# from main_e2e_oldInference import DEVICE

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        # self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        # curr_device = query.device
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k) #.to(curr_device)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k) #.to(curr_device)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k) #.to(curr_device)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return x # self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class SimpleConv1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dropout = 0.1):
        super().__init__()
        CHECK lESS params
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
    
    def forward(self, convd_speech, chunk_timestep_len):
        """Take mean

        Args:
            convd_speech (Tensor): B', T', D
            chunk_timestep_len (Tensor): for each chunk len of the timesteps 
            max_pad (_type_): _description_

        Returns:
            _type_: _description_
        """
        curr_device = convd_speech.device
        mask_seq_len = convd_speech.shape[1]
        B = convd_speech.shape[0]
        mask = torch.ones((B, mask_seq_len, self.hidden_dim), dtype = torch.bool).to(curr_device)
        row_idx = torch.arange(mask_seq_len).to(curr_device)
        mask[row_idx >= chunk_timestep_len] = False

        unpadded_convd_speech = convd_speech * mask
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
        # left over zeros cause issue
        # attn_mask also causes issue
        # wts = torch.randn((1, max_chunk, 5))
        # # print("cross", (wts @ Q_emb).size(), Q_emb.size())
        # return wts @ Q_emb
        batch_wts = []
        for num_chunk in segment_len:
            # wts 1, max_chunk, 5
            # print(num_chunk, max_chunk)
            wts = self.generate_weights(num_chunk.item(), max_chunk, Q_emb.device).unsqueeze(0)
            # print("wts", num_chunk, torch.isnan(wts).any())
            # wts.requires_grad = True
            batch_wts.append(wts)
                    #  B x max_chunk x 5

        #  B x max_chunk x 5
        batch_wts = torch.cat(batch_wts, dim=0).float()
        
        # B x 5 x D
        
        return batch_wts @ Q_emb
            
        
class Feat_Merger(nn.Module):

    def __init__(self, input_dim = 768, hidden_dim = 768, dropout = 0.1, PyramidalConv = False, kernel_size = 10):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(hidden_dim, hidden_dim)
        
        self.aggregation_layer = SimpleConv1D(input_dim, hidden_dim, kernel_size = kernel_size, stride = 2)
    
        self.mask_and_mean_layer = Mask_and_Mean(hidden_dim = hidden_dim, kernel_size = kernel_size)
        self.self_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout)

        self.gaussian_cross_attn_layer = Gaussian_cross_attn()
    
    def forward(self, speech, text, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask):
        '''
        Batch B, Timesteps T, Hidden Dim D, No. of Qs per Segment Q
        speech: (B, T, D)
        text: (B, Q, D)
        '''

        t_feat = F.normalize(t_feat, p=2, dim=-1)

        if speech is None: return t_feat

        convd_speech  = self.aggregation_layer(speech)
        agg_speech = self.mask_and_mean_layer(convd_speech, chunk_timestep_len)
        s_feat = self.fc_speech(agg_speech)

        # B x max_chunk x D
        '''
        check if grad propagates
        '''
        # CHECK
        if speech_padding_mask is not None:
            s_feat = self.sequence_to_padding(s_feat, segment_len)#.to(DEVICE)

        self_attd_chunk = self.self_attn_layer(s_feat, s_feat, s_feat, mask = speech_attn_mask)
        # self_attd_chunk = self.norm_self(self_attd_chunk)

        cross_attd_chunk = self.gaussian_cross_attn_layer(segment_len, self_attd_chunk.size()[1], t_feat)
        # cross_attd_chunk = self.norm_cross(cross_attd_chunk)

        self_attd_chunk = F.normalize(self_attd_chunk, p=2, dim=-1)
        cross_attd_chunk = F.normalize(cross_attd_chunk, p=2, dim=-1)

        return self_attd_chunk, cross_attd_chunk, t_feat

    def sequence_to_padding(self, x, length): 
        # declare the shape, it can work for x of any shape.
        ret_tensor = torch.zeros((length.shape[0], torch.max(length)) + tuple(x.shape[1:])).to(x.device) 
        print("check grad", ret_tensor)
        cum_len = 0  
        for i, l in enumerate(length): 
            ret_tensor[i, :l] = x[cum_len: cum_len+l] 
            cum_len += l 
        return ret_tensor