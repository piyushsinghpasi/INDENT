import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.espnet_attn import MultiHeadedAttention


# from nets.espnet_attn import MultiHeadedAttention
import scipy.stats as stats

class Gaussian_cross_attn(nn.Module):
    def __init__(self):
        super().__init__()
        
    def generate_weights(self, num_chunks, max_chunk, curr_device):
        std = 2.5

        frame = torch.zeros((max_chunk, 5)).float().to(curr_device)
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

    def __init__(self, input_dim = 768, hidden_dim = 768, dropout = 0.1, PyramidalConv = False, kernel_size = 10):
        super(Feat_Merger, self).__init__()
        self.fc_speech = nn.Linear(input_dim, hidden_dim)
        # self.fc_text = nn.Linear(input_dim, hidden_dim)

        self.self_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type="self")

        # self.gaussian_cross_attn_layer = MultiHeadedAttention(n_head = 1, n_feat = hidden_dim, dropout_rate = dropout, attn_type="cross") 
        self.gaussian_cross_attn_layer = Gaussian_cross_attn()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
 
        self.dropout1 = nn.Dropout(p=dropout)       
        self.dropout2 = nn.Dropout(p=dropout)       
    
    def forward(self, speech, text, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask = None, padding_cross_attn_mask = None):
        '''
        Batch B, Timesteps T, Hidden Dim D, No. of Qs per Segment Q (5 in this case)
        speech: ( [C_1, C_2, ...,C_B], D)
        text: (B, Q, D)
        '''

        t_feat = text
        # t_feat = self.fc_text(t_feat) 
        # t_feat = F.normalize(t_feat, p=2, dim=-1)

        if speech is None: return t_feat

        speech = self.norm1(speech)
        s_feat = speech + self.fc_speech(speech)
        s_feat = self.norm2(self.dropout1(s_feat))
        
        s_feat = self.sequence_to_padding(s_feat, segment_len)
        src1 = self.self_attn_layer(s_feat, s_feat, s_feat, mask = speech_attn_mask)
        
        src1 = s_feat + self.dropout2(src1)
        self_attd_chunk = self.self_attn_norm(src1)

        # cross_attd_chunk = self.gaussian_cross_attn_layer(s_feat, t_feat, t_feat, mask = padding_cross_attn_mask)
        cross_attd_chunk = self.gaussian_cross_attn_layer(segment_len, self_attd_chunk.size()[1], t_feat)

        return self_attd_chunk, cross_attd_chunk

    def sequence_to_padding(self, x, length): 
        ret_tensor = torch.zeros((length.shape[0], torch.max(length)) + tuple(x.shape[1:])).to(x.device) 
        cum_len = 0  
        for i, l in enumerate(length): 
            ret_tensor[i, :l] = x[cum_len: cum_len+l] 
            cum_len += l 
            
        return ret_tensor