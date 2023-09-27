
import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_style(nn.Module):
    def __init__(self, smoothing = 0.1, temperature = 1.0):
        super().__init__()
        self.temperature = temperature 
        self.smoothing = smoothing

    def forward(self, self_attd_chunk, cross_attd_chunk, padding_mask, valid_negs_mask, sampled_neg_inds):
        """Loss function
            B batch size, N groups, M max number of chunks, D dim
        Args:
            self_attd_chunk (Tensor): B x N x M x D
            cross_attd_chunk (Tensor): B x N x M x D
            padding_mask : B x N x M x M
            valid_negs_mask : B x N*M x N*M
            sample_neg_inds : B*N*M*num_negs x 3

        Returns:
            loss obj: Contrastive loss
        """
        
        B, N, M, D = self_attd_chunk.size()
    
        # B x N x M
        num = (self_attd_chunk * cross_attd_chunk).sum(dim=-1)
        numerator = num / self.temperature
        
        # B x N x M x M
        # sim_segment = self_attd_chunk @ cross_attd_chunk.permute(0, 1, 3, 2)
        # sim_segment = torch.exp( self_attd_chunk @ cross_attd_chunk.permute(0, 1, 3, 2) / self.temperature )
        
        # # B x N x M
        # sim_segment = sim_segment.masked_fill(padding_mask, 0.).sum(dim=-1)
        
        # B x N x M
        padding_mask_row = padding_mask[:,:,:,0].squeeze() # padded chunk -> true
        
        # B x N*M x D
        self_attd_chunk_unfolded = self_attd_chunk.view(B, N*M, D)
        cross_attd_chunk_unfolded = cross_attd_chunk.view(B, N*M, D)
        
        # B x N*M x N*M
        sim_grp = self_attd_chunk_unfolded @ cross_attd_chunk_unfolded.permute(0, 2, 1) / self.temperature
        
        invalid_pos = ~valid_negs_mask
        sim_grp = sim_grp.masked_fill(invalid_pos, 0.)
        
        sim_grp = sim_grp[sampled_neg_inds[:, 0], sampled_neg_inds[:, 1], sampled_neg_inds[:, 2]]
        
        # B x N x M x num_negs
        sim_grp = sim_grp.view(B, N, M, -1)
        _, _, _, num_negs = sim_grp.size()
        
        # B x N x M x (num_negs+1)
        pred = torch.cat([numerator.unsqueeze(-1), sim_grp], dim=-1)
        log_prob = pred.log_softmax(dim=-1)
        
        not_padded = ~padding_mask_row
    
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            if num_negs == 1:
                true_dist.fill_(0.)
                true_dist[:, :, :, 0] = 1.0
            else:
                true_dist.fill_(self.smoothing / (num_negs-1))
                true_dist[:, :, :, 0] = 1.0 - self.smoothing

        chunk_loss = torch.sum(-true_dist * log_prob, dim=-1)
        chunk_loss = chunk_loss[not_padded]
        
        ce_loss = chunk_loss.mean()

        num.clamp_(min=-1.0, max=1.0)
        # num_norm = (F.normalize(self_attd_chunk, p=2, dim=-1) * F.normalize(cross_attd_chunk, p=2, dim=-1)).sum(dim=-1)
        sim_loss = 1 - num
        sim_loss = sim_loss[not_padded]
        sim_loss = sim_loss.mean()

        return ce_loss, sim_loss