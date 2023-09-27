import torch
import torch.nn as nn

class contrastAcrossSegments(nn.Module):
    def __init__(self, temperature = 1.0):
        super().__init__()
        self.temperature = temperature 

    def forward(self, self_attd_chunk, cross_attd_chunk, padding_mask, valid_negs_mask, sampled_neg_inds):
        """Loss function
            B batch size, N groups, M max number of chunks, D dim
        Args:
            self_attd_chunk (Tensor): B x N x M x D
            cross_attd_chunk (Tensor): B x N x M x D

        Returns:
            loss obj: Contrastive loss
        """
        eps = 1e-5
        
        B, N, M, D = self_attd_chunk.size()
    
        
        # B x N x M
        num = (self_attd_chunk * cross_attd_chunk).sum(dim=-1)
        numerator = torch.exp( num / self.temperature )
        
        # B x N x M x M
        # sim_segment = self_attd_chunk @ cross_attd_chunk.permute(0, 1, 3, 2)
        # sim_segment = torch.exp( self_attd_chunk @ cross_attd_chunk.permute(0, 1, 3, 2) / self.temperature )
        
        # # B x N x M
        # sim_segment = sim_segment.masked_fill(padding_mask, 0.).sum(dim=-1)
        
        
        # set padded chunks to False
        # B x N x M
        padding_mask_row = padding_mask[:,:,:,0].squeeze()
        
        
        # B x N*M x D
        self_attd_chunk_unfolded = self_attd_chunk.view(B, N*M, D)
        cross_attd_chunk_unfolded = cross_attd_chunk.view(B, N*M, D)
        
        # B x N*M x N*M
        # sim_grp = self_attd_chunk_unfolded @ cross_attd_chunk_unfolded.permute(0, 2, 1)
        sim_grp = torch.exp( self_attd_chunk_unfolded @ cross_attd_chunk_unfolded.permute(0, 2, 1) / self.temperature )
        
        invalid_pos = ~valid_negs_mask
        sim_grp = sim_grp.masked_fill(invalid_pos, 0.) #.masked_fill(unfolded_in_segment_mask, 0.)
        
        sim_grp = sim_grp[sampled_neg_inds[:, 0], sampled_neg_inds[:, 1], sampled_neg_inds[:, 2]]
        
        # B x N x M
        sim_grp = sim_grp.view(B, N, M, -1)
        _, _, _, num_negs = sim_grp.size()
        denominator = sim_grp.sum(-1)
        

        m = 0.3
        # mmrl = max(0, - num + denominator + m)
        # print("num | ", num)
        # print("den |", denominator)
        log_exp = torch.log( numerator / ( numerator + denominator) )
        total_num_chunks = (~padding_mask_row).sum(-1).sum(-1).sum(-1)
        log_exp = log_exp.masked_fill(padding_mask_row, 0.).sum(-1).sum(-1).sum(-1) / total_num_chunks
        

        sim_loss = (1 - num).masked_fill(padding_mask_row, 0.).sum(-1).sum(-1).sum(-1) / total_num_chunks

        return -log_exp, sim_loss