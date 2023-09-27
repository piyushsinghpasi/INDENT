import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import random
import pickle
from os import listdir, makedirs, system
from os.path import isfile, join, exists
# from utils.get_features import extract_wav2vec, labse
from torch.nn.utils.rnn import pad_sequence
import copy


random.seed(42)
np.random.seed(42)
DEVICE_dataloader = "cpu"

def pad_N_set(N_set):
    N_set_len = [x.shape[0] for x in N_set]

    N_set = pad_sequence(N_set, batch_first=True, padding_value=0.)

    return N_set, N_set_len

def pad_collate(batch):
    '''
    each individual sample is a segment
    segment['chunks'] = list([T1, D]) of size C
                        C: Number of chunks
                        T: padded timesteps
                        D: hidden dim
    segment['Q_emb'] = 5 x D
    # segment['seg_Q_ID'] = Tensor of 5 
    #     (Qids int-transformed using some dict mapping;
    #      make mapping random and in range [1, num Qids]
    #     )
    segment['chunk_timestep_len'] = list of size C, C[i] = #timesteps of chunk i
    '''

    collapsed_segs = []
    Q_emb = []
    segment_len = [] # number of chunks per segment
    chunk_timestep_len = []
    max_chunk = 0
    B = len(batch)
    N = 0
    for i, grp in enumerate(batch):
        N = len(grp)
        for j, segment in enumerate(grp):
            seg_size = len(segment['chunks'])
            segment_len.append(seg_size)
            max_chunk = max(max_chunk, seg_size)
            chunk_timestep_len += segment['chunk_timestep_len']
            collapsed_segs += segment['chunks']

            # segment['Q_emb'] is 5 x D
            Q_emb  += [segment['seg_Q_ID_emb'].squeeze().unsqueeze(0)]
    
    
    segment_len = torch.Tensor(segment_len).long()

    # (C1+C2+C3+C4+C5) x 1
    chunk_timestep_len = torch.Tensor(chunk_timestep_len).long().unsqueeze(1)
    
    # (C1+C2+C3+C4+C5) x T x D
    batch_chunks = torch.from_numpy(torch.stack(collapsed_segs).cpu().detach().numpy())

    # B x num_Q_per_segment x D
    Q_emb = torch.from_numpy(torch.cat(Q_emb, dim=0).cpu().detach().numpy())
    # B x num_Q_per_segment

    T = batch_chunks.shape[1] #max pad timestep
    M = torch.max(segment_len)
    
    D = Q_emb.size()[-1]
        
    speech_attn_mask = torch.ones((B*N, M, M)).bool() # attention inside a segment
    for i in range(B*N):
        pad = nn.ZeroPad2d((0, M - segment_len[i], 0, M - segment_len[i]) )
        one = torch.ones((segment_len[i], segment_len[i])).long()
        speech_attn_mask[i] = pad(one).bool()
    speech_attn_mask = ~speech_attn_mask

    speech_padding_mask = speech_attn_mask.view(B, N, M, M).contiguous()   

    S = N*M
    unfolded_in_segment_mask = torch.zeros(N, N*M, N*M).long()
    cum_r, cum_h = 0, 0
    for i in range(N):  
        pad = torch.nn.ZeroPad2d((cum_r, S-cum_r-M, cum_h, S-cum_h-M))
        one = torch.ones((M, M)).long()
        unfolded_in_segment_mask[i] = pad(one).bool()
        cum_r += M
        cum_h += M
    unfolded_in_segment_mask = unfolded_in_segment_mask.sum(dim=0).repeat(B,1,1).bool()

    # set padded chunks to False
    # just row/column of padding mask will do
    # B x N x M
    padding_mask_row = speech_padding_mask[:,:,:,0].squeeze()
    not_padding_mask_row = ~padding_mask_row
    not_padding_mask_row_int = not_padding_mask_row.long()
    not_padding_mask_grp = (not_padding_mask_row_int.view(B, N*M, 1) @ not_padding_mask_row_int.view(B, N*M, 1).permute(0, 2, 1)).squeeze().bool()
    
    numQ = 5
    padding_cross_attn_mask = padding_mask_row.unsqueeze(-1).repeat(1, 1, 1, numQ).view(B*N, M, numQ)
    # print(padding_cross_attn_mask, "t", padding_cross_attn_mask.size(), padding_mask_row.size())
    
    # set padded chunks to True
    # B x N*M x N*M
    padding_mask_grp = ~not_padding_mask_grp
    
    valid_negs_mask = ~(unfolded_in_segment_mask + padding_mask_grp)
    
    neg_count = valid_negs_mask.sum(-1)
    num_negs = torch.min(neg_count[neg_count.nonzero(as_tuple=True)]).item()

    sampled_neg_inds = []
    for i in range(B):
        for j in range(N*M):
            ind = torch.argwhere(valid_negs_mask[i][j])
            if len(ind) == 0:
                ind = torch.arange(num_negs).unsqueeze(-1)
            else:
                idx = torch.randperm(len(ind))[:num_negs]
                ind = ind[idx]
            q = torch.Tensor([i, j]).int().repeat(ind.size()[0], 1)
            
            index = torch.cat((q, ind), dim=-1)
            sampled_neg_inds += [index]

    sampled_neg_inds = torch.cat(sampled_neg_inds, dim=0)
    
    return {
        "B": B,
        "T": T,
        "M": M,
        "D": D,
        "num_negs": num_negs,
        "segment_len": segment_len,
        "batch_chunks": batch_chunks,
        "speech_padding_mask": speech_padding_mask,
        "Q_emb": Q_emb,
        "chunk_timestep_len": chunk_timestep_len,
        "speech_attn_mask": speech_attn_mask,
        "numchunks": segment_len.sum(-1),
        "unfolded_in_segment_mask":unfolded_in_segment_mask,
        "padding_mask_grp": padding_mask_grp,
        "sampled_neg_inds":sampled_neg_inds,
        "valid_negs_mask":valid_negs_mask,
        "padding_cross_attn_mask":padding_cross_attn_mask,
    }

def pad_collate_test(batch):
    '''
    each individual sample is a segment
    segment['chunks'] = list([T1, D]) of size C
                        C: Number of chunks
                        T: padded timesteps
                        D: hidden dim

    segment['chunk_timestep_len'] = list of size C, C[i] = #timesteps of chunk i

    segment['audio_ID'] = Unqiue Int used to identify negative segment of same audio file in a Batch 
    '''

    collapsed_segs = []
    segment_len = [] # number of chunks per segment
    chunk_timestep_len = []
    chunk_start_end = []
    start = 0
    num_Q_per_segment = 5
    audio_ID = []
    segment_id = []

    for i, segment in enumerate(batch):
        # print(segment.keys())
        seg_size = len(segment['chunks'])
        segment_len.append(seg_size)
        chunk_timestep_len += segment['chunk_timestep_len']
        # end = start + seg_size - 1
        chunk_start_end += [ [0, seg_size - 1] ]
        # chunk_start_end += [ [start, end]*seg_size ]
        # start = end+1
        
        collapsed_segs += segment['chunks']
        
        # seg_Q_ID += [segment['seg_Q_ID']]
        segment_id += [segment['segment_id']]
    
    segment_len = torch.Tensor(segment_len).long()

    padded_segment_chunks = pad_sequence(collapsed_segs).permute(1, 0)
    max_pad = padded_segment_chunks.shape[1]

    # (C1+C2+C3+C4+C5) x 1
    chunk_timestep_len = torch.Tensor(chunk_timestep_len).long().unsqueeze(1)

    # (C1+C2+C3+C4+C5) x T x D
    batch_chunks = torch.from_numpy(padded_segment_chunks.cpu().detach().numpy())

    # B x num_Q_per_segment
    # seg_Q_ID = torch.cat(seg_Q_ID, dim = 0).cpu().detach().numpy()

    B = 1 #batch_chunks.shape[0]
    T = batch_chunks.shape[1] #max pad timestep
    
    dont_skip = torch.tensor(chunk_start_end)

    max_num_chunks = torch.max(segment_len)
    speech_padding_mask = torch.ones((B, max_num_chunks)).bool()
    speech_padding_mask[torch.arange(B).unsqueeze(1), dont_skip] = False
    
    
    speech_attn_mask = torch.ones((B, max_num_chunks, max_num_chunks)).bool()
    for i in range(B):
        pad = nn.ZeroPad2d((0, max_num_chunks - segment_len[i], 0, max_num_chunks - segment_len[i]) )
        one = torch.ones((segment_len[i], segment_len[i])).long()
        speech_attn_mask[i] = pad(one).bool()
        
    speech_attn_mask = ~speech_attn_mask
    return {
        "B": B,
        "T": T,
        "segment_len": segment_len,
        "batch_chunks": batch_chunks,
        "speech_padding_mask": speech_padding_mask,
        # "target_len": target_len,
        "chunk_timestep_len": chunk_timestep_len,
        "max_pad":max_pad,
        'segment_id':segment_id,
        "speech_attn_mask":speech_attn_mask,
    }
class CARE_dataset(Dataset):

    def __init__(self, segs_dir, nsample, qns_emb_file = './../care_india/questions/questions_00-02_hindi_labse_embeddings.json', train=None, transform = None):
        self.segs_dir = segs_dir
        self.transform = transform
        self.train = train
        self.nsample = nsample

        segs_list = []
        audio_id_mapping = {}
        audio_seg_mapping = {}

        #for each seg store -> {path_of_seg: , file: , seg_index: , num_chunks: , chunks_emb_list: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp)

        qns_emb = pickle.load(open(qns_emb_file, 'rb'))
        qns_list = list(qns_emb.keys())
        qns_ind = {}
        for ind, qn in enumerate(qns_list): qns_ind[qn] = ind

        for segs in seg_audio_files:
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)
            chunks_emb_list = []
            chunk_timestep_len = []

            seg_dict = pickle.load(open(segs + '/chunks_qns.pkl', 'rb'))
            seg_qns = seg_dict['questions']
            seg_qns_inds = [qns_ind[qn] for qn in seg_qns]
            seg_qns_emb = [qns_emb[qn] for qn in seg_qns]

            seg_Q_ID = torch.Tensor(seg_qns_inds).long()
            seg_Q_ID_emb = torch.stack((seg_qns_emb))

            matched_chunks = pickle.load(open(segs + '/matched_dict_sim_partition.pkl', 'rb'))
            list_matched_chunks = list(matched_chunks.keys())

            for ind, chunk in enumerate(chunk_files):
                feat_file = chunk.replace("vad_chunks", "vad_text_feat").replace("wav", "npy")
                if not os.path.exists(feat_file): continue
                check_matched = chunk.replace("chunks", "text").replace("wav", "txt")
                if check_matched not in list_matched_chunks: continue
                asr_labse = torch.from_numpy(np.load(feat_file))

                with open(check_matched, 'r') as f:
                    asr = f.read()                

                # if len(asr) < 10: continue

                # print("asr: ", asr, (asr_labse < 0.01).sum(-1))

                chunks_emb_list += [asr_labse]
                chunk_timestep_len += [chunks_emb_list[-1].shape[0]]
                # print("labse asr: ", asr_labse)

            if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
            else:
                audio_ID = len(list(audio_id_mapping.keys()))
                audio_id_mapping[audio] = audio_ID

            if len(chunks_emb_list) == 0: continue

            segs_list += [{'path': segs, 'audio': audio, 'audio_ID': audio_ID, 'seg_ind': seg_ind, 'chunks': chunks_emb_list, 'seg_Q_ID': seg_Q_ID, 'seg_Q_ID_emb': seg_Q_ID_emb, 'chunk_timestep_len': chunk_timestep_len}]

            if audio in audio_seg_mapping.keys(): audio_seg_mapping[audio] += [len(segs_list)-1]
            else: audio_seg_mapping[audio] = [len(segs_list)-1] 

        self.segs_list = segs_list
        # with open('../care_india_data/train_segs_list.pkl', 'rb') as f:
        #     self.segs_list = pickle.load(f)
        self.audio_seg_mapping = audio_seg_mapping
        self.minibatches = None
        self.Grpshuffle()

    def Grpshuffle(self):
        minibatches_list = []
        nsample = self.nsample

        for i in range(2):
            for audio in self.audio_seg_mapping.keys():
                shuffled_segs_list = self.audio_seg_mapping[audio]
                random.shuffle(shuffled_segs_list)

                ind = 0
                num_segs = len(shuffled_segs_list)
                while ind + nsample <= num_segs:
                    curr_minibatch = []
                    for i in range(nsample): curr_minibatch += [self.segs_list[shuffled_segs_list[ind+i]]]
                    ind += nsample
                    minibatches_list += [curr_minibatch]

                if num_segs%nsample != 0:
                    curr_minibatch = [self.segs_list[shuffled_segs_list[i]] for i in range(num_segs-nsample, num_segs)]
                    minibatches_list += [curr_minibatch]

        random.shuffle(minibatches_list)
        self.minibatches = minibatches_list

        # return minibatches_list
            
    def __len__(self):
        return len(self.minibatches)

    def __getitem__(self, idx):
        sample = self.minibatches[idx]
        if self.transform: sample = self.transform(sample)
        return sample

    def id_to_int(self, ids):
        int_ids = list(range(1, len(ids)+1))
        random.shuffle(int_ids)

        id_to_int_mapping = dict()
        for id, int_id in zip(ids, int_ids):
            id_to_int_mapping[id] = int_id
        return id_to_int_mapping


class Test_dataset(Dataset):
    def __init__(self, segs_dir, file_name, transform = None):
        self.segs_dir = segs_dir
        segments_list = []
        self.transform = transform
        
        # for each segment -> {file: , seg_index: , list_of_chunk_embs: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp)
        audio_id_mapping = {}

        for segs in seg_audio_files:
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            if audio != file_name: continue

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)

            # empty segment
            if chunk_files == [] : continue

            chunk_emb_list = []
            chunk_timestep_len = []

            for chunk in chunk_files:
                feat_file = chunk.replace("vad_chunks", "vad_text_feat").replace("wav", "npy")
                if not os.path.exists(feat_file): continue
                chunk_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunk_emb_list[-1].shape[0]]

            if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
            else:
                audio_ID = len(list(audio_id_mapping.keys()))
                audio_id_mapping[audio] = audio_ID

            # segment unique ID
            segment_unique_id = seg_ind # + str(audio_ID)
            print("sid", segment_unique_id)
            segments_list += [{'file': file_name, 'segment_id': segment_unique_id, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len}]

        self.segments_list = segments_list

    def __len__(self):
        return len(self.segments_list)


    def __getitem__(self, idx):
        if self.transform: return self.transform(self.segments_list[idx])
        return self.segments_list[idx]

class Test_dataset_fixed(Dataset):
    def __init__(self, segs_dir, file_name, num_chunks, transform = None):
        self.segs_dir = segs_dir
        # segments_list = []
        fixed_chunks_list = []
        self.transform = transform
        self.num_chunks = num_chunks
        
        # for each segment -> {file: , seg_index: , list_of_chunk_embs: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp)
        audio_id_mapping = {}
        num = 0

        for segs in seg_audio_files:
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            if audio != file_name: continue

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)

            # empty segment
            if chunk_files == [] : continue

            chunk_emb_list = []
            chunk_timestep_len = []

            for chunk in chunk_files:
                feat_file = chunk.replace("vad_chunks", "vad_text_feat").replace("wav", "npy")
                if not os.path.exists(feat_file): continue

                num += 1
                chunk_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunk_emb_list[-1].shape[0]]

                if num%num_chunks == 0:
                    fixed_chunks_list += [{'file': file_name, 'segment_id': audio + "__" + str(num), 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len}]
                    chunk_emb_list = []
                    chunk_timestep_len = []
        
        if len(chunk_emb_list) != 0:
            fixed_chunks_list += [{'file': file_name, 'segment_id': audio + "__" + str(num), 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len}]

        #     if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
        #     else:
        #         audio_ID = len(list(audio_id_mapping.keys()))
        #         audio_id_mapping[audio] = audio_ID

        #     # segment unique ID
        #     segment_unique_id = seg_ind # + str(audio_ID)
        #     segments_list += [{'file': file_name, 'segment_id': segment_unique_id, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len}]

        # self.segments_list = segments_list
        self.fixed_chunks_list = fixed_chunks_list

    def __len__(self):
        return len(self.fixed_chunks_list)


    def __getitem__(self, idx):
        if self.transform: return self.transform(self.fixed_chunks_list[idx])
        return self.fixed_chunks_list[idx]

class Question_dataset(Dataset):
    def __init__(self, file_name, test_file='./../care_india/audio_features_dict_valdata_5ques.json', qns_file = './../care_india/questions/questions_00-02_hindi.json', qns_emb_file = './../care_india/questions/questions_00-02_hindi_labse_embeddings.json', transform=None):
        self.transform = transform
        test_dict = pickle.load(open(test_file, 'rb'))
        # print(test_dict.keys())
        curr_segs = test_dict[file_name]
        qns_list = []
        qns_dict = pickle.load(open(qns_file, 'rb'))
        qns_emb = pickle.load(open(qns_emb_file, 'rb'))

        for seg_ind in curr_segs.keys():
            for qn in curr_segs[seg_ind]['questions']:
                qns_list += [{'qn_no': qn, 'segment_id': seg_ind, 'Question': qns_emb[qn]}]
                # print((qns_emb[qn] < 0.01).sum(-1))

        self.qns_list = qns_list
        self.curr_segs = curr_segs
        self.qns_dict = qns_dict

    def __len__(self):
        return len(self.qns_list)

    def __getitem__(self, idx):
        if self.transform: return self.transform(self.qns_list[idx])
        return self.qns_list[idx]

# class ToTensor(object):

#     def __call__(self, sample):
#         pos_s = sample['pos_s']
#         pos_t = sample['pos_t']
#         # neg_s_left = sample['neg_s_left']
#         # neg_t_left = sample['neg_t_left']
#         # neg_s_right = sample['neg_s_right']
#         # neg_t_right = sample['neg_t_right']
#         # more_pos_s_left = sample['more_pos_s_left']
#         # more_pos_s_right = sample['more_pos_s_right']
#         neg_s = sample['NS']
#         neg_t = sample['NT']
#         more_pos_s = sample['PS']
#         more_pos_t = sample['PT']

#         return {
#                     'pos_s': torch.from_numpy(pos_s), 'pos_t': pos_t, 
#                     # 'neg_s_left': torch.from_numpy(neg_s_left), 'neg_t_left': torch.from_numpy(neg_t_left),
#                     # 'neg_s_right': torch.from_numpy(neg_s_right), 'neg_t_right': torch.from_numpy(neg_t_right),
#                     # 'more_pos_s_left': torch.from_numpy(more_pos_s_left), #'more_pos_t_left': np.array(more_pos_t_left),
#                     # 'more_pos_s_right': torch.from_numpy(more_pos_s_right), #, 'more_pos_t_right': np.array(more_pos_t_right)
#                     'NS': [torch.from_numpy(x) for x in neg_s], 'NT': neg_t, 
#                     'PS': [torch.from_numpy(x) for x in more_pos_s], 'PT': more_pos_t
#                     # 'NS': [neg], 'NT': []
#                 }

class ToTensorChunk(object):

    def __call__(self, sample):

        return { 'file': sample['file'], 'segment_id': sample['segment_id'], 'chunks': sample['chunks'] }

class ToTensorQuestion(object):

    def __call__(self, sample):
        return sample


# if __name__ == "__main__":
    # question_dataset = Question_dataset(file_name='AA_0a7eaac7-d608-42a7-bb38-f621ddcf797e_AFTER_0S', test_file='./../care_india/audio_features_dict_traindata_5ques.json', transform = transforms.Compose([ToTensorQuestion()]))
    # print(question_dataset.__len__())
    # x = question_dataset.__getitem__(2)
    # print(question_dataset.qns_list)
    # print(x['Question'].shape)