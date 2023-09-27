import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import random
import pickle
from os import listdir, makedirs, system
from os.path import isfile, join, exists
# from utils.get_features import extract_wav2vec, labse
from torch.nn.utils.rnn import pad_sequence


random.seed(42)
np.random.seed(42)
DEVICE_dataloader = "cpu"


def getTime(s):
    # print(s)
    try:
        complex(s)
        return s
    except ValueError:
        s = s.split(":")[::-1]
        sec, mint, hr = 0, 0, 0
        if len(s) >= 1:
            sec = int(s[0])
        if len(s) >= 2:
            mint = int(s[1])
        if len(s) >= 3:
            hr = int(s[2])
        if hr > 1 and sec != 0:
            raise ValueError("Custom error: got hour time value greater than 1, could be some issue with time format")
        elif hr > 1 and sec == 0:
            sec = mint
            mint = hr
            hr = 0
        return sec + 60*mint + 60*60*hr


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
    audio_ID = []
    B = len(batch)
    N = 0
    grp_count = 0
    
    for i, grp in enumerate(batch):
        N = len(grp)
        for j, segment in enumerate(grp):
            seg_size = len(segment['chunks'])
            segment_len.append(seg_size)
            chunk_timestep_len += segment['chunk_timestep_len']
            collapsed_segs += segment['chunks']

            # segment['Q_emb'] is 5 x D
            Q_emb  += [segment['seg_Q_ID_emb'].squeeze().unsqueeze(0)]
    
    
    segment_len = torch.Tensor(segment_len).long()

    # (C1+C2+C3+C4+C5) x 1
    chunk_timestep_len = torch.Tensor(chunk_timestep_len).long().unsqueeze(1)
    
    # (C1+C2+C3+C4+C5) x T x D
    # print(collapsed_segs[0].size())
    collapsed_segs = [x.squeeze() for x in collapsed_segs]
    padded_segment_chunks = pad_sequence(collapsed_segs).permute(1, 0, 2)
    batch_chunks = torch.from_numpy(padded_segment_chunks.cpu().detach().numpy())

    # B*N x num_Q_per_segment x D
    Q_emb = torch.from_numpy(torch.cat(Q_emb, dim=0).cpu().detach().numpy())

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
        seg_size = len(segment['chunks'])
        segment_len.append(seg_size)
        chunk_timestep_len += segment['chunk_timestep_len']

        chunk_start_end += [ [0, seg_size - 1] ]
        
        collapsed_segs += segment['chunks']
        
        segment_id += [segment['segment_id']]
    
    segment_len = torch.Tensor(segment_len).long()

    collapsed_segs = [x.squeeze() for x in collapsed_segs]
    padded_segment_chunks = pad_sequence(collapsed_segs).permute(1, 0, 2)
    max_pad = padded_segment_chunks.shape[1]

    # (C1+C2+C3+C4+C5) x 1
    chunk_timestep_len = torch.Tensor(chunk_timestep_len).long().unsqueeze(1)

    # (C1+C2+C3+C4+C5) x T x D
    batch_chunks = torch.from_numpy(padded_segment_chunks.detach().cpu().numpy())
    

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

    def __init__(self, segs_dir, nsample, vad_chunks_feat_dir, D, qns_emb_file = '../../care_india/questions/questions_00-02_hindi_labse_embeddings.json', train=None, transform = None):
        self.segs_dir = segs_dir
        self.transform = transform
        self.train = train
        self.nsample = nsample

        segs_list = []
        audio_id_mapping = {}

        #for each seg store -> {path_of_seg: , file: , seg_index: , num_chunks: , chunks_emb_list: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp) # list of all segments - (audio)__(segment index)

        qns_emb = pickle.load(open(qns_emb_file, 'rb')) # qns_emb[qn] = qn_labse_emb
        qns_list = list(qns_emb.keys()) 
        qns_ind = {} # qns_ind[qn] = ind
        for ind, qn in enumerate(qns_list): qns_ind[qn] = ind

        audio_seg_mapping = {} # audio -> list of segments of that audio

        for segs in seg_audio_files:
            l = segs.split('/') # home/audio__segind
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f] # list of chunks paths of that segment
            chunk_files = sorted(temp)
            chunks_emb_list = []
            chunk_timestep_len = []

            seg_dict = pickle.load(open(segs + '/chunks_qns.pkl', 'rb')) 
            seg_qns = seg_dict['questions'] # 5 qns of that segment
            seg_qns_inds = [qns_ind[qn] for qn in seg_qns]
            seg_qns_emb = [qns_emb[qn] for qn in seg_qns]

            seg_Q_ID = torch.Tensor(seg_qns_inds).long() # qn embeddings ready for that segment
            seg_Q_ID_emb = torch.stack((seg_qns_emb))

            for ind, chunk in enumerate(chunk_files): # chunks embeddings ready for that segment
                feat_file = chunk.replace("vad_chunks", vad_chunks_feat_dir).replace("wav", "npy")
                # if not os.path.exists(feat_file): continue
                chunks_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunks_emb_list[-1].shape[0]]

            if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
            else:
                audio_ID = len(list(audio_id_mapping.keys()))
                audio_id_mapping[audio] = audio_ID

            if len(chunk_files) == 0: continue

            segs_list += [{'path': segs, 'audio': audio, 'audio_ID': audio_ID, 'seg_ind': seg_ind, 'chunks': chunks_emb_list, 'seg_Q_ID': seg_Q_ID, 'seg_Q_ID_emb': seg_Q_ID_emb, 'chunk_timestep_len': chunk_timestep_len}]
            # segs_list[i] -> audio j then audio_seg_mapping[j] will have i 

            if audio in audio_seg_mapping.keys(): audio_seg_mapping[audio] += [len(segs_list)-1]
            else: audio_seg_mapping[audio] = [len(segs_list)-1] 

        self.segs_list = segs_list
        # with open('../care_india_data/train_segs_list.pkl', 'rb') as f:
        #     self.segs_list = pickle.load(f)
        self.audio_seg_mapping = audio_seg_mapping
        self.minibatches = None
        self.D = D
        self.Grpshuffle()

    def Grpshuffle(self):
        minibatches_list = []
        nsample = self.nsample

        for i in range(self.D):
            for audio in self.audio_seg_mapping.keys():
                shuffled_segs_list = self.audio_seg_mapping[audio]
                random.shuffle(shuffled_segs_list)

                ind = 0
                num_segs = len(shuffled_segs_list)
                if num_segs < nsample: continue
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
            
    def __len__(self):
        return len(self.minibatches)

    def __getitem__(self, idx):
        sample = self.minibatches[idx]
        if self.transform: sample = self.transform(sample)
        return sample

class Test_dataset(Dataset):
    def __init__(self, segs_dir, file_name, vad_chunks_feat_dir, transform = None):
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
                feat_file = chunk.replace("vad_chunks", vad_chunks_feat_dir).replace("wav", "npy")
                # if not os.path.exists(feat_file): continue
                chunk_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunk_emb_list[-1].shape[0]]

            if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
            else:
                audio_ID = len(list(audio_id_mapping.keys()))
                audio_id_mapping[audio] = audio_ID

            # segment unique ID
            segment_unique_id = seg_ind # + str(audio_ID)
            segments_list += [{'file': file_name, 'segment_id': segment_unique_id, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len}]

        self.segments_list = segments_list

    def __len__(self):
        return len(self.segments_list)


    def __getitem__(self, idx):
        if self.transform: return self.transform(self.segments_list[idx])
        return self.segments_list[idx]

class Test_dataset_fixed(Dataset):
    def __init__(self, segs_dir, file_name, num_chunks, vad_chunks_feat_dir,transform = None):
        self.segs_dir = segs_dir
        # segments_list = []
        fixed_chunks_list = []
        self.transform = transform
        self.num_chunks = num_chunks
        
        # for each segment -> {file: , seg_index: , list_of_chunk_embs: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp, key=lambda path: int(path.split('___')[-1]))
        # seg_audio_files = sorted(temp)
        # print(seg_audio_files)
        audio_id_mapping = {}
        seg_ind = 0

        for segs in seg_audio_files:
            num = 0
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            # seg_ind = l1[-1]

            if audio != file_name: continue

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)

            seg_dict = pickle.load(open(segs + '/chunks_qns.pkl', 'rb'))
            vad_chunks = seg_dict['vad_chunks']
            seg_start_time = seg_dict['start_time']

            # empty segment
            if chunk_files == [] : continue

            chunk_emb_list = []
            chunk_timestep_len = []
            chunk_start_time = []
            chunk_end_time = []

            for chunk_ind, chunk in enumerate(chunk_files):
                feat_file = chunk.replace("vad_chunks", vad_chunks_feat_dir).replace("wav", "npy")
                # if not os.path.exists(feat_file): continue

                num += 1
                chunk_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunk_emb_list[-1].shape[0]]
                chunk_start_time += [seg_start_time + vad_chunks[chunk_ind][0]] # chunk start time 
                chunk_end_time += [seg_start_time + vad_chunks[chunk_ind][1]] # chunk end time

                if num%num_chunks == 0:
                    fixed_chunks_list += [{'file': file_name, 'segment_id': seg_ind, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len, 'chunk_start_time': chunk_start_time, 'chunk_end_time': chunk_end_time, 'segment_start_time': chunk_start_time[0], 'segment_end_time': chunk_end_time[-1]}]
                    chunk_emb_list = []
                    chunk_timestep_len = []
                    chunk_start_time, chunk_end_time = [], []
                    seg_ind += 1

        if len(chunk_emb_list) != 0:
            fixed_chunks_list += [{'file': file_name, 'segment_id': seg_ind, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len, 'chunk_start_time': chunk_start_time, 'chunk_end_time': chunk_end_time, 'segment_start_time': chunk_start_time[0], 'segment_end_time': chunk_end_time[-1]}]


        for seg in fixed_chunks_list:
            print(len(seg['chunks']))
            # print(seg['segment_id'], seg['segment_start_time'], seg['segment_end_time'])
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


class Test_dataset_fixed_new(Dataset):
    def __init__(self, segs_dir, file_name, num_chunks, vad_chunks_feat_dir, transform = None):
        self.segs_dir = segs_dir
        fixed_chunks_list = []
        self.transform = transform
        self.num_chunks = num_chunks

        audio = join(segs_dir, file_name)
        seg_ind = 0

        temp = [(audio + '/vad_chunks/' + f) for f in listdir(audio + '/vad_chunks/') if '.wav' in f]
        chunk_files = sorted(temp)

        aud_dict = pickle.load(open(audio + '/chunks_qns.pkl', 'rb'))
        vad_chunks = aud_dict['vad_chunks']

        # if chunk_files == [] : continue

        chunk_emb_list = []
        chunk_timestep_len = []
        chunk_start_time = []
        chunk_end_time = []

        num = 0
        for chunk_ind, chunk in enumerate(chunk_files):
            feat_file = chunk.replace("vad_chunks", vad_chunks_feat_dir).replace("wav", "npy")

            num += 1
            chunk_emb_list += [torch.from_numpy(np.load(feat_file))]
            chunk_timestep_len += [chunk_emb_list[-1].shape[0]]
            chunk_start_time += [vad_chunks[chunk_ind][0]] # chunk start time 
            chunk_end_time += [vad_chunks[chunk_ind][1]] # chunk end time

            if num%num_chunks == 0:
                fixed_chunks_list += [{'file': file_name, 'segment_id': seg_ind, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len, 'chunk_start_time': chunk_start_time, 'chunk_end_time': chunk_end_time, 'segment_start_time': chunk_start_time[0], 'segment_end_time': chunk_end_time[-1]}]
                chunk_emb_list = []
                chunk_timestep_len = []
                chunk_start_time, chunk_end_time = [], []
                seg_ind += 1

        if len(chunk_emb_list) != 0:
            fixed_chunks_list += [{'file': file_name, 'segment_id': seg_ind, 'chunks': chunk_emb_list, 'chunk_timestep_len': chunk_timestep_len, 'chunk_start_time': chunk_start_time, 'chunk_end_time': chunk_end_time, 'segment_start_time': chunk_start_time[0], 'segment_end_time': chunk_end_time[-1]}]

        self.fixed_chunks_list = fixed_chunks_list

    def __len__(self):
        return len(self.fixed_chunks_list)


    def __getitem__(self, idx):
        if self.transform: return self.transform(self.fixed_chunks_list[idx])
        return self.fixed_chunks_list[idx]


class Question_dataset_new(Dataset):
    def __init__(self, test_data_object, file_name, csv_dir, test_file, qns_file = '../../care_india/questions/questions_00-02_hindi.json', qns_emb_file = '../../care_india/questions/questions_00-02_hindi_labse_embeddings.json', transform=None):
        self.transform = transform
        # test_dict = pickle.load(open(test_file, 'rb'))
        # print(test_dict.keys())

        # curr_segs = test_dict[file_name]
        qns_list = []
        qns_dict = pickle.load(open(qns_file, 'rb'))
        qns_emb = pickle.load(open(qns_emb_file, 'rb'))

        # for seg_ind in list(curr_segs.keys()):
        #     for qn in curr_segs[seg_ind]['questions']:
        #         qns_list += [{'qn_no': qn, 'segment_id': seg_ind, 'Question': qns_emb[qn]}]
        csv_file_path = csv_dir + file_name + '.csv'
        csv_df = pd.read_csv(csv_file_path, usecols=[0, 1, 2], names=['Q_id', 'start_time', 'end_time'])

        fixed_chunks_list = test_data_object.fixed_chunks_list
        # print(fixed_chunks_list[0]['segment_id'])
        curr_seg_ind = 0
        
        # print(file_name)

        tot_q_dur = 0.
        for idx,row in csv_df.iterrows():
            qn, qn_start, qn_end = row['Q_id'], getTime(row['start_time']), getTime(row['end_time'])  
            
            tot_q_dur += qn_end - qn_start
            # !!!need to write len-1!!!
            while curr_seg_ind < len(fixed_chunks_list)-1 and qn_start > fixed_chunks_list[curr_seg_ind]['segment_end_time']: curr_seg_ind += 1
            
            # try:
            #     print(qn_start, qn_end, curr_seg_ind, fixed_chunks_list[curr_seg_ind]['segment_id'], fixed_chunks_list[curr_seg_ind]['segment_start_time'], fixed_chunks_list[curr_seg_ind]['segment_end_time']) 
            # except:
            #     print(qn, qn_start, qn_end)


            # print(curr_seg_ind, len(fixed_chunks_list))

            if fixed_chunks_list[curr_seg_ind]['segment_end_time'] - qn_start <= qn_end - fixed_chunks_list[curr_seg_ind]['segment_end_time'] and curr_seg_ind != len(fixed_chunks_list)-1: curr_seg_ind += 1

            qns_list += [{'qn_no': qn, 'qn_start_time': qn_start, 'qn_end_time': qn_end, 'Question': qns_emb[qn], 'segment_id': curr_seg_ind}]

        # print(tot_q_dur)
        self.qns_list = qns_list
        # self.curr_segs = curr_segs
        self.qns_dict = qns_dict

    def __len__(self):
        return len(self.qns_list)

    def __getitem__(self, idx):
        if self.transform: return self.transform(self.qns_list[idx])
        return self.qns_list[idx]


class Question_dataset(Dataset):
    def __init__(self, file_name, test_file, qns_file = '../../care_india/questions/questions_00-02_hindi.json', qns_emb_file = '../../care_india/questions/questions_00-02_hindi_labse_embeddings.json', transform=None):
        self.transform = transform
        test_dict = pickle.load(open(test_file, 'rb'))
        # print(test_dict.keys())
        curr_segs = test_dict[file_name]
        qns_list = []
        qns_dict = pickle.load(open(qns_file, 'rb'))
        qns_emb = pickle.load(open(qns_emb_file, 'rb'))

        for seg_ind in list(curr_segs.keys()):
            for qn in curr_segs[seg_ind]['questions']:
                qns_list += [{'qn_no': qn, 'segment_id': seg_ind, 'Question': qns_emb[qn]}]

        self.qns_list = qns_list
        self.curr_segs = curr_segs
        self.qns_dict = qns_dict

    def __len__(self):
        return len(self.qns_list)

    def __getitem__(self, idx):
        if self.transform: return self.transform(self.qns_list[idx])
        return self.qns_list[idx]
