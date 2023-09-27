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
    negs = []
    segment_len = [] # number of chunks per segment
    chunk_timestep_len = []
    chunk_start_end = []
    audio_ID = []
    max_chunk = 0
    
    for i, segment in enumerate(batch):
        seg_size = len(segment['chunks'])
        segment_len.append(seg_size)
        max_chunk = max(max_chunk, seg_size)
        chunk_timestep_len += segment['chunk_timestep_len']
        # end = start + seg_size - 1
        chunk_start_end += [ [0, seg_size - 1] ]
        # chunk_start_end += [ [start, end]*seg_size ]
        negs += segment['negs']
        # start = end+1
        
        collapsed_segs += segment['chunks']

        # segment['Q_emb'] is 5 x D
        Q_emb  += [segment['seg_Q_ID_emb'].squeeze().unsqueeze(0)]
        
        # seg_Q_ID += [segment['seg_Q_ID']]
    
    segment_len = torch.Tensor(segment_len).long()

    padded_segment_chunks = pad_sequence(collapsed_segs).permute(1, 0, 2)
    padded_negs = pad_sequence(negs).permute(1, 0, 2)
    
    max_pad = padded_segment_chunks.shape[1]

    for i, segment in enumerate(batch):
        audio_ID += [segment['audio_ID']]*max_chunk
    audio_ID = torch.Tensor(audio_ID).long()
    # (C1+C2+C3+C4+C5) x 1
    chunk_timestep_len = torch.Tensor(chunk_timestep_len).long().unsqueeze(1)
    
    # (C1+C2+C3+C4+C5) x T x D
    batch_chunks = torch.from_numpy(padded_segment_chunks.cpu().detach().numpy())

    # B x num_Q_per_segment x D
    Q_emb = torch.from_numpy(torch.cat(Q_emb, dim=0).cpu().detach().numpy())
    # Q_emb = torch.repeat_interleave(Q_emb, segment_len, dim=0)
    # B x num_Q_per_segment

    B = len(batch)
    T = batch_chunks.shape[1] #max pad timestep
    nsample = padded_negs.size()[1] // B
    
    dont_skip = torch.tensor(chunk_start_end)

    max_num_chunks = torch.max(segment_len)
    speech_padding_mask = torch.ones((B, max_num_chunks)).bool()
    # speech_padding_mask[torch.arange(B).unsqueeze(1), dont_skip] = False
    speech_padding_mask[torch.arange(max_num_chunks) < segment_len.unsqueeze(1)] = False
    # print("dfgh",speech_padding_mask)
    
    speech_attn_mask = torch.ones((B, max_num_chunks, max_num_chunks)).bool()
    for i in range(B):
        pad = nn.ZeroPad2d((0, max_num_chunks - segment_len[i], 0, max_num_chunks - segment_len[i]) )
        one = torch.ones((segment_len[i], segment_len[i])).long()
        speech_attn_mask[i] = pad(one).bool()
        
    # speech_attn_mask = ~speech_attn_mask
    
    return {
        "B": B,
        "T": T,
        "nsample": nsample,
        "segment_len": segment_len,
        "batch_chunks": batch_chunks,
        "speech_padding_mask": speech_padding_mask,
        # "seg_Q_ID": seg_Q_ID, # tensor of num_Q_per_segment ids int-transformed,
        # "seg_Q_ID_emb": Q_emb, # tensor
        # "text_attn_mask": text_attn_mask,
        "Q_emb": Q_emb,
        # "target_len": target_len,
        "chunk_timestep_len": chunk_timestep_len,
        "max_pad":max_pad,
        "audio_ID":audio_ID,
        "speech_attn_mask": speech_attn_mask,
        "negs":negs,
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
        # end = start + seg_size - 1
        chunk_start_end += [ [0, seg_size - 1] ]
        # chunk_start_end += [ [start, end]*seg_size ]
        # start = end+1
        
        collapsed_segs += segment['chunks']
        
        # seg_Q_ID += [segment['seg_Q_ID']]
        segment_id += [segment['segment_id']]
    
    segment_len = torch.Tensor(segment_len).long()

    padded_segment_chunks = pad_sequence(collapsed_segs).permute(1, 0, 2)
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
    # speech_padding_mask[torch.arange(B) >= segment_len.unsqueeze(1)] = False
    # print("dfgh",speech_padding_mask)
    speech_padding_mask[torch.arange(max_num_chunks) < segment_len.unsqueeze(1)] = False
    
    speech_attn_mask = torch.ones((B, max_num_chunks, max_num_chunks)).bool()
    for i in range(B):
        pad = nn.ZeroPad2d((0, max_num_chunks - segment_len[i], 0, max_num_chunks - segment_len[i]) )
        one = torch.ones((segment_len[i], segment_len[i])).long()
        speech_attn_mask[i] = pad(one).bool()
        
    # speech_attn_mask = ~speech_attn_mask
    
    return {
        "B": B,
        "T": T,
        "segment_len": segment_len,
        "batch_chunks": batch_chunks,
        "speech_padding_mask": speech_padding_mask,
        "chunk_timestep_len": chunk_timestep_len,
        "max_pad":max_pad,
        'segment_id':segment_id,
        "speech_attn_mask": speech_attn_mask,
    }

class CARE_dataset(Dataset):

    def __init__(self, segs_dir, qns_emb_file = './../care_india/questions/questions_00-02_hindi_labse_embeddings.json', train=None, transform = None):
        self.segs_dir = segs_dir
        self.transform = transform
        self.train = train

        segs_list = []
        audio_id_mapping = {}

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

            for ind, chunk in enumerate(chunk_files):
                feat_file = chunk.replace("vad_chunks", "vad_chunks_feat").replace("wav", "npy")
                if not os.path.exists(feat_file): continue
                chunks_emb_list += [torch.from_numpy(np.load(feat_file))]
                chunk_timestep_len += [chunks_emb_list[-1].shape[0]]

            if audio in audio_id_mapping.keys(): audio_ID = audio_id_mapping[audio]
            else:
                audio_ID = len(list(audio_id_mapping.keys()))
                audio_id_mapping[audio] = audio_ID

            if len(chunk_files) == 0: continue

            segs_list += [{'path': segs, 'audio': audio, 'audio_ID': audio_ID, 'seg_ind': seg_ind, 'chunks': chunks_emb_list, 'seg_Q_ID': seg_Q_ID, 'seg_Q_ID_emb': seg_Q_ID_emb, 'chunk_timestep_len': chunk_timestep_len}]

        self.segs_list = segs_list
        # with open('../care_india_data/train_segs_list.pkl', 'rb') as f:
        #     self.segs_list = pickle.load(f)
            
    def __len__(self):
        return len(self.segs_list)

    def __getitem__(self, idx):
        sample = self.segs_list[idx]
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
                feat_file = chunk.replace("vad_chunks", "vad_chunks_feat").replace("wav", "npy")
                if not os.path.exists(feat_file): continue
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

        self.qns_list = qns_list
        self.curr_segs = curr_segs
        self.qns_dict = qns_dict

    def __len__(self):
        return len(self.qns_list)

    def __getitem__(self, idx):
        if self.transform: return self.transform(self.qns_list[idx])
        return self.qns_list[idx]

class ToTensorChunk(object):

    def __call__(self, sample):

        return sample #{ 'file': sample['file'], 'segment_id': sample['segment_id'], 'chunks': sample['chunks'] }

class ToTensorQuestion(object):

    def __call__(self, sample):
        return sample


# if __name__ == "__main__":
    # question_dataset = Question_dataset(file_name='AA_0a7eaac7-d608-42a7-bb38-f621ddcf797e_AFTER_0S', test_file='./../care_india/audio_features_dict_traindata_5ques.json', transform = transforms.Compose([ToTensorQuestion()]))
    # print(question_dataset.__len__())
    # x = question_dataset.__getitem__(2)
    # print(question_dataset.qns_list)
    # print(x['Question'].shape)