import numpy as np
import torch
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

random.seed(0)

def pad_N_set(N_set):
    N_set_len = [x.shape[0] for x in N_set]

    N_set = pad_sequence(N_set, batch_first=True, padding_value=0.)

    return N_set, N_set_len

def pad_collate(batch):

    batch_pos_s, batch_pos_t, batch_NS, batch_NS_len, batch_NT_len, batch_NT = [], [], [], [], [], []
    batch_PS, batch_PS_len, batch_PT_len, batch_PT = [], [], [], []
    for sample_dict in batch:
        batch_pos_s.append(sample_dict['pos_s'])            
        batch_pos_t.append(sample_dict['pos_t'])

        batch_NS += sample_dict['NS']
        batch_PS += sample_dict['PS']

        NT = sample_dict['NT']
        PT = sample_dict['PT']

        batch_NT.append(torch.cat(NT, dim=0)) #.cpu().detach().numpy())
        batch_PT.append(torch.cat(PT, dim=0))
        # batch_NS_len.append(NS_len)
        # batch_NT_len.append(NT_len)

    pos_s_len = [x.shape[0] for x in batch_pos_s]
    pos_t_len = [x.shape[0] for x in batch_pos_s]

    pos_s_pad = pad_sequence(batch_pos_s, batch_first=True, padding_value=0.)
    pos_t = torch.cat(batch_pos_t, dim=0).cpu().detach().numpy()
    pos_t = torch.from_numpy(pos_t)

    batch_NT = torch.cat([x.unsqueeze(0) for x in batch_NT], dim=0).cpu().detach().numpy()
    batch_NT = torch.from_numpy(batch_NT)
    batch_PT = torch.cat([x.unsqueeze(0) for x in batch_PT], dim=0).cpu().detach().numpy()
    batch_PT = torch.from_numpy(batch_PT)

    ns_len = [x.shape[0] for x in batch_NS]
    nsample = int(len(ns_len) / len(batch))
    ps_len = [x.shape[0] for x in batch_PS]
    nsample_pos = int(len(ps_len) / len(batch))

    batch_NS_pad = pad_sequence(batch_NS, batch_first=True, padding_value=0.)
    batch_PS_pad = pad_sequence(batch_PS, batch_first=True, padding_value=0.)
    # batch_NS_pad = batch_NS_pad.permute(1,2,0,3)

    # print(pos_s_pad.shape, pos_t_pad.shape, pos_s_len, pos_t_len, NS, NT)
    return {
        "batch_size": len(batch),
        "nsample": nsample,
        "nsample_pos": nsample_pos,
        "pos_s_pad": pos_s_pad, 
        "pos_t_pad": pos_t, 
        "batch_NS_pad": batch_NS_pad,
        "batch_PS_pad": batch_PS_pad,
        "batch_NT": batch_NT,
        "batch_PT": batch_PT, 
        "pos_s_len": pos_s_len, 
        "pos_t_len": pos_t_len, 
        "batch_NS_len": ns_len,
        "batch_PS_len": ps_len,
        "batch_NT_len": batch_NT_len,
        "batch_PT_len": batch_PT_len,
    }

class CARE_dataset(Dataset):

    def __init__(self, segs_dir, matchings = 'matched_dict_sim_partition.pkl', qns_file = './../care_india/questions/questions_00-02_hindi.json', qns_emb_file = './../care_india/questions/questions_00-02_hindi_labse_embeddings.json', train=None, transform=None, k=0, nsample = 10):
        self.segs_dir = segs_dir
        self.transform = transform
        self.train = train
        self.k = k
        self.nsample = nsample

        chunks_list = []
        path_idx = {}
        tmp_name_path_idx = {}
        matched_list = []

        #for each chunk store -> {path_of_chunk: , file: , seg_index: , chunk_index: , chunk_emb: }
        #for each matched chunk store -> {idx in prev list: , Qn_number: , Qn_text: , Qn_emb: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp)

        for segs in seg_audio_files:
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)

            for ind, chunk in enumerate(chunk_files):
                # print(chunk)
                path_idx[chunk] = len(chunks_list)
                # tmp_name_path_idx["/home/shubham/"+chunk[5:]] = len(chunks_list)
                feat_file = chunk.replace("vad_chunks", "vad_chunks_feat").replace("wav", "npy")
                chunks_list += [{'path': chunk, 'audio': audio, 'seg_ind': seg_ind, 'chunk_ind': ind, 'chunk_feat': np.load(feat_file)}]

            matched_chunks = pickle.load(open(segs + '/matched_dict_sim_partition.pkl', 'rb'))
            qns_dict = pickle.load(open(qns_file, 'rb'))
            qns_emb = pickle.load(open(qns_emb_file, 'rb'))

            for chunk in matched_chunks.keys():
                # print(chunk)
                qn = matched_chunks[chunk][1]
                qn_text = qns_dict[qn]
                qn_emb = qns_emb[qn]

                chunk_path = chunk.replace("text", "chunks").replace("txt", "wav")
                # print(chunk_path)
                # matched_list += [{'chunk_list_idx' : tmp_name_path_idx[chunk_path], 'qn_no': qn, 'qn_text' : qn_text, 'qn_emb': qn_emb}]
                matched_list += [{'chunk_list_idx' : path_idx[chunk_path], 'qn_no': qn, 'qn_text' : qn_text, 'qn_emb': qn_emb}]

        self.chunks_list = chunks_list
        self.matched_list = matched_list
        self.path_idx = path_idx
        self.qns_emb = qns_emb
            
    def __len__(self):
        return len(self.matched_list)

    def __getitem__(self, idx):  # Negatives are matched samples - (s_i, t_i) - outside [-2k, 2k]
        k = self.k
        matched_list = self.matched_list
        chunks_list = self.chunks_list

        curr_chunk = matched_list[idx]
        chunk = chunks_list[curr_chunk['chunk_list_idx']]

        pos_s = chunk['chunk_feat']
        pos_t = curr_chunk['qn_emb']

        more_pos_s_right = []
        more_pos_s_left = []

        more_pos_t_left = []
        more_pos_t_right = []

        neg_s_right = []
        neg_s_left = []

        neg_t_left = []
        neg_t_right = []

        audio = chunk['audio']

        offset = 1
        # print("hello")
        while idx+offset < len(matched_list) and chunks_list[matched_list[idx+offset]['chunk_list_idx']]['audio'] == audio:
            if offset <= k:
                #add it to more_pos_s_right, more_pos_t_right
                more_pos_s_right += [chunks_list[matched_list[idx+offset]['chunk_list_idx']]['chunk_feat']]
                more_pos_t_right += [matched_list[idx+offset]['qn_emb']]
            if offset > 2*k:
                #add it to neg_s_right, neg_t_right
                neg_s_right += [chunks_list[matched_list[idx+offset]['chunk_list_idx']]['chunk_feat']]
                neg_t_right += [matched_list[idx+offset]['qn_emb']]

            offset += 1

        offset = 1
        while idx-offset >= 0 and chunks_list[matched_list[idx-offset]['chunk_list_idx']]['audio'] == audio:
            if offset <= k:
                #add it to more_pos_s_left, more_pos_t_left
                more_pos_s_left = [chunks_list[matched_list[idx-offset]['chunk_list_idx']]['chunk_feat']] + more_pos_s_left
                more_pos_t_left = [matched_list[idx-offset]['qn_emb']] + more_pos_t_left
            if offset > 2*k:
                #add it to neg_s_left, neg_t_left
                neg_s_left = [chunks_list[matched_list[idx-offset]['chunk_list_idx']]['chunk_feat']] + neg_s_left
                neg_t_left = [matched_list[idx-offset]['qn_emb']] + neg_t_left

            offset += 1

        more_pos_s = more_pos_s_left + more_pos_s_right
        more_pos_t = more_pos_t_left + more_pos_t_right

        
        pos_inds = random.sample(range(len(more_pos_s)), self.k)
        pos_s_sampled = [more_pos_s[i] for i in pos_inds]
        pos_t_sampled = [more_pos_t[i] for i in pos_inds]

        # Negatives s and t need not be matched

        # print(len(neg_s_left), len(neg_s_right))

        neg_s = neg_s_left + neg_s_right
        neg_t = neg_t_left + neg_t_right

        inds = random.sample(range(len(neg_s)), min(self.nsample, len(neg_s)))
        neg_s_sampled = [neg_s[i] for i in inds]
        neg_t_sampled = [neg_t[i] for i in inds]
        # print("pos_s", pos_s.shape)
        # print("pos_t", pos_t.shape)
        sample = { 'pos_s': pos_s, 'pos_t': pos_t, 'NS': neg_s_sampled, 'NT': neg_t_sampled, 'PS' : pos_s_sampled, 'PT' : pos_t_sampled }
                    # 'neg_s_left': np.array(neg_s_left), 'neg_t_left': np.array(neg_t_left),
                    # 'neg_s_right': np.array(neg_s_right), 'neg_t_right': np.array(neg_t_right),
                    # 'more_pos_s_left': np.array(more_pos_s_left), #'more_pos_t_left': np.array(more_pos_t_left),
                    # 'more_pos_s_right': np.array(more_pos_s_right)#, 'more_pos_t_right': np.array(more_pos_t_right)
                #}

        if self.transform: sample = self.transform(sample)

        return sample

class Test_dataset(Dataset):
    def __init__(self, segs_dir, file_name, transform = None):
        self.segs_dir = segs_dir
        segments_list = []
        self.transform = transform
        
        # for each segment -> {file: , seg_index: , list_of_chunk_embs: }

        temp = [join(segs_dir, f[:-4]) for f in listdir(segs_dir) if '.wav' in f]
        seg_audio_files = sorted(temp)

        for segs in seg_audio_files:
            l = segs.split('/')
            l1 = l[-1].split('___')

            audio = l1[0]
            seg_ind = l1[-1]

            if audio != file_name: continue

            temp = [(segs + '/vad_chunks/' + f) for f in listdir(segs + '/vad_chunks/') if '.wav' in f]
            chunk_files = sorted(temp)

            chunk_emb_list = []

            for chunk in chunk_files:
                feat_file = chunk.replace("vad_chunks", "vad_chunks_feat").replace("wav", "npy")
                chunk_emb_list += [np.load(feat_file)]

            segments_list += [{'file': file_name, 'segment_id': seg_ind, 'chunks': chunk_emb_list}]

        self.segments_list = segments_list

    def __len__(self):
        return len(self.segments_list)


    def __getitem__(self, idx):
        if self.transform: return self.transform(self.segments_list[idx])
        return self.segments_list[idx]

class Question_dataset(Dataset):
    def __init__(self, file_name, test_file='./../care_india/audio_features_dict_hindi_testdata_5ques.json', qns_file = './../care_india/questions/questions_00-02_hindi.json', qns_emb_file = './../care_india/questions/questions_00-02_hindi_labse_embeddings.json', transform=None):
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

class ToTensor(object):

    def __call__(self, sample):
        pos_s = sample['pos_s']
        pos_t = sample['pos_t']
        # neg_s_left = sample['neg_s_left']
        # neg_t_left = sample['neg_t_left']
        # neg_s_right = sample['neg_s_right']
        # neg_t_right = sample['neg_t_right']
        # more_pos_s_left = sample['more_pos_s_left']
        # more_pos_s_right = sample['more_pos_s_right']
        neg_s = sample['NS']
        neg_t = sample['NT']
        more_pos_s = sample['PS']
        more_pos_t = sample['PT']

        return {
                    'pos_s': torch.from_numpy(pos_s), 'pos_t': pos_t, 
                    # 'neg_s_left': torch.from_numpy(neg_s_left), 'neg_t_left': torch.from_numpy(neg_t_left),
                    # 'neg_s_right': torch.from_numpy(neg_s_right), 'neg_t_right': torch.from_numpy(neg_t_right),
                    # 'more_pos_s_left': torch.from_numpy(more_pos_s_left), #'more_pos_t_left': np.array(more_pos_t_left),
                    # 'more_pos_s_right': torch.from_numpy(more_pos_s_right), #, 'more_pos_t_right': np.array(more_pos_t_right)
                    'NS': [torch.from_numpy(x) for x in neg_s], 'NT': neg_t, 
                    'PS': [torch.from_numpy(x) for x in more_pos_s], 'PT': more_pos_t
                    # 'NS': [neg], 'NT': []
                }

class ToTensorChunk(object):

    def __call__(self, sample):

        return { 'file': sample['file'], 'segment_id': sample['segment_id'], 'chunks': [torch.from_numpy(x) for x in sample['chunks']] }

class ToTensorQuestion(object):

    def __call__(self, sample):
        return sample


if __name__ == "__main__":
    question_dataset = Question_dataset(file_name='AA_0a7eaac7-d608-42a7-bb38-f621ddcf797e_AFTER_0S', test_file='./../care_india/audio_features_dict_traindata_5ques.json', transform = transforms.Compose([ToTensorQuestion()]))
    print(question_dataset.__len__())
    x = question_dataset.__getitem__(2)
    print(question_dataset.qns_list)
    print(x['Question'].shape)