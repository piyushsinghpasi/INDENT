import torch
import torch.nn.functional as F
import numpy as np
from utils.plot_graphs import plot_chunks
from utils.featViz import display_tsne_scatterplot_3D
import os

# prev_all_scores = None

def calculate_mIoU(start_time_GT, end_time_GT, start_time_est, end_time_est):
    Intersection = (min(end_time_est, end_time_GT) - max(start_time_est, start_time_GT))
    Union = (max(end_time_est, end_time_GT) - min(start_time_est, start_time_GT))

    if Union == 0 or Intersection < 0 \
        or end_time_GT < start_time_est \
        or end_time_est < start_time_GT \
    :
        IoU = 0
    else:
        IoU = Intersection / Union
    return IoU

def eval(
    model, question_loader, chunk_loader, 
    result_logging_file_path = None, checkpoint="", hid_dim = 768, 
    show_plots = False, plot_save_path = './plots/', file_name = "",
    seg_to_Q = None, qns_dict  = None, K = 1, curr_device = "cpu",
    ):
    '''
    Eval done separately on each File (hence file_name is a parameter)
    Order of segment might not be temporally correct (though we have sorted)
    
    Hence, chunk_loader should load all segments  in the audio file
    and question_loader should load all question asked in the audio file
    '''
    # global prev_all_scores
    model.eval()
    seg_chunk_feat = dict()
    R_1, R_5, R_10 = 0., 0., 0.
    
    with torch.no_grad():
        dummy_text = torch.zeros((1, 5, hid_dim)).to(curr_device)
        for batch_idx, sample in enumerate(chunk_loader):
            chunks_feat, speech_padding_mask, segment_len, max_pad = sample['batch_chunks'].to(curr_device), sample['speech_padding_mask'].to(curr_device), sample['segment_len'].to(curr_device), sample['max_pad']
            speech_attn_mask = sample['speech_attn_mask'].to(curr_device)
            chunk_timestep_len = sample['chunk_timestep_len'].to(curr_device)
            max_pad = sample["max_pad"]
            self_attd_chunk, _ = model(chunks_feat, dummy_text, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask)
            seg_chunk_feat[sample['segment_id'][0]] = self_attd_chunk.squeeze()
            # print(self_attd_chunk)
                    
        # display_tsne_scatterplot_3D(seg_chunk_feat, [], None, None, None, 5, 500, 10000)
        
                
        for batch_idx, sample in enumerate(question_loader):

            segment_scores = dict()
            Question_feat, GT_seg_id = sample['Question'].to(curr_device), (sample['segment_id'])[0]

            Q_feat = model(None, Question_feat, None, None, None, None, None)
            Q_feat = Q_feat.squeeze()
            
            all_chunk_scores = []
            all_seg_ids = []
            for seg_id, chunk_feat in seg_chunk_feat.items():
                chunk_scores = (torch.matmul(chunk_feat, Q_feat))
                # print(chunk_scores)
                if (len(chunk_scores.size()) == 0): 
                    chunk_scores = chunk_scores.unsqueeze(0)
                all_chunk_scores += [chunk_scores.detach().cpu().numpy()]
                all_seg_ids += [seg_id]
                
                segment_scores[seg_id] = torch.topk(chunk_scores, min(K, chunk_scores.shape[-1]), dim=-1).values.mean(-1).item()

            all_scores = sorted(segment_scores, key=segment_scores.get, reverse=True)
            
            # Q_feat = Q_feat.unsqueeze(0)
            # Question_feat = Question_feat.squeeze().unsqueeze(0)
            # print(np.around(np.array(sorted(segment_scores.values(), reverse=True)[:10]), 2)) #, Q_feat@Q_feat.T, Question_feat@Question_feat.T)
            # break
            
            all_seg_ids_all_chunk_scores = sorted(zip(all_seg_ids, all_chunk_scores), key=lambda x: int(x[0]))
            all_seg_ids = [x for x,_ in all_seg_ids_all_chunk_scores]
            all_chunk_scores = [y for _,y in all_seg_ids_all_chunk_scores]
            # print(all_chunk_scores[0])

            prev_all_scores = all_scores
            qn_no = (sample['qn_no'])[0]
            print(qn_no)
            if (show_plots and all_chunk_scores):
                top_5_text = []
                Q_text = ""
                Q_text =  qns_dict[qn_no]
                print(Q_text)
                # for seg_id in all_scores[:5]:
                #     seg_id = str(seg_id)
                #     tmp_Q_text = qns_dict[seg_to_Q[seg_id]['questions']]
                #     top_5_text += [tmp_Q_text]
                
                plot_chunks(
                    all_chunk_scores, 
                    all_seg_ids, 
                    GT_seg_id, 
                    Q_text,
                    top_5_text,
                    0.,
                    save_path = os.path.join(plot_save_path, file_name+f"GT-seg{GT_seg_id}-{qn_no}.png")
                    
                )

            if GT_seg_id in all_scores[:1]:
                R_1 += 1
            if GT_seg_id in all_scores[:5]:
                R_5 += 1
            if GT_seg_id in all_scores[:10]:
                R_10 += 1
            

    R_1 = R_1/len(question_loader)
    R_5 = R_5/len(question_loader)
    R_10 = R_10/len(question_loader)
    R_avg = (R_1 + R_5 + R_10)/3.0

    # print(len(question_loader))
    # print(R_1*100, R_5*100, R_10*100, R_avg*100, len(question_loader))
    return R_1*100, R_5*100, R_10*100, R_avg*100
