def calculate_IoU(start_time_est, end_time_est, start_time_GT, end_time_GT):
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
    seg_to_Q = None, qns_dict  = None, K = 1,
    sliding_window_size = 12,
    ):
    '''
    each sample with contain fixed #chunks
    B = 1
    segment --> window_size x D

    '''
    model.eval()
    with torch.no_grad():
        total_IoU = 0.
        for q_idx, q_sample in enumerate(question_loader):
            Q_emb, Q_id_GT = q_sample['Question'].to(DEVICE), q_sample['Q_id']

            Q_start_time_GT, Q_end_time_GT = q_sample['start_time'], q_sample['end_time']

            for batch_idx, sample in enumerate(chunk_loader):
                # 1 x window_size x D
                segment = sample['chunks'].to(DEVICE)

                # window_size x num_Q
                log_prob = model(segment, Q_emb, None, None)

                pred_Q_ids = torch.argmax(log_prob, dim = -1) + 1 # IDs start from 1

                for i, Q_id_pred in enumerate(pred_Q_ids):
                    if Q_id_pred == Q_id_GT:
                        Q_start_time_pred = sample['vad_time'][i]['start_time']
                        Q_end_time_pred = sample['vad_time'][i]['end_time']

                        total_IoU += calculate_IoU(
                            Q_start_time_pred, 
                            Q_end_time_pred, 
                            Q_start_time_GT, 
                            Q_end_time_GT
                        )
            if (show_plots):
                top_5_text = []
                Q_text =  qns_dict[(sample['qn_no'])[0]]
                for seg_id in all_scores[:5]:
                    tmp_Q_text = qns_dict[seg_to_Q[seg_id]['questions'][0]]
                    top_5_text += [tmp_Q_text]
                print(Q_text, top_5_text)
                
                plot_chunks(
                    all_chunk_scores, 
                    all_seg_ids, 
                    GT_seg_id, 
                    Q_text,
                    top_5_text,
                    save_path = os.path.join(plot_save_path, file_name+f"GT-seg{GT_seg_id}-{batch_idx}.png")
                    
                )

    return total_IoU / len(question_loader)

