import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from dataloader_pos import *
from nets.network import Feat_Merger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys

import pandas as pd
import pickle as pkl

DEVICE = 'cuda'

class customContrastiveLoss(nn.Module):
    def __init__(self, temperature = 1.0):
        super(customContrastiveLoss, self).__init__()
        self.temperature = temperature 

    def forward(self, s, t, NS, NT, PS, PT):
        '''
        s.shape == t.shape -> Bxd

        k is for attention window

        [-k,+k] [-2k,2k]

        i, pos_s, pos_t, neg
        '''
        # FIX LOSS
        # print("loss shape", s.shape, t.shape, NS.shape, NT.shape)
        sim = torch.sum( s*t, dim=-1)
        numerator = torch.exp( torch.sum( s*t, dim=-1) / self.temperature)
        
        # denominator_N_set = torch.diag(torch.exp(torch.sum(N_set.unsqueeze(dim=2) * s, dim=-1) / self.temperature ).sum(dim=1))
        denominator_NS = torch.exp((NS * t.unsqueeze(dim=1)).sum(dim=-1) / self.temperature ).sum(dim=-1)
        denominator_NT = torch.exp((NT * s.unsqueeze(dim=1)).sum(dim=-1) / self.temperature ).sum(dim=-1)

        numerator_PS = torch.exp((PS * t.unsqueeze(dim=1)).sum(dim=-1) / self.temperature ).sum(dim=-1)
        numerator_PT = torch.exp((PT * s.unsqueeze(dim=1)).sum(dim=-1) / self.temperature ).sum(dim=-1)

        alpha = 5
        beta = 1

        num_t_s = numerator*alpha + numerator_PS*beta
        num_s_t = numerator*alpha + numerator_PT*beta

        # print("N, D", numerator.shape, denominator_NS.shape, torch.log(numerator/denominator_NS))
        # log_exp = (torch.log(numerator/denominator_N_set)).sum(dim=0) / s.shape[0]

        # return -log_exp
        # print('Num : ', numerator)
        # print('Den_NS : ', denominator_NS)
        # print('Den_NT : ', denominator_NT)
        log_t_s = torch.mean(torch.log(num_t_s/denominator_NS), dim=0)
        log_s_t = torch.mean(torch.log(num_s_t/denominator_NT), dim=0)

        
        similarity_loss = torch.mean((1 - sim), dim = 0)
        # print(-log_s_t, -log_t_s, similarity_loss)
        return  -log_s_t, -log_t_s, similarity_loss


def train(args, model, train_loader, optimizer, criterion, epoch, log_file=None, hid_dim=768):
    model.train()
    # exit(0)
    # batch_s_feat = torch.empty((0, s_feat.shape[-1])).to(DEVICE)
    for batch_idx, sample in enumerate(train_loader):

        # print(pos_s_pad.shape, pos_t.shape)
        optimizer.zero_grad()

        pos_s_pad = sample["pos_s_pad"].to(DEVICE)
        pos_t = sample["pos_t_pad"].to(DEVICE)
        batch_NT = sample["batch_NT"].to(DEVICE)
        batch_PT = sample["batch_PT"].to(DEVICE)
        
        batch_NS_pad = sample["batch_NS_pad"].to(DEVICE)
        batch_NS_len = sample["batch_NS_len"]

        batch_PS_pad = sample["batch_PS_pad"].to(DEVICE)
        batch_PS_len = sample["batch_PS_len"]

        # print(pos_s_pad.shape)
        # exit(0)
        batch_size, nsample, hdim, nsample_pos, hdim_pos = sample["batch_size"], sample["nsample"], batch_NS_pad.shape[-1], sample["nsample_pos"], batch_PS_pad.shape[-1]
        # reshaped_batch_NS_pad = p.view(batch*nsample, max_frame, hdim)
        # reshaped_batch_NS_len = [l for l in batch_NS_len for _ in range(nsample)]
        # batched_pack_NS = pack_padded_sequence(reshaped_batch_NS_pad, reshaped_batch_NS_len, batch_first=True, enforce_sorted=False)

        batched_pack_NS = pack_padded_sequence(batch_NS_pad, batch_NS_len, batch_first=True, enforce_sorted=False)
        batched_pack_PS = pack_padded_sequence(batch_PS_pad, batch_PS_len, batch_first=True, enforce_sorted=False)

        batch_pack_NS_feat, _ = model(speech = batched_pack_NS, text = None)
        _ , batch_pack_NT_feat = model(speech = None, text = batch_NT)

        batch_pack_PS_feat, _ = model(speech = batched_pack_PS, text = None)
        _ , batch_pack_PT_feat = model(speech = None, text = batch_PT)

        # reshaped_batch_unpack_NS_feat, lens_unpacked = pad_packed_sequence(batch_pack_NS_feat, batch_first=True)
        # batch_unpack_NS_feat, lens_unpacked = pad_packed_sequence(batch_pack_NS_feat, batch_first=True)
        batch_unpack_NS_feat = batch_pack_NS_feat.view(batch_size, nsample, hdim)
        batch_unpack_PS_feat = batch_pack_PS_feat.view(batch_size, nsample_pos, hdim_pos)
         
        pos_s_len = sample["pos_s_len"]
        # pos_t_len = sample["pos_t_len"]
        # batch_NT_len = sample["batch_NT_len"]

        pack_s = pack_padded_sequence(pos_s_pad, pos_s_len, batch_first=True, enforce_sorted=False)

        s_feat, t_feat = model(speech = pack_s, text = pos_t)

        # if 
        # new_batch_NS = []
        # for ns_idx, NS in enumerate(batch_NS):
        #     NS.requires_grad = True
        #     NS = NS.to(DEVICE)
        #     pack_NS = pack_padded_sequence(NS, batch_NS_len[ns_idx], batch_first=True, enforce_sorted=False)
        #     NS_feat, _ = model(speech = pack_NS, text = None)
        #     new_batch_NS += [NS_feat.unsqueeze(0)]


        L_speech_to_text, L_text_to_speech, similarity_loss = criterion(s_feat, t_feat, batch_unpack_NS_feat, batch_pack_NT_feat, batch_unpack_PS_feat, batch_pack_PT_feat)
        
        alpha = 1.
        beta = 1.
        gamma = 0.
        loss = alpha*L_speech_to_text + beta*L_text_to_speech + gamma*similarity_loss

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tL_speech_to_text:{:.3f}\tL_text_to_speech:{:.3}\tsimilarity_loss:{:.3f}\tLoss:{:.3f}'.format(
                epoch, batch_idx*s_feat.shape[0], len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), L_speech_to_text.item(), L_text_to_speech.item(), similarity_loss.item(), loss.item())
            print(out_loss)
    if log_file is not None:
        with open(log_file,'a') as f:
            f.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(L_speech_to_text.item(), L_text_to_speech.item(), similarity_loss.item(), loss.item()))



def eval(
    model, question_loader, chunk_loader, 
    result_logging_file_path = None, checkpoint="", hid_dim = 768, 
    show_plots = False, plot_save_path = './plots/', file_name = "",
    seg_to_Q = None, qns_dict  = None, K = 1,
    ):
    '''
    Src sep + Vad on test file segment
    Then get emb of each chunk and Q. from trained model
    
    1. we know segment Bound.
    2. Fixed segment
    3. Segment length Predictor
    4. Chunk retreival --> mIoU from test ann.
    
    Get sim. score and try avg, max.
    '''
    model.eval()
    # print("\nEvaluating...\n")
    # for all segment, for all_chunks
    # list of segment with chunk tensors, raggedtensor not supported in pytorch
    seg_chunk_feat = dict()
    num_chunk_per_file = None
    R_1, R_5, R_10 = 0., 0., 0.
    with torch.no_grad():
        for batch_idx, sample in enumerate(chunk_loader):
            
            chunks, seg_id = sample['chunks'], sample['segment_id'][0]
            if not chunks: continue
            chunks = [c.squeeze() for c in chunks]
            padded_chunks, len_padded_chunks = pad_N_set(chunks)
            padded_chunks = padded_chunks.to(DEVICE)
            batched_pack_NS = pack_padded_sequence(padded_chunks, len_padded_chunks, batch_first=True, enforce_sorted=False)
            chunk_feat, _ = model(batched_pack_NS, None)

            # chunk_feat = torch.empty((0, hid_dim)).to(DEVICE)
            # for _, chunk_i in enumerate(chunks):
            #     chunk_ix = chunk_i.to(DEVICE)
            #     x, _ = model(chunk_ix, None)
            #     x = x.squeeze().unsqueeze(dim=0)
            #     chunk_feat = torch.cat((chunk_feat, x), dim=0)

            seg_chunk_feat[seg_id] = chunk_feat.squeeze()

        
        
        for batch_idx, sample in enumerate(question_loader):
            segment_scores = dict()
            Question_feat, GT_seg_id = sample['Question'].to(DEVICE), (sample['segment_id'])[0]

            _, Q_feat = model(None, Question_feat)
            Q_feat = Q_feat.squeeze()
            
            all_chunk_scores = []
            all_seg_ids = []
            for seg_id, chunk_feat in seg_chunk_feat.items():

                chunk_scores = torch.matmul(seg_chunk_feat[seg_id], Q_feat)
                all_chunk_scores += [chunk_scores.cpu().detach().numpy()]
                all_seg_ids += [seg_id]
                # if (chunk_scores.shape[0] > 0):
                # we can use maxFeat_Merger
                if (len(chunk_scores.size()) == 0): 
                    segment_scores[seg_id] = chunk_scores.item()
                else :
                    segment_scores[seg_id] = torch.topk(chunk_scores, min(K, chunk_scores.shape[-1]), dim=-1).values.mean(-1)
                # segment_scores[seg_id] = chunk_scores.max(dim=-1)

            all_scores = sorted(segment_scores, key=segment_scores.get, reverse=True)
            
            
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
                # print("CHECK 25 ",batch_idx, all_chunk_scores[all_seg_ids.index(GT_seg_id)])
            # print(all_scores)
            # print("eexitsts", GT_seg_id in all_scores, GT_seg_id[0], GT_seg_id)
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


    return R_1*100, R_5*100, R_10*100, R_avg*100


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Aligning Questions with Long Audio Survey Interviews')
    parser.add_argument(
        "--train_segs_dir", type=str, default='', help="train audio dir")
    parser.add_argument(
        "--test_segs_dir", type=str, default='',
        help="test audio dir")
    parser.add_argument(
        "--val_segs_dir", type=str, default='',
        help="val audio dir")
    parser.add_argument(
        "--matchings", type=str, default="matched_dict_sim_partition.pkl", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='MMIL_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument(
        '--K', type=int, default=0, help='sets range [-2K, 2K] to be excluded for sampling negatives')
    parser.add_argument(
        '--nsample', type=int, default=10, help='sets range [-2K, 2K] to be excluded for sampling negatives')
    parser.add_argument(
        '--show_plots', action="store_true", help='saves plots Q. vs chunk scores in plots dir')
    parser.add_argument(
        '--result_logging_file_path', type=str, default="./results/", help='saves each epoch eval metrics'
    )
    parser.add_argument(
        '--topK', type=int, default=1, help='take avg of top K chunks scores in a segment for segment-level score'
    )
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.manual_seed(args.seed)

    model = Feat_Merger().to(DEVICE)

    if args.mode == 'train':
        train_dataset = CARE_dataset(
            segs_dir = args.train_segs_dir, 
            matchings = args.matchings, 
            qns_file = './../care_india/questions/questions_00-02_hindi.json', 
            k=args.K, 
            nsample = args.nsample,
            transform = transforms.Compose([ToTensor()])
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # tune temperature
        criterion = customContrastiveLoss(temperature = 1.0)

        val_files = set()
        for f in os.listdir(args.val_segs_dir) :
            if os.path.isdir(os.path.join(args.val_segs_dir,f)):
                filename = f.split("___")[0]
                val_files.add(filename)

        train_files = set()
        for f in os.listdir(args.train_segs_dir) :
            if os.path.isdir(os.path.join(args.train_segs_dir,f)):
                filename = f.split("___")[0]
                train_files.add(filename)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=12, 
            pin_memory = True,
            collate_fn=pad_collate,
        )

        best_R = 0
        with open(args.result_logging_file_path+args.checkpoint+'_val.txt', 'w') as result_file:
            result_file.write("R@1\tR@5\tR@10\tR-avg\n")
        with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'w') as result_file:
            result_file.write("R@1\tR@5\tR@10\tR-avg\n")
        with open(args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint),'w') as logfile:
            logfile.write("L_speech_to_text\tL_text_to_speech\tsimilarity_loss\tLoss\n")
        
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, log_file=args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint))
            scheduler.step()

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
            for file_name in train_files:
                chunk_dataset = Test_dataset(
                    segs_dir = args.train_segs_dir, 
                    file_name=file_name, 
                    transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset(
                    file_name=file_name, 
                    test_file='./../care_india/audio_features_dict_traindata_5ques.json', 
                    transform = transforms.Compose([ToTensorQuestion()])
                )
                chunk_loader = DataLoader(
                    chunk_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=1, 
                    pin_memory = True
                )
                question_loader = DataLoader(
                    question_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=1, 
                    pin_memory = True
                )
                a, b, c, d = eval(
                    model, 
                    question_loader, 
                    chunk_loader, 
                    args.result_logging_file_path, 
                    args.checkpoint+"_train",
                    K = args.topK,
                )
                tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d

            train_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( tot_R_1/len(train_files), tot_R_5/len(train_files), tot_R_10/len(train_files), tot_R_avg/len(train_files))
            print("Training metrics: ",train_metrics)
            
            with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'a') as result_file:
                result_file.write(train_metrics)

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
            for file_name in val_files:
                chunk_dataset = Test_dataset(
                    segs_dir = args.val_segs_dir, 
                    file_name=file_name, 
                    transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset(
                    file_name=file_name, 
                    transform = transforms.Compose([ToTensorQuestion()])
                )
                chunk_loader = DataLoader(
                    chunk_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=1, 
                    pin_memory = True
                )
                question_loader = DataLoader(
                    question_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=1, 
                    pin_memory = True
                )


                a, b, c, d = eval(
                    model, 
                    question_loader, 
                    chunk_loader, 
                    args.result_logging_file_path, 
                    args.checkpoint+"_val",
                    K = args.topK,
                )

                tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d
            
            print("val train file lengths",len(val_files), len(train_files))
            val_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( 
                tot_R_1/len(val_files), 
                tot_R_5/len(val_files), 
                tot_R_10/len(val_files), 
                tot_R_avg/len(val_files)
            )
            print("Validation metrics: ",val_metrics)
            
            with open(args.result_logging_file_path+args.checkpoint+'_val.txt', 'a') as result_file:
                result_file.write(val_metrics)
                
            avg_tot_R = tot_R_avg/len(val_files)
            if (avg_tot_R) >= best_R:
                best_R = avg_tot_R
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
        print("Done {}".format(args.checkpoint))

    elif args.mode == 'val':
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        val_files = set()
        for f in os.listdir(args.val_segs_dir) :
            if os.path.isdir(os.path.join(args.val_segs_dir,f)):
                filename = f.split("___")[0]
                val_files.add(filename)

        tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
        for file_name in val_files:
            chunk_dataset = Test_dataset(
                segs_dir = args.val_segs_dir, 
                file_name=file_name, 
                transform = transforms.Compose([ToTensorChunk()])
            )
            question_dataset = Question_dataset(
                file_name=file_name, 
                transform = transforms.Compose([ToTensorQuestion()])
            )
            chunk_loader = DataLoader(
                chunk_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                pin_memory = True
            )
            question_loader = DataLoader(
                question_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                pin_memory = True
            )
            a, b, c, d = eval(
                model, 
                question_loader, 
                chunk_loader, 
                args.result_logging_file_path, 
                args.checkpoint+"_val",
                K = args.topK,
            )

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d

        print("val train file lengths",len(val_files), len(train_files))
        val_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( 
            tot_R_1/len(val_files), 
            tot_R_5/len(val_files), 
            tot_R_10/len(val_files), 
            tot_R_avg/len(val_files)
        )
        print("Validation metrics: ",val_metrics)

    else:
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        test_files = set()
        for f in os.listdir(args.test_segs_dir) :
            if os.path.isdir(os.path.join(args.test_segs_dir,f)):
                filename = f.split("___")[0]
                test_files.add(filename)

        tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
        for file_name in test_files:
            chunk_dataset = Test_dataset(
                segs_dir = args.test_segs_dir, 
                file_name=file_name, 
                transform = transforms.Compose([ToTensorChunk()])
            )
            question_dataset = Question_dataset(
                file_name=file_name, 
                # test_file='./../care_india/audio_features_dict_traindata_5ques.json',
                transform = transforms.Compose([ToTensorQuestion()])
            )
            chunk_loader = DataLoader(
                chunk_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                pin_memory = True
            )
            question_loader = DataLoader(
                question_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                pin_memory = True
            )
            seg_to_Q = question_dataset.curr_segs
            qns_dict = question_dataset.qns_dict
            a, b, c, d = eval(
                model,
                question_loader, 
                chunk_loader, 
                args.result_logging_file_path, 
                args.checkpoint, 
                show_plots=args.show_plots, 
                file_name = file_name,
                seg_to_Q = seg_to_Q,
                qns_dict = qns_dict,
                K = args.topK,
            )
            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d

        test_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( 
            tot_R_1/len(test_files), 
            tot_R_5/len(test_files), 
            tot_R_10/len(test_files), 
            tot_R_avg/len(test_files)
        )
        print("Test metrics: ")
        print(test_metrics)


if __name__ == '__main__':
# python main_sim.py --train_segs_dir ./../care_india/annotations/audio_hindi_test_segments --val_segs_dir ./../care_india/annotations/audio_hindi_test_segments --test_segs_dir ./../shubham/care_india/annotations/audio_hindi_test_segments --mode train --checkpoint model1 --gpu 2 --batch-size 4
    main()
    