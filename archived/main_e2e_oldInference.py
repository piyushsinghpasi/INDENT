import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from dataloader_e2e import *
from nets.net_e2e import Feat_Merger
from utils.recall_eval import eval
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# torch.multiprocessing.set_start_method('spawn')

import sys

import pandas as pd
import pickle as pkl

DEVICE = 'cuda'
class contrastAcrossSegments(nn.Module):
    def __init__(self, temperature = 1.0):
        super().__init__()
        self.temperature = temperature 

    def forward(self, self_attd_chunk, cross_attd_chunk, Q_emb, audio_ID, speech_padding_mask):
        '''
        self_attd_chunk B x max_chunk x D
        Think of cont across segment (but same file or is same file even a concern??)
        audio_ID B*max_chunk
        '''
        _min, _max = -1.0, 1.0

        B, max_chunk, D = self_attd_chunk.size()
        eps = 1e-5

        # multiply (audio_ID * audio_ID.unsqueeze(1)) 
        # to remove values from segment of another audio file
        # B*max_chunk, D
        self_attd_chunk_unfolded = self_attd_chunk.reshape(B*max_chunk, D)
        cross_attd_chunk_unfolded = cross_attd_chunk.reshape(B*max_chunk, D)

        # dot = self_attd_chunk_unfolded[0] * Q_emb[0][0]
        # # print("pos: ", dot.sum(dim=-1))

        # audio_id_check = audio_ID[0]
        # # print("audio_id: ", audio_id_check)

        # for i in range(max_chunk, B*max_chunk):
        #     if audio_ID[i] == audio_id_check:
        #         dot1 = self_attd_chunk_unfolded[i] * Q_emb[0][0]
        #         # print("neg: ", dot1.sum(dim=-1))
        #         break

        # sim_all B*max_chunk x B*max_chunk
        # sim_segment B x max_chunk
        sim = torch.exp((self_attd_chunk @ cross_attd_chunk.permute(0, 2, 1)) / self.temperature)
        # print("SIM", torch.isnan(sim).any())
        # sim = torch.exp( torch.clip((self_attd_chunk @ self_attd_chunk.permute(0, 2, 1)), min = _min, max = _max) / self.temperature)

        sim_segment = (sim).sum(dim=-1)
        sim_all = (self_attd_chunk_unfolded @ cross_attd_chunk_unfolded.T) / self.temperature
        # print("SIMALL", torch.isnan(sim_all).any())
        # print(sim_all)
        # sim_all = torch.clip((self_attd_chunk_unfolded @ self_attd_chunk_unfolded.T), min = _min, max = _max) / self.temperature
        in_audio_mask = (audio_ID == audio_ID.unsqueeze(1))
        not_in_audio_mask = ~in_audio_mask
        # print("sim", sim_all)
        # print("in_audio_mask sum", in_audio_mask.sum(-1).flatten())
        '''
        for some reason this gives NAN
        sim_all_same_audio_negs = (torch.exp(sim_all )) * in_audio_mask
        '''
        sim_all_same_audio_negs = (torch.exp(sim_all)).masked_fill(not_in_audio_mask, 0.)

        denominator = sim_all_same_audio_negs.sum(dim=-1).view(B, max_chunk) - sim_segment
        denominator[denominator <= eps] = 0.
        denominator = denominator.masked_fill(speech_padding_mask, 0.)
        # print("DEN", torch.isnan(denominator).any())
        
        # check_negs = in_audio_mask.sum(-1).view(B, max_chunk)
        # for i, (neg_count, den, pad) in enumerate(zip(check_negs, denominator, speech_padding_mask)):
        #     print(i,"\n", neg_count, "\n", den, "\n", pad)
        #     if (i >= 16): print("--------------------------------------------------------------------------------------")
        # denominator = B x max_chunk
        # denominator = sim_segment - (sim * torch.eye(max_chunk).unsqueeze(dim=0).repeat(B, 1, 1)).sum(dim=1)

        # numerator B x max_chunk
        num = torch.sum( self_attd_chunk * cross_attd_chunk, dim=-1) / self.temperature
        numerator = torch.exp(num)
        # numerator = torch.exp( torch.clip(torch.sum( self_attd_chunk * cross_attd_chunk, dim=-1), min = _min, max = _max) / self.temperature)
        # print("NUM", torch.isnan(numerator).any())
        log_exp = torch.mean(torch.log(numerator/(numerator + denominator + eps)), dim=1).mean(dim=0)
        # print("num | ", torch.log(numerator), "den |", torch.log(denominator))

        # 1st and last Q matching known 
        
        sim_loss = (1 - num).mean(dim=-1).mean(dim=-1)
        # text_sim_loss = (2.0 - (Q_emb[0]*self_attd_chunk[0].T) - (Q_emb[-1]*self_attd_chunk[-1].T))/2.0
        return -log_exp, sim_loss

class customContrastiveLoss(nn.Module):
    def __init__(self, temperature = 1.0):
        super(customContrastiveLoss, self).__init__()
        self.temperature = temperature 

    def forward(self, self_attd_chunk, cross_attd_chunk, negs):
        '''
        self_attd_chunk B x max_chunk x D
        Think of cont across segment (but same file or is same file even a concern??)
        '''
        _min, _max = -1.0, 1.0
        B = self_attd_chunk.shape[0]
        max_chunk = self_attd_chunk.shape[1]
        eps = 1e-10

        #  sim B x max_chunk x max_chunk
        sim = self_attd_chunk.bmm(cross_attd_chunk.permute(0, 2, 1)) / self.temperature
        sim = torch.clip(sim, min=_min, max=_max)

        sim = torch.exp(sim)
        # print("sim", sim)
        # denominator = B x max_chunk
        denominator = sim.sum(dim=1) - (sim * torch.eye(max_chunk).unsqueeze(dim=0).repeat(B, 1, 1)).sum(dim=1)
        # print("den", denominator)

        # numerator B x max_chunk
        numerator = torch.exp( torch.clip(torch.sum( self_attd_chunk * cross_attd_chunk, dim=-1), min=_min, max=_max) / self.temperature)
        # print("num", numerator)

        log_exp = torch.mean(torch.log(numerator/(numerator + denominator + eps)), dim=1).mean(dim=0)
        
        # 1st and last Q matching known 
        text_sim_loss = 0.
        # text_sim_loss = (2.0 - (Q_emb[0]*self_attd_chunk[0].T) - (Q_emb[-1]*self_attd_chunk[-1].T))/2.0
        return  -log_exp, text_sim_loss


def train(args, model, train_loader, optimizer, criterion, epoch, log_file=None, hid_dim=768):
    model.train()
    running_loss, running_cont_loss, running_sim_loss = 0., 0., 0.
    for batch_idx, sample in enumerate(train_loader):

        optimizer.zero_grad()

        chunks_feat = sample['batch_chunks'].to(DEVICE)
        negs = sample['negs'].to(DEVICE)
        speech_padding_mask = sample['speech_padding_mask'].to(DEVICE)
        # Q_id_seq = sample['Q_id_seq'].to(DEVICE)
        Q_emb = sample['Q_emb'].to(DEVICE)
        # text_attn_mask = sample['text_attn_mask'].to(DEVICE)
        segment_len = sample['segment_len'].to(DEVICE)
        # target_len = sample['target_len'].to(DEVICE)
        max_pad = sample['max_pad']
        chunk_timestep_len = sample['chunk_timestep_len'].to(DEVICE)
        audio_ID = sample['audio_ID'].to(DEVICE)
        speech_attn_mask = sample['speech_attn_mask'].to(DEVICE)
        nsample = sample['nsample']

        # B x max_chunk x numQ
        self_attd_chunk, cross_attd_chunk, negs  = model(chunks_feat, Q_emb, negs, speech_padding_mask, segment_len, chunk_timestep_len, max_pad, speech_attn_mask)
        
        hybrid_loss, sim_loss = criterion(self_attd_chunk, cross_attd_chunk, negs, audio_ID, speech_padding_mask)
        # loss = hybrid_loss
        beta = 0.0
        loss = hybrid_loss + beta*sim_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cont_loss += hybrid_loss.item()
        running_sim_loss += sim_loss.item()
        if batch_idx % args.log_interval == 0:
            # out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tloss:{:.3f}'.format(
            #     epoch, batch_idx*self_attd_chunk.shape[0], len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), running_loss/(batch_idx+1))
            out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tCont.Loss:{:.3f}\tSim Loss:{:.3f}\tloss:{:.3f}'.format(
                epoch, batch_idx*self_attd_chunk.shape[0], len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), running_cont_loss/(batch_idx+1), running_sim_loss/(batch_idx+1), running_loss/(batch_idx+1))
            print(out_loss)
    if log_file is not None:
        with open(log_file,'a') as f:
            f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(running_cont_loss/(batch_idx+1), running_sim_loss/(batch_idx+1), running_loss/(batch_idx+1)))


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
    parser.add_argument(
        '--save_plots_dir', type=str, default='./plots/', help='save plots'
    )
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)


    os.system("rm -r {}*".format(args.save_plots_dir))
    model = Feat_Merger().to(DEVICE)

    if args.mode == 'train':
        train_dataset = CARE_dataset(
            segs_dir = args.train_segs_dir, 
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # criterion = nn.CTCLoss(blank = 0)
        # criterion = customContrastiveLoss(temperature = 1.0)
        criterion = contrastAcrossSegments(temperature = 1.0)

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
            
            # train_dataset.model = copy.deepcopy(model).to(DEVICE_dataloader)

            scheduler.step()

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
            for file_name in train_files:
                chunk_dataset = Test_dataset(
                    segs_dir = args.train_segs_dir, 
                    file_name=file_name, 
                    # transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset(
                    file_name=file_name, 
                    test_file='./../care_india/audio_features_dict_traindata_5ques_new.json', 
                    transform = transforms.Compose([ToTensorQuestion()])
                )
                chunk_loader = DataLoader(
                    chunk_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=1, 
                    pin_memory = True,
                    collate_fn=pad_collate_test,
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
                    curr_device=DEVICE,
                )
                # a, b, c, d = 0, 0, 0, 0
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
                    # transform = transforms.Compose([ToTensorChunk()])
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
                    pin_memory = True,
                    collate_fn=pad_collate_test,
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
                    curr_device=DEVICE,
                )
                # a, b, c, d = 0, 0, 0, 0

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
                # transform = transforms.Compose([ToTensorChunk()])
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
                pin_memory = True,
                collate_fn=pad_collate_test,
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
                curr_device=DEVICE,
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
                # transform = transforms.Compose([ToTensorChunk()])
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
                pin_memory = True,
                collate_fn=pad_collate_test,
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
                curr_device=DEVICE,
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
    