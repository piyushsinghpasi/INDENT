import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from dataloader import *
from nets.network import Feat_Merger

import sys

import pandas as pd
import pickle as pkl

DEVICE = 'cuda'

class customContrastiveLoss(nn.Module):
    def __init__(self, temperature = 1.0):
        super(customContrastiveLoss, self).__init__()
        self.temperature = temperature 

    def forward(self, s, t, NS, NT):
        '''
        s.shape == t.shape -> Bxd

        k is for attention window

        [-k,+k] [-2k,2k]

        i, pos_s, pos_t, neg
        '''

        
        numerator = torch.exp( torch.sum( s*t, dim=-1) / self.temperature).sum(dim=-1)
        
        # denominator_N_set = torch.diag(torch.exp(torch.sum(N_set.unsqueeze(dim=2) * s, dim=-1) / self.temperature ).sum(dim=1))
        denominator_NS = torch.diag(torch.exp(torch.sum(NS.unsqueeze(dim=2) * t, dim=-1) / self.temperature ).sum(dim=1))
        denominator_NT = torch.diag(torch.exp(torch.sum(NT.unsqueeze(dim=2) * s, dim=-1) / self.temperature ).sum(dim=1))

        # log_exp = (torch.log(numerator/denominator_N_set)).sum(dim=0) / s.shape[0]

        # return -log_exp

        log_exp_t_s = (torch.log(numerator/denominator_NS)).sum(dim=0) / NS.shape[1]
        log_exp_s_t = (torch.log(numerator/denominator_NT)).sum(dim=0) / NT.shape[1]

        return  -log_exp_s_t, -log_exp_t_s


def train(args, model, train_loader, optimizer, criterion, epoch, log_file=None):
    model.train()
    # exit(0)
    # batch_s_feat = torch.empty((0, s_feat.shape[-1])).to(DEVICE)
    for batch_idx, sample in enumerate(train_loader):
        
        s, t, NS, NT = sample['pos_s'].to(DEVICE), sample['pos_t'].to(DEVICE), sample['NS'], sample['NT']
        
        optimizer.zero_grad()
        s_feat, t_feat = model(s, t)

        # if 
        
        NS_feat = torch.empty((0, s_feat.shape[-1])).to(DEVICE)
        for ind, NS_i in enumerate(NS):
            NS_x = NS_i.to(DEVICE)
            x, _ = model(NS_x, None)
            x = x.squeeze().unsqueeze(dim=0)
            # print(NS_feat.shape, x.shape)
            # if ind == 0: NS_feat = x
            NS_feat = torch.cat((NS_feat, x), dim=0)

        NT_feat = torch.empty((0, t_feat.shape[-1])).to(DEVICE)
        # print(NT_feat)
        for _, NT_i in enumerate(NT):
            NT_x = NT_i.to(DEVICE)
            _, x = model(None, NT_x)
            x = x.squeeze().unsqueeze(dim=0)
            NT_feat = torch.cat((NT_feat, x), dim=0)

        s_feat = s_feat.unsqueeze(dim=0)
        t_feat = t_feat.unsqueeze(dim=0)
        NS_feat = NS_feat.unsqueeze(dim=0)
        NT_feat = NT_feat.unsqueeze(dim=0)

        L_speech_to_text, L_text_to_speech = criterion(s_feat, t_feat, NS_feat, NT_feat)
        
        # print(L_speech_to_text, L_text_to_speech)
        loss = L_speech_to_text + L_text_to_speech

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tL_speech_to_text:{:.6f}\tL_text_to_speech:{:.6f}\tLoss:{:.6f}'.format(
                epoch, batch_idx+1, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), L_speech_to_text.item(), L_text_to_speech.item(), loss.item())
            print(out_loss)
    if log_file is not None:
        with open(log_file,'a') as f:
            f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(L_speech_to_text.item(), L_text_to_speech.item(), loss.item()))



def eval(model, question_loader, chunk_loader, result_logging_file_path = None, checkpoint="", hid_dim = 768,):
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
    
    # for all segment, for all_chunks
    # list of segment with chunk tensors, raggedtensor not supported in pytorch
    seg_chunk_feat = dict()
    num_chunk_per_file = None
    R_1, R_5, R_10 = 0., 0., 0.
    with torch.no_grad():
        for batch_idx, sample in enumerate(chunk_loader):
            
            chunks, seg_id = sample['chunks'], sample['segment_id'][0]
            print(f"Evaluating {seg_id}")

            chunk_feat = torch.empty((0, hid_dim)).to(DEVICE)
            for _, chunk_i in enumerate(chunks):
                chunk_ix = chunk_i.to(DEVICE)
                x, _ = model(chunk_ix, None)
                x = x.squeeze().unsqueeze(dim=0)
                chunk_feat = torch.cat((chunk_feat, x), dim=0)

            seg_chunk_feat[seg_id] = chunk_feat

        for batch_idx, sample in enumerate(question_loader):
            segment_scores = dict()

            Question_feat, GT_seg_id = sample['Question'].to(DEVICE), sample['segment_id'][0]

            _, Q_feat = model(None, Question_feat)
            Q_feat = Q_feat.squeeze()
            
            for seg_id, chunk_feat in seg_chunk_feat.items():
                # [no. chunk, 1]
                chunk_scores = torch.matmul(seg_chunk_feat[seg_id], Q_feat)
                # we can use maxFeat_Merger
                segment_scores[seg_id] = chunk_scores.max(dim=-1)

            all_scores = sorted(segment_scores, key=segment_scores.get, reverse=True)
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

    print('R@1: {:.3f}'.format(R_1))
    print('R@5: {:.3f}'.format(R_5))
    print('R@10: {:.3f}'.format(R_10))
    print('R-Avg: {:.3f}'.format((R_1 + R_5 + R_10)/3.0))

    out_metrics = "{:.3f}\t{:.3f}\t{:.3f}\n".format(R_1, R_5, R_10)
    if result_logging_file_path is not None:
        with open(result_logging_file_path+checkpoint+'.txt', 'a') as result_file:
            result_file.write(out_metrics)


    return (R_1 + R_5 + R_10)/3.0


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
        '--result_logging_file_path', type=str, default="./results/", help='saves each epoch eval metrics')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.manual_seed(args.seed)

    model = Feat_Merger().to(DEVICE)

    if args.mode == 'train':
        train_dataset = CARE_dataset(
            segs_dir = args.train_segs_dir, 
            matchings = args.matchings, 
            qns_file = '/home/shubham/care_india/questions/questions_00-02_hindi.json', 
            k=0, 
            transform = transforms.Compose([ToTensor()])
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # tune temperature
        criterion = customContrastiveLoss(temperature = 1.2)

        val_files = set()
        for f in os.listdir(args.val_segs_dir) :
            if os.path.isdir(os.path.join(args.val_segs_dir,f)):
                filename = f.split("___")[0]
                val_files.add(filename)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)

        best_R = 0
        with open(args.result_logging_file_path+args.checkpoint+'.txt', 'w') as result_file:
            result_file.write("R@1\tR@5\tR@10\n")
        with open(args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint),'w') as logfile:
            logfile.write("L_speech_to_text\tL_text_to_speech\tLoss\n")
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, log_file=args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint))
            scheduler.step()
            
            tot_R = 0
            for file_name in val_files:
                chunk_dataset = Test_dataset(
                    segs_dir = args.val_segs_dir, 
                    file_name=file_name, 
                    transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset(file_name=file_name, transform = transforms.Compose([ToTensorQuestion()]))
                chunk_loader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
                question_loader = DataLoader(question_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
                
                R = eval(model, question_loader, chunk_loader, args.result_logging_file_path, args.checkpoint)
                tot_R += R
            avg_tot_R = tot_R/len(val_files)
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

        tot_R = 0
        for file_name in val_files:
            chunk_dataset = Test_dataset(
                    segs_dir = args.val_segs_dir, 
                    file_name=file_name, 
                    transform = transforms.Compose([ToTensorChunk()])
                )
            question_dataset = Question_dataset(file_name=file_name, transform = transforms.Compose([ToTensorQuestion()]))

            chunk_loader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
            question_loader = DataLoader(question_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
            
            R = eval(model, question_loader, chunk_loader, args.result_logging_file_path, args.checkpoint)
            print(R)
            tot_R += R
        avg_tot_R = tot_R/len(val_files)
        print(f"Avg recall:{avg_tot_R:3f}")
    else:
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        test_files = set()
        for f in os.listdir(args.test_segs_dir) :
            if os.path.isdir(os.path.join(args.test_segs_dir,f)):
                filename = f.split("___")[0]
                test_files.add(filename)

        tot_F = 0
        for file_name in test_files:
            chunk_dataset = Test_dataset(
                segs_dir = args.test_segs_dir,
                file_name=file_name, 
                transform = transforms.Compose([ToTensorChunk()])
            )
            question_dataset = Question_dataset(file_name=file_name, transform = transforms.Compose([ToTensorQuestion()]))
            chunk_loader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
            question_loader = DataLoader(question_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
            
            F = eval(model, question_loader, chunk_loader, args.result_logging_file_path, args.checkpoint)
            tot_F += F
        avg_tot_F = tot_F/len(test_files)
        print(f"Avg recall:{avg_tot_F:3f}")

if __name__ == '__main__':
    # python3.7 main.py --train_segs_dir /home/shubham/care_india/annotations/audio_hindi_test_segments --val_segs_dir /home/shubham/care_india/annotations/audio_hindi_test_segments --test_segs_dir /home/shubham/care_india/annotations/audio_hindi_test_segments --mode train --checkpoint model1 --gpu 2
    main()