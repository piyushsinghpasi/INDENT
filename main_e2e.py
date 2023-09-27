import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from dataloader_pos_hard_negs import *
from nets.network import Feat_Merger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys

import pandas as pd
import pickle as pkl

DEVICE = 'cuda'

    
def train(args, model, train_loader, optimizer, ctc_loss, epoch, log_file=None, hid_dim=768):
    model.train()
    running_loss = 0.
    for batch_idx, sample in enumerate(train_loader):

        optimizer.zero_grad()

        chunks_feat = sample['batch_chunks'].to(DEVICE)
        speech_padding_mask = sample['speech_padding_mask'].to(DEVICE)
        Q_id_seq = sample['Q_id_seq'].to(DEVICE)
        Q_emb = sample['Q_emb'].to(DEVICE)
        text_attn_mask = sample['text_attn_mask'].to(DEVICE)
        segment_len = sample['segment_len'].to(DEVICE)
        target_len = sample['target_len'].to(DEVICE)

        # B x max_chunk x numQ
        log_prob = model(chunks_feat, Q_emb, speech_padding_mask, text_attn_mask)

        # padded_input_log_prob = sequence_to_padding(log_prob, segment_len)

        loss = ctc_loss(log_prob, Q_id_seq, segment_len, target_len)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tloss:{:.3f}'.format(
                epoch, batch_idx*s_feat.shape[0], len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), running_loss/(batch_idx+1))
            print(out_loss)
    if log_file is not None:
        with open(log_file,'a') as f:
            f.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(running_loss/(batch_idx+1)))

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

        criterion = nn.CTCLoss(blank = 0)

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

        best_mIoU_val = 0.
        with open(args.result_logging_file_path+args.checkpoint+'_val.txt', 'w') as result_file:
            result_file.write("mIoU\n")
        with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'w') as result_file:
            result_file.write("mIoU\n")
        with open(args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint),'w') as logfile:
            logfile.write("CTCLoss\n")
        
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, log_file=args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint))
            
            scheduler.step()

            mIoU_train = 0.
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
                mIoU_train += eval(
                    model, 
                    question_loader, 
                    chunk_loader, 
                    args.result_logging_file_path, 
                    args.checkpoint+"_train",
                    K = args.topK,
                )

            mIoU_train /= len(train_files)
            train_metrics = "{:.4f}\n".format(mIoU_train)
            print("Training mIoU (across files): ".format(train_metrics))
            
            with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'a') as result_file:
                result_file.write(train_metrics)

            mIoU_val = 0.
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


                mIoU_val += eval(
                    model, 
                    question_loader, 
                    chunk_loader, 
                    args.result_logging_file_path, 
                    args.checkpoint+"_val",
                    K = args.topK,
                )

            mIoU_val /= len(val_files)
            val_metrics = "{:.4f}\n".format(mIoU_val)
            print("Validation metrics: ".format(val_metrics))
            print("val train file lengths",len(val_files), len(train_files))

            
            with open(args.result_logging_file_path+args.checkpoint+'_val.txt', 'a') as result_file:
                result_file.write(val_metrics)
                
            if (mIoU_val) >= best_mIoU_val:
                best_mIoU_val = mIoU_val
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
        print("Done {}".format(args.checkpoint))

    elif args.mode == 'val':
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        val_files = set()
        for f in os.listdir(args.val_segs_dir) :
            if os.path.isdir(os.path.join(args.val_segs_dir,f)):
                filename = f.split("___")[0]
                val_files.add(filename)

        mIoU_val = 0.
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
            mIoU_val += eval(
                model, 
                question_loader, 
                chunk_loader, 
                args.result_logging_file_path, 
                args.checkpoint+"_val",
                K = args.topK,
            )

        mIoU_val /= len(val_files)
        print("val train file lengths",len(val_files), len(train_files))
        val_metrics = "{:.4f}\n".format(mIoU_val)
        print("Validation metrics: ".format(val_metrics))

    else:
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        test_files = set()
        for f in os.listdir(args.test_segs_dir) :
            if os.path.isdir(os.path.join(args.test_segs_dir,f)):
                filename = f.split("___")[0]
                test_files.add(filename)

        mIoU_test = 0.
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
            mIoU_test += eval(
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

        mIoU_test /= len(test_files)
        test_metrics = "{:.4f}\n".format(mIoU_test)
        print("Test metrics: ".format(test_metrics))


if __name__ == '__main__':
# python main_sim.py --train_segs_dir ./../care_india/annotations/audio_hindi_test_segments --val_segs_dir ./../care_india/annotations/audio_hindi_test_segments --test_segs_dir ./../shubham/care_india/annotations/audio_hindi_test_segments --mode train --checkpoint model1 --gpu 2 --batch-size 4
    main()
    