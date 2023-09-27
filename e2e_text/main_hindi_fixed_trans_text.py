import argparse
import torch
# torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader_trans_text import *

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from net.net_e2e_asr import Feat_Merger
# from utils.recall_heuristic import eval
from utils.recall_eval import eval

from utils.CE_style_hinge import CE_style
from utils.contrastiveLoss import contrastAcrossSegments
from utils.plot_grad import plot_grad_flow

import os

DEVICE = 'cuda'


def train(args, model, train_loader, optimizer, criterion, epoch, N, log_file=None, hid_dim=768):
    model.train()

    running_loss, running_cont_loss, running_sim_loss, running_cont_loss_self, running_sim_loss_self, running_hinge_loss, running_hinge_loss_self = 0., 0., 0., 0., 0., 0., 0.
    for batch_idx, sample in enumerate(train_loader):

        chunks_feat = sample['batch_chunks'].to(DEVICE) 
        speech_padding_mask = sample['speech_padding_mask'].to(DEVICE)
        Q_emb = sample['Q_emb'].to(DEVICE) 
        Q_emb.requires_grad = False
        segment_len = sample['segment_len'].to(DEVICE)
        chunk_timestep_len = sample['chunk_timestep_len'].to(DEVICE)
        speech_attn_mask = sample['speech_attn_mask'].to(DEVICE)
        valid_negs_mask = sample['valid_negs_mask'].to(DEVICE)
        sampled_neg_inds = sample['sampled_neg_inds'].to(DEVICE)
        padding_cross_attn_mask = sample["padding_cross_attn_mask"].to(DEVICE)
        
        B = sample["B"]
        M = sample['M']
        D = sample["D"]

        optimizer.zero_grad()

        self_attd_chunk, cross_attd_chunk = model(chunks_feat, Q_emb, speech_padding_mask, segment_len, chunk_timestep_len, M, speech_attn_mask, padding_cross_attn_mask)
        
        self_attd_chunk = self_attd_chunk.view(B, N, M, D)
        cross_attd_chunk = cross_attd_chunk.view(B, N, M, D)
        # print(self_attd_chunk)

        cont_loss, sim_loss, hinge_loss = criterion(self_attd_chunk, cross_attd_chunk, speech_padding_mask, valid_negs_mask, sampled_neg_inds)
        cont_loss_self, sim_loss_self, hinge_loss_self = criterion(self_attd_chunk, self_attd_chunk, speech_padding_mask, valid_negs_mask, sampled_neg_inds)
        
        loss = cont_loss + cont_loss_self
        loss.backward()
        
        # if (batch_idx%args.log_interval) == 0:
        #     plot_grad_flow(model.named_parameters(), "./plots/E{}-B{}.png".format(epoch, batch_idx+1, first=batch_idx==0))
            
        optimizer.step()

        running_loss += loss.item()
        running_cont_loss += cont_loss.item()
        running_sim_loss += sim_loss.item()
        running_cont_loss_self += cont_loss_self.item()
        running_sim_loss_self += sim_loss_self.item()
        running_hinge_loss += hinge_loss.item()
        running_hinge_loss_self += hinge_loss_self.item()
        
        if (batch_idx % args.log_interval) == 0:
            out_loss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tCont.Loss:{:.3f}\tSim Loss:{:.3f}\tHinge:{:.3f}\tCont.Self:{:.3f}\tSim Self:{:.3f}\tHinge Self:{:.3f}\tloss:{:.3f}'.format(
                epoch, batch_idx*self_attd_chunk.shape[0], len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), running_cont_loss/(batch_idx+1), running_sim_loss/(batch_idx+1), running_hinge_loss/(batch_idx+1), running_cont_loss_self/(batch_idx+1), running_sim_loss_self/(batch_idx+1), running_hinge_loss_self/(batch_idx+1), running_loss/(batch_idx+1))
            print(out_loss)
    if log_file is not None:
        with open(log_file,'a') as f:
            f.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(running_cont_loss/(batch_idx+1), running_sim_loss/(batch_idx+1), running_hinge_loss/(batch_idx+1), running_cont_loss_self/(batch_idx+1), running_sim_loss_self/(batch_idx+1), running_hinge_loss_self/(batch_idx+1), running_loss/(batch_idx+1)))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print("Total Trainable Params: {} ~ {:.2f} M".format(total_params, (total_params/1e6)))
    exit(0)
    
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
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
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
    parser.add_argument(
        '--window_size', type=int, default=12, help='Fixed window size segments'
    )
    parser.add_argument(
        '--feat_model', type=str, help='which feat model to pick'
    )
    parser.add_argument(
        '--D', type=int, help='Times Data Augmentation'
    )
    parser.add_argument(
        '--std', type=float, help='standard deviation in Gaussian Layer', default=0.5,
    )
    parser.add_argument(
        '--lang', type=str, help='lang code for translat6ed Q. embeddings', 
    )
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # torch.manual_seed(args.seed)
    set_seed(1)

    args.checkpoint = "{}-B{}-E{}-lr{}-N{}-D{}-FM{}-window{}".format(args.checkpoint, args.batch_size, args.epochs, args.lr, args.nsample, args.D, args.feat_model, args.window_size)

    os.system("rm -r {}*".format(args.save_plots_dir))
    
    print("\n\nModel Checkpoint: {}".format(args.checkpoint))
    model = Feat_Merger(input_dim = 768, hidden_dim = 768).to(DEVICE)
    # model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
    # count_parameters(model)
    vad_text_feat_dir = None
    if args.feat_model == "vakyansh":
        vad_text_feat_dir = "vad_text_feat"
    elif args.feat_model == "indicASR":
        vad_text_feat_dir = "vad_text_indicASR_feat"
    # elif args.feat_model == "xlsr":
    #     vad_text_feat_dir = "vad_text_feat_xlsr"
    else:
        raise ValueError("Unknown model feature extractor: Pls pick from vakyansh or indicASR")

    if args.mode == 'train':
        train_dataset = CARE_dataset(
            segs_dir = args.train_segs_dir, 
            nsample = args.nsample,
            vad_text_feat_dir = vad_text_feat_dir,
            D = args.D,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # criterion = contrastAcrossSegments(temperature = 1.0)
        # criterion = contrastAcrossSegments(temperature = 0.5)
        criterion = CE_style(smoothing = 0.05, temperature = 1.0)

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
        best_val_metrics = ''
        
        
        with open(args.result_logging_file_path+args.checkpoint+'_val.txt', 'w') as result_file:
            result_file.write("R@1\tR@5\tR@10\tR-avg\n")
        with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'w') as result_file:
            result_file.write("R@1\tR@5\tR@10\tR-avg\n")
        with open(args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint),'w') as logfile:
            logfile.write("Cont.Loss\tSimLoss\tHingeLoss\tCont.SelfLoss\tSimSelfLoss\tHingeSelf\tLoss\n")
        
        for epoch in range(1, args.epochs + 1):
            print("\n")
            train_dataset.Grpshuffle()
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, N=args.nsample, log_file=args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint))
            # train_test(args, model, train_loader, optimizer, criterion, epoch=epoch, N=args.nsample, log_file=args.result_logging_file_path+"{}_train_loss.txt".format(args.checkpoint))
            
            # train_dataset.model = copy.deepcopy(model).to(DEVICE_dataloader)

            scheduler.step()

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
            for file_name in train_files:
                chunk_dataset = Test_dataset(
                    segs_dir = args.train_segs_dir, 
                    file_name=file_name, 
                    # num_chunks = args.window_size,
                    vad_text_feat_dir = vad_text_feat_dir,
                    # transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset(
                    file_name=file_name, 
                    test_file='../../care_india/audio_features_dict_traindata_5ques.json', 
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
                a, b, c, d = 0, 0, 0, 0
                # a, b, c, d = eval(
                #     model, 
                #     question_loader, 
                #     chunk_loader, 
                #     args.result_logging_file_path, 
                #     args.checkpoint+"_train",
                #     K = args.topK,
                #     curr_device=DEVICE,
                # )
                tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d

            train_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( tot_R_1/len(train_files), tot_R_5/len(train_files), tot_R_10/len(train_files), tot_R_avg/len(train_files))
            print("Training metrics: ",train_metrics)
            
            with open(args.result_logging_file_path+args.checkpoint+'_train.txt', 'a') as result_file:
                result_file.write(train_metrics)

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
            for file_name in val_files:
                chunk_dataset = Test_dataset_fixed_new(
                    segs_dir = args.val_segs_dir, 
                    file_name=file_name, 
                    num_chunks=args.window_size,
                    vad_text_feat_dir = vad_text_feat_dir,
                    # transform = transforms.Compose([ToTensorChunk()])
                )
                question_dataset = Question_dataset_new(
                    file_name=file_name, 
                    test_data_object=chunk_dataset,
                    csv_dir = '../../care_india_data/audio_hindi_val_csv/',
                    test_file='../../care_india/audio_features_dict_valdata_5ques.json', 
                    qns_emb_file='../../care_india/questions/questions_00-02_hindi_labse_embeddings-{}.json'.format(args.lang)
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
            
            print("Files: {}-val {}-train".format(len(val_files), len(train_files)))
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
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
        print("Done {}: {}".format(args.checkpoint, best_val_metrics))

    elif args.mode == 'val':
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))

        val_files = set()
        for f in os.listdir(args.val_segs_dir) :
            if os.path.isdir(os.path.join(args.val_segs_dir,f)):
                filename = f.split("___")[0]
                val_files.add(filename)

        tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
        for file_name in val_files:
            chunk_dataset = Test_dataset_fixed_new(
                    segs_dir = args.val_segs_dir, 
                    file_name=file_name, 
                    num_chunks=args.window_size,
                    vad_text_feat_dir=vad_text_feat_dir,
                    # transform = transforms.Compose([ToTensorChunk()])
            )
            question_dataset = Question_dataset_new(
                file_name=file_name, 
                test_data_object=chunk_dataset,
                csv_dir = '../../care_india_data/audio_hindi_val_csv/',
                test_file='../../care_india/audio_features_dict_valdata_5ques.json', 
                qns_emb_file='../../care_india/questions/questions_00-02_hindi_labse_embeddings-{}.json'.format(args.lang)

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
                args.checkpoint, 
                show_plots=args.show_plots, 
                file_name = file_name,
                # seg_to_Q = seg_to_Q,
                # qns_dict = qns_dict,
                K = args.topK,
                curr_device=DEVICE,
            )

            tot_R_1, tot_R_5, tot_R_10, tot_R_avg = tot_R_1+a, tot_R_5 +b, tot_R_10+c, tot_R_avg+d

        # print("val train file lengths",len(val_files), len(train_files))
        val_metrics = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format( 
            tot_R_1/len(val_files), 
            tot_R_5/len(val_files), 
            tot_R_10/len(val_files), 
            tot_R_avg/len(val_files)
        )
        print("Validation metrics: ",val_metrics)

    else:
        eval_dir = 'test'
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        # print(model.self_attn_layer.linear_v.weight)
        # exit(0)
        test_files = set()
        for f in os.listdir(args.test_segs_dir) :
            if os.path.isdir(os.path.join(args.test_segs_dir,f)):
                filename = f.split("___")[0]
                test_files.add(filename)

        tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
        for file_name in test_files:
            chunk_dataset = Test_dataset_fixed_new(
                segs_dir = args.test_segs_dir, 
                file_name=file_name, 
                num_chunks=args.window_size,
                vad_text_feat_dir=vad_text_feat_dir,
                # transform = transforms.Compose([ToTensorChunk()])
            )
            question_dataset = Question_dataset_new(
                file_name=file_name, 
                test_data_object=chunk_dataset,
                csv_dir = '../../care_india_data/audio_hindi_{}_csv/'.format(eval_dir),
                test_file='../../care_india/audio_features_dict_{}data_5ques.json'.format(eval_dir),
                qns_emb_file='../../care_india/questions/questions_00-02_hindi_labse_embeddings-{}.json'.format(args.lang)
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
            seg_to_Q = None
            qns_dict = None
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
    main()
    