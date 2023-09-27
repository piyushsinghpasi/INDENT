# import soundfile as sf
import os
import torch
import numpy as np
import torch.nn as nn
from sentence_transformers import SentenceTransformer
# import librosa
import random

import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1)

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

# scp -r shubham@swara.cse.iitb.ac.in:/home/shubham/care_india/questions ./care_india/
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
DEVICE = torch.device("cuda")

labse_model = SentenceTransformer('sentence-transformers/LaBSE').to(DEVICE)

def labse(sentence_file):
    with open(sentence_file, 'r') as txt_file:
        sentence = txt_file.readlines()# [line.rstrip() for line in txt_file]
    return labse_model.encode(sentence)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='save features')

	parser.add_argument(
		'--audio_dir', type=str, help='path where all segment dirs are',
		default="../../care_india_data/audio_bihari_test_segments"
	)

	args = parser.parse_args()

	vad_wav = "vad_text_indicASR"
	vad_chunks_feat = "vad_text_indicASR_feat"

	all_seg_dir = [os.path.join(args.audio_dir,x) for x in  os.listdir(args.audio_dir) if os.path.isdir(os.path.join(args.audio_dir,x))]
	all_seg_dir = sorted(all_seg_dir)

	for seg_dir in all_seg_dir:
		file_name = seg_dir.split("/")[-1]
		
		print("Processing", file_name)
		vad_wav_dir = os.path.join(seg_dir, vad_wav)

		all_files = [ os.path.join(vad_wav_dir, f) for f in os.listdir(vad_wav_dir)]
		all_files = sorted(all_files)

		vad_chunks_feat_dir = os.path.join(seg_dir, vad_chunks_feat)
		if not os.path.exists(vad_chunks_feat_dir):
			os.makedirs(vad_chunks_feat_dir)


		for f in all_files:
			chunk_feat = labse(f)
			audio_id = f.split('/')[-1]
			print(audio_id)
			chunk_feat_file = os.path.join(vad_chunks_feat_dir,"{}.npy".format(audio_id[:-4]))
			# with open(os.path.join(vad_chunks_feat_dir,"{}.npy".format(audio_id[:-4])),'w') as chunk_feat_file:
			np.save(chunk_feat_file, chunk_feat)
	print(args.audio_dir)
