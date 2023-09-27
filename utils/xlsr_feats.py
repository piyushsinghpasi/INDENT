# import soundfile as sf
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2FeatureExtractor, AutoModel
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2Model
# from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import torch.nn as nn
from sentence_transformers import SentenceTransformer
# import librosa
import torchaudio

import argparse

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
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
DEVICE = torch.device("cuda")

# asr_processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
# asr_model = Wav2Vec2Model.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200").to(DEVICE)

# MODEL_ID = "ai4bharat/indicwav2vec-hindi"
# asr_model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE)
# asr_processor = AutoProcessor.from_pretrained(MODEL_ID)
# count_parameters(asr_model)


asr_processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
asr_model = AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(DEVICE)

# count_parameters(asr_model)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

layer_name = "encoder.layers.23.final_layer_norm"
# asr_model.encoder.layers[11].final_layer_norm.register_forward_hook(get_activation(layer_name))
asr_model.encoder.layers[15].final_layer_norm.register_forward_hook(get_activation(layer_name))

def extract_wav2vec(wav_file):
	audio_input, sample_rate = torchaudio.load(wav_file)

	input_values = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.float()

	# Features
	asr_model.eval()
	with torch.no_grad():
		asr_model(input_values.squeeze(0).to(DEVICE))

		# feats = feats.last_hidden_state[0]
		feats = activation[layer_name]

		feats = feats.detach().cpu().numpy()
		print(feats.shape, end=" ")
		return feats

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='save features')

	parser.add_argument(
		'--audio_dir', type=str, help='path where all segment dirs are',
		# default=""
	)

	args = parser.parse_args()

	vad_wav = "vad_chunks"
	vad_chunks_feat = "vad_chunks_feat_xlsr_layer{}_final_norm".format(15)

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
			chunk_feat = extract_wav2vec(f)
			audio_id = f.split('/')[-1]
			print(audio_id)

			chunk_feat_file = os.path.join(vad_chunks_feat_dir,"{}.npy".format(audio_id[:-4]))
			# with open(os.path.join(vad_chunks_feat_dir,"{}.npy".format(audio_id[:-4])),'w') as chunk_feat_file:
			np.save(chunk_feat_file, chunk_feat)
   
   
	print(args.audio_dir)
