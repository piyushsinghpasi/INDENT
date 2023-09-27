# import soundfile as sf
import os

# import librosa
import random

import argparse
import librosa


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='save features')

    parser.add_argument(
        '--audio_dir', type=str, help='path where all segment dirs are',
    )

    args = parser.parse_args()

    vad_wav = "vad_chunks"
    # vad_chunks_feat = "vad_text_indicASR_feat"

    all_seg_dir = [os.path.join(args.audio_dir,x) for x in  os.listdir(args.audio_dir) if os.path.isdir(os.path.join(args.audio_dir,x))]
    all_seg_dir = sorted(all_seg_dir)
    num_chunks = 0
    for seg_dir in all_seg_dir:
        file_name = seg_dir.split("/")[-1]
        
        # print("Processing", file_name)
        vad_wav_dir = os.path.join(seg_dir, vad_wav)

        all_files = [ os.path.join(vad_wav_dir, f) for f in os.listdir(vad_wav_dir)]
        
        for wav_file in all_files:
            print(librosa.get_duration(filename=wav_file))
            exit(0)
            # duration = 
        num_chunks += len(all_files)
        
    print("avg chunks per segment", num_chunks/len(all_seg_dir))
        