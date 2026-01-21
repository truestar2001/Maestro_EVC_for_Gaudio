# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse, copy
import glob
import json
import os
import shutil
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path
import torch.nn.functional as F
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from dataset import CodeDataset, parse_manifest, mel_spectrogram, parse_manifest_for_inference, \
    MAX_WAV_VALUE
from utils import AttrDict
from models import CodeGenerator, EmotionClassifier
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
h = None
device = None
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.cuda.manual_seed(1234)
EMOTION = {"Angry":0, "Happy":1, "Neutral":2, "Sad":3, "Surprise":4}



def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def segment_embedding(embedded_seq):
    # embedded_seq = embedded_seq.transpose(0, 1)
    code = []
    durations = []
    indexs = []
    smoothed_durations = []
    prev = embedded_seq[0]
    count = 1
    start_idx = 0
    
    for i in range(1, len(embedded_seq)):
        if embedded_seq[i] == prev:
            count += 1
        else:
            code.append(prev)
            durations.append(count)
            indexs.append(start_idx)
            
            prev = embedded_seq[i]
            count = 1
            start_idx = i
    
    code.append(embedded_seq)
    durations.append(count)
    indexs.append(start_idx)
    d = torch.tensor(durations, dtype = torch.float32).numpy()
    smoothed_durations = [savgol_filter(d, window_length = 5, polyorder= 2, mode= "nearest")]
    
    return code, durations, indexs, smoothed_durations
    


def generate(h, generator, emotion_classifier, code, emo_x):
    start = time.time()
    unique_emo = [segment_embedding(emo_x['code'])]
    y_g_hat, spk, emo, predicted_energy, target_energy, predicted_f0, target_f0, target_vuv, predicted_log_scale_duration, src_duration, predicted_emotion_class, attention_weights, predicted_content_class, code = generator(target_energy = None, target_f0 = None, unique_emo = unique_emo, **code)
    y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf, predicted_energy, target_energy, predicted_f0, target_f0, target_vuv


@torch.no_grad()
def inference(item_index):
    code, gt_audio, filename, _ = dataset[item_index]
    code = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in code.items()}

    if a.parts:
        parts = Path(filename).parts
        fname_out_name = '_'.join(parts[-3:])[:-4]
    else:
        fname_out_name = Path(filename).stem
    folder_name = filename.split('/')[-2]
    people_name = filename.split('/')[-3]
    new_code = dict(code)
    audio, rtf = generate(h, generator, new_code)
    os.makedirs(os.path.join(a.output_dir, people_name, folder_name), exist_ok=True)
    output_file = os.path.join(a.output_dir, people_name, folder_name, fname_out_name + '.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

    
def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_code_file', default='./inference/gaudio_pairs.txt')
    parser.add_argument('--config', default='./configs/inference_config.json')
    parser.add_argument('--output_dir', default='./inference/output')
    parser.add_argument('--checkpoint_file', default = "./checkpoint/g_00300000")
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('-n', type=int, default=2000)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(8))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(a.input_code_file) as f:
        pair_lines = [ln.strip() for ln in f if ln.strip()]

    src_wavs, spk_wavs, emo_wavs = [], [], []
    for ln in pair_lines:
        parts = ln.split("|")
        if len(parts) != 3:
            raise ValueError(f"Bad line (expected 3 fields separated by |): {ln}")
        src_wavs.append(parts[0])
        spk_wavs.append(parts[1])
        emo_wavs.append(parts[2])

    # 이제 parse_manifest_for_inference는 그냥 identity로 써도 됨
    src_file_list = src_wavs
    spk_file_list = spk_wavs
    emo_file_list = emo_wavs
    
    src_dataset = CodeDataset(src_file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                            multispkr=h.get('multispkr', None),
                            pad=a.pad)
    spk_dataset = CodeDataset(spk_file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                            multispkr=h.get('multispkr', None),
                            pad=a.pad)
    emo_dataset = CodeDataset(emo_file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                            h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                            multispkr=h.get('multispkr', None),
                            pad=a.pad)

    from models import EmotionClassifier
    emotion_classifier = EmotionClassifier(emb_dim=1024,num_labels=5).to(device)
    emo_check = torch.load("/home/jyp/Expressive_VC/NAM2Speech-code-ESD/emotion_classifier/original_wavlm_classifier_checkpoints/model_best.pth") # diarization version
    emotion_classifier.load_state_dict(emo_check['model_state_dict'])
    for i, j in emotion_classifier.named_parameters():
        j.requires_grad = False
    emotion_classifier.eval()
    
    generator = CodeGenerator(h, "inference").to(device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])
    for name, param in generator.named_parameters():
        if 'weight_cont_emo' in name:
            print(name, param)
    generator.eval()
    generator.remove_weight_norm()


    # dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
    #                         h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
    #                         multispkr=h.get('multispkr', None),
    #                         pad=a.pad)

    ################################################################################################################
    for i in range(len(src_file_list[0])):
        src_x, src_y, src_file_name, src_y_mel, src_emo_id = src_dataset[i]
        spk_x, spk_y, spk_file_name, spk_y_mel, spk_emo_id = spk_dataset[i]
        emo_x, emo_y, emo_file_name, emo_y_mel, emo_emo_id = emo_dataset[i]
        emo_emo_id = None
        
        X = copy.deepcopy(src_x)
        X['spkr'] = spk_x['spkr']
        X['speaker_embedding'] = spk_x['speaker_embedding']
        # X['emotion_embedding'] = emo_x['emotion_embedding'] 
        # X['arousal_valence'] = np.array(X['arousal_valence'])
        # X['emotion_diarization_embedding'] = emo_x['emotion_diarization_embedding']
        X['full_emotion_diarization_embedding'] = emo_x['full_emotion_diarization_embedding']
        X['smoothed_energy'] = emo_x['smoothed_energy']
        X['energy'] = emo_x['energy']
        X['smoothed_f0'] = emo_x['smoothed_f0']
        X['f0'] = emo_x['f0']
        X['vuv'] = emo_x['vuv']
        X['src_vuv'] = src_x['src_vuv']
        
        X = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in X.items()}
        with torch.no_grad():

            src_file_name_list = src_file_name.split('/')
            save_src_file_name = src_file_name_list[-2] + '_' + src_file_name_list[-1]
            spk_file_name_list = spk_file_name.split('/')
            save_spk_file_name = spk_file_name_list[-2] + '_' + spk_file_name_list[-1]
            emo_file_name_list = emo_file_name.split('/')
            save_emo_file_name = emo_file_name_list[-2] + '_' + emo_file_name_list[-1]
            file_name = f"src_{save_src_file_name}_spk_{save_spk_file_name}_emo_{save_emo_file_name.replace('.wav', '')}"
            os.makedirs(os.path.join(a.output_dir, file_name), exist_ok = True)

            y_g_hat, rtf, predicted_energy, emotion_energy, predicted_f0, target_f0, target_vuv = generate(h, generator, emotion_classifier, X, emo_x)
            print(y_g_hat.shape, src_y.shape)

            output_file = os.path.join(a.output_dir, file_name, file_name + '.wav')
            
            shutil.copy(src_file_name, os.path.join(a.output_dir, file_name, save_src_file_name))
            shutil.copy(spk_file_name, os.path.join(a.output_dir, file_name, save_spk_file_name))
            shutil.copy(emo_file_name, os.path.join(a.output_dir, file_name, save_emo_file_name))
            audio = librosa.util.normalize(y_g_hat.astype(np.float32))
            write(output_file, h.sampling_rate, audio)
            print(f"{i}th Generate {output_file}")



if __name__ == '__main__':
    main()
