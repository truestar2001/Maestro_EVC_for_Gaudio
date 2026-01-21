# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import random
from pathlib import Path
import os
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import torch
import torch.utils.data
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
EMOTION = {"Angry":0, "Happy":1, "Neutral":2, "Sad":3, "Surprise":4}
MAX_WAV_VALUE = 32768.0
F0_MEAN = 209.2966
F0_STD = 46.9128

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # Use return_complex=True to get a complex tensor
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    # Convert complex to magnitude
    spec = torch.sqrt(torch.real(spec).pow(2) + torch.imag(spec).pow(2) + (1e-9))

    # Perform mel spectrogram conversion and normalization
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def compute_energy_waveform_based(y, frame_count=28):

    batch_size, _, total_length = y.shape
    frame_size = total_length // frame_count  # 320

    y = y.squeeze(1)  # (batch_size, time)
    energy_list = []

    for i in range(frame_count):
        start = i * frame_size
        end = start + frame_size
        frame = y[:, start:end]  # (batch_size, frame_size)
        rms = torch.sqrt((frame ** 2).mean(dim=1) + 1e-9)  # (batch_size,)
        energy_list.append(rms)

    energy = torch.stack(energy_list, dim=1)  # (batch_size, frame_count)
    return energy

def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output




def parse_manifest_for_inference(manifest):
    audio_files = []
    codes = []
    arousal_valence = []
    emotion_embedding = []
    speaker_embedding = []
    
    for i in manifest:
        if i[0] == "{":
            sample = eval(i.strip(), {"array": np.array, "float32":np.float32})
            k = 'hubert'
            codes += [torch.LongTensor(
                [int(x) for x in sample[k].split(' ')]
            ).numpy()]
            audio_files += [Path(sample["audio"])]
            arousal_valence += [sample['arousal_valence']]
            emotion_embedding += [str(sample['emotional_embedding'])]
            speaker_embedding += [str(sample['speaker_embedding'])]
        else:
            audio_files += [Path(i.strip())]
        
    return audio_files, codes, arousal_valence, emotion_embedding, speaker_embedding

def parse_manifest(manifest):
    audio_files = []
    codes = []
    arousal_valence = []
    emotion_embedding = []
    speaker_embedding = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                # print(line)
                sample = eval(line.strip(), {"array": np.array, "float32":np.float32})
                k = 'hubert'
                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
                arousal_valence += [sample['arousal_valence']]
                emotion_embedding += [str(sample['emotional_embedding'])]
                speaker_embedding += [str(sample['speaker_embedding'])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes, arousal_valence, emotion_embedding, speaker_embedding


def get_dataset_filelist(h):
    training_files, training_codes, training_arousal_valence, training_emotion_embedding, training_speaker_embedding = parse_manifest(h.input_training_file)
    validation_files, validation_codes, validation_arousal_valence, validation_emotion_embedding, validation_speaker_embedding = parse_manifest(h.input_validation_file)

    return (training_files, training_codes, training_arousal_valence, training_emotion_embedding, training_speaker_embedding), (validation_files, validation_codes, validation_arousal_valence, validation_emotion_embedding, validation_speaker_embedding)


def parse_speaker(path, method):
    if type(path) == str:
        path = Path(path)

    if method == '_':
        return '_'.join(path.name.split('_')[:2])
    if method == 'ESD':
        # print(path.name.split('_')[0])
        return path.name.split('_')[0]
    
    
    elif method == 'single':
        return 'A'
    else:
        raise NotImplementedError()


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, code_hop_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, multispkr=False, pad=None):
        self.audio_files = training_files
        random.seed(1234)
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

        self.multispkr = multispkr
        self.pad = pad
        if self.multispkr:
            spkrs = [parse_speaker(f, self.multispkr) for f in self.audio_files]
            spkrs = list(set(spkrs))
            spkrs.sort()

            self.id_to_spkr = spkrs
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}
            print(f"id_to_spkr: {self.id_to_spkr}")
            print(f"spkr_to_id: {self.spkr_to_id}")
            
            # id_to_spkr: ['bho_f', 'bho_m', 'en_f', 'en_m', 'gu_f', 'gu_m', 'hi_f', 'hi_m', 'kn_f', 'kn_m']
            # spkr_to_id: {'bho_f': 0, 'bho_m': 1, 'en_f': 2, 'en_m': 3, 'gu_f': 4, 'gu_m': 5, 'hi_f': 6, 'hi_m': 7, 'kn_f': 8, 'kn_m': 9}
    
    def _sample_interval_emotion_diarization_original_code(self, seqs, seq_len=None):
        """
        seqs: list where
            seqs[0] = audio
            seqs[1] = repeated_code (10ms)
            seqs[2] = code_20ms (reference)
            others = 10ms-resolution features
        """
        code_20ms = seqs[2]
        ref_len = code_20ms.shape[-1]  # 기준 길이

        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else ref_len * self.code_hop_size

        segment_frames = seq_len // self.code_hop_size  # 예: 1280//320 = 4 → 20ms 단위 frame 수

        interval_start = 0
        interval_end = ref_len - segment_frames
        if interval_end >= 3:
            interval_end -= 3

        start_frame = random.randint(interval_start, interval_end)
        end_frame = start_frame + segment_frames

        new_seqs = []
        for i, v in enumerate(seqs):
            if i == 0:
            # 가능한 케이스:
            # (1, T, 2) or (T, 2) or (2, T)
                if v.ndim == 3 and v.shape[-1] == 2:
                    # (1, T, 2) → (1, T)
                    v = v.mean(axis=-1)
                elif v.ndim == 2 and v.shape[0] == 2:
                    # (2, T) → (1, T)
                    v = v.mean(axis=0, keepdims=True)
                elif v.ndim == 2 and v.shape[-1] == 2:
                    # (T, 2) → (1, T)
                    v = v.mean(axis=-1, keepdims=True)
            v_len = v.shape[-1]
            ratio = v_len / ref_len  # 각 시퀀스의 해상도 비율

            mapped_start = int(round(start_frame * ratio))
            if i == 0:
                ratio = 320.0
            elif i == 2:
                ratio = 1.0
            else:
                ratio = 2.0
            target_len = int(round(segment_frames * ratio))
            mapped_end = mapped_start + target_len

            clip = v[..., mapped_start : mapped_end]

            # 보정: 길이 고정
            actual_len = clip.shape[-1]
            if actual_len > target_len:
                clip = clip[..., :target_len]
            elif actual_len < target_len:
                pad_len = target_len - actual_len
                pad = np.full(clip.shape[:-1] + (pad_len,), clip[..., -1], dtype=clip.dtype)
                clip = np.concatenate([clip, pad], axis=-1)

            new_seqs.append(clip)

        return new_seqs
    def _load_hubert_txt_as_int_array(self, txt_path: str) -> np.ndarray:
        # 파일 내용: "17 17 296 ..." (공백 구분 정수)
        with open(txt_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        if len(s) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.fromstring(s, sep=" ", dtype=np.int64)

    def __getitem__(self, index):
        emotion_id=0
        wav_path = self.audio_files[index]              # ✅ wav 경로
        filename = wav_path                             # ✅ 아래 로직이 filename 쓰고 있어서 유지
        wav_stem = str(Path(wav_path).with_suffix(""))  # "/.../foo"

        # (1) emotion diarization embedding
        with open(wav_stem + "_emotion_diarization_embedding.npy", "rb") as f:
            emotion_diarization_embedding = np.load(f)
        with open(wav_stem + "_speaker_embedding.npy", "rb") as f:
            speaker_embedding = np.load(f)
        # (2) energy
        with open(wav_stem + "_mel_energy_nonor.npy", "rb") as f:
            target_energy = np.load(f)
        with open(wav_stem + "_smoothed_energy_nonor.npy", "rb") as f:
            smoothed_energy = np.load(f)

        # (3) f0
        with open(wav_stem + "_nor_f0_clean.npy", "rb") as f:
            target_f0 = np.load(f)
        with open(wav_stem + "_smoothed_f0_clean.npy", "rb") as f:
            smoothed_f0 = np.load(f)

        # (4) vuv
        with open(wav_stem + "_vuv_clean.npy", "rb") as f:
            vuv = np.load(f)

        emotion_diarization_embedding = emotion_diarization_embedding.squeeze(0)

        target_f0 = target_f0 / F0_STD
        smoothed_f0 = smoothed_f0 / F0_STD

        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)

            audio = audio / MAX_WAV_VALUE
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        hubert_txt_path = wav_stem + "_hubert.txt"
        code_full = self._load_hubert_txt_as_int_array(hubert_txt_path)  

        code_length = min(audio.shape[0] // self.code_hop_size, code_full.shape[0])
        code = code_full[:code_length]
        audio = audio[:code_length * self.code_hop_size]

        assert audio.shape[0] // self.code_hop_size == code.shape[0], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"

        ### For repeated code
        repeated_code = np.repeat(code, 2)

        audio, repeated_code, code, smoothed_energy, target_energy, smoothed_f0, target_f0, vuv = \
            self._sample_interval_emotion_diarization_original_code(
                [audio, repeated_code, code, smoothed_energy, target_energy, smoothed_f0, target_f0, vuv]
            )
        
        mel_loss = mel_spectrogram(
            audio, self.n_fft, self.num_mels,
            self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
            center=False
        )

        feats = {"code": repeated_code.squeeze()}
        feats["original_code"] = code.squeeze()
        if self.multispkr:
            feats['spkr'] = self._get_spkr(index)

        feats['speaker_embedding'] = speaker_embedding
        feats['full_emotion_diarization_embedding'] = emotion_diarization_embedding
        feats['smoothed_energy'] = smoothed_energy
        feats['energy'] = target_energy
        feats['smoothed_f0'] = smoothed_f0
        feats['f0'] = target_f0
        feats['vuv'] = vuv.astype(np.int64)
        feats['src_vuv'] = vuv.astype(np.int64)

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze(), emotion_id

    def _get_spkr(self, idx):
        spkr_name = parse_speaker(self.audio_files[idx], self.multispkr)
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id


    def collate_fn(self, batch):
        feats, audio, filename, mel_loss, emo_id = zip(*batch)
        max_len = max([x['full_emotion_diarization_embedding'].shape[0] for x in feats])
        for x in feats:
            x['padding_mask'] = torch.tensor([1] * x['full_emotion_diarization_embedding'].shape[0] + [0] * (max_len - x['full_emotion_diarization_embedding'].shape[0]))      
        for i in range(len(feats)):
            feats[i]['full_emotion_diarization_embedding'] = np.pad(feats[i]['full_emotion_diarization_embedding'], ((0, max_len - feats[i]['full_emotion_diarization_embedding'].shape[0]), (0, 0)))
            # feats[i]["original_code"] = np.pad(feats[i]["original_code"], (0, max_len - feats[i]["original_code"].shape[0]))
        stacked_feats = {}
        for key in feats[0].keys():
            if isinstance(feats[0][key], torch.Tensor):
                stacked_feats[key] = torch.stack([x[key] for x in feats])
            else:
                stacked_feats[key] = torch.stack([torch.tensor(x[key]) for x in feats])
            # stacked_feats[key] = torch.stack([torch.tensor(x[key]) for x in feats])
        audio = torch.stack(audio)
        mel_loss = torch.stack(mel_loss)
        return stacked_feats, audio, filename, mel_loss, emo_id
    

    def __len__(self):
        return len(self.audio_files)