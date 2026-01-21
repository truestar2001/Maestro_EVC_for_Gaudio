# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')
import torch.nn as nn
import numpy as np
import itertools
from transformers import AutoProcessor, WavLMModel, Wav2Vec2FeatureExtractor, set_seed

import shutil
import time
import argparse
import json
from dataset import compute_energy_waveform_based
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from models import CodeGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, MutualInformationLoss, \
    KL_div_based_MutualInformationLoss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.diarization import Speech_Emotion_Diarization
# from wavLM import Emotion_encoder
# from mi_estimator import CLUBSample_group

torch.backends.cudnn.benchmark = True



import sys
EMOTION = {"Angry":0, "Happy":1, "Neutral":2, "Sad":3, "Surprise":4}

from torch.autograd import Function
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output
    
class FilterWarnings:
    def __init__(self, filter_message, original_stream):
        self.filter_message = filter_message
        self.original_stream = original_stream

    def write(self, message):
        # 특정 경고 메시지가 포함된 경우 출력하지 않음
        if self.filter_message in message:
            return  
        # 줄바꿈이 필요한 경우 정상적으로 출력
        if message.strip() == "":
            return
        # 정상적인 로그는 원래대로 출력
        self.original_stream.write(message + "\n")

    def flush(self):
        self.original_stream.flush()

# stdout과 stderr 모두 필터링하여 특정 경고 메시지만 제거 (줄바꿈 유지)
sys.stdout = FilterWarnings("WARNING: module.dict.weight has no gradient!", sys.__stdout__)
sys.stderr = FilterWarnings("WARNING: module.dict.weight has no gradient!", sys.__stderr__)

def gradient_reversal(x):
    return GradientReversalFunction.apply(x)


def mi_eval_forward(spk_emo_mi_net, spk, emo):
    spk_d = spk.detach()
    emo_d = emo.detach()
    spk_d = torch.permute(spk_d, (0, 2, 1))
    spk_d = spk_d[:, 0, :].squeeze(1)
    emo_d = torch.permute(emo_d, (0, 2, 1))
    if isinstance(spk_emo_mi_net, DistributedDataParallel):
        lld_spk_emo_loss = -spk_emo_mi_net.module.loglikeli(spk_d, emo_d)
        mi_spk_emo_loss = spk_emo_mi_net.module.mi_est(spk_d, emo_d)
    else:
        lld_spk_emo_loss = -spk_emo_mi_net.loglikeli(spk_d, emo_d)
        mi_spk_emo_loss = spk_emo_mi_net.mi_est(spk_d, emo_d)
    return lld_spk_emo_loss, mi_spk_emo_loss

def mi_second_forward(spk_emo_mi_net, spk, emo):
    spk = torch.permute(spk, (0, 2, 1))
    spk = spk[:, 0, :].squeeze(1)
    emo = torch.permute(emo, (0, 2, 1))
    if isinstance(spk_emo_mi_net, DistributedDataParallel):
        spk_emo_loss = spk_emo_mi_net.module.mi_est(spk, emo)
    else:
        spk_emo_loss = spk_emo_mi_net.mi_est(spk, emo)
    return spk_emo_loss

def mi_first_forward(spk_emo_mi_net, optimizer_spk_emo_mi_net, spk, emo):
    optimizer_spk_emo_mi_net.zero_grad()
    spk_d = spk.detach()
    emo_d = emo.detach()
    spk_d = torch.permute(spk_d, (0, 2, 1))
    spk_d = spk_d[:, 0, :].squeeze(1)
    emo_d = torch.permute(emo_d, (0, 2, 1))
    if isinstance(spk_emo_mi_net, DistributedDataParallel):    
        lld_spk_emo_loss = -spk_emo_mi_net.module.loglikeli(spk_d, emo_d)
    else:
        lld_spk_emo_loss = -spk_emo_mi_net.loglikeli(spk_d, emo_d)
    lld_spk_emo_loss.backward()
    optimizer_spk_emo_mi_net.step()
    return optimizer_spk_emo_mi_net, lld_spk_emo_loss





def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))
    torch.cuda.set_device(device)
    generator = CodeGenerator(h, "train").to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    from models import EmotionClassifier
    emotion_classifier = EmotionClassifier(emb_dim=1024,num_labels=5).to(device)
    emo_check = torch.load("./emotion_classifier/model_best.pth") # diarization version
    emotion_classifier.load_state_dict(emo_check['model_state_dict'])
    for i, j in emotion_classifier.named_parameters():
        j.requires_grad = False
    emotion_classifier.eval()
    
    writer = SummaryWriter(log_dir="./runs/ten_logs")
    # using loss
    using_spk_emo_vCLUB_mi_loss = False
    using_KL_MI_loss = False
    using_emotion_classifier_loss = False
    using_energy_prediction_loss = False
    using_pitch_prediction_loss = False
    using_duration_prediction_L1_loss = True
    using_attention_weight_loss = False
    using_spk_cos_sim_loss = False
    using_auxiliary_loss = False
    using_energy_mse_loss = True
    using_y_hat_energy_mse_loss = False
    using_f0_mse_loss = True
    # using_vuv_bce_loss = True
    
    using_generated_speaker_mse_loss = False # generated audio에서 ECAPA-TDNN 기반 speaker verification loss
    using_generated_emotion_classifier_loss = False # generated audio에서 WavLM을 추출한 다음 pretrained emotion classifier에 넣은 loss

    using_GRL_emotion_classifier_loss = True # GRL을 이용하여 emotion classifier loss로 spk 학습
    using_GRL_content_classifier_loss = True

    using_spk_triplet_loss = True
    # if using_spk_emo_vCLUB_mi_loss:
    #     spk_emo_mi_net = CLUBSample_group(192, 192, 384).to(device)
    
    ###########################################################################################################################################################    
    speaker_encoder = EncoderClassifier.from_hparams(source = "speechbrain/spkrec-ecapa-voxceleb", run_opts = {"device":device}) # speaker verification score로도 표현이 가능하대
    for i, j in speaker_encoder.named_parameters():
        j.requires_grad = False
    speaker_encoder.eval()
       
    ### WavLM feature extractor    
    WavLM_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
    WavLM_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    WavLM_model.to(device)
    WavLM_model.eval()    

    ###########################################################################################################################################################    


    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        if using_spk_emo_vCLUB_mi_loss:
            cp_spk_emo_mi_net = scan_checkpoint(a.checkpoint_path, "spk_emo_mi_net_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        
        # load spk_emo_mi_net
        if using_spk_emo_vCLUB_mi_loss:
            state_dict_spk_emo_mi_net = load_checkpoint(cp_spk_emo_mi_net, device)
            spk_emo_mi_net.load_state_dict(state_dict_spk_emo_mi_net['spk_emo_mi_net'])
        
        
    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank],
            # find_unused_parameters=('f0_quantizer' in h),
            find_unused_parameters = True,
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[local_rank], find_unused_parameters = True,).to(device)
        msd = DistributedDataParallel(msd, device_ids=[local_rank], find_unused_parameters = True,).to(device)

        if using_spk_emo_vCLUB_mi_loss:
            spk_emo_mi_net = DistributedDataParallel(spk_emo_mi_net, device_ids=[local_rank], find_unused_parameters = True,).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])


    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])
    
    if using_spk_emo_vCLUB_mi_loss:
        optimizer_spk_emo_mi_net = torch.optim.Adam(spk_emo_mi_net.parameters(), lr = 3e-4)

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    
        if using_spk_emo_vCLUB_mi_loss:
            optimizer_spk_emo_mi_net.load_state_dict(state_dict_do['optimizer_spk_emo_mi_net'])
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, multispkr=h.get('multispkr', None))

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=16, shuffle=False, sampler=train_sampler, persistent_workers = True, prefetch_factor=2,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True, collate_fn=trainset.collate_fn)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                               fmax_loss=h.fmax_for_loss, device=device,multispkr=h.get('multispkr', None))
        validation_loader = DataLoader(validset, num_workers = 16, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True, collate_fn=validset.collate_fn)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    for name, param in generator.named_parameters():
        if not param.requires_grad:
            print(name)
    for name, param in mpd.named_parameters():
        if not param.requires_grad:
            print(name)
    for name, param in msd.named_parameters():
        if not param.requires_grad:
            print(name)
    
    generator.train()
    mpd.train()
    msd.train()
    
    if using_spk_emo_vCLUB_mi_loss:
        spk_emo_mi_net.train()
    
    for epoch in range(max(0, last_epoch), a.training_epochs):
        lld_spk_emo_loss = None
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, filename, y_mel, emo_id = batch
            emo_id = None
            
            
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}

            y_energy = compute_energy_waveform_based(y)
            target_normalized_f0 = None
            
            
            

            # y_g_hat = generator(**new_x)
            y_g_hat, spk, emo, predicted_energy, target_energy, predicted_f0, target_f0, target_vuv, predicted_log_scale_duration, target_duration, predicted_emotion_class, attention_weights, predicted_content_class, code = generator(y_energy, target_normalized_f0, emo_id, **x)

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

            def compute_energy_batch(y_g_hat, sample_rate=16000, frame_size_ms=20):
                frame_size = int(sample_rate * frame_size_ms / 1000)
                
                batch_size, _, total_len = y_g_hat.shape
                signal = y_g_hat.squeeze(1)  # (batch, total_len)

                energies = []
                for start in range(0, total_len, frame_size):
                    end = start + frame_size
                    if end > total_len:
                        break
                    frame = signal[:, start:end]  # (batch, frame_size)
                    energy = torch.sqrt((frame ** 2).mean(dim=1)) # (batch,)
                    energies.append(energy)

                energy_tensor = torch.stack(energies, dim=1).unsqueeze(1)  # (batch, 1, frame_count)
                return energy_tensor
            y_g_hat_energy = compute_energy_batch(y_g_hat) *10          
            
            
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            if using_spk_emo_vCLUB_mi_loss:
                for i in range(5):
                    optimizer_spk_emo_mi_net, lld_spk_emo_loss = mi_first_forward(spk_emo_mi_net, optimizer_spk_emo_mi_net, spk, emo)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            
            
            # Mutual Information loss between spk and emo
            if using_KL_MI_loss:
                loss_MI = MutualInformationLoss(spk, emo)
                loss_KL_MI = KL_div_based_MutualInformationLoss(spk, emo)
                
                
            # Using vCLUB MI loss
            if using_spk_emo_vCLUB_mi_loss:
                loss_spk_emo_MI = mi_second_forward(spk_emo_mi_net, spk, emo)
            
            # Energy loss
            if using_energy_prediction_loss:
                loss_energy = F.mse_loss(energy_prediction, y_energy) * 0.1

            # normalized_f0 loss
            # valid_mask = []
            if using_pitch_prediction_loss:
                valid_mask = [sample.sum() > 0 for sample in target_normalized_f0]
                loss_normalized_f0 = F.l1_loss(predicted_f0[valid_mask], target_normalized_f0[valid_mask]) * 10
            
            
            # emotion_classifier_loss
            if using_emotion_classifier_loss:
                pred_emo = emotion_classifier(emo, "classification")
                pred = F.softmax(pred_emo, dim = 1)
                label = []
                for name in filename:
                    label.append(EMOTION[name.split("/")[-2]])
                label = torch.tensor(label).to(device)
                loss_emo_classification = torch.nn.CrossEntropyLoss()(pred, label)
            else:
                loss_emo_classification = None
            
            # get speaker embedding cosine-sim loss            
            if using_spk_cos_sim_loss:
                one_spk = spk[:, :, 0]
                loss_spk_cos_sim = F.cosine_similarity(one_spk, x['avg_neutral_speaker_embedding'].squeeze(1).squeeze(1), dim = -1)
                loss_spk_cos_sim = (1 - loss_spk_cos_sim).mean()

            # get auxiliary loss
            if using_auxiliary_loss:
                # get emotion_classification_loss from spk
                reversed_spk_emb = gradient_reversal(spk)
                pred_emo_from_spk = emotion_classifier(reversed_spk_emb, "classification")
                pred_from_spk = F.softmax(pred_emo_from_spk, dim = 1)
                label = []
                for name in filename:
                    label.append(EMOTION[name.split("/")[-2]])
                label = torch.tensor(label).to(device)
                loss_emo_classification_from_spk = torch.nn.CrossEntropyLoss()(pred_from_spk, label)

                # get spk_cos_sim_loss from emo
                loss_spk_cos_sim_from_emo = F.cosine_similarity(torch.mean(emo, dim = 2), x['avg_neutral_speaker_embedding'].squeeze(1).squeeze(1), dim = -1)
                loss_spk_cos_sim_from_emo = loss_spk_cos_sim_from_emo.abs().mean()
                auxiliary_loss = loss_emo_classification_from_spk + loss_spk_cos_sim_from_emo
            
            if using_energy_mse_loss:
                mse_loss = nn.MSELoss()
                loss_energy_mse = mse_loss(predicted_energy, target_energy)
                
            if using_f0_mse_loss:
                mask = (target_vuv == 1).float()  # 1인 부분은 1, 나머지는 0
                target_f0 = torch.nan_to_num(target_f0, nan=0.0)
                # MSE 손실 계산 시 마스킹된 부분만 계산
                mse_loss = nn.MSELoss(reduction='none')  # element-wise 손실 계산
                raw_loss = mse_loss(predicted_f0, target_f0)
                raw_loss = torch.nan_to_num(raw_loss, nan = 0.0)

                # 마스크 적용해서 필요한 부분만 loss에 반영
                masked_loss = raw_loss * mask

                # 평균을 마스크된 부분만 기준으로 계산
                if mask.sum() > 0:
                    loss_f0_mse = masked_loss.sum() / mask.sum()
                else:
                    loss_f0_mse = torch.tensor(0.0, device=predicted_f0.device)
                
            # if using_vuv_bce_loss:
            #     bce_loss = nn.BCELoss()
            #     loss_vuv_bce = bce_loss(predicted_vuv, target_vuv)
                 
            if using_y_hat_energy_mse_loss:
                mse_loss = nn.MSELoss()
                loss_y_hat_energy_mse = mse_loss(y_g_hat_energy, target_energy)
            
            if using_duration_prediction_L1_loss:
                mask = target_duration > 0
                if len(predicted_log_scale_duration.shape) > 2:
                    log_duration_prediction = predicted_log_scale_duration.squeeze(2)
                else:
                    log_duration_prediction = predicted_log_scale_duration
                log_duration = log_duration_prediction
                target_log_duration = torch.log(target_duration.float() + 1)
                
                loss_duration = F.l1_loss(log_duration, target_log_duration, reduction = "none")
                masked_loss_duration = loss_duration * mask
                loss_duration = masked_loss_duration.sum() / mask.sum()
            
            
            ### For GRL
            if using_GRL_emotion_classifier_loss:
                pred = predicted_emotion_class
                # pred = F.softmax(pred, dim = 1)
                label = []
                for name in filename:
                    label.append(EMOTION[name.split("/")[-2]])
                label = torch.tensor(label).to(device)
                loss_GRL_emo_classification = torch.nn.CrossEntropyLoss()(pred, label)
            
             ### For content GRL
            if using_GRL_content_classifier_loss:
                pred = predicted_content_class # batch, num_classes, seq_len
                # pred = F.softmax(pred, dim = 1)
                original_code = code[:, ::2] # batch, seq_len
                label = original_code
                loss_GRL_content_classification = torch.nn.CrossEntropyLoss()(pred, label)
            
            if using_spk_triplet_loss:
                generated_spk_emb = spk[:,:,0]
                spk_id = x["spkr"]
                anchor, positive, negative = build_speaker_triplets(generated_spk_emb, spk_id)
                loss_spk_triplet = triplet_loss(anchor, positive, negative)
                
            
            # generated speaker loss
            if using_generated_speaker_mse_loss:
                speaker_encoder.eval()
                generated_speaker_emb = []
                for i in y_g_hat:
                    gen_speaker_emb = speaker_encoder.encode_batch(i)
                    generated_speaker_emb.append(gen_speaker_emb)
                generated_speaker_emb = torch.stack(generated_speaker_emb).to(device)
                target_speaker_embedding = x["speaker_embedding"]
                generated_speaker_mse_loss = F.mse_loss(generated_speaker_emb, target_speaker_embedding) * 0.01
                
            
            # generated emotion loss
            if using_generated_emotion_classifier_loss:
                generated_emotion_embedding = []
                feature_lens = []
                for i in y_g_hat:
                    audios = WavLM_feature_extractor(i.squeeze(0), sampling_rate = 16000, return_tensors = "pt", padding = True).to(device)
                    output = WavLM_model(**audios, output_hidden_states = True)
                    hidden_states = output.hidden_states
                    stacked_outputs = []
                    for hidden in hidden_states[1:]:
                        stacked_outputs.append(hidden)
                    generated_emotion_embedding.append(torch.stack(stacked_outputs, dim = 0))
                    feature_lens.append(generated_emotion_embedding[0].shape[-2])
                # generated_emotion_embedding shape => (batch_size, 24, 1, length, 1024)
                generated_emotion_embedding = torch.stack(generated_emotion_embedding).squeeze(2).to(device) # (batch_size, 24, length, 1024)
                pred, final_emb = emotion_classifier(generated_emotion_embedding, feature_lens)
                pred = F.softmax(pred, dim = 1)
                label = []
                for name in filename:
                    label.append(EMOTION[name.split("/")[-2]])
                label = torch.tensor(label).to(device)
                generated_emotion_cls_loss = torch.nn.CrossEntropyLoss()(pred, label)
                


            
            # if valid_mask.any():
            #     loss_normalized_f0 = F.l1_loss(pitch_prediction[valid_mask], target_normalized_f0[valid_mask]) * 10
            # else:
            #     loss_normalized_f0 = F.l1_loss(pitch_prediction, target_normalized_f0) * 10
            
            
            ###########################################################################################################################################################
            # # speaker embedding loss
            # loss_speaker_embedding = F.l1_loss(generated_speaker_embedding, target_speaker_embedding) * 0.1
            # loss_speaker_embedding = F.cosine_similarity(generated_speaker_embedding.squeeze(1).squeeze(1), target_speaker_embedding.squeeze(1).squeeze(1), dim = -1)
            # loss_speaker_embedding = (1 - loss_speaker_embedding).mean() # 0.5정도?
            
            # # emotion embedding loss
            # if generated_emotion_embedding.shape[1] != target_emotion_embedding.shape[1]:
            #     if generated_emotion_embedding.shape[1] > target_emotion_embedding.shape[1]:
            #         generated_emotion_embedding = generated_emotion_embedding[:, :-1, :]
            #     else:
            #         target_emotion_embedding = target_emotion_embedding[:, :-1, :]
            # # loss_emotion_embedding = F.l1_loss(generated_emotion_embedding, target_emotion_embedding) * 0.1
            # loss_emotion_embedding = F.cosine_similarity(generated_emotion_embedding, target_emotion_embedding, dim = -1)
            # loss_emotion_embedding = (1 - loss_emotion_embedding).mean() # 0.3정도?
            ###########################################################################################################################################################            
            
            # attention_weight loss
            # if using_attention_weight_loss:
            #     diag_elements = torch.diagonal(attention_weight, dim1 = -1, dim2 = -2)
            #     loss_attention_weight = F.l1_loss(diag_elements, torch.ones_like(diag_elements))
            # else:
            #     loss_attention_weight = torch.tensor(0.0).to(device)
            
            
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            
            
            
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            
            
            

            # using nothing
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel     
            loss_components = [
                        ("Gen Loss Total", loss_gen_all),
                        ("loss_gen_s", loss_gen_s),
                        ("loss_gen_f", loss_gen_f),
                        ("loss_fm_s", loss_fm_s),
                        ("loss_fm_f", loss_fm_f),
                        ("Mel-Spec. Error / 45", loss_mel / 45),
                    ]
            # using KL_MI loss
            if using_KL_MI_loss:
                loss_gen_all += loss_KL_MI
                loss_components.append(("KL_MI_Loss", loss_KL_MI))
            # using vCLUB MI loss
            if using_spk_emo_vCLUB_mi_loss:
                loss_gen_all += loss_spk_emo_MI
                loss_components.append(("vCLUB_spk_emo_MI_Loss", loss_spk_emo_MI))
            # using emotion classifier loss
            if using_emotion_classifier_loss:
                loss_gen_all += loss_emo_classification
                loss_components.append(("Emotion_Classification_Loss", loss_emo_classification))
            if using_energy_prediction_loss:
                loss_gen_all += loss_energy
                loss_components.append(("Energy_Loss", loss_energy))
            if using_pitch_prediction_loss:
                loss_gen_all += loss_normalized_f0
                loss_components.append(("Normalized_f0_Loss", loss_normalized_f0))
            # if using_attention_weight_loss:
            #     loss_gen_all += loss_attention_weight
            #     loss_components.append(("Attention_Weight_Loss", loss_attention_weight))
            if using_spk_cos_sim_loss:
                loss_gen_all += loss_spk_cos_sim
                loss_components.append(("Speaker_Cosine_Similarity_Loss", loss_spk_cos_sim))
            if using_auxiliary_loss:
                loss_gen_all += auxiliary_loss
                loss_components.append(("Emo_classification_from_spk_Loss", loss_emo_classification_from_spk))
                loss_components.append(("Spk_cos_sim_with_emo", loss_spk_cos_sim_from_emo))
                loss_components.append(("Auxiliary_Loss", auxiliary_loss))
                
            if using_energy_mse_loss:
                loss_gen_all += loss_energy_mse
                loss_components.append(("loss_energy_mse", loss_energy_mse))
                
            if using_f0_mse_loss:
                loss_gen_all += loss_f0_mse
                loss_components.append(("loss_f0_mse", loss_f0_mse))
                
            # if using_vuv_bce_loss:
            #     loss_gen_all += loss_vuv_bce
            #     loss_components.append(("loss_vuv_bce", loss_vuv_bce))
                
            if using_y_hat_energy_mse_loss:
                loss_gen_all += loss_y_hat_energy_mse
                loss_components.append(("loss_y_hat_energy_mse", loss_y_hat_energy_mse))
            
            if using_duration_prediction_L1_loss:
                loss_gen_all += loss_duration
                loss_components.append(("loss_duration", loss_duration))
            
            if using_GRL_emotion_classifier_loss:
                loss_gen_all += loss_GRL_emo_classification
                loss_components.append(("loss_GRL_emotion_classifier", loss_GRL_emo_classification))
            
            if using_GRL_content_classifier_loss:
                loss_gen_all += loss_GRL_content_classification
                loss_components.append(("loss_GRL_content_classifier", loss_GRL_content_classification))
            
            if using_spk_triplet_loss:
                loss_gen_all += loss_spk_triplet
                loss_components.append(("loss_spk_triplet", loss_spk_triplet))
                
            
            if using_generated_speaker_mse_loss:
                loss_gen_all += generated_speaker_mse_loss
                loss_components.append(("loss_generated_speaker_mse", generated_speaker_mse_loss))
            
            if using_generated_emotion_classifier_loss:
                loss_gen_all += generated_emotion_cls_loss
                loss_components.append(("loss_generated_emotion_cls", generated_emotion_cls_loss))
            
            for name, value in loss_components:
                writer.add_scalar(f"Train/{name}", value.item(), steps)
                writer.flush()
                
            
            # using attention_weight_loss
            # loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_attention_weight
            
            
            # using KL_based MI loss
            # if not energy_prediction is None and not pitch_prediction is None:
            #     loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_KL_MI + loss_energy + loss_normalized_f0 + loss_speaker_embedding
            # else:
            #     loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_KL_MI + loss_speaker_embedding
                        
                            
            
            loss_gen_all.backward()
            # loss.backward() 호출 후 실행
            for name, param in generator.named_parameters():
                if param.grad is None:
                    if "_bins" in name:
                        continue
                    elif "neutral_spk_encoder" in name:
                        continue
                    else:
                        print(f"WARNING: {name} has no gradient!")

            for name, param in mpd.named_parameters():
                if param.grad is None:
                    print(f"WARNING: {name} has no gradient!")

            for name, param in msd.named_parameters():
                if param.grad is None:
                    print(f"WARNING: {name} has no gradient!")

            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print_message = "Steps : {:d}".format(steps)
                    for name, value in loss_components:
                        print_message += ", {}: {:4.5f}".format(name, value)
                    print_message += ", s/b, : {:4.5f}".format(time.time() - start_b)
                    print(print_message)
                
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                      'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                      'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                      'steps': steps, 'epoch': epoch})
                    if using_spk_emo_vCLUB_mi_loss:
                        checkpoint_path = "{}/spk_emo_mi_net_{:08d}".format(a.checkpoint_path, steps)
                        save_checkpoint(checkpoint_path, {'spk_emo_mi_net': (spk_emo_mi_net.module if h.num_gpus > 1 else spk_emo_mi_net).state_dict()})
                    
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/gen_loss_s", loss_gen_s, steps)
                    sw.add_scalar("training/gen_loss_f", loss_gen_f, steps)
                    sw.add_scalar("training/feature_matching_s", loss_fm_s, steps)
                    sw.add_scalar("training/feature_matching_f", loss_fm_f, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    if using_KL_MI_loss:
                        sw.add_scalar("training/kl_mi_loss", loss_KL_MI, steps)
                    if using_attention_weight_loss:
                        sw.add_scalar("training/attention_weight_loss", loss_attention_weight, steps)
                    if using_energy_prediction_loss:
                        sw.add_scalar("training/energy_loss", loss_energy, steps)
                    if using_pitch_prediction_loss:
                        sw.add_scalar("training/normalized_f0_loss", loss_normalized_f0, steps)
                    if using_duration_prediction_L1_loss:
                        sw.add_scalar("training/log_duration_prediction_loss", loss_duration, steps)
                    if using_emotion_classifier_loss:
                        sw.add_scalar("training/emotion_classification_loss", loss_emo_classification, steps)
                    if using_spk_emo_vCLUB_mi_loss:
                        sw.add_scalar("training/spk_emo_MI_loss", loss_spk_emo_MI, steps)
                    if using_spk_cos_sim_loss:
                        sw.add_scalar("training/speaker_cosine_similarity_loss", loss_spk_cos_sim, steps)
                    if using_auxiliary_loss:
                        sw.add_scalar("training/emo_classification_from_spk_loss", loss_emo_classification_from_spk, steps)
                        sw.add_scalar("training/spk_cos_sim_with_emo", loss_spk_cos_sim_from_emo, steps)
                        sw.add_scalar("training/auxiliary_loss", auxiliary_loss, steps)
                    
                    if using_GRL_emotion_classifier_loss:
                        sw.add_scalar("training/GRL_emotion_classifier_loss", loss_GRL_emo_classification, steps)
                    
                    if using_GRL_content_classifier_loss:
                        sw.add_scalar("training/GRL_content_classifier_loss", loss_GRL_content_classification, steps)
                    
                    if using_spk_triplet_loss:
                        sw.add_scalar("training/spk_triplet_loss", loss_spk_triplet, steps)
                    
                    if using_generated_speaker_mse_loss:
                        sw.add_scalar("training/generated_speaker_mse_loss", generated_speaker_mse_loss, steps)
                    if using_generated_emotion_classifier_loss:
                        sw.add_scalar("training/generated_emotion_cls_loss", generated_emotion_cls_loss, steps)
                        
                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    if using_spk_emo_vCLUB_mi_loss:
                        spk_emo_mi_net.eval()
                    torch.cuda.empty_cache()
                    val_KL_MI_Loss_tot = 0
                    val_energy_loss_tot = 0
                    val_normalized_f0_tot = 0
                    val_err_tot = 0
                    val_attention_weight_loss = 0
                    val_emo_classification_loss = 0
                    val_spk_emo_MI_tot = 0
                    val_spk_emo_loglilkeli_tot = 0
                    val_spk_cos_sim_loss_tot = 0
                    val_auxiliary_loss_tot = 0
                    val_emo_classification_from_spk_tot = 0
                    val_spk_cos_sim_from_emo_tot = 0
                    
                    val_duration_loss_tot = 0
                    val_f0_mse_loss_tot = 0
                    val_energy_mse_loss_tot = 0
                    
                    val_GRL_emo_classification_tot = 0
                    val_GRL_content_classification_tot = 0
                    
                    val_generated_speaker_mse_loss_tot = 0
                    val_generated_emotion_cls_loss_tot = 0
                    
                    val_spk_triplet_loss_tot = 0
                    
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, filename, y_mel, emo_id = batch
                            emo_id = None
                            
                            y = y.unsqueeze(1)
                            x = {k: v.to(device, non_blocking=False) for k, v in x.items()}
                            target_energy = compute_energy_waveform_based(y).to(device)
                            

                            y_g_hat, spk, emo, predicted_energy, target_energy, predicted_f0, target_f0, target_vuv, predicted_log_scale_duration, target_duration, predicted_emotion_class, attention_weights, predicted_content_class, code = generator(target_energy, None, emo_id, **x)
                            
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            # y_energy = compute_energy_hubert_aligned(y_mel_for_energy, x['code'].size(1)).to(device)
                            y_energy = compute_energy_waveform_based(y)
                            
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if using_KL_MI_loss:
                                # val_MI_loss_tot += MutualInformationLoss(spk, emo)
                                val_KL_MI_Loss_tot += KL_div_based_MutualInformationLoss(spk, emo)
                            
                            if using_energy_prediction_loss:
                                val_energy_loss_tot += F.mse_loss(energy_prediction, y_energy) * 0.1
                            
                            if using_pitch_prediction_loss:
                                loss_normalized_f0 += F.l1_loss(predicted_f0[valid_mask], target_normalized_f0[valid_mask]) * 10                       
                            
                            if using_f0_mse_loss:
                                mask = (target_vuv == 1).float()  # 1인 부분은 1, 나머지는 0
                                target_f0 = torch.nan_to_num(target_f0, nan=0.0)
                                # MSE 손실 계산 시 마스킹된 부분만 계산
                                mse_loss = nn.MSELoss(reduction='none')  # element-wise 손실 계산
                                raw_loss = mse_loss(predicted_f0, target_f0)
                                raw_loss = torch.nan_to_num(raw_loss, nan = 0.0)

                                # 마스크 적용해서 필요한 부분만 loss에 반영
                                masked_loss = raw_loss * mask

                                # 평균을 마스크된 부분만 기준으로 계산
                                if mask.sum() > 0:
                                    loss_f0_mse = masked_loss.sum() / mask.sum()
                                else:
                                    loss_f0_mse = torch.tensor(0.0, device=predicted_f0.device)
                                val_f0_mse_loss_tot += loss_f0_mse
                                
                            if using_y_hat_energy_mse_loss:
                                mse_loss = nn.MSELoss()
                                loss_y_hat_energy_mse = mse_loss(y_g_hat_energy, target_energy)
                                val_energy_mse_loss_tot += loss_y_hat_energy_mse
                                
                            if using_duration_prediction_L1_loss:
                                mask = target_duration > 0
                                if len(predicted_log_scale_duration.shape) > 2:
                                    log_duration_prediction = predicted_log_scale_duration.squeeze(2)
                                else:
                                    log_duration_prediction = predicted_log_scale_duration
                                log_duration = log_duration_prediction
                                target_log_duration = torch.log(target_duration.float() + 1)
                                
                                loss_duration = F.l1_loss(log_duration, target_log_duration, reduction = "none")
                                masked_loss_duration = loss_duration * mask
                                val_loss_duration = masked_loss_duration.sum() / mask.sum()
                                val_duration_loss_tot += val_loss_duration
                                
                            if using_attention_weight_loss:
                                # diag_elements = torch.diagonal(attention_weight, dim1 = -1, dim2 = -2)
                                val_attention_weight_loss += F.l1_loss(diag_elements, torch.ones_like(diag_elements))
                            
                            if using_emotion_classifier_loss:
                                pred_emo = emotion_classifier(emo, "classification")
                                pred = F.softmax(pred_emo, dim = 1)
                                label = []
                                for name in filename:
                                    label.append(EMOTION[name.split("/")[-2]])
                                label = torch.tensor(label).to(device)
                                val_emo_classification_loss += torch.nn.CrossEntropyLoss()(pred, label)
                            
                            if using_spk_emo_vCLUB_mi_loss:
                                val_spk_emo_loglilkeli, val_spk_emo_MI = mi_eval_forward(spk_emo_mi_net, spk, emo)
                                val_spk_emo_MI_tot+= val_spk_emo_MI
                                val_spk_emo_loglilkeli_tot+= val_spk_emo_loglilkeli
                            
                            if using_spk_cos_sim_loss:
                                one_spk = spk[:, :, 0]
                                val_spk_cos_sim_loss = F.cosine_similarity(one_spk, x['avg_neutral_speaker_embedding'].squeeze(1).squeeze(1), dim = -1)
                                val_spk_cos_sim_loss = (1 - val_spk_cos_sim_loss).mean()
                                val_spk_cos_sim_loss_tot += val_spk_cos_sim_loss
                            
                            if using_auxiliary_loss:
                                # get emotion_classification_loss from spk
                                pred_emo_from_spk = emotion_classifier(spk, "classification")
                                pred_from_spk = F.softmax(pred_emo_from_spk, dim = 1)
                                label = []
                                for name in filename:
                                    label.append(EMOTION[name.split("/")[-2]])
                                label = torch.tensor(label).to(device)
                                loss_emo_classification_from_spk = torch.nn.CrossEntropyLoss()(pred_from_spk, label)
                                val_emo_classification_from_spk_tot += loss_emo_classification_from_spk
                                
                                # get spk_cos_sim_loss from emo
                                loss_spk_cos_sim_from_emo = F.cosine_similarity(torch.mean(emo, dim = 2), x['avg_neutral_speaker_embedding'].squeeze(1).squeeze(1), dim = -1)
                                loss_spk_cos_sim_from_emo = loss_spk_cos_sim_from_emo.abs().mean()
                                val_spk_cos_sim_from_emo_tot += loss_spk_cos_sim_from_emo
                                auxiliary_loss = loss_emo_classification_from_spk + loss_spk_cos_sim_from_emo
                                val_auxiliary_loss_tot += auxiliary_loss
                            
                            if using_GRL_emotion_classifier_loss:
                                pred = predicted_emotion_class
                                # pred = F.softmax(pred, dim = 1)
                                label = []
                                for name in filename:
                                    label.append(EMOTION[name.split("/")[-2]])
                                label = torch.tensor(label).to(device)
                                val_GRL_emo_classification_tot += torch.nn.CrossEntropyLoss()(pred, label)

                            if using_GRL_content_classifier_loss:
                                pred = predicted_content_class
                                # pred = F.softmax(pred, dim = 1)
                                original_code = code[:, ::2] # batch, seq_len
                                label = original_code
                                val_GRL_content_classification_tot += torch.nn.CrossEntropyLoss()(pred, label)
                            
                            if using_spk_triplet_loss:
                                generated_spk_emb = spk[:,:,0]
                                spk_id = x["spkr"]
                                anchor, positive, negative = build_speaker_triplets(generated_spk_emb, spk_id)
                                val_spk_triplet_loss_tot += triplet_loss(anchor, positive, negative)
                            
                            
                            if using_generated_speaker_mse_loss:
                                speaker_encoder.eval()
                                generated_speaker_emb = []
                                for i in y_g_hat:
                                    gen_speaker_emb = speaker_encoder.encode_batch(i)
                                    generated_speaker_emb.append(gen_speaker_emb)
                                generated_speaker_emb = torch.stack(generated_speaker_emb).to(device)
                                target_speaker_embedding = x["speaker_embedding"]
                                generated_speaker_mse_loss = F.mse_loss(generated_speaker_emb, target_speaker_embedding)* 0.01
                                val_generated_speaker_mse_loss_tot += generated_speaker_mse_loss
                                
                            if using_generated_emotion_classifier_loss:
                                generated_emotion_embedding = []
                                feature_lens = []
                                for i in y_g_hat:
                                    audios = WavLM_feature_extractor(i.squeeze(0), sampling_rate = 16000, return_tensors = "pt", padding = True).to(device)
                                    output = WavLM_model(**audios, output_hidden_states = True)
                                    hidden_states = output.hidden_states
                                    stacked_outputs = []
                                    for hidden in hidden_states[1:]:
                                        stacked_outputs.append(hidden)
                                    generated_emotion_embedding.append(torch.stack(stacked_outputs, dim = 0))
                                    feature_lens.append(generated_emotion_embedding[0].shape[-2])
                                # generated_emotion_embedding shape => (batch_size, 24, 1, length, 1024)
                                generated_emotion_embedding = torch.stack(generated_emotion_embedding).squeeze(2).to(device) # (batch_size, 24, length, 1024)
                                pred, final_emb = emotion_classifier(generated_emotion_embedding, feature_lens)
                                pred = F.softmax(pred, dim = 1)
                                label = []
                                for name in filename:
                                    label.append(EMOTION[name.split("/")[-2]])
                                label = torch.tensor(label).to(device)
                                generated_emotion_cls_loss = torch.nn.CrossEntropyLoss()(pred, label)
                                val_generated_emotion_cls_loss_tot += generated_emotion_cls_loss
                            
                            
                                                       
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        if using_KL_MI_loss:
                            val_KL_MI_Loss = val_KL_MI_Loss_tot / (j + 1)
                        if using_energy_prediction_loss:
                            val_energy_loss = val_energy_loss_tot / (j + 1)
                        if using_pitch_prediction_loss:
                            val_normalized_f0 = val_normalized_f0_tot / (j + 1)
                        if using_duration_prediction_L1_loss:
                            val_duration_loss = val_duration_loss_tot / (j + 1)
                        if using_attention_weight_loss:
                            val_attention_weight_loss = val_attention_weight_loss / (j + 1)
                        if using_emotion_classifier_loss:
                            val_emo_classification_loss = val_emo_classification_loss / (j + 1)
                        if using_spk_emo_vCLUB_mi_loss:
                            val_spk_emo_MI = val_spk_emo_MI_tot / (j + 1)
                            val_spk_emo_loglilkeli = val_spk_emo_loglilkeli_tot / (j + 1)
                        if using_spk_cos_sim_loss:
                            val_spk_cos_sim_loss = val_spk_cos_sim_loss_tot / (j + 1)
                        if using_auxiliary_loss:
                            val_auxiliary_loss = val_auxiliary_loss_tot / (j + 1)
                        
                        if using_GRL_emotion_classifier_loss:
                            val_GRL_emo_classification_loss = val_GRL_emo_classification_tot / (j + 1)

                        if using_GRL_content_classifier_loss:
                            val_GRL_content_classification_loss = val_GRL_content_classification_tot / (j + 1)
                        
                        if using_spk_triplet_loss:
                            val_spk_triplet_loss = val_spk_triplet_loss_tot / (j + 1)
                            
                        if using_generated_speaker_mse_loss:
                            val_generated_speaker_mse_loss = val_generated_speaker_mse_loss_tot / (j + 1)
                            
                        if using_generated_emotion_classifier_loss:
                            val_generated_emotion_cls_loss = val_generated_emotion_cls_loss_tot / (j + 1)

                        
                        
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        if using_KL_MI_loss:
                            sw.add_scalar("Validation/kl_mi_loss", val_KL_MI_Loss, steps)
                        if using_energy_prediction_loss:
                            sw.add_scalar("Validation/energy_loss", val_energy_loss, steps)
                        if using_pitch_prediction_loss:
                            sw.add_scalar("Validation/normalized_f0_loss", val_normalized_f0, steps)
                        if using_duration_prediction_L1_loss:
                            sw.add_scalar("Validation/log_duration_prediction_loss", val_duration_loss, steps)
                        if using_attention_weight_loss:
                            sw.add_scalar("Validation/attention_weight_loss", val_attention_weight_loss, steps)
                        if using_emotion_classifier_loss:
                            sw.add_scalar("Validation/emotion_classification_loss", val_emo_classification_loss, steps)
                        if using_spk_emo_vCLUB_mi_loss:
                            sw.add_scalar("Validation/spk_emo_MI_loss", val_spk_emo_MI, steps)
                            sw.add_scalar("Validation/spk_emo_loglilkelihood", val_spk_emo_loglilkeli, steps)
                        if using_spk_cos_sim_loss:
                            sw.add_scalar("Validation/speaker_cosine_similarity_loss", val_spk_cos_sim_loss, steps)
                        if using_auxiliary_loss:
                            sw.add_scalar("Validation/emo_classification_from_spk_loss", loss_emo_classification_from_spk, steps)
                            sw.add_scalar("Validation/spk_cos_sim_with_emo", loss_spk_cos_sim_from_emo, steps)
                            sw.add_scalar("Validation/auxiliary_loss", val_auxiliary_loss, steps)
                        
                        if using_GRL_emotion_classifier_loss:
                            sw.add_scalar("Validation/GRL_emotion_classifier_loss", val_GRL_emo_classification_loss, steps)
                        if using_GRL_content_classifier_loss:
                            sw.add_scalar("Validation/GRL_content_classifier_loss", val_GRL_content_classification_loss, steps)
                        
                        if using_spk_triplet_loss:
                            sw.add_scalar("Validation/spk_triplet_loss", val_spk_triplet_loss, steps)
                        
                        if using_generated_speaker_mse_loss:
                            sw.add_scalar("Validation/loss_generated_speaker_mse", val_generated_speaker_mse_loss, steps)
                        if using_generated_emotion_classifier_loss:
                            sw.add_scalar("Validation/loss_generated_emotion_cls", val_generated_emotion_cls_loss, steps)
                            
                    generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')

def build_speaker_triplets(embeddings, speaker_labels):
    """
    embeddings: (B, D) speaker embedding
    speaker_labels: (B,)
    """
    anchors, positives, negatives = [], [], []
    B = embeddings.size(0)
    speaker_labels = speaker_labels.squeeze(1)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            if speaker_labels[i] == speaker_labels[j]:  # positive
                # pick a negative
                neg_indices = (speaker_labels != speaker_labels[i]).nonzero(as_tuple=False).squeeze()
                if len(neg_indices) == 0:
                    continue
                k = neg_indices[torch.randint(0, len(neg_indices), (1,))]

                anchors.append(embeddings[i])
                positives.append(embeddings[j])
                negatives.append(embeddings[k].squeeze(0))

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def triplet_loss(anchor, positive, negative, margin=0.3):
    if anchor is None:
        return torch.tensor(0.0, device = positive.device, requires_grad=True)
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    
    # dist_ap = F.pairwise_distance(anchor, positive, p=2)  # L2 거리
    # dist_an = F.pairwise_distance(anchor, negative, p=2)
    # loss = F.relu(dist_ap - dist_an + margin)
    # return loss.mean()

    sim_ap = F.cosine_similarity(anchor, positive, dim=1)  # → 1이면 완전 같은 방향
    sim_an = F.cosine_similarity(anchor, negative, dim=1)

    loss = F.relu(sim_an - sim_ap + margin).mean()
    return loss
    
   

def main():
    print('Initializing Training Process..')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='./runs/checkpoints')
    parser.add_argument('--config', default='./configs/config.json')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--training_steps', default=600000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    os.makedirs(a.checkpoint_path,exist_ok=True)

    config_name = a.config.split('/')[-1]

    build_env(a.config, config_name, a.checkpoint_path)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = int(os.environ['LOCAL_RANK'])  # Automatically set by the torch.distributed.run
        rank = local_rank
        print('Batch size per GPU:', h.batch_size)
    else:
        rank = 0
        local_rank = 0
        h.num_gpus = 1

    # Print number of GPUs being used
    print(f"Number of GPUs used: {h.num_gpus}")

    train(rank, local_rank, a, h)


if __name__ == '__main__':
    main()
    