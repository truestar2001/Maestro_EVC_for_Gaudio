# adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
from scipy.interpolate import interp1d
import math
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random
from pytorch_revgrad import RevGrad
import attentions

LRELU_SLOPE = 0.1

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                        padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                   padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                       padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Emotion_diarization_single_Content_new_CrossAttention_relative_positional_encoding(nn.Module):
    def __init__(self, embed_dim, target_embed, emo_dim, num_head = 4, dropout = 0.1, max_relative_position = 64):
        super(Emotion_diarization_single_Content_new_CrossAttention_relative_positional_encoding, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_relative_position = max_relative_position
        
        # Preprocessing layers
        self.pre_emo = nn.Sequential(
            nn.Conv1d(emo_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

        self.pre_content = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

        # Relative position embedding: [-max_pos, ... , +max_pos] -> index 0~2*max_pos
        self.rel_pos_embedding = nn.Embedding(2 * max_relative_position + 1, embed_dim)
        
    def get_relative_position_index(self, len_q, len_k, device):
        """
        Create a (len_q, len_k) matrix where each entry is (i - j) clipped and shifted to 0~2*max
        """
        
        q_ids = torch.arange(len_q, device = device).unsqueeze(1) # (len_q, 1)
        k_ids = torch.arange(len_k, device = device).unsqueeze(0) # (1, len_k)
        rel_pos = q_ids - k_ids # (len_q, len_k)
        rel_pos = rel_pos.clamp(-self.max_relative_position, self.max_relative_position) # (len_q, len_k)
        rel_pos = rel_pos + self.max_relative_position # (len_q, len_k)
        
        return rel_pos

    def forward(self, content_emb, emotion_emb, padding_mask = None):
        """
        content_emb: (B, D, T_c)
        emotion_emb: (B, D, T_e)
        """

        B, _, T_c = content_emb.shape
        _, _, T_e = emotion_emb.shape
        
        # Preprocess content and emotion embeddings
        pre_content = self.pre_content(content_emb) # (B, D, T_c), train때는 T_c가 56
        pre_emotion = self.pre_emo(emotion_emb) # (B, D, T_e)
        
        # Transpose for batch matmul:(B, T, D)
        q = pre_content.transpose(1, 2) # (B, T_c, D)
        k = pre_emotion.transpose(1, 2) # (B, T_e, D)
        
        # Dot-product attention base score
        sim = torch.bmm(q, k.transpose(1, 2)) # (B, T_c, T_e)
        
        # Relative positional bias
        rel_index = self.get_relative_position_index(T_c, T_e, device = content_emb.device) # (T_c, T_e)
        rel_bias = self.rel_pos_embedding(rel_index) # (T_c, T_e, D)
        rel_score = torch.einsum('bqd,qkd->bqk', q, rel_bias)
        
        # Final attention score
        sim = sim + rel_score
        
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(-1, T_c, -1) # (B, T_c, T_e)
            sim = sim.masked_fill(mask == 0, float("-inf")) # (B, T_c, T_e)
        
        # Normalize attention
        attn_weights = F.softmax(sim / math.sqrt(self.embed_dim), dim = -1) # (B, T_c, T_e)
        
        return attn_weights # (B, T_c, T_e)
        
        

class Emotion_diarization_single_Content_new_CrossAttetion(nn.Module):
    def __init__(self, embed_dim, target_embed, emo_dim, num_head = 4, dropout = 0.1):
        """
        각 content embedding들과 emotion embedding sequence 사이 similarity를 기준으로
        weight를 구한 다음
        weight와 emotion embedding을 곱하여 content embedding에 더해 
        post content embedding을 생성
        """
        super(Emotion_diarization_single_Content_new_CrossAttetion, self).__init__()
                
                
        # self.positional_encoding = sinusoidal_positional_encoding(256, max_len = 5000)
        # preprocessing
        # self.prepre_emo = nn.Conv1d(1024, embed_dim, kernel_size = 3, stride = 1, padding = 1)
        
        # weignt norm ver
        self.pre_emo = weight_norm(nn.Conv1d(1024, embed_dim, kernel_size = 3, stride = 1, padding = 1))
        self.pre_content = weight_norm(nn.Conv1d(embed_dim, embed_dim, kernel_size = 3, stride = 1, padding = 1))
        
        # non weight norm ver
        # self.pre_emo = nn.Conv1d(1024, embed_dim, kernel_size = 3, stride = 1, padding = 1)
        # self.pre_content = nn.Conv1d(embed_dim, embed_dim, kernel_size = 3, stride = 1, padding = 1)
        
       
        # self.post_conv = nn.Conv1d(embed_dim * 2, target_embed, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, content_emb, emotion_emb, padding_mask = None, mode = "dot_product"):
        # content_emb = (batch_size, 256, length), emotion_emb = (batch_size, 256, length)
        
        # content_emb = torch.permute(content_emb, (0, 2, 1)) # batch, length, 256
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, length, 256
        # pre_emotion_emb = self.pre_emo(emotion_emb) # batch, length, 256
        
        # preprocessing
        pre_content_emb = self.pre_content(content_emb) # batch, length, 256
        pre_emotion_emb = self.pre_emo(emotion_emb) # batch, length, 256
        
        if mode == "cos_sim":
            if not padding_mask is None:
                pre_emotion_emb = pre_emotion_emb * padding_mask.unsqueeze(1)
            content_norm = torch.norm(pre_content_emb, p=2, dim = 1, keepdim = True) # batch, length, 1
            emotion_norm = torch.norm(pre_emotion_emb, p=2, dim = 1, keepdim = True) # batch, length, 1
            pre_content_emb_norm = pre_content_emb / (content_norm + 1e-8)
            pre_emotion_emb_norm = pre_emotion_emb / (emotion_norm + 1e-8)
            
            sim = torch.bmm(pre_content_emb_norm.transpose(1, 2), pre_emotion_emb_norm) # batch, length, length
            
            if not padding_mask is None:
                padding_mask = padding_mask.unsqueeze(1)
                mask = padding_mask.repeat(1, sim.shape[1], 1)
                sim = sim.masked_fill(mask == 0, float("-inf"))
            attn_weights = F.softmax(sim / torch.sqrt(torch.tensor(256.0)), dim = -1)
            
        else:       
            # calculate similarity
            sim = torch.bmm(pre_content_emb.transpose(1, 2), pre_emotion_emb) # batch, length, length
            
            # Masking
            if not padding_mask is None:
                padding_mask = padding_mask.unsqueeze(1)
                mask = padding_mask.repeat(1, sim.shape[1], 1)
                sim = sim.masked_fill(mask == 0, float("-inf"))
            
            
            # calculate attention weight
            attn_weights = F.softmax(sim / torch.sqrt(torch.tensor(256.0)), dim = -1) # batch, length, length
        
        return attn_weights




class Emotion_diarization_single_ContentCrossAttetion(nn.Module):
    def __init__(self, embed_dim, target_embed, emo_dim, num_head = 4, dropout = 0.1):
        """
        각 content embedding들과 emotion embedding sequence 사이 similarity를 기준으로
        weight를 구한 다음
        weight와 emotion embedding을 곱하여 content embedding에 더해 
        post content embedding을 생성
        """
        super(Emotion_diarization_single_ContentCrossAttetion, self).__init__()
       
        # attention_weight
        # self.attention_weight = nn.Parameter(torch.tensor(0.8)) # 초기값 0.8
        
        # Feed Forward Layer (변환된 Content Embedding을 교정)
        self.post_conv = nn.Conv1d(embed_dim * 2, target_embed, kernel_size = 3, stride = 1, padding = 1)
        
        # Layer Normalization
        # self.layer_norm = nn.LayerNorm(target_embed)
    
    def forward(self, content_emb, emotion_emb):
        # content_emb = (batch_size, 256, length), emotion_emb = (batch_size, 256, length)
        
        content_emb = torch.permute(content_emb, (0, 2, 1)) # batch, length, 256
        emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, length, 256
        
        # calculate similarity
        sim = torch.bmm(content_emb, emotion_emb.transpose(1, 2)) # batch, length, length
        
        # clculate attention weight
        attn_weights = F.softmax(sim / torch.sqrt(torch.tensor(256.0)), dim = -1) # batch, length, length
        weighted_emo = torch.bmm(attn_weights, emotion_emb) # batch, length, 256
        transformed_content = torch.concat([content_emb, weighted_emo], dim=-1) # batch, 512, length
        transformed_content = torch.permute(transformed_content, (0, 2, 1)) # batch, length, 512
        transformed_content = self.post_conv(transformed_content) # batch, 256, length
        
        # transformed_content = content_emb + transformed_content # batch, 256, length

        return transformed_content
            
class EmotionContentCrossAttetion(nn.Module):
    def __init__(self, embed_dim, target_embed, emo_dim, num_head = 4, dropout = 0.1):
        """
        Cross-Attention을 사용하여 Emotion Embedding을 기반으로 Content Embedding을 변환
        Args:
        - embed_dim: int, embedding dimension
        - num_head: int, number of heads in multi-head attention
        - dropout: float, dropout rate 
        """
        super(EmotionContentCrossAttetion, self).__init__()
        self.emotion_projection = nn.Linear(emo_dim, embed_dim)
        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_head, dropout = dropout, batch_first = True)
        
        # attention_weight
        # self.attention_weight = nn.Parameter(torch.tensor(0.8)) # 초기값 0.8
        
        # Feed Forward Layer (변환된 Content Embedding을 교정)
        self.ffn = nn.Sequential(nn.Linear(embed_dim*2, target_embed))
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(target_embed)
    
    def forward(self, content_emb, emotion_emb):
        ##########################################################
        # query로 emo를 쓸지, key & value로 emo를 쓸 지 애매하네...
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1))
        # emotion_emb = self.emotion_projection(emotion_emb)
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1))
        
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1))
        # content_emb = torch.permute(content_emb, (0, 2, 1))
        # attention_output, _ = self.cross_attention(query = emotion_emb, key = content_emb, value = content_emb) # query로 emo 사용
        
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1))
        # content_emb = torch.permute(content_emb, (0, 2, 1))
        # attention_output = torch.permute(attention_output, (0, 2, 1))
        # # transformed_content = self.emotion_weight * attended_content + (1 - self.emotion_weight) * content_emb # query를 content로 사용한 경우
        
        # transformed_content = content_emb + self.emotion_weight*attention_output # query로 emo를 사용한 경우
        # # emotion을 query로 사용했으니, attention_output은 변형된 content embedding이 될 것이다.
        # # transformed_content로 사용가능한 경우의 수가 여러 가지다.
        # # 1) transformed_content = content_emb + self.emotion_weight * attention_output # content 정보를 어느 정도 많이 유지하고 싶은 경우
        # # 2) transformed_content = attention_output + emotion_emb # emotion에 맞춰 변형된 content에 emotion 정보를 한번 더 더해 감정 정보를 많이 반영
        # # 3) transformed_content = content_emb + self.emotion_weight * attention_output + emotion_emb # content 정보를 어느 정도 유지하면서 감정 정보를 많이 반영
        # transformed_content = torch.permute(transformed_content, (0, 2, 1))
        # transformed_content = self.ffn(transformed_content)
        
        # # Layer Normalization
        # transformed_content = self.layer_norm(transformed_content)
        # transformed_content = torch.permute(transformed_content, (0, 2, 1))
        ##########################################################

        # 생각해보니, emotion을 key & value로 사용하고, query로 content를 사용하자!!
        # 이를 토대로 emotion의 variaion embedding을 구하고, content embedding과 concatenate를 하자!!
        emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, length, 64
        emotion_emb = self.emotion_projection(emotion_emb) # batch, length, 256
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, 256, length
        
        # emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, length, 256
        content_emb = torch.permute(content_emb, (0, 2, 1)) # batch, length, 256
        attention_output, _ = self.cross_attention(query = content_emb, key = emotion_emb, value = emotion_emb) # query로 content 사용, batch, length, 256
        
        emotion_emb = torch.permute(emotion_emb, (0, 2, 1)) # batch, 256, length
        content_emb = torch.permute(content_emb, (0, 2, 1)) # batch, 256, length 
        attention_output = torch.permute(attention_output, (0, 2, 1)) # batch, 256, length
        
        transformed_content = torch.concat([content_emb, attention_output], dim=1) # batch, 512, length
        transformed_content = torch.permute(transformed_content, (0, 2, 1)) # batch, length, 512
        transformed_content = self.ffn(transformed_content) # batch, length, 256
        transformed_content = torch.permute(transformed_content, (0, 2, 1)) # batch, 256, length
        transformed_content = content_emb + transformed_content # batch, 256, length
        transformed_content = torch.permute(transformed_content, (0, 2, 1)) # batch, length, 256
        
        
        # Layer Normalization
        transformed_content = self.layer_norm(transformed_content)
        transformed_content = torch.permute(transformed_content, (0, 2, 1)) # batch, 256, length

        return transformed_content
    
class AdaptiveTimeWeight(nn.Module):
    def __init__(self, dim):
        super(AdaptiveTimeWeight, self).__init__()
        self.attention_layer = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, content_embedding, emotion_embedding):
        # weight = self.time_weight[:, :, :embeddings.size(2)]
        # weight = torch.sigmoid(weight)
        # output = weight * embeddings
        
        batch_size, dim, audio_len = content_embedding.shape
        combined = torch.cat([content_embedding, emotion_embedding], dim = 1)
        attention_score = self.attention_layer(combined.permute(0, 2, 1))
        attention_score = self.sigmoid(attention_score)
        attention_score = attention_score.permute(0, 2, 1)
        weighted_emo = emotion_embedding * attention_score
        return weighted_emo

class AdaptiveTimeWeight_for_diarization_embedding(nn.Module):
    def __init__(self, dim):
        super(AdaptiveTimeWeight_for_diarization_embedding, self).__init__()
        self.attention_layer = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, content_embedding, emotion_embedding, padding_mask = None):
        batch_size, dim, audio_len = content_embedding.shape
        combined = torch.cat([content_embedding, emotion_embedding], dim = 1)
        attention_score = self.attention_layer(combined.permute(0, 2, 1))
        
        
        if not padding_mask is None:
            padding_mask = padding_mask.unsqueeze(1)
            mask = padding_mask.repeat(1, sim.shape[1], 1)
            sim = sim.masked_fill(mask == 0, float("-inf"))
        
        
        # calculate attention weight
        attn_weights = F.softmax(sim / torch.sqrt(torch.tensor(256.0)), dim = -1) # batch, length, length

        attention_score = self.sigmoid(attention_score)
        attention_score = attention_score.permute(0, 2, 1)
        weighted_emo = emotion_embedding * attention_score
        return weighted_emo

class DurationPredictor_fastspeech_with_100_dim_global_emotion_embedding(nn.Module):
    def __init__(self):
        super(DurationPredictor_fastspeech_with_100_dim_global_emotion_embedding, self).__init__()
        self.duration_predictor = VariancePredictor_with_100_dim_global_emotion_embedding()
    
    def forward(self, unique_x):
        log_duration_prediction = self.duration_predictor(unique_x.transpose(1, 2), None)
        
        return log_duration_prediction
    

class DurationPredictor_fastspeech(nn.Module):
    def __init__(self):
        super(DurationPredictor_fastspeech, self).__init__()
        self.duration_predictor = VariancePredictor()
    
    def forward(self, unique_x):
        log_duration_prediction = self.duration_predictor(unique_x.transpose(1, 2), None)
        
        return log_duration_prediction
    
def median_smooth_batch(energy, window_size=9):
    """
    배치 입력 (32,1,28) 형태의 energy 텐서에 대해 마지막 차원(28)에 median filter 적용
    """
    energy = energy.cpu().numpy()  # GPU -> CPU 변환 후 NumPy 배열로 변경
    smoothed_energy = energy.copy()  # 원본 데이터 유지

    half_window = window_size // 2
    batch_size, channels, seq_len = energy.shape  # (32, 1, 28)

    # 각 데이터 샘플별로 median filter 적용
    for b in range(batch_size):
        for c in range(channels):
            for i in range(seq_len):
                start = max(0, i - half_window)
                end = min(seq_len, i + half_window + 1)
                window = energy[b, c, start:end]
                smoothed_energy[b, c, i] = np.median(window)

    smoothed_energy = torch.tensor(smoothed_energy).cuda()  # 다시 Torch 텐서로 변환하여 GPU로 이동

    return smoothed_energy






LRELU_SLOPE = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                self.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, self.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, self.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(
            x, self.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, self.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final
    
    def convert_pad_shape(self, pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x_mask = x_mask.unsqueeze(-1).transpose(1, 2)
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, self.convert_pad_shape(padding))
        return x
    
    def convert_pad_shape(self, pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape




class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
    
    
    
class Drawspeech_transformer_encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=5, # 원래는 4였는데, savgol_filter의 window_length가 5이기 때문에 맞춰줌
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
        self.last_projection = nn.Linear(hidden_channels, 1)

    def forward(self, x, x_mask):
        if x_mask is not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask.unsqueeze(-1)
            x = x.transpose(1, 2)
            for i in range(self.n_layers):
                y = self.attn_layers[i](x, x, attn_mask)
                y = self.drop(y)
                x = self.norm_layers_1[i](x + y)

                y = self.ffn_layers[i](x, x_mask)
                y = self.drop(y)
                x = self.norm_layers_2[i](x + y)
            result = self.last_projection(x.transpose(1, 2)) # batch, dim, length
            result = result * x_mask.unsqueeze(-1) # batch, 1, length
            return result
        else:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask.unsqueeze(-1)
            x = x.transpose(1, 2)
            for i in range(self.n_layers):
                y = self.attn_layers[i](x, x, None)
                y = self.drop(y)
                x = self.norm_layers_1[i](x + y)

                y = self.ffn_layers[i](x, None)
                y = self.drop(y)
                x = self.norm_layers_2[i](x + y)
            result = self.last_projection(x.transpose(1, 2)) # batch, dim, length
            result = result * x_mask.unsqueeze(-1) # batch, 1, length
            return result

class sinusoidal_positional_encoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(sinusoidal_positional_encoding, self).__init__()
        
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype = torch.float32) * -(math.log(10000.0) / d_model)
        ) # (d_model // 2, )
        
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # even number index
        pe[:, 1::2] = torch.cos(position * div_term) # odd number index
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        self.register_buffer('pe', pe) # buffer에 저장
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added        
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].detach()
        return x



class Sketch2ContourPredictor(nn.Module):
    def __init__(self, pitch_embedding_dim=256, energy_embedding_dim=256, ffn_dim=512, n_heads=2, n_layers=2):
        super(Sketch2ContourPredictor, self).__init__()

        assert pitch_embedding_dim == energy_embedding_dim
        embedding_dim = pitch_embedding_dim
        self.encoder = attentions.Encoder(embedding_dim, ffn_dim, n_heads, n_layers, kernel_size=3, p_dropout=0.1)
        self.linear_layer = nn.Linear(embedding_dim, 2)

    def forward(self, x_smoothed_energy_f0, mask=None):
        '''
        x: expanded text embedding, [b, t, h]
        mask: [b, t], 1 for real data, 0 for padding
        '''

        # if pitch_sketch is None and energy_sketch is None:
        #     return None, None

        # if pitch_sketch is None:
        #     pitch_sketch_embedding = 0
        # else:
        #     pitch_sketch_embedding = self.get_pitch_embedding(pitch_sketch, mask)
        
        # if energy_sketch is None:
        #     energy_sketch_embedding = 0
        # else:
        #     energy_sketch_embedding = self.get_energy_embedding(energy_sketch, mask)

        # x = x + smoothed_f0 + smoothed_energy

        # x_smoothed_energy_f0 = x_smoothed_energy_f0.transpose(1, 2)  # [b, h, t]
        # mask = mask.unsqueeze(1).to(x.dtype)  # [b, 1, t], 1 for real data, 0 for padding
        mask = torch.ones(x_smoothed_energy_f0.shape[0], x_smoothed_energy_f0.shape[2], dtype=torch.float32, device=x_smoothed_energy_f0.device)  # [B, T]
        mask = mask.unsqueeze(1)
        x = self.encoder(x_smoothed_energy_f0, mask)

        out = self.linear_layer(x.transpose(1, 2))
        pitch, energy = out.chunk(2, dim=-1)

        pitch = pitch.squeeze(-1)
        energy = energy.squeeze(-1)

        return pitch, energy

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        
        self.f0_embedding_dim = 256
        self.energy_embedding_dim = 256
        self.vuv_embedding_dim = 256
        self.n_bins = 256
        self.f0_bins = nn.Parameter(torch.linspace(-4, 4, self.n_bins - 1),requires_grad=False)
        self.energy_bins = nn.Parameter(torch.linspace(0, 6, self.n_bins - 1),requires_grad=False)
        
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_emo = weight_norm(
            Conv1d(1024, 256, kernel_size = 3, stride = 1, padding = 1)
        )
        
        self.conv_pre = weight_norm(
            Conv1d(640, 512, kernel_size = 7, stride = 1, padding=3)
        )
        
        
        
        self.emo_content_cross_attention = Emotion_diarization_single_Content_new_CrossAttention_relative_positional_encoding(256, 256, 256)

      
        ### Duration predictor from DrawSpeech
        self.duration_predictor = Drawspeech_transformer_encoder(hidden_channels = 512, filter_channels = 512, n_heads = 2, n_layers = 2, kernel_size = 3, p_dropout = 0.1)
        self.duration_linear_layer = nn.Linear(1, 64)
        
        ### For GRL
        self.linear_spk_layer = nn.Sequential(
            nn.Linear(192, 192),
            nn.LeakyReLU(0.1),
            nn.Linear(192, 192),
        )
        self.emo_clf = torch.nn.Sequential(
                RevGrad(),
                nn.Linear(192, 1024),
                nn.Mish(),
                nn.Linear(1024, 256),
                nn.Linear(256, 5),
            )
        
        ### For Content GRL
        self.conv_content_layer = nn.Sequential(
            Conv1d(256, 192, kernel_size = 3, padding = 1, stride = 1),
            nn.LeakyReLU(0.1),
            Conv1d(192, 192, kernel_size = 3, padding = 1, stride = 1),
        )
        self.content_clf = torch.nn.Sequential(
                RevGrad(),
                Conv1d(192, 512, kernel_size = 3, padding = 1, stride = 1),
                nn.Mish(),
                Conv1d(512, 512, kernel_size = 3, padding = 1, stride = 1),
                Conv1d(512, 500, kernel_size = 1),
        )
        
        self.transblock = Sketch2ContourPredictor()
        
        ### nn.Embedding 기반
        self.f0_embedding = nn.Embedding(self.n_bins + 1, self.f0_embedding_dim)
        self.energy_embedding = nn.Embedding(self.n_bins + 1, self.energy_embedding_dim)
        self.vuv_embedding = nn.Embedding(2, self.vuv_embedding_dim)
        
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)), k,
                                u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
    

    def forward(self, x, unique_x, original_x, unique_original_x, speaker_embedding, full_emotion_diarization_embedding, target_energy, smoothed_energy, smoothed_f0, target_f0, target_vuv, unique_emo, src_vuv, mode, padding_mask = None, code = None):
        
        target_energy = target_energy.float() * 10
        smoothed_energy = smoothed_energy.float() * 10
        target_f0 = target_f0.float()
        smoothed_f0 = smoothed_f0.float()
        target_vuv = target_vuv.float()
        src_vuv = src_vuv
        
        # speaker_embedding = self.neutral_spk_encoder(speaker_embedding.transpose(1, 2))
        speaker_embedding = F.normalize(speaker_embedding.transpose(1, 2), p=2, dim=-1) # (B, T, 192)
        speaker_embedding = self.linear_spk_layer(speaker_embedding)
        speaker_embedding = speaker_embedding.transpose(1, 2) # (B, T, 192) -> (B, 192, T)
        speaker_embedding_for_emo_cls = speaker_embedding[:, :, 0] # (B, 192, T) -> (B, 192)
        
        ### For GRL
        predicted_emotion_class = self.emo_clf(speaker_embedding_for_emo_cls) # (B, 5)
        
        ### For Content GRL layer
        full_emotion_diarization_embedding = self.conv_emo(full_emotion_diarization_embedding)

        ### Doing Cross Attention with emotion and content embedding method
        attention_weights = self.emo_content_cross_attention(original_x, full_emotion_diarization_embedding, padding_mask) # batch, 128, length, inference할 때도 original_x와 full_emotion_diarization embedding을 사용하는 것이 맞음
        attention_emo = torch.bmm(attention_weights, full_emotion_diarization_embedding.transpose(1, 2))
        post_emo = self.conv_content_layer(attention_emo.transpose(1, 2)) # (B, 192, T), train 때 not repeated content를 이용하니 T는 56
        
        ### For Content GRL layer
        predicted_content_class = self.content_clf(post_emo) # (B, 192, T) -> (B, 192)
        
        post_emo = post_emo.repeat_interleave(2, dim = 2)
        # 이 post_emo는 content embedding을 고려하여 적용된 weight
        
        
        ### get unique unit duration & Warping
        if mode == "train":

            warped_smoothed_duration = []
            warping_mode = random.randint(0,1)
            if warping_mode == 0:
                ## shifting
                for i in range(len(unique_x)):
                    smoothed_dur = unique_x[i][3].squeeze(0, 1).to(device = x.device)
                    shift = random.randint(-15, 15)
                    smoothed_dur = torch.roll(smoothed_dur, shifts = shift, dims = -1)
                    warped_smoothed_duration.append(smoothed_dur)
            if warping_mode == 1:
                ## piecewise stretch
                for i in range(len(unique_x)):
                    smoothed_dur = unique_x[i][3].squeeze(0, 1).to(device = x.device)
                    smoothed_dur = self.piecewise_time_warp_single_datapoint(smoothed_dur.unsqueeze(-1).transpose(0,1))
                    warped_smoothed_duration.append(smoothed_dur)

            padded_unique_x_content_emb = [] # 각 datapoint마다 unique content embedding length가 다르기 때문에, padding이 필요
            padded_unique_x_content_emb_with_emo = [] # content embedding과 global emotion embedding을 concatenate

            target_duration = []
            smoothed_duration = []
            padded_warped_smoothed_duration = []
            padding_mask = []
            max_unique_content_emb_len = 0
            for i in range(len(unique_x)):
                if unique_x[i][0].shape[0] > max_unique_content_emb_len:
                    max_unique_content_emb_len = unique_x[i][0].shape[0] # padding을 위해 구한 batch에서 가장 긴 length
            
            for i in range(len(unique_x)): # 여기서부터 padding 진행
                emb = unique_x[i][0]
                index= unique_x[i][2].to(device = emb.device)
                # emo = emotion_embedding[i][:, :emb.shape[0]].transpose(0,1).to(device = emb.device)
                # emb_with_emo = torch.concat([emb, emo], dim = 1) # unique content embedding과 global emotion embedding을 concatenate
                emo_diar = post_emo[i][:, index].transpose(0, 1) # unique index에 맞춰서 뽑한 emo_diarization embedding
                emb_with_emo = torch.concat([emb, emo_diar], dim = 1) # unique content embedding과 emotion_diarization embedding을 concatenate
                
                dur = unique_x[i][1]
                warped_smoothed_dur = warped_smoothed_duration[i].to(device = emb.device)
                smoothed_dur = unique_x[i][3].squeeze(0, 1).to(device = emb.device)
                pad_len = max_unique_content_emb_len - emb_with_emo.shape[0]

                if pad_len > 0:
                    padded = F.pad(emb, (0, 0, 0, pad_len))
                    padded_with_emo = F.pad(emb_with_emo, (0, 0, 0, pad_len))
                    padded_dur = F.pad(dur, (0, pad_len))
                    padded_smoothed_dur = F.pad(smoothed_dur, (0, pad_len))
                    padded_warped_smoothed_dur = F.pad(warped_smoothed_dur, (0, pad_len))
                else:
                    padded = emb
                    padded_with_emo = emb_with_emo
                    # padded_emo = unique_emotion_diarization
                    padded_dur = dur
                    padded_smoothed_dur = smoothed_dur
                    padded_warped_smoothed_dur = warped_smoothed_dur

                padding_mask.append(torch.sum(padded, dim = 1) != 0)
                padded_unique_x_content_emb.append(padded)
                padded_unique_x_content_emb_with_emo.append(padded_with_emo)
                target_duration.append(padded_dur)
                smoothed_duration.append(padded_smoothed_dur)
                padded_warped_smoothed_duration.append(padded_warped_smoothed_dur)

            padding_mask = torch.stack(padding_mask).to(x.device).float()
            padded_unique_x_content_emb = torch.stack(padded_unique_x_content_emb).to(x.device)
            target_duration = torch.stack(target_duration).to(x.device)
            smoothed_duration = torch.stack(smoothed_duration).to(x.device)
            padded_unique_x_content_emb = padded_unique_x_content_emb.transpose(1, 2)
            padded_unique_x_content_emb_with_emo = torch.stack(padded_unique_x_content_emb_with_emo).to(x.device).transpose(1, 2)
            padded_warped_smoothed_duration = torch.stack(padded_warped_smoothed_duration).to(x.device)

            ### Now, predict duration!
            # post_x = torch.concat([x, post_emotion_diarization_embedding], dim = 1) # original x랑 emotion_diarization embedding에 conv 적용한 값을 concatenate
            # padded_post_unique_x = torch.concat([padded_unique_x_content_emb, padded_unique_emotion_diarization_emb], dim = 1) # padded_unique_x_content_emb이랑 padded_unique_emotion_diarization embedding을 concatenate
            ### using padded warped smoothed duration
            smoothed_duration_emb = self.duration_linear_layer(padded_warped_smoothed_duration.unsqueeze(2)).transpose(1,2) # smoothed duration을 linear layer에 통과시켜서 embedding
            predicted_log_scale_duration = self.duration_predictor(torch.concat([padded_unique_x_content_emb_with_emo, smoothed_duration_emb], dim = 1).transpose(1,2), padding_mask).squeeze(-1) # 위 concatenate result를 가지고 duration 예측
            # train 할 때는 padding masking을 duration_predictor에 전달 안해도 됨
            # inference 시에도 결국 우리는 1개 sample로 inference를 하기 때문에 masking 안해도 됨
            
        ### inference 
        else:
            if src_vuv.shape[-1] != x.shape[-1]:
                pad_len = x.shape[-1] - src_vuv.shape[-1]
                if pad_len > 0:
                    src_vuv = F.pad(src_vuv, (0, pad_len))
                else:
                    src_vuv = src_vuv[:, :, :x.shape[-1]]

            unique_x_vuv = [] # unique_embedding의 v/uv를 구하는 부분
            for uniq in unique_x:
                uniq_ind = uniq[2]
                # uniq_vuv = src_vuv.squeeze(0, 1)[uniq_ind]
                uniq_vuv = src_vuv[0,0, uniq_ind]
                unique_x_vuv.append(uniq_vuv)
            unique_x_vuv = torch.stack(unique_x_vuv).to(x.device)
            
            # interporlated_post_emotion_diarization_embedding, _ = self.match_and_interp(x, post_emotion_diarization_embedding, target_vuv, src_vuv) # emotion audio와 content audio에 맞춰 emotion diarization에 interpolation 진행
            # inference code상 각 sample마다 inference를 진행해서 아래 부분이 필요 없지만, 혹시 나중에 batch 단위로 수정할 수도 있으니, 이렇게 작성해놓음
            padded_unique_x_content_emb = []
            padded_unique_x_content_emb_with_emo = []
            target_duration = []
            smoothed_duration = []
            interpolated_smoothed_duration = []
            interpolated_smoothed_unique_duration = []
            padding_mask = []
            
            max_unique_content_emb_len = 0
            for i in range(len(unique_x)):
                if unique_x[i][0].shape[0] > max_unique_content_emb_len:
                    max_unique_content_emb_len = unique_x[i][0].shape[0]
                    
            for i in range(len(unique_x)):
                emb = unique_x[i][0]
                index= unique_x[i][2].to(device = emb.device)
                
                emo_diar = post_emo[i][:, index].transpose(0, 1) # unique index에 맞춰서 뽑한 emo_diarization embedding
                emb_with_emo = torch.concat([emb, emo_diar], dim = 1) # unique content embedding과 emotion_diarization embedding을 concatenate

                
                # unique_emotion_diarization = post_emotion_diarization_embedding[i,:,index]
                dur = unique_x[i][1] # src audio의 unique
                smoothed_dur = torch.tensor(unique_emo[i][3]).unsqueeze(0) # emotion reference audio의 unique
                
                # interpolated_smoothed_dur = self.match_and_interp_for_duration(x, smoothed_dur, target_vuv, src_vuv, unique_emo[i])[0] # 여기가 좀 애매한 듯?
                interpolated_smoothed_dur = self.match_and_interp_for_duration_consider_start_vuv_not_zeropadding(x, smoothed_dur, target_vuv, src_vuv, unique_emo[i])[0] # 여기가 좀 애매한 듯?

                pad_len = max_unique_content_emb_len - emb_with_emo.shape[0]
                if pad_len > 0:
                    padded = F.pad(emb, (0, 0, 0, pad_len))
                    padded_with_emo = F.pad(emb_with_emo, (0, 0, 0, pad_len))
                    padded_dur = F.pad(dur, (0, pad_len))
                    padded_smoothed_dur = F.pad(smoothed_dur, (0, pad_len))
                    padded_interpolated_smoothed_dur = F.pad(interpolated_smoothed_dur, (0, pad_len))
                else:
                    padded = emb
                    padded_with_emo = emb_with_emo
                    padded_dur = dur
                    padded_smoothed_dur = smoothed_dur
                    padded_interpolated_smoothed_dur = interpolated_smoothed_dur
                
                padding_mask.append(torch.sum(padded, dim = 1) != 0)    
                interpolated_smoothed_unique_duration.append(interpolated_smoothed_dur[:,:,unique_x[i][2]])
                padded_unique_x_content_emb.append(padded)
                padded_unique_x_content_emb_with_emo.append(padded_with_emo)
                target_duration.append(padded_dur)
                smoothed_duration.append(padded_smoothed_dur)
                interpolated_smoothed_duration.append(padded_interpolated_smoothed_dur)
            
            padding_mask = torch.stack(padding_mask).to(x.device).float()    
            padded_unique_x_content_emb = torch.stack(padded_unique_x_content_emb).to(x.device)
            target_duration = torch.stack(target_duration).to(x.device)
            padded_unique_x_content_emb = padded_unique_x_content_emb.transpose(1, 2)
            padded_unique_x_content_emb_with_emo = torch.stack(padded_unique_x_content_emb_with_emo).to(x.device).transpose(1, 2)
            smoothed_duration = torch.stack(smoothed_duration).to(x.device)
            interpolated_smoothed_duration = torch.stack(interpolated_smoothed_duration).squeeze(0).to(x.device)
            interpolated_smoothed_unique_duration = torch.stack(interpolated_smoothed_unique_duration).squeeze(0).to(x.device)
            
            smoothed_duration_emb = self.duration_linear_layer(interpolated_smoothed_duration.transpose(1, 2)).transpose(1, 2)
            smoothed_unique_duration_emb = self.duration_linear_layer(interpolated_smoothed_unique_duration.transpose(1, 2)).transpose(1,2)
            predicted_log_scale_duration = self.duration_predictor(torch.concat([padded_unique_x_content_emb_with_emo, smoothed_unique_duration_emb], dim = 1).transpose(1,2), padding_mask).squeeze(-1) # 위 concatenate result를 가지고 duration 예측
        

        if mode == 'train':
            if warping_mode == 0:
                ## shifting
                shift = random.randint(-15, 15)
                smoothed_energy = torch.roll(smoothed_energy, shifts=shift, dims=-1)
                smoothed_f0 = torch.roll(smoothed_f0, shifts=shift, dims=-1)
                shifted_target_vuv = torch.roll(target_vuv, shifts=shift, dims=-1)
            if warping_mode == 1:
                ## piecewise stretch
                smoothed_energy, smoothed_f0 = self.piecewise_time_warp(smoothed_energy, smoothed_f0)

        src_vuv_256 = self.get_vuv_embedding(src_vuv)
        if mode == 'train':
            smoothed_energy_256 = self.get_energy_embedding(smoothed_energy)
            smoothed_f0_256 = self.get_f0_embedding(smoothed_f0)
            x_smoothed_energy_f0 = x + smoothed_energy_256 + smoothed_f0_256 + src_vuv_256
            
        else:
            duration = torch.expm1(predicted_log_scale_duration)
            duration_rounded = torch.clamp(duration.round(), min = 1)

            
            expanded_unique_x = []
            masked_expanded_unique_x = [] 
            
            expanded_post_unique_x = []
            masked_expanded_post_unique_x = []
            
            expanded_vuv = []
            masked_expanded_vuv = []
            
            expanded_emotion_diarization_embedding = []
            
            for duration, emb, src, uniq, uniq_vuv, emo_diar in zip(duration_rounded, padded_unique_x_content_emb_with_emo, src_vuv, unique_x, unique_x_vuv, post_emo):
                if len(duration.shape) > 1:
                    duration = duration.squeeze(1)
                src_end = (src == 1).nonzero(as_tuple=True)[-1]
                if src_end.numel() == 0:
                    src_end = src.shape[-1] - 1
                else:
                    src_end = src_end.max().item()
                    
                unique_index = torch.where(uniq[2] <= src_end)[0]# 이 부분에서 v_uv랑 duration이랑 unique index가 안맞는 경우 발생
                if len(unique_index) >0:
                    unique_index = unique_index[-1]
                else:
                    unique_index = None
                    
                
                expanded_x = []
                masked_expanded_x = []
                
                expanded = []
                masked_expanded = []
                
                vuv_expanded = []
                vuv_masked_expanded = []
                
                expanded_emo_diar = []
                
                for i in range(len(duration)):
                    vec = emb[:, i].unsqueeze(1)
                    repeated = vec.repeat(1, int(duration[i].item()))
                    expanded.append(repeated)
                    
                    vuv = uniq_vuv[i]
                    vuv_expanded.append(vuv.repeat(int(duration[i].item())))
                    
                    unique_content = uniq[0][i, :].unsqueeze(1)
                    expanded_x.append(unique_content.repeat(1, int(duration[i].item())))
                    
                    
                for i in range(unique_index.item()):
                    vec = emb[:, i].unsqueeze(1)
                    repeated = vec.repeat(1, int(duration[i].item()))
                    masked_expanded.append(repeated)
                    vuv = uniq_vuv[i]
                    vuv_masked_expanded.append(vuv.repeat(int(duration[i].item())))
                    unique_content = uniq[0][i, :].unsqueeze(1)
                    masked_expanded_x.append(unique_content.repeat(1, int(duration[i].item())))
                
                expanded_x = torch.cat(expanded_x, dim = 1)
                expanded = torch.cat(expanded, dim = 1)
                masked_expanded = torch.cat(masked_expanded, dim = 1)
                vuv_expanded = torch.cat(vuv_expanded, dim = 0)
                vuv_masked_expanded = torch.cat(vuv_masked_expanded, dim = 0)
                masked_expanded_x = torch.cat(masked_expanded_x, dim = 1)
                
                expanded_unique_x.append(expanded_x)
                masked_expanded_unique_x.append(masked_expanded_x)
                
                expanded_post_unique_x.append(expanded)
                masked_expanded_post_unique_x.append(masked_expanded)
                
                expanded_vuv.append(vuv_expanded)
                masked_expanded_vuv.append(vuv_masked_expanded)
                
                # expanded_emotion_diarization_embedding.append(torch.cat(expanded_emo_diar, dim = 1))
                
            expanded_unique_x = torch.stack(expanded_unique_x).to(x.device)
            masked_expanded_unique_x = torch.stack(masked_expanded_unique_x).to(x.device)
            
            expanded_post_unique_x = torch.stack(expanded_post_unique_x).to(x.device)
            masked_expanded_post_unique_x = torch.stack(masked_expanded_post_unique_x).to(x.device)
            
            expanded_vuv = torch.stack(expanded_vuv).to(x.device)
            masked_expanded_vuv = torch.stack(masked_expanded_vuv).to(x.device)
            
            ### for using emo diarization embedding
            # expanded_emo_diar1 = torch.stack(expanded_emotion_diarization_embedding).to(x.device)
            expanded_emotion_diarization_embedding = expanded_post_unique_x[:, -192:, :]
            
            ########################################################################################
            # src_vuv_64 = self.src_vuv_proj_pre(expanded_vuv).unsqueeze(0)
            
            expanded_vuv_256 = self.get_vuv_embedding(expanded_vuv)  
            smoothed_energy, _ = self.match_and_interp(expanded_post_unique_x, smoothed_energy, target_vuv, expanded_vuv) # emotion speech에 맞춰 interpolation 진행
            smoothed_f0, _ = self.match_and_interp(expanded_post_unique_x, smoothed_f0, target_vuv, expanded_vuv) # emotion speech에 맞춰 interpolation 진행
            target_energy, _ = self.match_and_interp(expanded_post_unique_x, target_energy, target_vuv, expanded_vuv) # emotion speech에 맞춰 interpolation 진행
            target_f0, target_vuv = self.match_and_interp(expanded_post_unique_x, target_f0, target_vuv, expanded_vuv) # emotion speech에 맞춰 interpolation 진행
           
            ## make src_vuv_mask
            if src_vuv.dtype != torch.bool:
                src_vuv_mask = src_vuv == 0  # unvoiced mask
            else:
                src_vuv_mask = ~src_vuv
            
            smoothed_f0_masked = smoothed_f0.clone()
            vuv_len = src_vuv_mask.shape[-1]
            f0_len = smoothed_f0_masked.shape[-1]

            # 길이 맞추기
            if vuv_len > f0_len:
                src_vuv_mask = src_vuv_mask[..., :f0_len]  # 자르기
            elif vuv_len < f0_len:
                # 마지막 값을 복제해서 길이 맞추기 (혹은 0 또는 False로 패딩해도 됨)
                pad_size = f0_len - vuv_len
                last_value = src_vuv_mask[..., -1:]
                src_vuv_mask = torch.cat([src_vuv_mask, last_value.expand(*src_vuv_mask.shape[:-1], pad_size)], dim=-1)

            # 적용
            smoothed_f0_masked[src_vuv_mask] = float('nan')
            
            smoothed_energy_256 = self.get_energy_embedding(smoothed_energy)
            smoothed_f0_256 = self.get_f0_embedding(smoothed_f0_masked)
            x_smoothed_energy_f0 = expanded_unique_x + smoothed_energy_256 + smoothed_f0_256 + expanded_vuv_256
        
        predicted_f0, predicted_energy = self.transblock(x_smoothed_energy_f0)
        predicted_f0 = predicted_f0.unsqueeze(1)
        predicted_energy = predicted_energy.unsqueeze(1)
        
        #######################################################
        if mode == 'train':
            target_energy_256 = self.get_energy_embedding(target_energy)
            target_f0_256 = self.get_f0_embedding(target_f0)
            src_vuv_256 = self.get_vuv_embedding(src_vuv)
            x = torch.cat([x + target_energy_256 + target_f0_256 + src_vuv_256, speaker_embedding, post_emo], dim=1)

        else:
            # 아직 inference는 수정 안함
            
            if expanded_vuv.dtype != torch.bool:
                expanded_vuv_bool = expanded_vuv ==0
            else:
                expanded_vuv_bool = ~expanded_vuv
            
            expanded_vuv_bool = expanded_vuv_bool.unsqueeze(0)
            predicted_energy_masked = predicted_energy.clone()
            predicted_energy_masked[predicted_energy_masked <= 0.1] = 0.0
            predicted_f0_masked = predicted_f0.clone()
            
            predicted_energy_256 = self.get_energy_embedding(predicted_energy_masked)
            predicted_f0_256 = self.get_f0_embedding(predicted_f0_masked)
            
            target_energy_256 = self.get_energy_embedding(target_energy)
            target_f0_256 = self.get_f0_embedding(target_f0)
            
            expanded_vuv_256 = self.get_vuv_embedding(expanded_vuv)
            
            if speaker_embedding.shape[-1] > expanded_vuv_256.shape[-1]:
                speaker_embedding = speaker_embedding[:, :, :expanded_vuv_256.shape[-1]]
            else:
                speaker_embedding = torch.cat([speaker_embedding, speaker_embedding[:,:,-1:].expand(-1,-1, expanded_vuv_256.shape[-1] - speaker_embedding.shape[-1])], dim = 2)            
            x = torch.cat([expanded_unique_x + predicted_energy_256 + predicted_f0_256 + expanded_vuv_256, speaker_embedding, expanded_emotion_diarization_embedding], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x, speaker_embedding, post_emo, predicted_energy, target_energy, predicted_f0, target_f0, target_vuv, predicted_log_scale_duration, target_duration, predicted_emotion_class, attention_weights, predicted_content_class, code

    def get_f0_embedding(self, x):
        nan_mask = torch.isnan(x)
        x_no_nan = torch.where(nan_mask, torch.tensor(0.0, device=x.device), x)
        embedding_idx = torch.bucketize(x_no_nan, self.f0_bins)
        embedding_idx = torch.where(nan_mask, torch.tensor(len(self.f0_bins), device=x.device), embedding_idx)
        embedding = self.f0_embedding(embedding_idx)
        embedding = embedding.squeeze(1).permute(0,2,1)
        return embedding

    def get_energy_embedding(self, x):
        nan_mask = torch.isnan(x)
        x_no_nan = torch.where(nan_mask, torch.tensor(0.0, device=x.device), x)
        embedding_idx = torch.bucketize(x_no_nan, self.energy_bins)
        embedding_idx = torch.where(nan_mask, torch.tensor(len(self.energy_bins), device=x.device), embedding_idx)
        embedding = self.energy_embedding(embedding_idx)
        embedding = embedding.squeeze(1).permute(0,2,1)
        return embedding
    
    def get_vuv_embedding(self, x):
        embedding = self.vuv_embedding(x)
        embedding = embedding.squeeze(1).permute(0,2,1)
        return embedding
    
    def piecewise_time_warp(self, tensor1, tensor2, num_pieces=5, min_factor=0.4, max_factor=1.6):
        assert tensor1.shape == tensor2.shape, "두 입력 텐서의 shape가 같아야 합니다."
        batch_size, _, T = tensor1.shape

        tensor1 = tensor1.squeeze(1)  # shape: (batch_size, T)
        tensor2 = tensor2.squeeze(1)  # shape: (batch_size, T)

        warped1_batch = []
        warped2_batch = []

        for i in range(batch_size):
            np_tensor1 = tensor1[i].cpu().numpy()
            np_tensor2 = tensor2[i].cpu().numpy()

            # ✅ 랜덤한 구간 개수 및 경계 설정 (두 텐서에 동일하게 사용)
            curr_num_pieces = random.randint(2, num_pieces)
            piece_boundaries = np.linspace(0, T, num=curr_num_pieces + 1, dtype=int)

            warped1 = []
            warped2 = []

            for j in range(curr_num_pieces):
                start = piece_boundaries[j]
                end = piece_boundaries[j + 1]

                piece1 = np_tensor1[start:end]
                piece2 = np_tensor2[start:end]

                # ✅ 같은 factor 사용
                factor = np.random.uniform(min_factor, max_factor)
                new_len = max(1, int(len(piece1) * factor))

                stretched1 = np.interp(
                    np.linspace(0, len(piece1) - 1, new_len),
                    np.arange(len(piece1)),
                    piece1
                )
                stretched2 = np.interp(
                    np.linspace(0, len(piece2) - 1, new_len),
                    np.arange(len(piece2)),
                    piece2
                )

                warped1.append(stretched1)
                warped2.append(stretched2)

            warped1 = np.concatenate(warped1)
            warped2 = np.concatenate(warped2)

            # 🔁 T 길이로 리샘플링
            interp1 = interp1d(np.linspace(0, 1, len(warped1)), warped1, kind='linear', fill_value="extrapolate")
            interp2 = interp1d(np.linspace(0, 1, len(warped2)), warped2, kind='linear', fill_value="extrapolate")

            warped1_resampled = interp1(np.linspace(0, 1, T))
            warped2_resampled = interp2(np.linspace(0, 1, T))

            warped1_batch.append(warped1_resampled)
            warped2_batch.append(warped2_resampled)

        # (batch, 1, T) 형태로 반환
        warped1_batch = np.array(warped1_batch)
        warped2_batch = np.array(warped2_batch)

        warped1_tensor = torch.tensor(warped1_batch, dtype=tensor1.dtype, device=tensor1.device).unsqueeze(1)
        warped2_tensor = torch.tensor(warped2_batch, dtype=tensor2.dtype, device=tensor2.device).unsqueeze(1)

        return warped1_tensor, warped2_tensor

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_emo)
    
    def piecewise_time_warp_single_datapoint(self, tensor, num_pieces = 5, min_factor = 0.4, max_factor = 1.6):
        num_pieces = random.randint(2, 5)
        _, T = tensor.shape
        tensor = tensor.squeeze(0)
        
        np_tensor = tensor.cpu().numpy()
        piece_boundaries = np.linspace(0, T, num=num_pieces + 1, dtype = int)
        warped = []
        for j in range(num_pieces):
            start = piece_boundaries[j]
            end = piece_boundaries[j+1]
            piece = np_tensor[start:end]
            factor = np.random.uniform(min_factor, max_factor)
            new_len = max(1, int(len(piece) * factor))
            stretched = np.interp(
                np.linspace(0, len(piece) - 1, new_len),
                np.arange(len(piece)),
                piece
            )
            warped.append(stretched)
        
        warped = np.concatenate(warped)

        # 🔁 원래 길이 T로 리샘플링
        interp_fn = interp1d(np.linspace(0, 1, len(warped)), warped, kind='linear', fill_value="extrapolate")
        warped_resampled = interp_fn(np.linspace(0, 1, T))
        return torch.tensor(warped_resampled, dtype=tensor.dtype, device=tensor.device)
        
    def match_and_concat(self, x, smoothed, vuv = None, mode = None):
        x_len = x.shape[-1]  # x의 length
        se_len = smoothed.shape[-1]  # smoothed_energy의 length
        
        if se_len > x_len:
            smoothed = smoothed[..., :x_len]  # 자르기
            if vuv != None:
                vuv = vuv[..., :x_len]
        elif se_len < x_len:
            pad_size = x_len - se_len
            smoothed = torch.nn.functional.pad(smoothed, (0, pad_size))  # Zero padding
            if vuv != None:
                vuv = torch.nn.functional.pad(vuv, (0, pad_size))
        return smoothed, vuv

    def match_and_interp_for_duration_not_considering_start_vuv_new_version(self, x, smoothed, vuv = None, src_vuv = None):
        x_len = x.shape[-1]
        se_len = smoothed.shape[-1]
        x_voice_indices = (src_vuv == 1).nonzero(as_tuple=True)[-1]
        if x_voice_indices.numel() == 0:
            x_voice_start = 0
            x_voice_end = x_len - 1
        else:
            x_voice_start = x_voice_indices.min().item()
            x_voice_end = x_voice_indices.max().item()
        
        smoothed_voice_indices = (vuv == 1).nonzero(as_tuple=True)[-1]
        if smoothed_voice_indices.numel() == 0:
            smoothed_voice_start = 0
            smoothed_voice_end = se_len - 1
        else:
            smoothed_voice_start = smoothed_voice_indices.min().item()
            smoothed_voice_end = smoothed_voice_indices.max().item()

        def interp_to_match(src, src_len, tgt_len):
            src = src[..., :src_len]  # 필요한 구간만 자르기
            src = F.interpolate(src, size=tgt_len, mode='linear', align_corners=True)
            return src.squeeze(0)

        smoothed_interp = interp_to_match(smoothed, smoothed_voice_end + 1, x_voice_end + 1)
        if vuv is not None:
            vuv_interp = interp_to_match(vuv, smoothed_voice_end + 1, x_voice_end + 1)
        else:
            vuv_interp = None

        # 나머지 부분은 0으로 padding
        pad_len = x_len - (x_voice_end + 1)
        pad_value = smoothed_interp[:, -1]
        pad_value = pad_value.unsqueeze(0).expand(-1, pad_len)
        smoothed_final = torch.cat([smoothed_interp, pad_value], dim=-1)
        # smoothed_final = F.pad(smoothed_interp, (0, pad_len))
        if vuv_interp is not None:
            vuv_final = F.pad(vuv_interp, (0, pad_len))
        else:
            vuv_final = None

        return smoothed_final.unsqueeze(0), vuv_final


    def match_and_interp_for_duration_consider_start_vuv_not_zeropadding(self, x, smoothed, vuv = None, src_vuv = None, unique_emo = None):
        x_len = x.shape[-1]
        se_len = smoothed.shape[-1]

        # x의 voice 시작과 끝 인덱스
        x_voice_indices = (src_vuv == 1).nonzero(as_tuple=True)[-1]
        if x_voice_indices.numel() == 0:
            x_voice_start = 0
            x_voice_end = x_len - 1
        else:
            x_voice_start = x_voice_indices.min().item()
            x_voice_end = x_voice_indices.max().item()

        # smoothed의 voice 시작과 끝 인덱스
        
        smoothed_voice_indices = (vuv == 1).nonzero(as_tuple=True)[-1]
        if smoothed_voice_indices.numel() == 0:
            smoothed_voice_start = 0
            smoothed_voice_end = se_len - 1
        else:
            smoothed_voice_start = smoothed_voice_indices.min().item()
            smoothed_voice_end = smoothed_voice_indices.max().item()
        
        def find_nearest_le_index(lst, target):
            # target보다 작거나 같은 값 중 가장 가까운 값 찾기
            candidates = [i for i, v in enumerate(lst) if v <= target]
            if not candidates:
                return 0  # 혹은 예외 처리
            return candidates[-1]

        unique_emo_start_index = find_nearest_le_index(unique_emo[2], smoothed_voice_start + 1)
        unique_emo_end_index = find_nearest_le_index(unique_emo[2], smoothed_voice_end + 1)

        # interpolate 대상 길이
        target_interp_len = x_voice_end - x_voice_start + 1
        source_interp_len = smoothed_voice_end - smoothed_voice_start + 1

        # 해당 구간 잘라서 interpolate
        def interp_to_match(src, start, end, tgt_len):
            src = src[..., start:end+1]  # 필요한 구간만 자르기
            src = F.interpolate(src, size=tgt_len, mode='linear', align_corners=True)
            return src.squeeze(0)

        smoothed_interp = interp_to_match(smoothed, unique_emo_start_index, unique_emo_end_index, target_interp_len)
        if vuv is not None:
            vuv_interp = interp_to_match(vuv, unique_emo_start_index, unique_emo_end_index, target_interp_len)
        else:
            vuv_interp = None

        # 앞뒤로 zero-padding 추가
        pre_pad = x_voice_start
        post_pad = x_len - x_voice_end - 1

        pre_pad_value = smoothed_interp[:, 0]
        pre_pad_value = pre_pad_value.unsqueeze(0).expand(-1, pre_pad)
        
        post_pad_value = smoothed_interp[:, -1]
        post_pad_value = post_pad_value.unsqueeze(0).expand(-1, post_pad)
        smoothed_final = torch.cat([pre_pad_value, smoothed_interp, post_pad_value], dim=-1)
        # smoothed_final = F.pad(smoothed_interp, (pre_pad, post_pad))
        if vuv_interp is not None:
            vuv_final = F.pad(vuv_interp, (pre_pad, post_pad))
        else:
            vuv_final = None

        return smoothed_final.unsqueeze(0), vuv_final


    def match_and_interp_for_duration_not_considering_start_vuv(self, x, smoothed, vuv = None, src_vuv = None):
        x_len = x.shape[-1]
        se_len = smoothed.shape[-1]

        x_voice_indices = (src_vuv == 1).nonzero(as_tuple=True)[-1]
        if x_voice_indices.numel() == 0:
            x_voice_start = 0
            x_voice_end = x_len - 1
        else:
            x_voice_start = x_voice_indices.min().item()
            x_voice_end = x_voice_indices.max().item()

        # smoothed의 voice 시작과 끝 인덱스
        
        smoothed_voice_indices = (vuv == 1).nonzero(as_tuple=True)[-1]
        if smoothed_voice_indices.numel() == 0:
            smoothed_voice_start = 0
            smoothed_voice_end = se_len - 1
        else:
            smoothed_voice_start = smoothed_voice_indices.min().item()
            smoothed_voice_end = smoothed_voice_indices.max().item()

        # smoothed와 vuv의 voice 구간을 x의 voice 구간 길이에 맞게 interpolate
        def interp_to_match(src, src_len, tgt_len):
            src = src[..., :src_len]  # 필요한 구간만 자르기
            src = F.interpolate(src, size=tgt_len, mode='linear', align_corners=True)
            return src.squeeze(0)

        smoothed_interp = interp_to_match(smoothed, smoothed_voice_end + 1, x_voice_end + 1)
        if vuv is not None:
            vuv_interp = interp_to_match(vuv, smoothed_voice_end + 1, x_voice_end + 1)
        else:
            vuv_interp = None

        # 나머지 부분은 0으로 padding
        pad_len = x_len - (x_voice_end + 1)
        smoothed_final = F.pad(smoothed_interp, (0, pad_len))
        if vuv_interp is not None:
            vuv_final = F.pad(vuv_interp, (0, pad_len))
        else:
            vuv_final = None

        return smoothed_final.unsqueeze(0), vuv_final

    def match_and_interp_for_duration(self, x, smoothed, vuv = None, src_vuv = None, unique_emo = None):
        x_len = x.shape[-1]
        se_len = smoothed.shape[-1]

        # x의 voice 시작과 끝 인덱스
        x_voice_indices = (src_vuv == 1).nonzero(as_tuple=True)[-1]
        if x_voice_indices.numel() == 0:
            x_voice_start = 0
            x_voice_end = x_len - 1
        else:
            x_voice_start = x_voice_indices.min().item()
            x_voice_end = x_voice_indices.max().item()

        # smoothed의 voice 시작과 끝 인덱스
        
        smoothed_voice_indices = (vuv == 1).nonzero(as_tuple=True)[-1]
        if smoothed_voice_indices.numel() == 0:
            smoothed_voice_start = 0
            smoothed_voice_end = se_len - 1
        else:
            smoothed_voice_start = smoothed_voice_indices.min().item()
            smoothed_voice_end = smoothed_voice_indices.max().item()
        
        def find_nearest_le_index(lst, target):
            # target보다 작거나 같은 값 중 가장 가까운 값 찾기
            candidates = [i for i, v in enumerate(lst) if v <= target]
            if not candidates:
                return 0  # 혹은 예외 처리
            return candidates[-1]

        unique_emo_start_index = find_nearest_le_index(unique_emo[2], smoothed_voice_start + 1)
        unique_emo_end_index = find_nearest_le_index(unique_emo[2], smoothed_voice_end + 1)

        # interpolate 대상 길이
        target_interp_len = x_voice_end - x_voice_start + 1
        source_interp_len = smoothed_voice_end - smoothed_voice_start + 1

        # 해당 구간 잘라서 interpolate
        def interp_to_match(src, start, end, tgt_len):
            src = src[..., start:end+1]  # 필요한 구간만 자르기
            src = F.interpolate(src, size=tgt_len, mode='linear', align_corners=True)
            return src.squeeze(0)

        smoothed_interp = interp_to_match(smoothed, unique_emo_start_index, unique_emo_end_index, target_interp_len)
        if vuv is not None:
            vuv_interp = interp_to_match(vuv, unique_emo_start_index, unique_emo_end_index, target_interp_len)
        else:
            vuv_interp = None

        # 앞뒤로 zero-padding 추가
        pre_pad = x_voice_start
        post_pad = x_len - x_voice_end - 1

        smoothed_final = F.pad(smoothed_interp, (pre_pad, post_pad))
        if vuv_interp is not None:
            vuv_final = F.pad(vuv_interp, (pre_pad, post_pad))
        else:
            vuv_final = None

        return smoothed_final.unsqueeze(0), vuv_final
    

    def match_and_interp(self, x, smoothed, vuv=None, src_vuv=None): # (expanded_x_post, smoothed, target_vuv, expanded_vuv) 
        x_len = x.shape[-1]
        se_len = smoothed.shape[-1]

        # x의 voice 시작과 끝 인덱스
        x_voice_indices = (src_vuv == 1).nonzero(as_tuple=True)[-1]
        if x_voice_indices.numel() == 0:
            x_voice_start = 0
            x_voice_end = x_len - 1
        else:
            x_voice_start = x_voice_indices.min().item()
            x_voice_end = x_voice_indices.max().item()

        # smoothed의 voice 시작과 끝 인덱스
        
        smoothed_voice_indices = (vuv == 1).nonzero(as_tuple=True)[-1]
        if smoothed_voice_indices.numel() == 0:
            smoothed_voice_start = 0
            smoothed_voice_end = se_len - 1
        else:
            smoothed_voice_start = smoothed_voice_indices.min().item()
            smoothed_voice_end = smoothed_voice_indices.max().item()
            

        # interpolate 대상 길이
        target_interp_len = x_voice_end - x_voice_start + 1
        source_interp_len = smoothed_voice_end - smoothed_voice_start + 1

        # 해당 구간 잘라서 interpolate
        def interp_to_match(src, start, end, tgt_len):
            src = src[..., start:end+1]  # 필요한 구간만 자르기
            src = F.interpolate(src, size=tgt_len, mode='linear', align_corners=True)
            return src.squeeze(0)

        smoothed_interp = interp_to_match(smoothed, smoothed_voice_start, smoothed_voice_end, target_interp_len)
        if vuv is not None:
            vuv_interp = interp_to_match(vuv, smoothed_voice_start, smoothed_voice_end, target_interp_len)
        else:
            vuv_interp = None

        # 앞뒤로 zero-padding 추가
        pre_pad = x_voice_start
        post_pad = x_len - x_voice_end - 1

        smoothed_final = F.pad(smoothed_interp, (pre_pad, post_pad))
        if vuv_interp is not None:
            vuv_final = F.pad(vuv_interp, (pre_pad, post_pad))
        else:
            vuv_final = None

        return smoothed_final.unsqueeze(0), vuv_final


class EmotionIDEmbedding(nn.Module):
    def __init__(self, num_emotions = 5, emb_dim = 192):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, emb_dim)
        
    def forward(self, emotion_ids):
        return self.embedding(emotion_ids)



class CodeGenerator(Generator):
    def __init__(self, h, mode):
        super().__init__(h)
        self.dict = nn.Embedding(num_embeddings = h.num_embeddings, embedding_dim = h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.multispkr = h.get('multispkr', None)
        self.mode = mode
        # if self.multispkr:
            # self.spkr = nn.Embedding(1, h.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal


    def segment_embedding(self, embedded_seq):
        embedded_seq = embedded_seq.transpose(0, 1)
        unique_embs = []
        durations = []
        smoothed_durations = []
        indexs = []
        prev = embedded_seq[0]
        count = 1
        start_idx = 0
        for i, curr in enumerate(embedded_seq[1:], start = 1):
            if torch.allclose(curr, prev):
                count += 1
            else:
                unique_embs.append(prev)
                durations.append(count)
                indexs.append(start_idx)
                prev = curr
                count = 1
                start_idx = i
        unique_embs.append(prev)
        durations.append(count)
        indexs.append(start_idx)
        d = torch.tensor(durations, dtype = torch.float32).numpy()
        # smoothed_durations.append(np.array([gaussian_filter1d(d, sigma = 2)]))
        smoothed_durations.append(np.array([savgol_filter(d, window_length = 5, polyorder = 2, mode = 'nearest')]))
        
        return (
            torch.stack(unique_embs),
            torch.tensor(np.array(durations)),
            torch.tensor(np.array(indexs)),
            torch.tensor(np.array(smoothed_durations))
        )
        # return torch.stack(unique_embs), torch.tensor(durations), torch.tensor(indexs), torch.tensor(smoothed_durations)

    def extract_unique_duration(self, inputs):
        results = []
        for seq in inputs:
            # seq = seq.transpose(0, 1)
            unique_values = []
            durations = []
            prev = seq[0]
            count = 1
            for curr in seq[1:]:
                if curr == prev:
                    count += 1
                else:
                    unique_values.append(prev)
                    durations.append(count)
                    prev = curr
                    count = 1
            unique_values.append(prev)
            durations.append(count)
            results.append((unique_values, durations))
        return results


    def forward(self, target_energy, target_f0, emo_id = None, unique_emo = "None", **kwargs,):
        target_energy = target_energy
        x = self.dict(kwargs['code']).transpose(1, 2)
        code = kwargs['code']
        unique_x = [self.segment_embedding(seq) for seq in x]
        
        original_x = self.dict(kwargs['original_code']).transpose(1, 2).to(x.device)
        unique_original_x = [self.segment_embedding(seq) for seq in original_x]
        
        # 32, 128, 28
        ######################################################
        speaker_embedding = kwargs['speaker_embedding'].unsqueeze(0).unsqueeze(0) # 32, 1, 1, 192
        # speaker_embedding = kwargs['speaker_embedding']
        speaker_embedding = speaker_embedding.squeeze(1).transpose(1, 2).repeat(1, 1, x.shape[2])
        # emotion_embedding = kwargs['emotion_embedding'] # 32, 100
        # emotion_embedding = emotion_embedding.unsqueeze(2).repeat(1, 1, x.shape[2])
        full_emotion_diarization_embedding = kwargs["full_emotion_diarization_embedding"].transpose(1, 2)
        smoothed_energy = kwargs['smoothed_energy'].unsqueeze(1)
        target_energy = kwargs['energy'].unsqueeze(1)
        smoothed_f0 = kwargs['smoothed_f0'].unsqueeze(1)
        target_f0 = kwargs['f0'].unsqueeze(1)
        target_vuv = kwargs['vuv'].unsqueeze(1)
        src_vuv = kwargs['src_vuv'].unsqueeze(1)
        if "padding_mask" in kwargs.keys(): # for emotion diarization & original content embedding
            padding_mask = kwargs["padding_mask"] 
        else:
            padding_mask = None
            
        if emo_id is not None:
            emo_id = torch.tensor([emo_id])
            emo_id_embedding = EmotionIDEmbedding()(emo_id).transpose(0, 1)    

        if emo_id is not None and padding_mask is not None:
            return super().forward(x, unique_x, original_x, unique_original_x, speaker_embedding, full_emotion_diarization_embedding, target_energy, smoothed_energy, smoothed_f0, target_f0, target_vuv, unique_emo, src_vuv, emo_id_embedding, self.mode, padding_mask, code = code)  
        elif emo_id is not None and padding_mask is None:
            return super().forward(x, unique_x, original_x, unique_original_x, speaker_embedding, full_emotion_diarization_embedding, target_energy, smoothed_energy, smoothed_f0, target_f0, target_vuv, unique_emo, src_vuv, emo_id_embedding, self.mode, code = code)
        elif emo_id is None and padding_mask is not None:
            return super().forward(x, unique_x, original_x, unique_original_x, speaker_embedding, full_emotion_diarization_embedding, target_energy, smoothed_energy, smoothed_f0, target_f0, target_vuv, unique_emo, src_vuv, self.mode, padding_mask, code = code)
        else:
            return super().forward(x, unique_x, original_x, unique_original_x,speaker_embedding, full_emotion_diarization_embedding, target_energy, smoothed_energy, smoothed_f0, target_f0, target_vuv, unique_emo, src_vuv, self.mode, code = code)

            
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))), ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11), ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
             norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
             norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
             norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
             norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2)), ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS(), ])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def MutualInformationLoss(spk, emo):
    batch_size = spk.shape[0]
    
    # Concatenate joint distribution
    joint_pairs = torch.cat([spk, emo], dim = -1)
    
    # Create marginal distribution (P(spk)P(emo)) by shuffling emo
    emo_perm = emo[torch.randperm(batch_size)]  
    marginal_pairs = torch.cat([spk, emo_perm], dim = -1)
    
    # Compute scores usign a simple neural network function
    def T(x):
        return torch.tanh(torch.sum(x, dim = -1, keepdim = True))
    
    T_joint = T(joint_pairs) # Logis for joint distribution
    T_marginal = T(marginal_pairs) # Logits for marginal distribution
    
    mutual_info = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))
    
    return -mutual_info # negative returns since to minimize the loss
    
    
def KL_div_based_MutualInformationLoss(spk, emo):
    batch_size = spk.shape[0]
    
    # Concatenate joint distribution
    # joint_pairs = torch.cat([spk, emo], dim = -1)
    joint_pairs = spk*emo
    
    # Create marginal distribution (P(spk)P(emo)) by shuffling emo
    emo_perm = emo[torch.randperm(batch_size)] 
    # marginal_pairs = torch.cat([spk, emo_perm], dim = -1)
    marginal_pairs = spk * emo_perm
    
    # Discriminator function D(X, Y) -> Probability Score
    def D(x):
        return torch.sigmoid(torch.norm(x, dim = -1, keepdim = True)) # Sigmoid to get probability
    # D(x)가 별로면, 결국 neural network를 학습시켜서 사용해야 할 수도 있음
    
    D_joint = D(joint_pairs) # P(X, Y)에서 sampling된 점수
    D_marginal = D(marginal_pairs) # P(X)P(Y)에서 sampling된 점수

    # KL Divergence 기반 MI Loss (Binary Cross Entropy 형태)
    mi_loss = (torch.mean(torch.log(D_joint + 1)) - torch.mean(torch.log((1 - D_marginal) + 1)))
    
    return mi_loss 



class EmotionClassifier(nn.Module):
    def __init__(self, emb_dim, num_labels, hidden_dim = 100):
        super().__init__()
        ##################################################################
        # self.emb_dim = emb_dim       
        # self.proj = nn.Linear(emb_dim, hidden_dim)
        # # self.proj2 = nn.Linear(512, hidden_dim)
        # self.out = nn.Linear(hidden_dim, num_labels)
        # nn.init.xavier_uniform_(self.proj.weight)
        # nn.init.xavier_uniform_(self.out.weight)
        ################################################################## 
               
        # original version
        self.layer_num = 24
        self.emb_dim = emb_dim

        self.weights = nn.Parameter(torch.randn(self.layer_num))
        self.proj = nn.Linear(emb_dim,hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward(self, feature, feature_lens, mode = "train"):
        if mode == "inference":
            final_emb = feature
            pred = self.out(final_emb)
            return pred
        
        else:
            
            norm_weights = F.softmax(self.weights, dim = -1)
            norm_weights = norm_weights.view(1, -1 , 1 ,1)
            weighted_feature = (norm_weights * feature).sum(dim=1)
            
            # average pooling
            agg_vec_list = []
            for i in range(len(weighted_feature)):
                agg_vec = torch.mean(weighted_feature[i][:feature_lens[i]], dim=0)
                agg_vec_list.append(agg_vec)

            avg_emb = torch.stack(agg_vec_list)

            # classifier
            final_emb = self.proj(avg_emb)
            pred = self.out(final_emb)
            return pred, final_emb