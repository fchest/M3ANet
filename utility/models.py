import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
input_length = 1624



#Depth-Wise Conv1D block from Conv-TasNet
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super(DepthConv1d, self).__init__()

        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))


        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual
        

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)



class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ConvCrossAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, in_ch, kernel_size, dilation, dropout=0.1):
        super(ConvCrossAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.w_ks = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.w_vs = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.GroupNorm(1, in_ch, eps=1e-08)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q, _ = self.w_qs(q)
        k, _ = self.w_ks(k)
        v, _ = self.w_vs(v)
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_v, -1)
        out = self.dropout(out)
        out += residual

        out = self.layer_norm(out)

        return out

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, input_size, layer, in_ch, kernel_size, dilation):
        super(MultiLayerCrossAttention, self).__init__()
        self.audio_encoder = nn.ModuleList()
        self.spike_encoder = nn.ModuleList()
        self.layer = layer
        self.projection = nn.Conv1d(in_ch*4, in_ch, kernel_size, padding='same')
        self.LayernormList_audio = nn.ModuleList()
        self.LayernormList_spike = nn.ModuleList()
        self.layernorm_out = nn.GroupNorm(1, in_ch, eps=1e-08)
        for i in range(layer):
            self.LayernormList_audio.append(nn.GroupNorm(1, in_ch, eps=1e-08))
            self.LayernormList_spike.append(nn.GroupNorm(1, in_ch, eps=1e-08))
        for i in range(layer):
            self.audio_encoder.append(ConvCrossAttention(n_head=1, d_model=input_size, d_k=input_size, d_v=input_size,
                                                  in_ch=in_ch, kernel_size=kernel_size, dilation=dilation))
            self.spike_encoder.append(ConvCrossAttention(n_head=1, d_model=input_size, d_k=input_size, d_v=input_size,
                                                         in_ch=in_ch, kernel_size=kernel_size, dilation=dilation))


    def forward(self, audio, spike):
        out_audio = audio
        out_spike = spike
        skip_audio = 0.
        skip_spike = 0.
        residual_audio = audio
        residual_spike = spike
        for i in range(self.layer):
            out_audio = self.audio_encoder[i](out_spike, out_audio, out_audio)
            out_spike = self.spike_encoder[i](out_audio, out_spike, out_spike)
            out_audio = out_audio + residual_audio
            out_audio = self.LayernormList_audio[i](out_audio)
            out_spike = out_spike + residual_spike
            out_spike = self.LayernormList_spike[i](out_spike)
            residual_audio = out_audio
            residual_spike = out_spike
            skip_audio += out_audio
            skip_spike += out_spike
        out = torch.cat((skip_audio, audio, out_spike, spike), dim=1)
        out = self.projection(out)
        out = self.layernorm_out(out)
        return out        

class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out



# The separation network adapted from Conv-TasNet
class DPRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer, kernel=3,
                 rnn_type='LSTM', norm='ln', K=160, dropout=0,
                 bidirectional=True,  
                 CMCA_layer_num=3
                 ):
        super(DPRNN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        self.K = K
        self.layer = layer
        self.norm = nn.GroupNorm(1, input_dim, eps=1e-8)
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, 1, bias=False)

        self.fusion = MultiLayerCrossAttention(input_size=input_length, layer=CMCA_layer_num, in_ch=hidden_dim, kernel_size=kernel,
                                               dilation=1)
        
        self.DPRNN= nn.ModuleList([])
        for i in range(self.layer):
            self.DPRNN.append(Dual_RNN_Block(hidden_dim, hidden_dim,
                                     rnn_type=rnn_type, norm=norm, dropout=dropout,
                                     bidirectional=bidirectional))
                    
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()    
        # output layer
        # self.output = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 1),
        #                             nn.Tanh()
        #                             )
        # self.output_gate = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 1),
        #                                  nn.Sigmoid()
        #                                  )
        self.end_conv1x1 = nn.Conv1d(hidden_dim, output_dim, 1, bias=False)
        self.linear = nn.Conv1d(hidden_dim*2,hidden_dim,1)
        
    def forward(self, input, spike):
        
        # input shape: (B, N, L)
        
        # normalization
        input = self.norm(input)    ##8，128，1624
        
        input = self.conv1d(input)  #(8,64,1624)


        spike = F.pad(spike, (0, (input.size(2) - spike.size(2))))  #(8,64,1624)
        input = self.fusion(input, spike)

        audio, gap = self._Segmentation(input, self.K)  ## 8，64，100，34

        for i in range(self.layer):
            audio = self.DPRNN[i](audio)    
        
        
        B, _, K, S = audio.shape
        audio = audio.view(B,-1, K, S)
        # [B*spks, N, L]
        output = self._over_add(audio, gap)  #(8,64,1624)        
        output = self.prelu(output)
        output = self.end_conv1x1(output)   ##(8,128,1624)
        output = self.activation(output)

        
        return output
    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
