import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
# from espnet2.enh.decoder.stft_decoder import STFTDecoder
# from espnet2.enh.encoder.stft_encoder import STFTEncoder
# from espnet2.enh.layers.complex_utils import new_complex_like
# from espnet2.enh.separator.abs_separator import AbsSeparators
# from espnet2.torch_utils.get_layer_from_string import get_layer
# from BASEN import AudioEncoder, EEGEncoder
# from utility.CMFusion import CMFusion


# def new_complex_like(
#     ref: Union[torch.Tensor, ComplexTensor],
#     real_imag: Tuple[torch.Tensor, torch.Tensor],
# ):
#     if isinstance(ref, ComplexTensor):
#         return ComplexTensor(*real_imag)
#     elif is_torch_complex_tensor(ref):
#         return torch.complex(*real_imag)
#     else:
#         raise ValueError(
#             "Please update your PyTorch version to 1.9+ for complex support."
#         )

class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        # approx_qk_dim=512,
        eps=1e-5
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        E = 4
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.ReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    nn.ReLU(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    nn.ReLU(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.ReLU(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape    #8,4,256,128
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))  #实际上前后的T/Q是一样的

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q] #
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q] 32,4,256,128
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q] 32,256,512
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]   #512

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T] 32,256,256
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class TFGridNet(nn.Module):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        n_layers=6,
        lstm_hidden_units=16,   #emb_dim的四倍
        attn_n_head=4,
        # attn_approx_qk_dim=512,
        emb_dim=4,
        emb_ks=4,
        emb_hs=1,
        eps=1.0e-5,
    ):
        super().__init__()
        self.n_srcs = 1
        self.n_layers = n_layers
        self.n_imics = 1
        # assert n_fft % 2 == 0
        # n_freqs = n_fft // 2 + 1
        # self.ref_channel = ref_channel

        # self.enc = STFTEncoder(
        #     n_fft, n_fft, stride, window=window, use_builtin_complex=use_builtin_complex
        # )
        # self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)
        

        n_freqs = 1624 #F==T
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        # self.attention = CMFusion(128, 4, emb_dim)    #通道数固定为32
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    # approx_qk_dim=attn_approx_qk_dim,
                    eps=eps,
                )
            )
        # self.convfusion = nn.Sequential(nn.Conv2d(4,8,1),
        #                                 nn.Conv2d(8,4,1)
        #                                 )

        # self.deconv = nn.ConvTranspose2d(emb_dim, self.n_srcs * 2, ks, padding=padding)
        self.proj = nn.Conv2d(emb_dim, 2, 1)
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(64, 128, 1)
                                   )
        self.norm = nn.GroupNorm(1, 128, eps=1e-8)
        self.conv1d = nn.Conv1d(128, 64, 1, bias=False)

    def forward(self, input, spike):
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """
        # normalization
        input = self.norm(torch.squeeze(input,-1))
        
        input = self.conv1d(input)  #(8,64,7315)

        #fusion
        # output = self.BN(output)
        spike = F.pad(spike, (0, (input.size(2) - spike.size(2))))  #(8,64,1624)
        input = torch.unsqueeze(input,1)
        spike = torch.unsqueeze(spike,1)
        batch = torch.cat([input,spike],1)

        batch = self.conv(batch)  # [B, -1, T, F] 

        #分离网络
        for ii in range(self.n_layers):
            # batch = self.attention(batch)
            batch = self.blocks[ii](batch)  # [B, -1, T, F]
            # batch = self.convfusion(batch)

        batch = self.proj(batch)  # [B, n_srcs*2, T, F]
        batch = self.output(batch)  #8,2,256,128

        # batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        # batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))

        # batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]  # [B, n_srcs, -1]

        # batch = self.pad2(batch.view([n_batch, self.num_spk, -1]), n_samples)

        # batch = batch * mix_std_  # reverse the RMS normalization

        # batch = [batch[:, src] for src in range(self.num_spk)]

        return batch

    # @property
    # def num_spk(self):
    #     return self.n_srcs

    # @staticmethod
    # def pad2(input_tensor, target_len):
    #     input_tensor = torch.nn.functional.pad(
    #         input_tensor, (0, target_len - input_tensor.shape[-1])
    #     )
    #     return input_tensor

'''class ExpandGridNet(nn.Module):
    def __init__(self, enc_channel=64, feature_channel=32, encoder_kernel_size=32, layer_per_stack=8, stack=2,
                 kernel=3):
        super().__init__()

        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        # self.encoder_kernel = encoder_kernel_size
        self.encoder_kernel = 1
        self.stride = 8
        self.win = 32
        self.layer = layer_per_stack
        self.stack = stack
        self.kernel = kernel
        # EEG encoder
        self.spike_encoder = EEGEncoder(layer=8, enc_channel=enc_channel, feature_channel=feature_channel)
        # audio encoder
        self.audio_encoder = AudioEncoder(self.enc_channel, self.feature_channel, self.feature_channel*4,
                              self.layer, self.kernel)
        # self.orig_audio = nn.Conv1d(1, self.enc_channel, self.encoder_kernel, bias=False, stride=self.stride)
        self.conv2d = nn.Conv2d(257,256,self.encoder_kernel)

        # TCN separation network from Conv-TasNet
        self.gridnet = TFGridNet()

        # self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        # self.decoder = nn.ConvTranspose1d(self.enc_channel, 1, self.encoder_kernel, bias=False, stride=self.stride)
        self.Tconv2d = nn.ConvTranspose2d(256, 257, self.encoder_kernel)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
    
    def forward(self,input,spike_input):
        # enc_output_orig = self.orig_audio(input)
        output = torch.stft(input.squeeze(),n_fft=254, hop_length=114, return_complex=False)  #(8,128,257,2)  (B, F, T, C)  这里是原始音频      #128:114
        output = self.conv2d(output.transpose(1,2)) #改变大小--(8,256,128,2)
        output = output.permute(0,3,1,2)    #(8,2,256,128) (B, C, T, F)
        # output, rest = self.pad_signal(input)
        #spike, rest = self.pad_signal(spike_input)
        batch_size = output.size(0)
        
        
        # waveform encoder
        enc_output = self.audio_encoder(output)  # B, N, L
        enc_output_spike = self.spike_encoder(spike_input).permute(0,2,1) #(8,256,128)
        #扩展维度
        # enc_output_spike = F.pad(enc_output_spike, (0, (enc_output.size(2) - enc_output_spike.size(2))))
        expand = torch.cat((enc_output, enc_output_spike.unsqueeze(1)), dim=1) #(8,3,256,128)
        # generate masks
        # masks = torch.sigmoid(self.gridnet(output)).view(batch_size, self.num_spk, self.enc_channel, -1)  # B, C, N, L
        masks = torch.sigmoid(self.gridnet(expand)) # B, C, N, L (8,2,256,128)
        masked_multi = output * masks  # B, 2, T, F
        masked_output = self.Tconv2d(masked_multi.transpose(1,2)).permute(0,3,1,2) # B, F, T, C

        
        # waveform decoder
        # output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_channel, -1))  # B*C, 1, L
        # output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = torch.istft(masked_output, n_fft=254, hop_length=114)
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        # return output, enc_output_spike.unsqueeze(1).repeat(1, 2, 1, 1), masked_multi
        return output

def test_conv_tasnet():
    x = torch.rand(8, 1, 29184)
    y = torch.rand(8, 64, 256)
    nnet = ExpandGridNet()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()'''