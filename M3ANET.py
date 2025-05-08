import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utility.layers import GraphConvolution,Linear
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D, ConvTrans1D
from utility import models as models
import numpy  as np
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_
from models.ss2d import SS2D
from models.csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
from models.csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4
from models.groupmamba import FFN,PVT2FFN


class Chebynet(nn.Module):
    def __init__(self, in_channel=128, k_adj=3):
        super(Chebynet, self).__init__()
        self.K = k_adj
        self.gc = nn.ModuleList()
        for i in range(k_adj):
            self.gc.append(GraphConvolution(in_channel, in_channel))

    def forward(self, x ,L):
        
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result


class EEGEncoder(nn.Module):
    def __init__(self, num_electrodes=128, k_adj=3, enc_channel=128, feature_channel=64, kernel_size=8,
                 rnn_type='LSTM', norm='ln', K=160,  dropout=0, bidirectional=False, kernel=3, skip=True):
        super(EEGEncoder, self).__init__()
        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.kernel = kernel
        self.proj_kernel_size = kernel_size
        self.stride = 4
        self.K=k_adj
        self.K = K
        self.BN1 = nn.BatchNorm1d(29184)
        self.layer1 = Chebynet(128, k_adj)
        self.projection = nn.Conv1d(128, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes , num_electrodes).cuda())
        nn.init.xavier_normal_(self.A)

        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            ResBlock(feature_channel, feature_channel),
            ResBlock(feature_channel, enc_channel),
            ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, feature_channel, 1),
        )

    def forward(self, spike):


        spike = self.BN1(spike.transpose(1, 2)).transpose(1, 2) ##8,128,29184
        L = normalize_A(self.A)
        output = self.layer1(spike, L)  ##8,128,29184
        
        output = self.projection(output)    
        output = self.eeg_encoder(output)   




        return output

  
class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):  
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x

class  VSSS(nn.Module):
    def __init__(self, input_dim, layer) -> None:
        super(VSSS, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//4, bias=True)
        self.fc2 = nn.Linear(input_dim//4, input_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.d_state = 1
        self.d_conv =3
        self.expand = 1
        #share parameters
        self.mamba_block = SS2D(
            d_model=1,
            d_state=self.d_state,
            ssm_ratio=self.expand,
            d_conv=self.d_conv
        )
        
        self.layer = layer
        
        self.channel = 4
        self.mlp = PVT2FFN(in_features=self.channel, hidden_features=int(self.channel * 4))
        self.norm= nn.LayerNorm(4)
        self.H = 128
        self.W = 1624
    
    def forward(self, w1, w2, w3, w4):
        w1, w2, w3, w4 = w1.unsqueeze(-1), w2.unsqueeze(-1), w3.unsqueeze(-1), w4.unsqueeze(-1)
        batch = w1.size(0)
        x_mamba = torch.cat([w1, w2, w3, w4], dim=-1)

        for i in range(self.layer):
            
            if i!=0:
                w1, w2, w3, w4 = torch.chunk(x_mamba, 4, dim=-1)
            
            # Channel Affinity
            channel = rearrange(x_mamba, 'b h w c -> b (h w) c', b=batch, h=self.H, w=self.W, c=4)
            z = channel.permute(0, 2, 1).mean(dim=2)    #(8,4)
            fc_out_1 = self.relu(self.fc1(z))   #(8,1)
            fc_out_2 = self.sigmoid(self.fc2(fc_out_1))     #(8,4)

            mamba1 = self.mamba_block(w1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)   
            mamba2 = self.mamba_block(w2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
            mamba3 = self.mamba_block(w3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
            mamba4 = self.mamba_block(w4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)
            x_mamba = torch.cat([mamba1, mamba2, mamba3, mamba4], dim=-1)   #      #(8,128,1624,4)
            
            x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=batch, h=self.H, w=self.W, c=4) 
            # Channel Modulation
            x_mamba = x_mamba * fc_out_2.unsqueeze(1)
            x_mamba = self.norm(x_mamba)
            
            x_mamba = channel + x_mamba 
            
            x_mamba = x_mamba + self.mlp(self.norm(x_mamba), self.H, self.W)
            x_mamba = rearrange(x_mamba, 'b (h w) c -> b h w c', b=batch, h=self.H, w=self.W, c=4)
            

        return x_mamba  



class M3ANET(nn.Module):
    def __init__(self, L1=0.0025, L2=0.005, L3=0.01, L4=0.02, enc_channel=128, feature_channel=64, encoder_kernel_size=8, layers=4,
                rnn_type='LSTM', norm='ln', K=250, dropout=0, bidirectional=True, CMCA_kernel=3,
                CMCA_layer_num=3):
        super(M3ANET, self).__init__()

        # hyper parameters
        #self.num_spk = num_spk
        self.L1 = int(L1 * 14700)
        self.L2 = int(L2 * 14700)
        self.L3 = int(L3 * 14700)
        self.L4 = int(L4 * 14700) 
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        self.encoder_kernel = encoder_kernel_size
        self.CMCA_kernel = CMCA_kernel
        self.stride = 16
        self.win = 64
        self.layer = layers
        self.K = K
        # EEG encoder
        self.spike_encoder = EEGEncoder(enc_channel=enc_channel, feature_channel=feature_channel)
        # audio encoder
        self.encoder_1d_L1 = Conv1D(1, enc_channel, self.L1, stride=self.L1 // 2, padding=0)
        self.encoder_1d_L2 = Conv1D(1, enc_channel, self.L2, stride=self.L1 // 2, padding=0)
        self.encoder_1d_L3 = Conv1D(1, enc_channel, self.L3, stride=self.L1 // 2, padding=0)
        self.encoder_1d_L4 = Conv1D(1, enc_channel, self.L4, stride=self.L1 // 2, padding=0)   #xm
        
        self.vsss_block = VSSS(input_dim = 4, layer= 2) 

        self.ln = ChannelwiseLayerNorm(4*enc_channel)
        self.ln1 = ChannelwiseLayerNorm(enc_channel)
        # n x N x T => n x O x T
        self.proj = Conv1D(4*enc_channel, enc_channel, 1)
        # DPRNN separation network
        self.DPRNN = models.DPRNN(self.enc_channel, self.enc_channel, self.feature_channel,
                              self.layer, self.CMCA_kernel ,rnn_type=rnn_type, norm=norm, K=self.K, dropout=dropout,
                               bidirectional=bidirectional,  CMCA_layer_num=CMCA_layer_num)

        # output decoder
        self.decoder = ConvTrans1D(enc_channel, 1, self.L1 , stride=self.L1 // 2, bias=True)
        self.linear = nn.Linear(4,1)
        self.conv1d = nn.Conv1d(self.enc_channel, self.feature_channel, 1)
        
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
        
    def forward(self, input, spike_input):  #input(4,1,29184)  spike_input(4,128,29184)
        
        # padding
        output, rest = self.pad_signal(input) 

        batch_size = output.size(0)
        enc_output_spike = self.spike_encoder(spike_input) #（8，64，270）

        w1 = F.relu(self.encoder_1d_L1(output))  #(8,128,1624)
        T = w1.shape[-1]    #1624
        xlen1 = output.shape[-1]    #29264
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2  # 29287
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3  # 29508
        xlen4 = (T - 1) * (self.L1 // 2) + self.L4  # 29361
        
        w2 = F.relu(self.encoder_1d_L2(F.pad(output, (0, xlen2 - xlen1), "constant", 0)))  #（8，128，1624）
        w3 = F.relu(self.encoder_1d_L3(F.pad(output, (0, xlen3 - xlen1), "constant", 0)))   #（8，128，1624）
        w4 = F.relu(self.encoder_1d_L4(F.pad(output, (0, xlen4 - xlen1), "constant", 0)))     # （8，128，1624）
        
        enc_output = torch.cat([w1, w2, w3, w4], 1) #8,512,1624

        mamba_enc = self.vsss_block(w1, w2, w3, w4) #B，C，T，4   
        
        x_mamba = torch.sigmoid(self.DPRNN(torch.squeeze(self.linear(mamba_enc), -1), enc_output_spike)).view(batch_size, self.enc_channel, -1) #8，128，1624
        masks = self.ln1(x_mamba)   
        enc_output = self.ln(enc_output)    #8，512，1624
        enc_output = self.proj(enc_output)  #8，128，1624

        masked_output = enc_output * masks  #8，128，1624
        # waveform decoder
        output = self.decoder(masked_output)  # B*C, 1, L 
        output = F.pad(output, (0, xlen1 - output.size(1)), "constant", 0) 
        output = torch.unsqueeze(output, dim=1)   
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L  8，1，29184
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        pad_spike = F.pad(enc_output_spike, (0, 1624-270)).view(batch_size, -1)
        mamba_enc = self.conv1d(torch.squeeze(self.linear(mamba_enc),-1)).view(batch_size, -1)

        return output, pad_spike, mamba_enc
    


def test_conv_tasnet():
    x = torch.rand(8, 1, 29184)
    y = torch.rand(8, 128, 29184)
    nnet = M3ANET()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
