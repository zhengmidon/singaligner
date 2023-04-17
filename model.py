from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from module.layers import ConvNorm, LinearNorm
from module.common_layers import MultiheadAttention, EncSALayer, DecSALayer, SinusoidalPositionalEmbedding
from module.common_layers import Block, ResBlock
from utils.utils import to_gpu, get_mask_from_lengths
from utils.data_utils import build_phone_encoder
from conformer.encoder import ConformerBlock


class Conv2dPrenet(nn.Module):
    def __init__(self, in_dim, prenet_dim, prenet_k_sizes, prenet_d_sizes, strides, pool_k_sizes):
        super(Conv2dPrenet, self).__init__()
        assert len(prenet_k_sizes) == len(prenet_d_sizes)
        layers = []
        for i in range(len(prenet_k_sizes)):
            se = nn.Sequential(
                Block(in_channels=1, out_channels=prenet_dim, \
                kernel_size=prenet_k_sizes[i], stride=strides[i], \
                padding=(0, int(prenet_d_sizes[i][1] * (prenet_k_sizes[i][1] - 1) / 2)), 
                dilation=prenet_d_sizes[i], groups=8),
                nn.MaxPool2d(kernel_size=pool_k_sizes[i], stride=1, padding=(0,1), dilation=1)
                )
            layers.append(se)
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.convs[0](x)
        for layer in self.convs[1:]:
            y = y + layer(x)
        y = F.dropout(F.relu(y), p=0.2, training=True)
        y = y.squeeze(2).transpose(1,2)
        return y # [B,N,prenet_dim]

class Conv1dPrenet(nn.Module):
    def __init__(self, in_dim, prenet_dim, prenet_k_sizes, prenet_d_sizes):
        super(Conv1dPrenet, self).__init__()
        assert len(prenet_k_sizes) == len(prenet_d_sizes)
        self.convs = nn.Sequential(
            nn.ConstantPad1d((prenet_d_sizes[0] * (prenet_k_sizes[0] - 1), 0), 0.0),
            nn.Conv1d(in_channels=in_dim, out_channels=prenet_dim, \
            kernel_size=prenet_k_sizes[0], stride=1, \
            # padding=int(prenet_d_sizes[0] * (prenet_k_sizes[0] - 1) / 2), \
            padding=0,
            dilation=prenet_d_sizes[0], bias=True),
            nn.BatchNorm1d(prenet_dim),
            nn.ConstantPad1d((prenet_d_sizes[1] * (prenet_k_sizes[1] - 1), 0), 0.0),
            nn.Conv1d(in_channels=prenet_dim, out_channels=prenet_dim, \
            kernel_size=prenet_k_sizes[1], stride=1, \
            # padding=int(prenet_d_sizes[1] * (prenet_k_sizes[1] - 1) / 2), \
            padding=0,
            dilation=prenet_d_sizes[1], bias=True),
            nn.BatchNorm1d(prenet_dim),
                )

    def forward(self, x):
        y = self.convs(x)
        y = y.transpose(1,2)
        return y # [B,N,hidden_size]

class LinearPrenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(LinearPrenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.2, training=True)
        return x

class MelEncoder(nn.Module):
    def __init__(self, hparams):
        super(MelEncoder, self).__init__()
        self.hparams = hparams

        if hparams.prenet_type == 'conv1d':
            self.prenet = Conv1dPrenet(
                hparams.n_mel_channels,
                hparams.hidden_size, hparams.prenet_k_sizes_1d, 
                hparams.prenet_d_sizes_1d)
        elif hparams.prenet_type == 'linear':
                self.prenet = LinearPrenet(
                hparams.n_mel_channels ,
                [hparams.hidden_size, hparams.hidden_size])
        elif hparams.prenet_type == 'conv2d':
            self.prenet = Conv2dPrenet(
                hparams.n_mel_channels,
                hparams.hidden_size, hparams.prenet_k_sizes_2d, 
                hparams.prenet_d_sizes_2d, hparams.strides_2d, hparams.pool_k_sizes_2d)
        else:
            raise NotImplementedError

        self.lstm = nn.LSTM(hparams.hidden_size,
                            int(hparams.hidden_size / 2), num_layers = hparams.melencoder_layers,
                            batch_first=True, bidirectional=True)

        self.layer_norm = nn.LayerNorm(hparams.hidden_size)

    def forward(self, x, input_lengths):
        x = self.prenet(x) # [B, N, hidden_size]
        # outputs = x
        # x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响

        self.lstm.flatten_parameters() # 把参数存放成连续块
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = self.layer_norm(outputs)
        
        return outputs  # [B,N,hidden_size]


class CTCDecoder(nn.Module):
    def __init__(self, hparams, vocab_size):
        super(CTCDecoder, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.ctcdecoder_layers = hparams.ctcdecoder_layers
        self.vocab_size = vocab_size

        self.ctcdecoder_lstm = nn.LSTM(hparams.hidden_size,
            int(hparams.hidden_size / 2), num_layers = hparams.ctcdecoder_layers,
                             bidirectional=True)

        self.ctc_linear = LinearNorm(hparams.hidden_size, vocab_size, bias=False,
                                       w_init_gain='tanh')

    def forward(self, x, input_lengths):
        '''
        x: [N, B, 2C]
        '''
        tgt_mask = get_mask_from_lengths(input_lengths) # [B,N]
        tgt_mask = tgt_mask.transpose(0,1).unsqueeze(2)
        input_lengths = input_lengths.cpu().numpy() # [B, N]
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响

        self.ctcdecoder_lstm.flatten_parameters() # 把参数存放成连续块
        x, _ = self.ctcdecoder_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x)

        outputs = F.dropout(F.relu(x), 0.2, self.training)
        logits = self.ctc_linear(outputs)
        log_prob = logits.log_softmax(2)
        return x, log_prob * tgt_mask# [N, B, hidden_size], [N, B, vocab_size]

class MELDecoder(nn.Module):
    def __init__(self, hparams):
        super(MELDecoder, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.meldecoder_layers = hparams.meldecoder_layers
        self.mel_bins = hparams.n_mel_channels

        self.meldecoder_lstm = nn.LSTM(hparams.hidden_size,
            int(hparams.hidden_size / 2), num_layers = hparams.meldecoder_layers,
                             bidirectional=True)

        self.mel_linear = LinearNorm(hparams.hidden_size, hparams.n_mel_channels, bias=False,
                                       w_init_gain='tanh')

    def forward(self, x, input_lengths):
        '''
        x: [N, B, ctcdecoder_lstm_dim]
        '''
        input_lengths = input_lengths.cpu().numpy() # [B, N]
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False) #去除PAD，压紧成一维，从而消除PAD对RNN的影响
        self.meldecoder_lstm.flatten_parameters() # 把参数存放成连续块
        x, _ = self.meldecoder_lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x)
        outputs = F.dropout(F.relu(x), 0.2, self.training)
        outputs = self.mel_linear(x)
        return outputs  # [N, B, n_mel_channels]


class ATTNAligner(nn.Module):
    def __init__(self, hparams):
        super(ATTNAligner, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.text_encoder = build_phone_encoder(hparams.data_dir)
        self.melencoder = MelEncoder(hparams)
        self.ctcdecoder = CTCDecoder(hparams, self.text_encoder.vocab_size)
        self.meldecoder = MELDecoder(hparams)
        self.layer_norm = nn.LayerNorm(hparams.hidden_size)
        self.bottleneck_linear = LinearNorm(hparams.hidden_size * 2, \
                                    hparams.hidden_size, bias=False,
                                       w_init_gain='tanh')

        self.register_buffer('spec_min', torch.FloatTensor([-12.0]))
        self.register_buffer('spec_max', torch.FloatTensor([0.0]))

    def parse_batch(self, batch):
        text_padded, text_lengths, mel_padded, mel_lengths, text, split_gt = batch
        text_padded = to_gpu(text_padded).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        mel_lengths = to_gpu(mel_lengths).long()

        return (
            (text_padded, text_lengths, mel_padded, mel_lengths),
            text_padded, text, split_gt)

    def mask_attentions(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths) # [B,N]
            B, N, T = outputs.shape
            attn_mask = mask[:,None,:].expand(B, T, N).permute(0, 2, 1)
            outputs.data.masked_fill_(attn_mask, 0.0)

        return outputs

    # 放缩到[-1,1]
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, inputs):
        # [B,T],[B],[B,MB,N],[B]
        text_inputs, text_lengths, mels, mel_lengths = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data
        mels = self.norm_spec(mels)

        mel_hidden = self.melencoder(mels, mel_lengths) # [B,N,hidden_size]

        decoder_input = mel_hidden.transpose(0,1)

        ctc_hidden, log_prob = self.ctcdecoder(decoder_input, mel_lengths) # [N,B,hidden_size], [N,B,vocab_size]
        bottleneck_hidden = torch.cat([decoder_input, ctc_hidden], dim = 2)
        bottleneck_hidden = self.bottleneck_linear(bottleneck_hidden)
        bottleneck_hidden = self.layer_norm(bottleneck_hidden)
        mel_output = self.meldecoder(bottleneck_hidden, mel_lengths) # [N,B,n_mel_channels]

        return self.denorm_spec(mel_output), log_prob, 0 # [N,B,n_mel_channels], [N,B,vocab_size], [B,N,T]
