import torch
import torch.nn as nn
from torch.nn import functional as F
from .SpatialChannelTrans import SpatialChannelTrans
from einops import rearrange
from mamba_ssm import Mamba2
from argparse import Namespace
from models import register

class STaRNet(nn.Module):

    def __init__(self, args):
        input_angle = args.input_angle
        output_depth = args.output_depth
        interp_angle = args.interp_angle
        in_channel = args.in_channel
        up_scale_times = args.up_scale_times
        embedding_dim = args.embedding_dim
        window_size = args.window_size
        num_heads = args.num_heads
        hidden_dim = args.hidden_dim
        num_transBlock = args.num_transBlock
        attn_dropout_rate = args.attn_dropout_rate
        f_maps = args.f_maps
        input_dropout_rate = args.input_dropout_rate
        act_layer = args.act_layer
        is_conv3d = args.is_conv3d
        down_sample_d_size = args.down_sample_d_size
        is_GDFN = args.is_GDFN
        trans_gate = args.trans_gate
        trans_gate_share_weight = args.trans_gate_share_weight
        out_channel = args.out_channel
        spatial_block_type = getattr(args, 'spatial_block_type', 'mamba')
        mamba_d_state = getattr(args, 'mamba_d_state', 16)
        mamba_d_conv = getattr(args, 'mamba_d_conv', 4)
        mamba_expand = getattr(args, 'mamba_expand', 2)
        mamba_headdim = getattr(args, 'mamba_headdim', 4)
        super().__init__()
        if trans_gate is None:
            trans_gate = [0 for _ in range(len(f_maps))]
        self.output_depth = output_depth
        self.act_layer = act_layer
        self.is_conv3d = is_conv3d
        self.down_sample_d_size = down_sample_d_size
        interp_angle = [input_angle] + interp_angle
        layers = []
        for idx in range(1, len(interp_angle)):
            layers.append(nn.Conv2d(interp_angle[idx - 1], interp_angle[idx], 3, padding=(3 - 1) // 2, stride=1))
        layers.append(nn.BatchNorm2d(interp_angle[-1]))
        if act_layer == 'nn.ReLU()':
            layers.append(nn.ReLU())
        elif act_layer == 'nn.LeakyReLU(0.1, inplace=True)':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        else:
            raise Exception('invalid act_layer: ' + act_layer)
        self.f1 = nn.Sequential(*layers)
        kSize = 3
        layers = []
        if isinstance(interp_angle, list):
            interp_angle = interp_angle[-1]
        self.interp_angle = interp_angle
        self.f_maps = f_maps
        self.encoders, self.encoder_trans = self.temporalSqueeze(f_maps=[in_channel] + f_maps, interp_angle=interp_angle, act_layer=act_layer, num_transBlock=num_transBlock, num_heads=num_heads, window_size=window_size, attn_dropout_rate=attn_dropout_rate, is_GDFN=is_GDFN, input_dropout_rate=input_dropout_rate, embedding_dim_max=embedding_dim, trans_gate=trans_gate, spatial_block_type=spatial_block_type, mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand, mamba_headdim=mamba_headdim)
        self.decoders, self.decoder_trans = self.temporalExcitation(f_maps=f_maps[::-1] + [out_channel], interp_angle=interp_angle, act_layer=act_layer, num_transBlock=num_transBlock, num_heads=num_heads, window_size=window_size, attn_dropout_rate=attn_dropout_rate, is_GDFN=is_GDFN, input_dropout_rate=input_dropout_rate, embedding_dim_max=embedding_dim, trans_gate=trans_gate, spatial_block_type=spatial_block_type, mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand, mamba_headdim=mamba_headdim)
        if trans_gate_share_weight:
            del self.decoder_trans
            self.decoder_trans = nn.ModuleList([])
            for _ in range(len(self.encoder_trans)):
                self.decoder_trans.append(self.encoder_trans[len(self.encoder_trans) - _ - 1])
        self.trans_layers = self.prepare_trans(interp_angle // 2 ** len(f_maps), f_maps[-1], embedding_dim, act_layer, num_transBlock, num_heads, hidden_dim, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, spatial_block_type, mamba_d_state, mamba_d_conv, mamba_expand, mamba_headdim)

    def prepare_trans(self, seq_length, dim_before_trans, embedding_dim, act_layer, num_transBlock, num_heads, hidden_dim, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, spatial_block_type='mamba', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=4):
        conv_before_trans = SingleConv(dim_before_trans, embedding_dim, kernel_size=3, stride=1, padding=1, act_layer=act_layer)
        conv_after_trans = SingleConv(embedding_dim, dim_before_trans, kernel_size=3, stride=1, padding=1, act_layer=act_layer)
        layers = nn.ModuleList([])
        layers.append(conv_before_trans)
        for _ in range(num_transBlock):
            layers.append(SpatialChannelTrans(seq_length=seq_length, embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim, space_window_size=window_size, attn_dropout_rate=attn_dropout_rate, input_dropout_rate=input_dropout_rate, is_GDFN=is_GDFN, spatial_block_type=spatial_block_type, mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand, mamba_headdim=mamba_headdim))
        layers.append(conv_after_trans)
        layers = nn.Sequential(*layers)
        return layers

    def temporalSqueeze(self, f_maps, interp_angle, act_layer, num_transBlock, num_heads, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, embedding_dim_max, trans_gate, spatial_block_type='mamba', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=4):
        model_list = nn.ModuleList([])
        trans_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            encoder_layer = SqueezeLayer(in_channels=f_maps[idx - 1], out_channels=f_maps[idx], act_layer=self.act_layer, is_conv3d=self.is_conv3d, down_sample_d_size=self.down_sample_d_size)
            model_list.append(encoder_layer)
            if trans_gate[idx - 1] == 1:
                seq_length = interp_angle // 2 ** idx
                dim_before_trans = f_maps[idx]
                embedding_dim = min(dim_before_trans * 4, embedding_dim_max)
                hidden_dim = embedding_dim * 4
                trans_layer = self.prepare_trans(seq_length, dim_before_trans, embedding_dim, act_layer, num_transBlock, num_heads, hidden_dim, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, spatial_block_type, mamba_d_state, mamba_d_conv, mamba_expand, mamba_headdim)
            else:
                trans_layer = nn.Identity()
            trans_list.append(trans_layer)
        return (model_list, trans_list)

    def temporalExcitation(self, f_maps, interp_angle, act_layer, num_transBlock, num_heads, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, embedding_dim_max, trans_gate, spatial_block_type='mamba', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=4):
        model_list = nn.ModuleList([])
        trans_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            decoder_layer = ExcitationLayer(in_channels=f_maps[idx - 1], out_channels=f_maps[idx], if_up_sample=True, act_layer=self.act_layer)
            model_list.append(decoder_layer)
            if trans_gate[len(f_maps) - idx - 1] == 1:
                seq_length = interp_angle // 2 ** (len(f_maps) - idx)
                dim_before_trans = f_maps[idx - 1]
                embedding_dim = min(dim_before_trans * 4, embedding_dim_max)
                hidden_dim = embedding_dim * 4
                trans_layer = self.prepare_trans(seq_length, dim_before_trans, embedding_dim, act_layer, num_transBlock, num_heads, hidden_dim, window_size, attn_dropout_rate, input_dropout_rate, is_GDFN, spatial_block_type, mamba_d_state, mamba_d_conv, mamba_expand, mamba_headdim)
            else:
                trans_layer = nn.Identity()
            trans_list.append(trans_layer)
        return (model_list, trans_list)

    def forward(self, x):
        batch, seq, angle, H, W = x.shape
        x = torch.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        x = self.f1(x)
        x = torch.reshape(x, (batch, seq, -1, H, W))
        encoders_features = []
        for encoder_id in range(len(self.encoders)):
            encoder = self.encoders[encoder_id]
            before_down, x = encoder(x)
            encoders_features.insert(0, before_down)
            encoder_trans_layer = self.encoder_trans[encoder_id]
            x = encoder_trans_layer(x)
        x = self.trans_layers(x)
        for decoder, encoder_features, decoder_trans_layer in zip(self.decoders, encoders_features, self.decoder_trans):
            x = decoder_trans_layer(x)
            x = decoder(x, encoder_features)
        H, W = (x.shape[3], x.shape[4])
        x = x.permute(0, 1, 3, 4, 2)
        x = F.interpolate(x, size=(H, W, self.output_depth), mode='trilinear', align_corners=False)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.squeeze(1)
        return x

class SqueezeLayer(nn.Module):

    def __init__(self, in_channels, out_channels, act_layer, down_sample_d_size=3, is_conv3d=True, kernel_size=3):
        super(SqueezeLayer, self).__init__()
        self.conv_net = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, if_encoder=True, act_layer=act_layer, is_conv3d=is_conv3d)
        self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=(down_sample_d_size, 3, 3), stride=(2, 1, 1), padding=(down_sample_d_size // 2, 1, 1))

    def forward(self, x):
        before_down = self.conv_net(x)
        x = self.down_sample(before_down)
        return (before_down, x)

class ExcitationLayer(nn.Module):

    def __init__(self, in_channels, out_channels, act_layer, if_up_sample=True, kernel_size=3):
        super(ExcitationLayer, self).__init__()
        self.conv_net = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, if_encoder=False, act_layer=act_layer)
        self.if_up_sample = if_up_sample
        self.up_sample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))

    def forward(self, x, encoder_features):
        if self.if_up_sample:
            x = self.up_sample(x)
        x += encoder_features
        x = self.conv_net(x)
        return x

class SingleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, act_layer, is_conv3d=True, stride=1, padding=1):
        super(SingleConv, self).__init__()
        if is_conv3d:
            self.add_module('Conv3d', nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        else:
            self.add_module('Conv3d', nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), padding=(0, padding, padding), stride=stride))
        if act_layer == 'nn.ReLU()':
            self.add_module('ReLU', nn.ReLU(inplace=True))
        elif act_layer == 'nn.LeakyReLU(0.1, inplace=True)':
            self.add_module('LeakyReLU', nn.LeakyReLU(0.1, inplace=True))
        else:
            raise Exception('invalid act_layer: ' + act_layer)

class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, if_encoder, act_layer, is_conv3d=True, kernel_size=3):
        super(DoubleConv, self).__init__()
        if if_encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = (conv1_out_channels, out_channels)
        else:
            conv1_in_channels, conv1_out_channels = (in_channels, out_channels)
            conv2_in_channels, conv2_out_channels = (out_channels, out_channels)
        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=1, act_layer=act_layer, is_conv3d=is_conv3d))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=1, act_layer=act_layer, is_conv3d=is_conv3d))

@register('STaRNet_multi_mamba')
def make_STaRNet_multi_mamba(input_angle=13, output_depth=39, interp_angle=64, in_channel=1, out_channel=1, up_scale_times=2, embedding_dim=128, num_heads=8, hidden_dim=128 * 4, window_size=7, num_transBlock=1, attn_dropout_rate=0.1, f_maps=[16, 32, 64], input_dropout_rate=0, act_layer='nn.ReLU()', is_conv3d=True, down_sample_d_size=3, is_GDFN=False, trans_gate=None, trans_gate_share_weight=False, spatial_block_type='mamba', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_headdim=4):
    args = Namespace()
    args.input_angle = input_angle
    args.output_depth = output_depth
    args.interp_angle = interp_angle
    args.in_channel = in_channel
    args.up_scale_times = up_scale_times
    args.embedding_dim = embedding_dim
    args.num_heads = num_heads
    args.hidden_dim = hidden_dim
    args.window_size = window_size
    args.num_transBlock = num_transBlock
    args.attn_dropout_rate = attn_dropout_rate
    args.f_maps = f_maps
    args.input_dropout_rate = input_dropout_rate
    args.act_layer = act_layer
    args.is_conv3d = is_conv3d
    args.down_sample_d_size = down_sample_d_size
    args.is_GDFN = is_GDFN
    args.trans_gate = trans_gate
    args.trans_gate_share_weight = trans_gate_share_weight
    args.out_channel = out_channel
    args.spatial_block_type = spatial_block_type
    args.mamba_d_state = mamba_d_state
    args.mamba_d_conv = mamba_d_conv
    args.mamba_expand = mamba_expand
    args.mamba_headdim = mamba_headdim
    return STaRNet(args)
STaRNet_multi_mamba = STaRNet
