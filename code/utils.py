import os
import shutil
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD


cmap = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 255],
        [255, 0, 255],
        [255, 0, 0],
        [255, 255, 0],
        [255, 255, 255],
    ],
    dtype=torch.float32,
).unsqueeze(0).unsqueeze(1)
cmap = torch.nn.functional.interpolate(
    cmap,
    size=[256, 3],
    mode="bilinear",
    align_corners=True,
).squeeze().to(torch.uint8)


def get_rTG_3D_v2(image, in_channels):

    rTG = torch.stack([torch.max(torch.abs(image - image[i]), dim=0).values for i in range(in_channels)])

    rTG /= (torch.max(image).float() + 1e-7)
    rTG[torch.isnan(rTG)] = 0
    rTG[torch.isinf(rTG)] = 0

    return rTG


class LPIPS_vgg16(nn.Module):
    def __init__(self, pretrained=True, net='vgg', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True, is_single_channel=False, is_zero_one=False):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS_vgg16, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer(is_single_channel=is_single_channel, is_zero_one=is_zero_one)
        self.is_single_channel = is_single_channel

        if(self.pnet_type in ['vgg','vgg16']):
            self.chns = [64,128,256,512,512]
        else:
            raise NotImplementedError
        
        self.L = len(self.chns)

        if self.is_single_channel:
            self.net = vgg16_singlechannel(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        else:
            self.net = vgg16(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
            
        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'pretrained_weights/lpips_weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1
            
        if self.is_single_channel:
            in0_input = in0.reshape(-1,1,in0.shape[-2], in0.shape[-1])
            in1_input = in1.reshape(-1,1,in1.shape[-2], in1.shape[-1])
        else:
            in0_input = in0
            in1_input = in1
        in0_input, in1_input = (self.scaling_layer(in0_input), self.scaling_layer(in1_input)) if self.version=='0.1' else (in0_input, in1_input)

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.is_single_channel):
            if(self.lpips):
                if(self.spatial):
                    res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True).mean(dim=0,keepdim=True) for kk in range(self.L)]
            else:
                if(self.spatial):
                    res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True).mean(dim=0,keepdim=True) for kk in range(self.L)]
        else:
            if(self.lpips):
                if(self.spatial):
                    res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            else:
                if(self.spatial):
                    res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]
        
        
        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)):
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ScalingLayer(nn.Module):
    def __init__(self, is_single_channel=False, is_zero_one=False):
        super(ScalingLayer, self).__init__()

        if is_single_channel:
            if is_zero_one:
                self.shift =  torch.Tensor([0.485, 0.456, 0.406]).mean(dim=0).item()
                self.scale =  torch.Tensor([0.229, 0.224, 0.225]).mean(dim=0).item()
            else:
                self.shift =  torch.Tensor([-.030,-.088,-.188]).mean(dim=0).item()
                self.scale =  torch.Tensor([.458,.448,.450]).mean(dim=0).item()
        else:
            if is_zero_one:
                self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None])
                self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None])
            else:
                self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
                self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class vgg16_singlechannel(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, lpips=True, use_dropout=False):
        super(vgg16_singlechannel, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        self.slice1.add_module(str(0), torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1) ))

        self.slice1[0].weight = nn.Parameter(vgg_pretrained_features[0].weight.sum(dim=1,keepdim=True))
        self.slice1[0].bias = nn.Parameter(vgg_pretrained_features[0].bias)

        for x in range(1, 4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        if not np.isnan(v):
            self.v = (self.v * self.n + v * n) / (self.n + n)
            self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') and input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            print(1)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
