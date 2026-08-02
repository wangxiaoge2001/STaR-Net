import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register

class _BaseRLFM(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder_spec = encoder_spec
        self.encoder = models.make(encoder_spec)
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 27
            imnet_in_dim += 3
            if self.cell_decode:
                imnet_in_dim += 3
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def _resize_prediction(self, feat, inp, scale):
        if len(feat.shape) == 4:
            if len(inp.shape) == 5:
                target_h = inp.shape[3]
                target_w = inp.shape[4]
            else:
                target_h = inp.shape[2]
                target_w = inp.shape[3]
            return F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False).unsqueeze(1)
        target_d = inp.shape[1]
        target_h = inp.shape[2]
        target_w = inp.shape[3]
        return F.interpolate(feat, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)

@register('rlfm_1p')
class RLFM_1p(_BaseRLFM):

    def forward(self, inp, scale):
        self.feat = self.encoder(inp)
        return self.feat

@register('rlfm')
class RLFM(_BaseRLFM):

    def forward(self, inp, scale):
        self.feat = self.encoder(inp)
        self.feat = self._resize_prediction(self.feat, inp, scale)
        return self.feat

@register('rlfm-seq')
class RLFM_seq(_BaseRLFM):

    def forward(self, inp, scale):
        self.feat = self.encoder(inp)
        if len(self.feat.shape) == 4:
            self.feat = self._resize_prediction(self.feat, inp, scale)
        return self.feat
