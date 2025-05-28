import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

import numpy as np
from .build_fn import MODELS
from experiments.model.mamba3d.utils import misc
from experiments.model.mamba3d.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from experiments.model.mamba3d.utils.logger import *
import random
from knn_cuda import KNN
from experiments.model.mamba3d.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .pointnet_util import PointNetFeaturePropagation
### Mamba import start ###
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple

from .bimamba_ssm.modules.mamba_simple import Mamba
from .bimamba_ssm.utils.generation import GenerationMixin
from .bimamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from .rope import *
import random

try:
    from .bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

### Mamba import end ###

###ordering
import math
from experiments.model.mamba3d.models.z_order import *


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.GroupNorm
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.GroupNorm
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(8, oc),
                Swish(),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(390, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 390)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, features):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3
        _, _, feature_dim = features.shape  # B N C
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M : get M idx for every center
        assert idx.size(1) == self.num_group  # G center
        assert idx.size(2) == self.group_size  # M knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        feature_group = features.view(batch_size * num_points, -1)[idx, :]
        feature_group = feature_group.view(batch_size, self.num_group, self.group_size, feature_dim).contiguous()

        # normalize: relative distance
        neighborhood = neighborhood - center.unsqueeze(2)
        # relative distance normalization : sigmoid
        # neighborhood = torch.sigmoid(neighborhood)
        return neighborhood, center, feature_group


class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz)  # B N K : get K idx for every center
        assert idx.size(1) == num_points  # N center
        assert idx.size(2) == self.group_size  # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()  # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]  # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size,
                                                   feat.shape[-1]).contiguous()  # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, neighborhood_feat


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Local Geometry Aggregation
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # get knn xyz and feature
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)  # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w


# Max Pooling
class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        e_x = torch.exp(knn_x_w)  # B 2C G K
        up = (knn_x_w * e_x).mean(-1)  # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x


# shared MLP
class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute

    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)


## MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# K_Norm + K_Pool + Shared MLP
class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128,
                 act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm, ):
        super().__init__()
        '''features.permute(0, 2, 1).contiguous()
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x

        '''
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()

    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        # cls_token = feat[:,0,:].view(B, 1, C)
        feat = feat[:, :, :]  # B G C

        lc_x_w = self.lga(center, feat)  # B 2C G K

        lc_x_w = self.kpool(lc_x_w)  # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))  # pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))  # B G C : 1 128 384

        lc_x = self.act(lc_x)

        # lc_x = torch.cat((cls_token, lc_x), dim=1) # B G+1 C : 1 129 384
        return lc_x


class Mamba3DBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU,
                 norm_layer=nn.LayerNorm,
                 k_group_size=8,
                 alpha=100,
                 beta=1000,
                 num_group=128,
                 num_heads=6,
                 bimamba_type="v2",
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(384, 6 * 384, bias=True)
        )
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.lfa = LNPBlock(lga_out_dim=dim * 2,
                            k_group_size=self.k_group_size,
                            alpha=alpha,
                            beta=beta,
                            mlp_in_dim=dim * 2,
                            mlp_out_dim=dim,
                            num_group=self.num_group,
                            act_layer=act_layer,
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            )
        self.mixer = Mamba(dim, bimamba_type=bimamba_type)

    def forward(self, center, x, temd, use_lfa=True):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temd).chunk(6, dim=2)

        if use_lfa:
            x = x + gate_msa * self.drop_path(self.lfa(center, modulate(self.norm1(x), shift_msa, scale_msa)))
        else:
            x = x + gate_msa * modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_mlp * self.drop_path(self.mixer(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x



class Mamba3DEncoder(nn.Module):
    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6,
                 bimamba_type="v2", ):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim,
                k_group_size=self.k_group_size,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
            )
            for i in range(depth)])
        self.lfa_used = False  # 在 Encoder 层管理 lfa 的使用

    def forward(self, center, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            if not self.lfa_used:
                x = block(center, x, pos, use_lfa=True)
                self.lfa_used = True  # 标志 lfa 已经被使用过
            else:
                x = block(center, x, pos, use_lfa=False)  # 调用不使用 lfa 的 forward
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list



class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(1152, 2048)
        self.key_linear = nn.Linear(1152, 2048)
        self.value_linear = nn.Linear(d_model, 2048)

    def forward(self, query, key, value):
        # Linear transformations
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Scaled dot-product attention
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        output = attn_weights @ V

        return output


@MODELS.register_module()
class Mamba3D(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.2
        self.num_classes = num_classes
        self.num_heads = 6

        self.group_size = 32
        self.num_group = 128
        self.encoder_dims = 384

        # self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])
        self.ordering = False
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.k_group_size = 4  # default=8

        self.bimamba_type = "v4"

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # define the encoder
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
        )
        # embed_dim=768, depth=4, drop_path_rate=0.

        self.norm = nn.LayerNorm(self.trans_dim)
        self.sw = Swish()
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.label_smooth = 0.0
        self.build_loss_func()
        self.embedf = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims),
        )
        self.classifire = nn.Sequential(
            SharedMLP(3717, 128, dim=1),
            nn.Dropout(0.1),
            nn.Conv1d(128, num_classes, 1)
        )
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.convs1 = nn.Conv1d(3328, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 256, 1)
        self.convs3 = nn.Conv1d(256, self.num_classes, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(256)
        self.cross = CrossAttention(387)
        self.relu = nn.ReLU()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                # print_log('missing_keys', logger='Transformer')
                # print_log(
                #     get_missing_parameters_message(incompatible.missing_keys),
                #     logger='Transformer'
                # )
                print(1)
            if incompatible.unexpected_keys:
                # print_log('unexpected_keys', logger='Transformer')
                # print_log(
                #     get_unexpected_parameters_message(incompatible.unexpected_keys),
                #     logger='Transformer'
                # )
                print(1)
            # print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            # print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, t):
        t_emb = get_timestep_embedding(self.encoder_dims, t, inputs.device).float()
        t_emb = self.embedf(t_emb)[:, :, None].expand(-1, -1, self.num_group).permute(0, 2, 1).contiguous()
        pts = inputs[:, :3, :].permute(0, 2, 1).contiguous()  # (B, 3, N)
        features = inputs[:, 3:, :].permute(0, 2, 1).contiguous()

        # (B, 3 + S, N)
        B = pts.size(0)
        N = pts.size(1)
        # neighborhood
        # print(t[0])
        neighborhood, center, group_feature = self.group_divider(pts, features)
        neighborhood_f=torch.cat((neighborhood,group_feature),3)
        # B G K 3
        # neighborhood_f=torch.cat((neighborhood,group_feature),dim=3)
        # group_feature_max = torch.max(group_feature, 2)[0]
        # group_feature_mean = torch.mean(group_feature, 2)
        # group_global_feature = torch.cat((group_feature_max, group_feature_mean), 2)
        group_input_tokens = self.encoder(neighborhood_f)  # B G C
        # (1,128,384)
        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # x=group_input_tokens
        # x=torch.cat((group_input_tokens,group_feature),dim=1)
        pos = self.pos_embed(center)  # B G C

        # x = torch.cat((pos, group_input_tokens), dim=1)
        x = group_input_tokens + pos
        # pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        # 32 129 384
        feature_list = self.blocks(center, x, t_emb)  # enter transformer blocks
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_global_feature=self.cross(x_max_feature.permute(0,2, 1),x_avg_feature.permute(0,2, 1),features)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)
        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
        x = torch.cat((f_level_0, x_global_feature), 1)
        # x = self.relu(self.bns1(self.convs1(x)))
        # x = self.dp1(x)
        # ret = self.relu(self.bns2(self.convs2(x)))
        # x = self.sw(self.norm(x))
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)
        # concat_f = concat_f.unsqueeze(1).expand(group_input_tokens.size(0), 8192, -1)
        # # 调整输入维度适应 classifier: (b, 8192, 768) -> (b, 768, 8192)
        # concat_f = concat_f.transpose(1, 2)
        # #
        # # # 通过 classifier 处理: (b, 768, 8192) -> (b, 3, 8192)
        # ret = self.classifire(x)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        # x = F.log_softmax(x, dim=2)
        ret = x

        # ret = self.cls_head_finetune(concat_f)
        return ret

