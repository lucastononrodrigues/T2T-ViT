""" 
The original Vision Transformer (ViT) from timm, copyright belongs to / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

from functools import partial
import numpy as np
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch._six import container_abcs


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
try:
    from .SPE import *
except:
    from SPE import *

import performer_pytorch.performer_pytorch as prf_torch

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vip_smalltest': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.attn = Attention(
        #    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn= AttentionPerf(dim, dim//num_heads, head_cnt=num_heads, kernel_ratio=0.5, dp1=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        to_2tuple = _ntuple(2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #BxCxHxW
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) 
        # B x embed_dim=Channels x H' x W' -> B x d x (H'*W') -> B x (H'*W') x d
        return x
    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1),
                                                model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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


    
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class AttentionPerf(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1,spe=None):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.qkv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.head_dim=in_dim
        self.norm1 = nn.LayerNorm(dim)
        #self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.head_dim * kernel_ratio)
        #self.w = torch.randn(self.m, self.head_dim)
        #self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)
        
        self.create_projection = partial(prf_torch.gaussian_orthogonal_random_matrix, nb_rows = self.m, nb_columns = self.head_dim)
        self.w = self.create_projection()
        
        
        self.kernel='sm'
        spe='SineSPE'
        self.spe=spe
        self.num_realizations=64
        if spe is not None:
            if spe =='SineSPE':
                self.spe = SineSPE(num_heads=head_cnt, in_features=in_dim, num_sines=5, num_realizations=self.num_realizations)
                self.filter = SPEFilter(gated=True,code_shape=self.spe.code_shape)

    def attn(self, x):
        B, N, C = x.shape
        device = x.device
        qkv = self.qkv(x).reshape(B, N, 3, self.head_cnt, C // self.head_cnt).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #B x h x N x C//h
        if self.spe is not None:
            q=q.permute(0,2,1,3) #Fix this!! We will need to change either the spe algorithm or the performer one
            k=k.permute(0,2,1,3)
        #Filter is adapted for B N h d, so we need to twist it a little bit before sending inside the function
            q,k = self.filter(q,k,self.spe(q.shape[:2])) #We want to select the Batch and Token dimensions
            q=q.permute(0,2,1,3)/(self.num_realizations**0.25)
            k=k.permute(0,2,1,3)/(self.num_realizations**0.25)
        
        #print('q,k shapes',q.shape,k.shape)    
        if self.kernel=='sm':
            qp,kp= prf_torch.softmax_kernel(q,projection_matrix=self.w.to(device),is_query=True,device=device),prf_torch.softmax_kernel(k,projection_matrix=self.w.to(device),is_query=False,device=device)
        elif self.kernel=='relu':
            #print(q.device,device)
            qp,kp= prf_torch.generalized_kernel(q,projection_matrix=self.w.to(device),device=device),prf_torch.generalized_kernel(k,projection_matrix=self.w.to(device),device=device)
        #print('qp kp shapes',qp.shape,kp.shape)
        #y = prf_torch.linear_attention(qp,kp,v)
        y = prf_torch.linear_attention(qp,kp,v)
        #print('linear attention shape y ', y.shape)
        # skip connection
        if self.spe:
            #print('y shape',y.shape)
            y = y.permute(0,2,1,3).flatten(-2) / math.sqrt(self.num_realizations)
            v = v.permute(0,2,1,3).flatten(-2)
            #print('final y shape',y.shape)
            y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        else:
            y=y.permute(0,2,1,3).flatten(-2)
            #print('final y shape',y.shape)
            y = self.dp(self.proj(y))
        
        return y
    
        
    def forward(self, x):
        #print(x.shape)
        x = self.attn(self.norm1(x))
        #x = x + self.mlp(self.norm2(x)) Removing MLP + skip because already in Block
        return x


def mod_linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

class AttentionPerformer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1,spe=None):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.qkv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.head_dim=in_dim
        self.norm1 = nn.LayerNorm(dim)
        #self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.head_dim)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)
        
        self.spe=spe
        self.num_realizations=64
        if spe is not None:
            if spe =='SineSPE':
                self.spe = SineSPE(num_heads=head_cnt, in_features=in_dim, num_sines=5, num_realizations=self.num_realizations)
                self.filter = SPEFilter(gated=True,code_shape=self.spe.code_shape)

    def prm_exp(self,x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1,1, self.m) / 2
        wtx = torch.einsum('bhti,mi->bhtm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)
    
    def prm_exp2(self,x,is_query=False):
        d=x.shape[-1]
        if self.spe is not None:
            normal=1
        else:    
            normal=1.0/(d**0.25)
        
        ratio = 1 /(self.m ** 0.25)
        dash = torch.einsum("bhnc,mc->bhnm",x,self.w)
        diag=torch.square(x)
        diag=diag.sum(dim=-1,keepdim=False)
        diag=diag/2.0
        diag=diag*(normal**2)
        diag=diag.unsqueeze(-1)
        if is_query:
            dash = torch.exp(dash - diag - torch.max(dash,dim=-1, keepdims=True)[0]) + self.epsilon
            dash= ratio * dash
        else:
            dash = torch.exp(dash - diag - torch.max(dash)) +  self.epsilon
            dash= ratio * dash
        return dash #b h n m
    def attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head_cnt, C // self.head_cnt).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #B x h x N x C//h
        if self.spe is not None:
            q=q.permute(0,2,1,3) #Fix this!! We will need to change either the spe algorithm or the performer one
            k=k.permute(0,2,1,3)
        #v=v.permute(0,2,1,3)
        #Filter is adapted for B N h d, so we need to twist it a little bit before sending inside the function
            q,k = self.filter(q,k,self.spe(q.shape[:2])) #We want to select the Batch and Token dimensions
            q=q.permute(0,2,1,3)/(self.num_realizations**0.25)
            k=k.permute(0,2,1,3)/(self.num_realizations**0.25)
        kp, qp = self.prm_exp2(k), self.prm_exp2(q,is_query=True)  # B x h x N x m
          
        #print(kp.shape,qp.shape)
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=3)  # 
        #print(D.shape)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)  # 'bhnd,bhnm->bhdm'
        #print(qp.shape,kptv.shape)
        y = torch.einsum('bhnm,bhdm->bhnd', qp, kptv) # bhnm,bhdm->bhnd
        y= y / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon) # bhnd / bhnd
        #print(y.shape)
        # skip connection
        y = y.permute(0,2,1,3).flatten(-2) / math.sqrt(self.num_realizations)
        v = v.permute(0,2,1,3).flatten(-2)
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        
        #print(y.shape)
        return y
    
        
    def forward(self, x):
        #print(x.shape)
        x = self.attn(self.norm1(x))
        #x = x + self.mlp(self.norm2(x)) Removing MLP + skip because already in Block
        return x
        
class AttentionPerformerOriginal(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)
        
    def prm_exp(self,x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)
    
    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        print(k.shape)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x    
    
@register_model
def vip_smalltest(pretrained=False,**kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=4, num_heads=6,**kwargs)
    model.default_cfg = default_cfgs['vip_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def vip_smalltest_cifar100(pretrained=False,**kwargs):
    model = VisionTransformer(patch_size=4, embed_dim=384, depth=4, num_heads=6,**kwargs)
    model.default_cfg = default_cfgs['vip_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

