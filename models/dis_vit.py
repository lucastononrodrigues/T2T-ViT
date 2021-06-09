import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from itertools import repeat
from torch._six import container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

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
    'disvit_base_patch16_224': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_smalltest': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_wope_smalltest': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_wope_base_patch16_224': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_wopesak_d14': _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_7':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_10':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_12':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_14':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_19':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'disvit_24':_cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}




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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 relative_attention=True,pos_att_type='c2p|p2c',position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512,share_att_key=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self._all_head_size= head_dim * self.num_heads
        
        self.query_proj = nn.Linear(dim, self._all_head_size, bias=True)
        self.key_proj = nn.Linear(dim, self._all_head_size, bias=True)
        self.value_proj = nn.Linear(dim, self._all_head_size, bias=True)
        
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_drop = nn.Dropout(attn_drop)
        self.max_relative_positions=dim
        self.pos_ebd_size=self.max_relative_positions
        
        self.share_att_key=share_att_key
        self.relative_attention=relative_attention

        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')] # c2p|p2c
        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            if not self.share_att_key:
                self.pos_key_proj = nn.Linear(dim, self._all_head_size, bias=True)
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            if not self.share_att_key:
                self.pos_query_proj = nn.Linear(dim, self._all_head_size)


    

  
    def forward(self, x,relative_pos=None,rel_embeddings=None):
        B, N, C = x.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #BN3hd->3BhNd
        q= self.query_proj(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        k= self.key_proj(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        v= self.value_proj(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        #q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #B x h x N x C//h
        q,k,v = q.reshape(q.size(0)*q.size(1),N,C//self.num_heads),k.reshape(k.size(0)*k.size(1),N,C//self.num_heads),v.reshape(v.size(0)*v.size(1),N,C//self.num_heads)
        # (B*h) x N x C//h
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        self.scale = math.sqrt(q.size(-1)*scale_factor)
        
        attn = (q @ k.transpose(-2, -1)) / self.scale
        
        if self.relative_attention:
            rel_embeddings=self.pos_drop(rel_embeddings)
            #print(q.shape,k.shape,relative_pos.shape,rel_embeddings.shape)
            #print(relative_pos.shape)
            rel_att = self.disentangled_attention_bias(q, k, relative_pos, rel_embeddings, scale_factor)
        
        if rel_att is not None:
            attn = (attn + rel_att)
        
        attn = attn.view(-1,self.num_heads,attn.size(-2),attn.size(-1))
        
        attn_probs = attn.softmax(dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        x = (attn_probs.view(-1,attn_probs.size(-2),attn_probs.size(-1)) @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
            if relative_pos is None:
                q = query_layer.size(-2)
                relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
            if relative_pos.dim()==2:
                relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
            elif relative_pos.dim()==3:
                relative_pos = relative_pos.unsqueeze(1)
            # bxhxqxk
            elif relative_pos.dim()!=4:
                raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
            #print(relative_pos)
            att_span = self.pos_ebd_size
            relative_pos = relative_pos.long().to(query_layer.device)
            #print(relative_pos.max(),relative_pos.min())
            #print(att_span)
    
            rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
            if self.share_att_key:
                pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_heads)\
                    .repeat(query_layer.size(0)//self.num_heads, 1, 1) #.split(self.all_head_size, dim=-1)
                pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_heads)\
                    .repeat(query_layer.size(0)//self.num_heads, 1, 1) #.split(self.all_head_size, dim=-1)
                    
                    ## REMOVE UNUSED PARAMETERS
            else:
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_heads)\
                        .repeat(query_layer.size(0)//self.num_heads, 1, 1) #.split(self.all_head_size, dim=-1)
                        
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_heads)\
                        .repeat(query_layer.size(0)//self.num_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            
            score = 0
            # content->position
            if 'c2p' in self.pos_att_type:
                scale = math.sqrt(pos_key_layer.size(-1)*scale_factor)
                c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
                c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
                #print(c2p_pos,torch.max(c2p_pos),torch.min(c2p_pos))
                idx=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])
                #print(relative_pos,att_span)
                #print(c2p_att.shape,idx.shape,torch.max(idx),torch.min(idx))
                c2p_att = torch.gather(c2p_att, dim=-1, index=idx)
                score += c2p_att/scale
                
            # position->content
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                scale = math.sqrt(pos_query_layer.size(-1)*scale_factor)
                if key_layer.size(-2) != query_layer.size(-2):
                    r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                    r_pos = r_pos.unsqueeze(0)
                else:
                    r_pos = relative_pos
    
                p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
                if query_layer.size(-2) != key_layer.size(-2):
                    pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
    
            if 'p2c' in self.pos_att_type:
                p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
                p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
                if query_layer.size(-2) != key_layer.size(-2):
                    p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
                score += p2c_att/scale
    
            # position->position
            if 'p2p' in self.pos_att_type:
                pos_query = pos_query_layer[:,:,att_span:,:]
                p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
                p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
                if query_layer.size(-2) != key_layer.size(-2):
                    p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
                p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
                score += p2p_att
    
            return score

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, LAI=False,disentangled=True,share_att_key=False):
        super().__init__()
        self.LAI=LAI
        self.disentangled=disentangled
        #print(f'Disentangled attention: {self.disentangled}')
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_attention=disentangled,share_att_key=share_att_key)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, relative_pos=None,rel_embeddings=None):
        if self.disentangled:
            x = x + self.drop_path(self.attn(self.norm1(x),relative_pos=relative_pos,rel_embeddings=rel_embeddings))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x,relative_pos,rel_embeddings
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x



class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 LAI=False,disentangled=True,share_att_key=False):
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
        self.disentangled= self.relative_attention=disentangled
        self.position_buckets=-1
        self.max_relative_positions=embed_dim
        self.pos_ebd_size=self.max_relative_positions*2
        #self.rel_embeddings = nn.Embedding(self.pos_ebd_size, embed_dim)
        self.rel_embeddings = nn.Parameter(torch.zeros(self.pos_ebd_size,embed_dim))
        print(f'Disentangled attention: {disentangled}')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.LayerNorm= nn.LayerNorm(embed_dim)
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
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                disentangled=self.disentangled,share_att_key=share_att_key)
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
        trunc_normal_(self.rel_embeddings,std=0.02)
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
        return {'rel_embeddings','pos_embed', 'cls_token', 'dist_token'}

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
    
        if self.disentangled:
            relative_pos=self.get_rel_pos(x)
            rel_embeddings=self.get_rel_embedding()            
        
        if self.disentangled: #disentangled
            x=self.pos_drop(x+self.pos_embed)
            
            for module in self.blocks._modules.values():
                x,_,_ = module(x,relative_pos,rel_embeddings)
            #Sequential for multiple inputs
        else:
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
    
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        #rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None:
          rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings        
    
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
          q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
          relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos




class VisionTransformerWOPE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 LAI=False,disentangled=True,share_att_key=False):
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
        self.disentangled= self.relative_attention=disentangled
        self.position_buckets=-1
        self.max_relative_positions=embed_dim
        self.pos_ebd_size=self.max_relative_positions*2
        #self.rel_embeddings = nn.Embedding(self.pos_ebd_size, embed_dim)
        self.rel_embeddings=nn.Parameter(torch.zeros(self.pos_ebd_size,embed_dim))
        
        print(f'Disentangled attention: {disentangled}')
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.LayerNorm= nn.LayerNorm(embed_dim)
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                disentangled=self.disentangled,share_att_key=share_att_key)
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
        trunc_normal_(self.rel_embeddings, std=.02)
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
    
        if self.disentangled:
            relative_pos=self.get_rel_pos(x)
            rel_embeddings=self.get_rel_embedding()            
        
        if self.disentangled: #disentangled
            x=self.pos_drop(x)
            
            for module in self.blocks._modules.values():
                x,_,_ = module(x,relative_pos,rel_embeddings)
            #Sequential for multiple inputs
        else:
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
    
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        if rel_embeddings is not None:
          rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings        
    
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
          q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
          relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos
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

def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
  q_ids = np.arange(0, query_size)
  k_ids = np.arange(0, key_size)
  rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def make_log_bucket_position(relative_pos, bucket_size, max_position):
  sign = np.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
  log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
  return bucket_pos

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

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



@register_model
def disvit_wopesak_smalltest(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=384, depth=4, num_heads=6,disentangled=True,share_att_key=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_wope_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def disvit_wope_smalltest(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=384, depth=4, num_heads=6,disentangled=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_wope_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
@register_model
def disvit_sak_smalltest(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=4, num_heads=6,disentangled=True,share_att_key=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
    

@register_model
def disvit_smalltest(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=4, num_heads=6,disentangled=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_smalltest']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
    

@register_model
def disvit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,disentangled=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def disvit_wope_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=768, depth=12, num_heads=12,disentangled=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_wope_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
    
@register_model
def disvit_wopesak_d14(pretrained=False,**kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=384, depth=14, num_heads=6,disentangled=True,**kwargs)
    model.default_cfg = default_cfgs['disvit_wopesak_d14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


#Experiments based on T2T-ViT

@register_model
def disvit_7(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['disvit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def disvit_10(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['disvit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def disvit_12(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['disvit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def disvit_14(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['disvit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def disvit_19(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['disvit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def disvit_24(pretrained=False, **kwargs):
    model = VisionTransformerWOPE(patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['disvit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


