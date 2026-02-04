"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Based on timm code base
 https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from lavis.models.base_model import BaseEncoder

DEFAULT_THRESHOLD = None

""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Binarizer, self).__init__()

    @staticmethod
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.lt(threshold)] = 0
        outputs[inputs.ge(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return (gradOutput, None)
    

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class MaskAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mask_init='1s',
        mask_scale=1e-2,
        threshold_fn='binarizer',
        threshold=None, 
        mask_attn=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.mask_attn = mask_attn

        if self.mask_attn:
            self.mask_real = nn.Parameter(torch.Tensor(3 * dim, dim))
            self.mask_real_proj = nn.Parameter(torch.Tensor(dim, dim))
            if mask_init == '1s':
                self.mask_real.data.fill_(mask_scale)
                self.mask_real_proj.data.fill_(mask_scale)
            elif mask_init == 'uniform':
                self.mask_real.data.uniform_(0, mask_scale)
                self.mask_real_proj.data.uniform_(0, mask_scale)

            self.threshold_fn = threshold_fn
            self.mask_scale = mask_scale
            self.threshold = threshold 

            self.soft = False
            if threshold_fn == 'binarizer':
                self.threshold_fn = Binarizer().apply
            
        # self.weight = self.qkv.weight.data.clone()

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, threshold=None, register_hook=False):

        B, N, C = x.shape
        if threshold != None:
            self.threshold = threshold
            if self.training:
                mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
                mask_thresholded_proj = self.threshold_fn(self.mask_real_proj, self.threshold)
            else:
                mask_thresholded = (self.mask_real >= self.threshold)
                mask_thresholded_proj = (self.mask_real_proj >= self.threshold)

            # Clone the qkv weight to ensure the original weights are not changed
            qkv_weight = self.qkv.weight.clone() * mask_thresholded
            qkv_bias = self.qkv.bias.clone()
            proj_weight = self.proj.weight.clone() * mask_thresholded_proj
            proj_bias = self.proj.bias.clone()
        else:
            qkv_weight = self.qkv.weight.clone()
            qkv_bias = self.qkv.bias.clone()
            proj_weight = self.proj.weight.clone()
            proj_bias = self.proj.bias.clone()

        qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = F.linear(x,proj_weight,proj_bias)
        x = self.proj_drop(x)
        return x


class MaskBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_grad_checkpointing=False,
        mask_init: str = 'ls', 
        mask_scale: float = 1e-2, 
        threshold_fn: str = 'binarizer',
        threshold: float = 5e-3,
        mask_attn: bool = True
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_init=mask_init,
            mask_scale=mask_scale,
            threshold_fn=threshold_fn,
            threshold=threshold, 
            mask_attn=mask_attn
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, threshold = None, register_hook=False):
        if threshold != None:
            x = x + self.drop_path(self.attn(self.norm1(x), threshold, register_hook=register_hook))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MaskPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 mask_init='1s', mask_scale=1e-2,threshold_fn='binarizer', threshold=None, apply_mask=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.apply_mask = apply_mask

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        if apply_mask:
            # Initialize real-valued mask weights.
            self.mask_real = self.proj.weight.data.new(self.proj.weight.size())
            self.threshold = threshold

            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = nn.Parameter(self.mask_real)

            self.soft=False
            # Initialize the thresholder.
            if threshold_fn == 'binarizer':
                self.threshold_fn = Binarizer().apply
            

    def forward(self, x, threshold=None):
        
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if threshold != None:
            self.threshold = threshold
            if self.soft:
                mask_thresholded = self.threshold_fn(self.mask_real, "simple")
            elif self.training:
                mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
            else:
                mask_thresholded = (self.mask_real>=self.threshold) #.float()

            # Mask weights with above mask.
            weight_thresholded = mask_thresholded * self.proj.weight.clone() 
            weight_bias = self.proj.bias.clone()
        else:
            weight_thresholded = self.proj.weight.clone() 
            weight_bias = self.proj.bias.clone()

        x = F.conv2d(x, weight_thresholded, weight_bias, self.proj.stride, self.proj.padding, self.proj.dilation, self.proj.groups)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
class MaskVisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_grad_checkpointing=False,
        ckpt_layer=0,
        mask_init: str = 'ls', 
        mask_scale: float = 1e-2, 
        threshold_fn: str = 'binarizer', 
        threshold: float = 5e-3,
        apply_mask: bool = True
    ):
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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = MaskPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
             mask_init=mask_init, 
             mask_scale=mask_scale,
             threshold_fn=threshold_fn, 
             threshold=threshold, 
             apply_mask=apply_mask
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                MaskBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_grad_checkpointing=(
                        use_grad_checkpointing and i >= depth - ckpt_layer
                    ),
                    mask_init=mask_init,
                    mask_scale=mask_scale,
                    threshold_fn=threshold_fn,
                    threshold=threshold, 
                    mask_attn=apply_mask
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x, threshold=None, register_blk=-1):
        B = x.shape[0]
        if threshold != None:
            x = self.patch_embed(x,threshold)
        else:
            x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        fea = None
        for i, blk in enumerate(self.blocks):
            if threshold != None:
                x = blk(x, threshold, register_blk == i)
            else:
                x = blk(x, register_blk == i)
            if fea == None:
                fea = x.unsqueeze(0)
            else:
                fea = torch.cat((fea,x.unsqueeze(0)),dim=0)
        x = self.norm(x)

        return x[:,0,:],fea

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


@torch.no_grad()
def _load_weights(model: MaskVisionTransformer, checkpoint_path: str, prefix: str = ""):
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    # w = np.load(checkpoint_path)
    w = checkpoint_path
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    print('yes')
    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    #     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
    #         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
    #         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    #     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
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
    print("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bicubic", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print(
            "reshape position embedding from %d to %d" % (orig_size**2, new_size**2)
        )

        return new_pos_embed
    else:
        return pos_embed_checkpoint


class VisionTransformerEncoder(MaskVisionTransformer, BaseEncoder):
    @classmethod
    def from_config(cls, cfg, from_pretrained=False):

        vit_type = cfg.get("vit_type", "base")
        image_size = cfg.get("image_size", 384)
        ckpt_layer = cfg.get("vit_ckpt_layer", 0)
        drop_path_rate = cfg.get("vit_drop_path_rate", 0)
        norm_layer_eps = cfg.get("vit_layer_norm_epsilon", -1)
        use_grad_checkpointing = cfg.get("vit_grad_ckpt", False)

        if norm_layer_eps == -1:
            norm_layer = None
        else:
            norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
        assert vit_type in ["base", "large"], "vit parameter must be base or large"
        if vit_type == "base":
            vision_width = 768
            visual_encoder = cls(
                img_size=image_size,
                patch_size=16,
                embed_dim=vision_width,
                depth=12,
                num_heads=12,
                use_grad_checkpointing=use_grad_checkpointing,
                ckpt_layer=ckpt_layer,
                drop_path_rate=0 or drop_path_rate,
                norm_layer=norm_layer,
            )

            if from_pretrained:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu",
                    check_hash=True,
                )
                state_dict = checkpoint["model"]
                state_dict["pos_embed"] = interpolate_pos_embed(
                    state_dict["pos_embed"], visual_encoder
                )
                msg = visual_encoder.load_state_dict(state_dict, strict=False)

        elif vit_type == "large":
            vision_width = 1024
            visual_encoder = cls(
                img_size=image_size,
                patch_size=16,
                embed_dim=vision_width,
                depth=24,
                num_heads=16,
                use_grad_checkpointing=use_grad_checkpointing,
                ckpt_layer=ckpt_layer,
                drop_path_rate=0.1 or drop_path_rate,
                norm_layer=norm_layer,
            )
            if from_pretrained:
                from timm.models.helpers import load_custom_pretrained
                from timm.models.vision_transformer import default_cfgs

                load_custom_pretrained(
                    visual_encoder, default_cfgs["vit_large_patch16_224_in21k"]
                )

        visual_encoder.vision_width = vision_width
        return visual_encoder

    def forward_features(self, x, register_blk=-1):
        return super().forward(x, register_blk)
