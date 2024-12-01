# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""
import os
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from collections import OrderedDict

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock, TrajectoryAttentionBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.stem_helper import PatchEmbed
from slowfast.models.utils import round_width
from .ORViT import ORViT as ORViT

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from einops import rearrange, repeat
from sympy import divisors
import torchvision
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from .moco  import builder as moco_builder
import torchvision.models as torchvision_models
import timm

        
class AggTransformer(nn.Module):
    def __init__(self, cfg, hidden_dim=728):
        super().__init__()
        self.cfg = cfg
        self.temporal_extent = cfg.DATA.NUM_FRAMES
        self.class_token = nn.Parameter(torch.zeros(1, cfg.DATA.NUM_FRAMES, hidden_dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                        nhead=cfg.MF.NUM_HEADS)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=cfg.MODEL.EXTRA_ENCODER_DEPTH)

        
    def forward(self, x):
        # Shape x : batch_size, num_tokens, hiden_dim
        bs, num_patches, hidden_dim = x.shape
        class_tokens = self.class_token.repeat(bs, 1, 1)
        check_norm = torch.norm(x)
        x = torch.cat([class_tokens, x], dim=1)
        x = self.encoder(x)
        new_norm = torch.norm(x[:, self.temporal_extent:])
        x = x[:, :self.temporal_extent]
        if self.cfg.DEBUG:
            print('input_norm:', check_norm, 'output_norm:', new_norm)
        return x
        
        
        





@MODEL_REGISTRY.register()
class Pointformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        # self.patch_size = cfg.MF.PATCH_SIZE
        if cfg.MODEL.FEAT_EXTRACTOR == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            vit_mode = dino_config.split("_")[1]
            if 'vits' in vit_mode:
                vit_type = 'vits'
                self.embed_dim = self.dino_feat_size = 384
            elif 'vitb' in vit_mode:
                vit_type = 'vitb'
                self.embed_dim = self.dino_feat_size = 768
            elif 'vitl' in vit_mode:
                vit_type = 'vitl'
                self.embed_dim = self.dino_feat_size = 1024
                
            else:
                raise NotImplementedError("Only supports ViT-B and ViT-S for DINO")
            self.patch_size = int(vit_mode.replace(vit_type, ""))
        else:
            raise NotImplementedError('Feature extractor not implemented')

        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens" and cfg.TASK == 'classification':
            self.num_classes = [97, 300]  
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES
        
        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.DATA.NUM_FRAMES
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.cfg = cfg
        self.num_patches = (224 // self.patch_size) ** 2
        if cfg.POINT_INFO.ENABLE:
            self.point_grid_size = self.get_point_grid_size()
                
        else:
             self.point_grid_size = int(self.num_patches ** 0.5)


        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # # Positional embedding

        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        ##
        blocks = []
        for i in range(self.depth):
            # pt_attention is introduced, for now its just space-time attention
            _block = TrajectoryAttentionBlock(
                cfg = cfg,
                dim=self.embed_dim, 
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio, 
                qkv_bias=self.qkv_bias, 
                drop=self.drop_rate, 
                attn_drop=self.attn_drop_rate, 
                drop_path=dpr[i],
                norm_layer=norm_layer,
                pt_attention=cfg.MF.PT_ATTENTION,
                use_pt_visibility=cfg.MF.USE_PT_VISIBILITY or cfg.POINT_INFO.USE_PT_QUERY_MASK,
            )

            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh() 
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d"%a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes) 
                if self.num_classes > 0 else nn.Identity())
        self.dino_num_patch_side = 224 // self.patch_size
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.dino_num_patch_side, 
                                                        self.dino_num_patch_side))
        
        trunc_normal_(self.spatial_pos_embed, std=.02)
        if cfg.MODEL.FEAT_EXTRACTOR == 'resnet':
            #TODO(pulkit): Remove hard coding
            self.space_pos_embed = nn.Parameter(torch.zeros(1,49, self.embed_dim))
        else:
            self.space_pos_embed = nn.Parameter(torch.zeros(1,self.num_patches, self.embed_dim))

        self.time_pos_embed = nn.Parameter(torch.zeros(1,self.cfg.DATA.NUM_FRAMES, self.embed_dim))
        trunc_normal_(self.space_pos_embed, std=.02)
        trunc_normal_(self.time_pos_embed, std=.02)
        self.space_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.time_pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)


        
        self.spatial_pos_embed_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        self.layer_to_use = None
        
        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)
        if cfg.MODEL.FEAT_EXTRACTOR == "dino":
            dino_config  = cfg.MODEL.DINO_CONFIG
            local_path = os.path.join(os.environ['TORCH_HOME'], 'hub')
            if 'v2' in dino_config:
                local_path = os.path.join(local_path , 'facebookresearch_dinov2_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
            else:
                local_path = os.path.join(local_path , 'facebookresearch_dino_main')
                self.dino = torch.hub.load(local_path, dino_config, source='local')
                self.dino.num_register_tokens = 0
                
            self.feat_dict = dict()
            #output of last norm to be taken.
            layer = self.dino.norm
            self.hook = layer.register_forward_hook(self.hook_fn(self.feat_dict, 'dino'))

            self.dino.cuda()
        else:
            raise NotImplementedError('Feature extractor not implemented')
        


    def hook_fn(self, feat_dict, layer_name):
        def hook(module, input, output):
            # Store the extracted features as an attribute of the model
            feat_dict[layer_name] = output
        return hook
    
    def backward_hook_fn(self, feat_dict, layer_name):
        def hook(module, grad_inputs, grad_outputs):
            # Store the extracted features as an attribute of the model
            grad_out_norm = np.mean([torch.norm(grad_output).item() for grad_output in grad_outputs])
            grad_in_norm = np.mean([torch.norm(grad_input).item() for grad_input in grad_inputs])
            feat_dict[layer_name] = {
                'grad_out': np.around(grad_out_norm, 3),
                'grad_in': np.around(grad_in_norm, 3)
            }
        return hook
        

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MF.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head
    

    def get_point_grid_size(self):
        all_divisors = divisors(self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)
        return all_divisors[len(all_divisors) // 2]

    
    def get_dino_features(self, x):
        self.dino.eval()
        batch_size, num_frames, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        if self.cfg.MODEL.TRAIN_BACKBONE:
            _ = self.dino(x)
        else:
            with torch.no_grad():
                _ = self.dino(x)
        #using hooks to get the patch tokens
        feat = self.feat_dict['dino'][:, self.dino.num_register_tokens + 1 :]
        feat_size = feat.shape[-1]
        #dino patch side is fine
        feat = feat.view(batch_size, num_frames, self.dino_num_patch_side,
                         self.dino_num_patch_side, feat_size)
        return feat

    
    def pt_forward(self, x, metadata):
        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_mask = metadata['pred_query_mask'] # [BS, T, N]
        else:
            pt_mask = metadata['pred_visibility'] # [BS, T, N]
            
        bs, temporal_dim, num_points, feat_dim = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t n d -> b n t d')
        pt_mask = rearrange(pt_mask, 'b t n -> b n t')
        x = rearrange(x, 'b n t d -> b (n t) d')
        pt_mask = rearrange(pt_mask, 'b n t -> b (n t)')
        cls_tokens = self.cls_token.expand(bs, -1, -1) # [BS, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]
        cls_token_mask = torch.ones(bs, 1).bool().to(x.device)
        pt_mask = torch.cat((cls_token_mask, pt_mask), dim=1) # [BS, N, dim]
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, self.point_grid_size, 
            int(num_points / self.point_grid_size)]
        for i, blk in enumerate(self.blocks):
            x, _ = blk(
                x,
                metadata,
                thw,
                pt_mask
            )
        x = self.norm(x)
        cls_x, patch_x = x[:, 0], x[:, 1:]
        cls_x = self.pre_logits(cls_x)
        # Taking the patch tokens back to the input shape
        patch_x = rearrange(patch_x, 'b (n t) d -> b t n d', t=temporal_dim)
        if not torch.isfinite(x).all():
            print("WARNING: nan in features out")
        return cls_x, patch_x

    def add_st_pos_embeddings(self, x):
        bs, temporal_dim, sp_dim_1, sp_dim_2, feat_dim = x.shape
        # reshaping the input according to the attention block
        x = rearrange(x, 'b t p q d -> b t (p q) d')
        x = x + self.space_pos_embed.unsqueeze(0)
        x = self.space_pos_drop(x)
        x = rearrange(x, 'b t p d -> b p t d')
        x = x + self.time_pos_embed.unsqueeze(0)
        x = self.time_pos_drop(x)
        x = rearrange(x, 'b (p q) t d -> b t p q d', p=sp_dim_1, q=sp_dim_2)
        return x




    def forward(self, x, metadata):

        if self.cfg.MODEL.FEAT_EXTRACTOR == "dino":
            feat_to_use = self.get_dino_features(x)
        else:
            raise NotImplementedError('Feature extractor not implemented')
        if self.cfg.MF.USE_BASE_POS_EMBED:
            feat_to_use = self.add_st_pos_embeddings(feat_to_use)

        if self.cfg.POINT_INFO.ENABLE:
            bs, num_frames, num_patch, num_patch, feat_dim = feat_to_use.shape
            feat_to_use = rearrange(feat_to_use, 'b t p q d -> (b t) p q d')
            feat_to_use = rearrange(feat_to_use, 'b p q d -> b d p q')
            num_x, num_y = feat_to_use.shape[-2:]
            assert self.num_patches == num_x * num_y, "Number of patches mismatch"
            pred_tracks = metadata['pred_tracks']
            pred_tracks = pred_tracks.view(bs * num_frames, -1,1,2) 
            spatial_pos_embed = self.spatial_pos_embed.repeat(bs * num_frames, 1, 1, 1)
            sampled_feat = F.grid_sample(feat_to_use, pred_tracks, align_corners=True,
                                        mode=self.cfg.MODEL.FEAT_EXTRACT_MODE) 
            if (self.cfg.MF.USE_PT_SPACE_POS_EMBED and self.cfg.FEW_SHOT.USE_MODEL
                and not self.cfg.MF.USE_BASE_POS_EMBED):
                sample_pos_embedding = F.grid_sample(spatial_pos_embed, pred_tracks,
                                                    align_corners=True,
                                                    mode='bilinear')
                sampled_feat = sampled_feat + sample_pos_embedding
            sampled_feat = rearrange(sampled_feat, 'b d p q -> b p q d')
            #Removing the extra added dim
            sampled_feat = sampled_feat.squeeze(-2)
            sampled_feat = rearrange(sampled_feat, '(b t) p d -> b t p d', t=num_frames)

        else:
            sampled_feat = rearrange(feat_to_use, 'b t p q d -> b t (p q) d')
            self.point_grid_size = int(sampled_feat.shape[2] ** 0.5)
        
        cls_x, patch_x = self.pt_forward(sampled_feat, metadata)
        
        # x = self.forward_features(x, metadata) # [BS, d]
        x = self.head_drop(cls_x)
       
        x = self.head(x)
        # previously there was a softmax here for validation which messed up the loss computation
        if self.cfg.TASK == 'few_shot':
            return x, patch_x
        return x
