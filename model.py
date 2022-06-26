from turtle import forward
from backbone import VisionTransformer
import torch 
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone_sequence import VisionTransformer as sequence_vit
class ClassifyModel(nn.Module):
    def __init__(self, num_classes = 10, image_size =224, patch_size =16, in_channels = 3, embed_dim = 384, n_heads = 12, qvk_bias = True, attn_p = 0., proj_p = 0., mlp_hidden_ratio = 4, mlp_p = 0., p = 0.):
        super().__init__()
        self.backbone = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        n_heads=n_heads,
        qvk_bias=qvk_bias,
        attn_p=attn_p,
        proj_p=proj_p,
        mlp_hidden_ratio=mlp_hidden_ratio,
        mlp_p=mlp_p,
        p = p)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x[:, 0]
        x = self.head(x)
        return x


class DigitRecognitionModel(nn.Module):
    def __init__(self, max_tokens = 16, image_size = [32,256], embed_dim = 384, n_classes = 11, in_channels = 1, depth = 2):
        super().__init__()
        self.max_tokens = max_tokens
        self.backbone = sequence_vit(image_size= image_size, depth=depth, embed_dim=embed_dim, in_channels=in_channels)
        self.linear1 = nn.Linear(embed_dim, n_classes)
    def forward(self, x): # (batch_size, n_channels, h, w)
        x = self.backbone(x)
        x = self.linear1(x) #(batch_size, 1+ n_tokens, n_classes)
        sequence = x[:, 1:]
        sequence = sequence.permute(0,2,1).type(torch.float32)
        return sequence
