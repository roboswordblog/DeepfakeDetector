import pandas as pd
import torch.nn as nn
import torch
import pathlib
import torch.nn.functional as F

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int=16, emb_size:int = 768, img_size-224):
        self.patch_size = patch_size
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) -> b (h w) e')
        )

    def  forward(self, x:Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.embed(x)
        return x
        
