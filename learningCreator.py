import pandas as pd
import torch.nn as nn
import torch
import pathlib
import torch.nn.functional as F

def PositionalEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_lean):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j/emb_size)) if j % 2 == 0 else np.cos(i / pow(10000, (j-1) / emb_size)))
    return torch.tensor(embeddings)

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int=16, emb_size:int = 768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionalEmbedding((img_size // patch_size)**2 + 1, emb_size))
    
    def  forward(self, x:Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.embed(x)
        return x

class MultiHead(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.query =  nn.Linear(emb_size, emb_size)
        self.att_dr = nn.Dropout(0.1)
        
