from turtle import pos
import torch.nn as nn
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.nn.models import Node2Vec
from torch_geometric.graphgym.register import register_node_encoder
from torch.utils.data import DataLoader
import itertools


@register_node_encoder('Learnable')
class Node2VecLearnableEncoder(nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        
        assert cfg
        self.pecfg = cfg.posenc_Node2VecLearnable
        dim_in = cfg.share.dim_in
        self.dim_pe = self.pecfg.dim_pe
        self.num_nodes = cfg.posenc_Learnable.num_nodes
        self.embeddings = nn.Embedding(self.num_nodes, self.dim_pe)
    
        if self.pecfg.model == "Linear":
            self.encoder = nn.Linear(self.dim_pe, self.dim_pe)
        else:
            self.encoder = None

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - self.dim_pe)

        self.expand_x = expand_x

        
        if self.pecfg.raw_norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(self.dim_pe)
        else:
            self.raw_norm = None

    def forward(self, batch):
        pos_enc = self.embeddings.weight
        if self.encoder:
            pos_enc = self.encoder(pos_enc)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        if self.expand_x:
            h = self.linear_x(batch.x.to(torch.float32))
        else:
            h = batch.x.to(torch.float32)
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        return batch
