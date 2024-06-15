import torch.nn as nn
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('Node2Vec')
class Node2VecEncoder(nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        
        pecfg = cfg.posenc_Node2Vec
        dim_in = cfg.share.dim_in
        dim_pe= pecfg.dim_pe
        self.hid_dim = pecfg.hid_dim

        if pecfg.model == "Linear":
            self.encoder = nn.Linear(dim_pe, dim_pe)
        elif pecfg.model == "FFN":
             self.encoder = nn.Sequential(nn.Linear(dim_pe, self.hid_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hid_dim, dim_pe))
        else:
            self.encoder = None

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)

        self.expand_x = expand_x

        
        if pecfg.raw_norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim_pe)
        elif  pecfg.raw_norm_type.lower() == 'layernorm':
            self.raw_norm = nn.LayerNorm(dim_pe)
        else:
            self.raw_norm = None

    def forward(self, batch):
        if not (hasattr(batch, 'Node2VecEmb')):
            raise ValueError("Precomputed node2vec vectors are "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_Node2Vec.enable' to True")
        pos_enc = batch.Node2VecEmb
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
