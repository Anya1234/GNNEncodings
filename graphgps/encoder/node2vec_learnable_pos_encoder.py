from turtle import pos
import torch.nn as nn
import torch
from torch_geometric.graphgym import cfg
from torch_geometric.nn.models import Node2Vec
from torch_geometric.graphgym.register import register_node_encoder
from torch.utils.data import DataLoader
import itertools


@register_node_encoder('Node2VecLearnable')
class Node2VecLearnableEncoder(nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        
        assert cfg
        self.pecfg = cfg.posenc_Node2VecLearnable
        dim_in = cfg.share.dim_in
        self.dim_pe= self.pecfg.dim_pe
        self.num_nodes = cfg.posenc_Node2VecLearnable.num_nodes
        self.model = None
        self.num_negative_samples = 1
    
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

    def init_model(self, batch):
        self.model = Node2Vec(edge_index=batch.edge_index,
                              embedding_dim=self.dim_pe,
                              walk_length=self.pecfg.walk_length,
                              context_size=self.pecfg.window_size,
                              walks_per_node=self.pecfg.num_walks,
                              q=self.pecfg.q,
                              p=self.pecfg.p,
                              num_nodes=self.num_nodes,
                              num_negative_samples=self.num_negative_samples)
        self.model = self.model.to(batch.x.device)

    def forward(self, batch):
        if self.model is None:
            self.init_model(batch)

        loader = self.model.loader()
        pos_rw = []
        neg_rw = []
        for data in loader:
            pos_rw.append(data[0])
            neg_rw.append(data[1])
        pos_rw = torch.cat(pos_rw, axis=0)
        neg_rw = torch.cat(neg_rw, axis=0)
        device = batch.x.device
        pos_enc = self.model.forward()
        # pos_rw, neg_rw = next(self.loader_iter)
        pos_rw = pos_rw.to(device)
        neg_rw = neg_rw.to(device)
        pos_rw.to()
        batch.node2vec_loss = self.model.loss(pos_rw, neg_rw)
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
