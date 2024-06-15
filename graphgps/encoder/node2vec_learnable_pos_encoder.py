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
        self.num_negative_samples = 5
    
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

    def init_model(self, cfg, edge_index):
        self.model = Node2Vec(edge_index=edge_index,
                              embedding_dim=self.dim_pe,
                              walk_length=self.pecfg.walk_length,
                              context_size=self.pecfg.window_size,
                              walks_per_node=1,
                              q=self.pecfg.q,
                              p=self.pecfg.p,
                              num_nodes=self.num_nodes,
                              num_negative_samples=self.num_negative_samples)
        self.model = self.model.to(cfg.accelerator)
        self.loader =  DataLoader(list(range(self.num_nodes)) * self.pecfg.num_walks, collate_fn=self.model.sample, batch_size=128, shuffle=True)
        self.loader_iter = iter(self.loader)
    
    def get_next_data(self):
        try:
            data = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            data  = next(self.loader_iter)
        return data[0], data[1]

    def forward(self, batch):
        if self.model is None:
            self.init_model(cfg, batch.edge_index)

        pos_rw, neg_rw = self.get_next_data()
        pos_enc = self.model.forward()
        batch.node2vec_loss = self.model.loss(pos_rw.to(cfg.accelerator), neg_rw.to(cfg.accelerator))

        if self.pecfg.norm:
            pos_enc = torch.nn.functional.normalize(pos_enc)
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
