import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
from collections.abc import Iterable
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils.convert import to_networkx
import logging
import torch
import tempfile

from torch_sparse import random_walk


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int32)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
	    return J[kk]

def norm_vectors(x):
    norms = torch.linalg.vector_norm(x, dim=1)
    i = torch.argmin(norms)
    x_normed = x - x[i, :].reshape(1, -1)
    norms = torch.linalg.vector_norm(x_normed, dim=1)
    j = torch.argmax(norms)
    x_normed /= norms[j]
    return x_normed
	
# def learn_embeddings(data, cfg):
# 	'''
# 	Learn embeddings by optimizing the Skipgram objective using SGD.
# 	'''
# 	logging.disable()
# 	model = Node2Vec(edge_index=data.edge_index,
#                               embedding_dim=cfg.posenc_Node2Vec.dim_pe,
#                               walk_length=cfg.posenc_Node2Vec.walk_length,
#                               context_size=cfg.posenc_Node2Vec.window_size,
#                               walks_per_node=cfg.posenc_Node2Vec.num_walks,
#                               q=cfg.posenc_Node2Vec.q,
#                               p=cfg.posenc_Node2Vec.p,
#                               num_nodes=data.num_nodes,
#                               num_negative_samples=cfg.posenc_Node2Vec.num_negative_samples)

# 	loader = model.loader(batch_size=128, shuffle=True, num_workers=cfg.num_workers)
# 	walks = torch.cat([walks for walks, _ in loader], axis=0).tolist()

# 	wv_model = Word2Vec(walks, vector_size = cfg.posenc_Node2Vec.dim_pe, window=cfg.posenc_Node2Vec.window_size, min_count=0, sg=1, workers=cfg.num_workers)
# 	logging.disable(logging.DEBUG)
# 	embeddings = torch.from_numpy(wv_model.wv.vectors)
# 	if cfg.posenc_Node2Vec.norm:
# 		return embeddings / np.linalg.norm(embeddings , axis=1).reshape(-1, 1)
# 	else:
# 		return embeddings 

def learn_embeddings(data, cfg):
	model = Node2Vec(edge_index=data.edge_index,
                              embedding_dim=cfg.posenc_Node2Vec.dim_pe,
                              walk_length=cfg.posenc_Node2Vec.walk_length,
                              context_size=cfg.posenc_Node2Vec.window_size,
                              walks_per_node=cfg.posenc_Node2Vec.num_walks,
                              q=cfg.posenc_Node2Vec.q,
                              p=cfg.posenc_Node2Vec.p,
                              num_nodes=data.num_nodes,
                              num_negative_samples=cfg.posenc_Node2Vec.num_negative_samples).to(cfg.accelerator)
	model.train()

	# Initialize the model loader
	loader = model.loader(batch_size=128, shuffle=True, num_workers=cfg.num_workers)

	# Initial learning rate and minimum learning rate
	start_alpha = 0.025
	min_alpha = 0.0001
	num_epochs = 3

	# Create optimizer
	total_batches = len(loader) * num_epochs
	optimizer = torch.optim.Adam(list(model.parameters()), lr=start_alpha)
	# lambda_lr = lambda batch: max(min_alpha, start_alpha - (start_alpha - min_alpha) * (batch / total_batches))
	# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

	total_loss = 0
	losses = []
	for epoch in range(num_epochs):
		for batch_idx, (pos_rw, neg_rw) in enumerate(loader):
			optimizer.zero_grad()
			loss = model.loss(pos_rw.to(cfg.accelerator), neg_rw.to(cfg.accelerator))
			loss.backward()
			optimizer.step()
			# scheduler.step()
			losses.append(loss.item())

	embeddings = model().detach().cpu()
	if cfg.posenc_Node2Vec.norm:
		return embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
	else:
		return embeddings


		