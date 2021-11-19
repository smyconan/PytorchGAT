import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

IF_SPARSE = True

class GATLayer(nn.Module):
	def __init__(self, in_dim, out_dim, dropout_rate, alpha):
		super(GATLayer, self).__init__()
		self.dropout_rate = dropout_rate
		self.W = nn.Parameter(torch.zeros(in_dim, out_dim))
		self.a = nn.Parameter(torch.zeros(2*out_dim, 1))
		self.activate = nn.LeakyReLU(alpha)
		self.initialize()
	def initialize(self):
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)
		nn.init.xavier_uniform_(self.a.data, gain = 1.414)
	def forward(self, X, Adj):
		h = torch.mm(X, self.W) 															# BS * out_dim
		N, D = h.size()																		# N = BS, D = out_dim
		concatnated = torch.cat([h.repeat(1,N).view(N*N,D), h.repeat(N,1)], dim = 1)		# N^2 * 2out_dim
		e = self.activate(torch.mm(concatnated, self.a))									# N^2 * 1
		e = torch.exp(e).view(N,N,1)[:,:,0].masked_fill(mask = (Adj == 0),value = 0.0)		# N * N
		attention = (e.t() / torch.sum(e, dim = 1)).t()										# N * N
		attention = F.dropout(attention, self.dropout_rate, training = self.training)		# N * N
		return torch.mm(attention, h)														# N * out_dim

class SparseGATLayer(nn.Module):
	def __init__(self, in_dim, out_dim, dropout_rate, alpha):
		super(SparseGATLayer, self).__init__()
		self.dropout_rate = dropout_rate
		self.W = nn.Parameter(torch.zeros(in_dim, out_dim))
		self.a = nn.Parameter(torch.zeros(2*out_dim, 1))
		self.activate = nn.LeakyReLU(alpha)
		self.initialize()
	def initialize(self):
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)
		nn.init.xavier_uniform_(self.a.data, gain = 1.414)
	def forward(self, X, edges):																	
		h = torch.mm(X, self.W)																# BS * out_dim
		N, D = h.size()																		# N = BS, D = out_dim
		from_node_ids, to_node_ids = edges[0,:], edges[1,:]
		concatnated = torch.cat([h[from_node_ids,:], h[to_node_ids,:]], dim = 1)			# E * 2out_dim
		e = self.activate(torch.mm(concatnated, self.a))									# E * 1
		e = torch.sparse_coo_tensor(edges, e[:,0], size = (N, N))							# Sparse N * N
		attention = torch.sparse.softmax(e, dim = 1)										# Sparse N * N
		indices, values = attention.coalesce().indices(), attention.coalesce().values()
		values = F.dropout(values, self.dropout_rate, training = self.training)
		attention = torch.sparse_coo_tensor(indices, values, size = (N,N))					# Sparse N * N
		return torch.sparse.mm(attention, h)

class GAT(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, attention_dropout_rate, dropout_rate, alpha, num_heads_hidden, num_heads_out, num_layers, if_sparse, edge_types):
		super(GAT, self).__init__()
		hidden_head_dim = int(hidden_dim / num_heads_hidden)
		hidden_dim = hidden_head_dim * num_heads_hidden
		layer_in_dims = [in_dim] + [hidden_dim]*(num_layers - 1)
		layer_out_dims = [hidden_head_dim]*(num_layers - 1) + [out_dim]
		layer_num_heads = [num_heads_hidden]*(num_layers - 1) + [num_heads_out]
		GatModelLayer = SparseGATLayer if if_sparse else GATLayer
		self.hidden_head_dim = hidden_head_dim
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		self.num_heads_hidden = num_heads_hidden
		self.num_heads_out = num_heads_out
		self.if_sparse = if_sparse
		self.edge_types = edge_types
		self.layer_in_dims = layer_in_dims
		self.layer_out_dims = layer_out_dims
		self.attentions = nn.ModuleList([
							nn.ModuleList([
								nn.ModuleList([
									GatModelLayer(layer_in_dims[i], layer_out_dims[i], attention_dropout_rate, alpha) 
									for _ in range(layer_num_heads[i])
								])
								for _ in range(edge_types)
							]) 
							for i in range(num_layers)
						  ])
		self.edge_type_attentions = nn.ModuleList([nn.Linear(edge_types,1) for i in range(num_layers)])
	def forward(self, X, graphs):
		X = F.dropout(X, self.dropout_rate, training = self.training)
		for i in range(self.num_layers - 1):
			tmp = torch.zeros((X.size(0), self.layer_in_dims[i+1], self.edge_types))
			for j in range(self.edge_types):
				heads = self.attentions[i][j]
				Xj = torch.cat([F.elu(gat_head(X,graphs[j])) for gat_head in heads], dim = 1)
				Xj = F.dropout(Xj, self.dropout_rate, training = self.training)
				tmp[:,:,j] = Xj
			X = F.elu(self.edge_type_attentions[i](tmp)[:,:,0])
		tmp = torch.zeros((X.size(0), self.layer_out_dims[-1], self.edge_types))
		for j in range(self.edge_types):
			out_heads = self.attentions[-1][j]
			gat_out_j = F.elu(sum([gat_head(X,graphs[j]) for gat_head in out_heads]) / self.num_heads_out)		# N * out_dim
			tmp[:,:,j] = gat_out_j
		return F.elu(self.edge_type_attentions[-1](tmp)[:,:,0])