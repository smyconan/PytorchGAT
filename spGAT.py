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
	def __init__(self, in_dim, hidden_dim, out_dim, attention_dropout_rate, dropout_rate, alpha, num_heads_hidden, num_heads_out, num_layers, if_sparse):
		super(GAT, self).__init__()
		hidden_head_dim = int(hidden_dim / num_heads_hidden)
		hidden_dim = hidden_head_dim * num_heads_hidden
		layer_in_dims = [in_dim] + [hidden_dim]*(num_layers - 1)
		layer_out_dims = [hidden_head_dim]*(num_layers - 1) + [out_dim]
		layer_num_heads = [num_heads_hidden]*(num_layers - 1) + [num_heads_out]
		GatModelLayer = SparseGATLayer if if_sparse else GATLayer
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		self.num_heads_hidden = num_heads_hidden
		self.num_heads_out = num_heads_out
		self.if_sparse = if_sparse
		self.attentions = nn.ModuleList([
							nn.ModuleList([
								GatModelLayer(layer_in_dims[i], layer_out_dims[i], attention_dropout_rate, alpha) 
								for _ in range(layer_num_heads[i])
							]) 
							for i in range(num_layers)
						  ])
	def forward(self, X, graph):
		X = F.dropout(X, self.dropout_rate, training = self.training)
		for i in range(self.num_layers - 1):
			heads = self.attentions[i]
			X = torch.cat([F.elu(gat_head(X,graph)) for gat_head in heads], dim = 1)
			X = F.dropout(X, self.dropout_rate, training = self.training)
		out_heads = self.attentions[-1]
		gat_out = F.elu(sum([gat_head(X,graph) for gat_head in out_heads]) / self.num_heads_out) 	# N * out_dim
		return gat_out

def read_cora_data():
	f = open("cora/cora.content")
	id2idx = {}
	X, Y = [], []
	idx = -1
	for line in f:
		idx += 1
		tmp = line.split("\n")[0].split("\t")
		id2idx[int(tmp[0])] = idx
		X.append([float(i) for i in tmp[1:-1]])
		Y.append(tmp[-1])
	f.close()
	A = torch.eye(len(Y))
	f = open("cora/cora.cites")
	for line in f:
		tmp = line.split("\t")
		cited, citing = id2idx[int(tmp[0])], id2idx[int(tmp[1])]
		A[citing, cited] = 1
	f.close()
	classes = list(set(Y))
	classes.sort()
	str2cls = {classes[i]:i for i in range(len(classes))}
	Y = torch.LongTensor([str2cls[y] for y in Y])
	X = torch.tensor(X)
	print(X.size(), Y.size(), len(classes), A.size())
	print(str2cls)
	print()
	return X, A, Y, classes

def generate_random_data():
	X = torch.randn(500,128)
	A = torch.zeros(500,500)
	candidates = [(i,j) for i in range(500) for j in range(500) if i != j]
	random.shuffle(candidates)
	for edge in candidates[:10000]:
		A[edge[0],edge[1]] = 1
	return X, A

if __name__ == '__main__':

	# read data
	X, A, Y, classes = read_cora_data()
	if IF_SPARSE:
		N = X.size(0)
		A = torch.LongTensor([[i,j] for i in range(N) for j in range(N) if A[i,j] == 1]).t()

	# build model
	model = GAT(in_dim = X.size(1),
				hidden_dim = 8,
				out_dim = len(classes),
				attention_dropout_rate = 0.3,										# dropout rate for attention matrix
				dropout_rate = 0.3,													# dropout rate for feature matrix
				alpha = 0.2,
				num_heads_hidden = 8,												# hidden heads will be concatnated
				num_heads_out = 1,													# out heads will be averaged
				num_layers = 2,
				if_sparse = IF_SPARSE)

	# build dataset
	train_size, val_size, test_size = 20*len(classes), 500, 1000					# 140 for training | 500 for validation | 1000 for test
	idxes = list(range(X.size(0)))
	random.shuffle(idxes)
	train_idxes = [([idxes[i] for i in range(len(idxes)) if Y[idxes[i]] == c])[:train_size//len(classes)] for c in range(len(classes))]
	train_set = set([train_idxes[i][j] for i in range(len(train_idxes)) for j in range(len(train_idxes[i]))])
	train_idxes = np.array(list(train_set))
	val_idxes = np.array([idx for idx in idxes if idx not in train_set][:val_size])
	test_idxes = np.array([idx for idx in idxes if idx not in train_set][val_size:val_size+test_size])
	print("Training Nodes:", len(train_idxes), "\nValidation Nodes:", len(val_idxes), "\nTest Nodes:", len(test_idxes))

	optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
	loss_func = nn.CrossEntropyLoss()
	
	for epoch in range(10000):
		# train
		model.train()
		optimizer.zero_grad()
		out = model(X,A)
		loss = loss_func(out[train_idxes,:], Y[train_idxes])
		pred = torch.argmax(F.softmax(out[train_idxes,:], dim = 1), dim = 1)
		acc = torch.sum(pred == Y[train_idxes]) / float(train_size)
		loss.backward()
		optimizer.step()
		# validate
		model.eval()
		out_eval = model(X,A)
		loss_eval = loss_func(out_eval[val_idxes,:],Y[val_idxes])
		pred_eval = torch.argmax(F.softmax(out_eval[val_idxes,:], dim = 1), dim = 1)
		acc_eval = torch.sum(pred_eval == Y[val_idxes]) / float(Y[val_idxes].size(0))
		print("EPOCH","%5d"%epoch,"LOSS","%.8f"%loss.item(),"\tACC","%.8f"%acc.item(), "\tVDLOSS","%.8f"%loss_eval.item(),"\tVDACC","%.8f"%acc_eval.item())
		# test
		if epoch % 30 == 0:
			loss_test = loss_func(out_eval[test_idxes,:],Y[test_idxes])
			pred_test = torch.argmax(F.softmax(out_eval[test_idxes,:], dim = 1), dim = 1)
			acc_test = torch.sum(pred_test == Y[test_idxes]) / float(Y[test_idxes].size(0))
			print("TEST_LOSS","%.8f"%loss_test.item(),"\tTEST ACC","%.8f"%acc_test.item())
