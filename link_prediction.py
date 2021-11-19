from spGAT import GAT, IF_SPARSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

NEGATIVE_SAMPLING_RATE = 5

class LinkClassifier(nn.Module):
	def __init__(self, node_embed_dim, edge_types):
		super(LinkClassifier, self).__init__()
		self.classifiers = nn.ModuleList([nn.Linear(node_embed_dim*2,2) for _ in range(edge_types)]) # Linear Binary Classifier for each edge type. 
	def forward(self, node_pairs_by_edge_type):
		predictions = []
		for i in range(len(node_pairs_by_edge_type)):
			node_pairs = node_pairs_by_edge_type[i]
			model = self.classifiers[i]
			predictions.append(model(node_pairs))
		return predictions

def negative_sampling(true_edges, edge_types, N):

	false_edges = [set() for _ in range(len(true_edges))]
	for i in range(len(true_edges)):
		for pair in true_edges[i]:
			for _ in range(NEGATIVE_SAMPLING_RATE):
				target = int(random.random()*N)
				while ((pair[0], target) in true_edges[i]) or ((pair[0], target) in false_edges[i]):
					target = int(random.random()*N)
				false_edges[i].add((pair[0], target))

	train_edges, train_label = [[] for _ in range(edge_types)], [[] for _ in range(edge_types)]
	for i in range(len(true_edges)):
		for _ in range(NEGATIVE_SAMPLING_RATE):
			for pair in true_edges[i]:
				train_edges[i].append([pair[0], pair[1]])
				train_label[i].append(1)
		for pair in false_edges[i]:
			train_edges[i].append([pair[0], pair[1]])
			train_label[i].append(0)
		train_edges[i] = torch.LongTensor(train_edges[i])
		train_label[i] = torch.LongTensor(train_label[i])

	return false_edges, train_edges, train_label

def read_example_data():

	# read node features
	f = open("example/feature.txt")
	lines = [line for line in f]
	f.close()
	tmp = lines[0][:-1].split(" ")
	N, dim = int(tmp[0]), int(tmp[1])
	nodename2id, nid = {}, -1
	X = torch.zeros((N,dim))
	for i in range(1,len(lines)):
		tmp = lines[i][:-1].split(" ")
		nid += 1
		nodename2id[tmp[0]] = nid
		X[nid,:] = torch.tensor([float(t) for t in tmp[1:]])
	print("Nodes:",len(nodename2id))
	
	# read training edges
	f = open("example/train.txt")
	As = []
	edgename2id, eid, true_edges = {}, -1, []
	for line in f:
		tmp = line[:-1].split(" ")
		edgename = tmp[0]
		x,y = tmp[1],tmp[2]
		if edgename not in edgename2id:
			eid += 1
			edgename2id[edgename] = eid
			A = torch.eye(N)
			As.append(A)
		A = As[edgename2id[edgename]]
		A[nodename2id[x], nodename2id[y]] = 1
		if edgename2id[edgename] > len(true_edges):
			print("Index Error in reading edges.")
			exit()
		if edgename2id[edgename] == len(true_edges):
			true_edges.append(set())
		true_edges[edgename2id[edgename]].add((nodename2id[x], nodename2id[y]))
	print("Edges:",len(As))
	f.close()
	false_edges, train_edges, train_label = negative_sampling(true_edges, len(As), N)
	print("Training True Edges:",sum([len(true_edges[i]) for i in range(len(true_edges))]))
	print("Training False Edges (negative sampling):",sum([len(false_edges[i]) for i in range(len(false_edges))]))

	# read validating edges
	f = open("example/valid.txt")
	valid_true_edges = [[] for _ in range(len(true_edges))]
	valid_false_edges = [[] for _ in range(len(true_edges))]
	for line in f:
		tmp = line[:-1].split(" ")
		eid = edgename2id[tmp[0]]
		x, y = nodename2id[tmp[1]], nodename2id[tmp[2]]
		label = int(tmp[3])
		if label == 1:
			valid_true_edges[eid].append((x, y))
		else:
			valid_false_edges[eid].append((x, y))
	f.close()
	print("Valid True Edges:", sum([len(valid_true_edges[i]) for i in range(len(valid_true_edges))]))
	print("Valid False Edges:", sum([len(valid_false_edges[i]) for i in range(len(valid_false_edges))]))

	valid_edges, valid_label = [[] for _ in range(len(As))], [[] for _ in range(len(As))]
	for i in range(len(valid_true_edges)):
		for pair in valid_true_edges[i]:
			valid_edges[i].append([pair[0], pair[1]])
			valid_label[i].append(1)
		for pair in valid_false_edges[i]:
			valid_edges[i].append([pair[0], pair[1]])
			valid_label[i].append(0)
		valid_edges[i] = torch.LongTensor(valid_edges[i])
		valid_label[i] = torch.LongTensor(valid_label[i])

	return X, As, train_edges, train_label, valid_edges, valid_label, true_edges

if __name__ == '__main__':	

	X, As, train_edges, train_label, valid_edges, valid_label, original_true_training_edges = read_example_data()
	N = X.size(0)
	if IF_SPARSE:
		As = [torch.LongTensor([[i,j] for i in range(N) for j in range(N) if A[i,j] == 1]).t() for A in As]

	model = GAT(in_dim = X.size(1),
				hidden_dim = 64,
				out_dim = 16,
				attention_dropout_rate = 0.1,										# dropout rate for attention matrix
				dropout_rate = 0.1,													# dropout rate for feature matrix
				alpha = 0.2,
				num_heads_hidden = 8,												# hidden heads will be concatnated
				num_heads_out = 1,													# out heads will be averaged
				num_layers = 4,
				if_sparse = IF_SPARSE,
				edge_types = len(As))
	classifier = LinkClassifier(node_embed_dim = 16, edge_types = len(As))
	optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": classifier.parameters()}], lr = 0.01)
	loss_func = nn.CrossEntropyLoss()

	for epoch in range(10000):
		model.train()
		classifier.train()
		optimizer.zero_grad()
		out = model(X,As)
		node_pairs_by_edge_type = [out[train_edges[i]].view(train_edges[i].size()[0],-1) for i in range(len(train_edges))]
		predictions = classifier(node_pairs_by_edge_type)
		loss = sum([loss_func(predictions[i], train_label[i])*train_edges[i].size()[0] for i in range(len(train_edges))]) / float(sum([train_edges[i].size()[0] for i in range(len(train_edges))]))
		preds = [torch.argmax(F.softmax(predictions[i], dim = 1), dim = 1) for i in range(len(train_edges))]
		acc = sum([torch.sum(preds[i] == train_label[i]) for i in range(len(train_edges))]) / float(sum([train_edges[i].size()[0] for i in range(len(train_edges))]))
		loss.backward()
		optimizer.step()

		print("EPOCH", epoch, "LOSS", loss.item(), "ACC", acc.item(), end = " ")

		model.eval()
		classifier.eval()
		out = model(X,As)
		node_pairs_by_edge_type = [out[valid_edges[i]].view(valid_edges[i].size()[0],-1) for i in range(len(valid_edges))]
		predictions = classifier(node_pairs_by_edge_type)
		loss = sum([loss_func(predictions[i], valid_label[i])*valid_edges[i].size()[0] for i in range(len(valid_edges))]) / float(sum([valid_edges[i].size()[0] for i in range(len(valid_edges))]))
		preds = [torch.argmax(F.softmax(predictions[i], dim = 1), dim = 1) for i in range(len(valid_edges))]
		acc = sum([torch.sum(preds[i] == valid_label[i]) for i in range(len(valid_edges))]) / float(sum([valid_edges[i].size()[0] for i in range(len(valid_edges))]))
		
		print("VDLOSS", loss.item(), "VDACC", acc.item())

		_, train_edges, train_label = negative_sampling(original_true_training_edges, len(As), N)

		if epoch % 200 == 0:
			for group in optimizer.param_groups:
				group['lr'] = group['lr'] * 0.7


