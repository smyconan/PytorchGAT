from spGAT import GAT, IF_SPARSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

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
	A1 = torch.eye(len(Y))
	A2 = torch.eye(len(Y))
	f = open("cora/cora.cites")
	for line in f:
		tmp = line.split("\t")
		cited, citing = id2idx[int(tmp[0])], id2idx[int(tmp[1])]
		A1[citing, cited] = 1
		A2[cited, citing] = 1
	f.close()
	classes = list(set(Y))
	classes.sort()
	str2cls = {classes[i]:i for i in range(len(classes))}
	Y = torch.LongTensor([str2cls[y] for y in Y])
	X = torch.tensor(X)
	print(X.size(), Y.size(), len(classes), A1.size(), A2.size())
	print(str2cls)
	print()
	return X, [A1, A2], Y, classes

if __name__ == '__main__':

	# read data
	X, As, Y, classes = read_cora_data()
	if IF_SPARSE:
		N = X.size(0)
		As = [torch.LongTensor([[i,j] for i in range(N) for j in range(N) if A[i,j] == 1]).t() for A in As]

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
				if_sparse = IF_SPARSE,
				edge_types = 2)

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
		out = model(X,As)
		loss = loss_func(out[train_idxes,:], Y[train_idxes])
		pred = torch.argmax(F.softmax(out[train_idxes,:], dim = 1), dim = 1)
		acc = torch.sum(pred == Y[train_idxes]) / float(train_size)
		loss.backward()
		optimizer.step()

		# validate
		model.eval()
		out_eval = model(X,As)
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
