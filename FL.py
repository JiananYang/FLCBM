

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def fedAvg(w):
	"""
	w_avg: dicts, keys are 
	"""

	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1,len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k],len(w))
	return w_avg


class DatasetSplit(object):
	def __init__(self,dataset,idx):
		self.dataset = dataset
		self.idxs = list(idx)

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self,item):
		img,label = self.dataset[self.idxs[item]]
		return img, label


class CentralUpdate:
	
	def __init__(self,args,dataset):
		self.args = args
		self.loss_func = nn.CrossEntropyLoss()
		self.ld_train = DataLoader(dataset, batch_size=self.args.local_bs,shuffle=True)

	def train(self,net):
		net.train()
		optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
		epoch_loss = []

		for iter in range(self.args.ep):
			batch_loss = []
			for batch_idx, (img,labels) in enumerate(self.ld_train):
				img,labels = img.to(self.args.device), labels.to(self.args.device)
				net.zero_grad()
				log_probs,min_distances = net(img)
				loss = self.loss_func(log_probs,labels)
				loss.backward()
				optimizer.step()
				if batch_idx % 100 == 0:
					print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						iter, batch_idx * len(img), len(self.ld_train.dataset),
							100. * batch_idx / len(self.ld_train), loss.item()))

				batch_loss.append(loss.item())
			epoch_loss.append(sum(batch_loss)/len(batch_loss))
		return net.state_dict(), sum(epoch_loss) / len(epoch_loss)






class LocalUpdate:
	
	def __init__(self,args,dataset,idx):
		self.args = args
		self.ld_train = DataLoader(DatasetSplit(dataset,idx),batch_size=self.args.local_bs,shuffle=True)
		self.loss_func = nn.CrossEntropyLoss()

	def train(self,net,class_specific=True,coefs=None):

		net.train()
		optimizer = torch.optim.SGD(net.parameters(),lr=self.args.lr,momentum=self.args.momentum)

		epoch_loss = []
		total_cross_entropy = 0
		total_cluster_cost = 0
		# separation cost is meaningful only for class_specific
		total_separation_cost = 0
		total_avg_separation_cost = 0
		for iter in range(self.args.local_ep):
			batch_loss = []
			for batch_idx, (img,labels) in enumerate(self.ld_train):
				# img,labels = img.to(self.args.device),labels.to(self.args.device)
				img, labels = img.cuda(), labels.cuda()

				net.zero_grad()
				log_probs,min_distances = net(img)
				cross_entropy = self.loss_func(log_probs,labels)
				# print (log_probs,labels)
				if class_specific:
					max_dist = (net.module.prototype_shape[1]
							* net.module.prototype_shape[2]
							* net.module.prototype_shape[3])

					# prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
					# calculate cluster cost
					prototypes_of_correct_class = torch.t(net.module.prototype_class_identity[:,labels]).cuda()
					inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
					cluster_cost = torch.mean(max_dist - inverted_distances)

					# calculate separation cost
					prototypes_of_wrong_class = 1 - prototypes_of_correct_class
					inverted_distances_to_nontarget_prototypes, _ = \
						torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
					separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

					# calculate avg cluster cost
					avg_separation_cost = \
						torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
					avg_separation_cost = torch.mean(avg_separation_cost)
				
					# if use_l1_mask:
					# 	l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
					# 	l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
					# else:
					# 	l1 = model.module.last_layer.weight.norm(p=1) 
				else:
					min_distance, _ = torch.min(min_distances, dim=1)
					cluster_cost = torch.mean(min_distance)
					l1 = net.last_layer.weight.norm(p=1)

				if class_specific:
					if coefs is not None:
						loss = (coefs['crs_ent'] * cross_entropy
						  + coefs['clst'] * cluster_cost
						  + coefs['sep'] * separation_cost
						  + coefs['l1'] * l1)
					else:
						loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
				else:
					if coefs is not None:
						loss = (coefs['crs_ent'] * cross_entropy
							+ coefs['clst'] * cluster_cost
							+ coefs['l1'] * l1)
					else:
						# loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
						loss = cross_entropy
				loss = cross_entropy
				loss.backward()
				optimizer.step()
				if batch_idx % 100 == 0:
					print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						iter, batch_idx * len(img), len(self.ld_train.dataset),
							100. * batch_idx / len(self.ld_train), loss.item()))

				batch_loss.append(loss.item())
			epoch_loss.append(sum(batch_loss)/len(batch_loss))
		return net.state_dict(), sum(epoch_loss) / len(epoch_loss)