

import numpy as np


def CUB_iid(dataset,k):
	num_items = int(len(dataset) / k)

	dict_users, all_idxs = {}, [i for i in range(len(dataset))]

	for i in range(k):
		dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))#select #num_items indexs from all indexs
		# all_idxs.pop(dict_users[i])
		all_idxs = list(set(all_idxs) - dict_users[i])#remove idx selected

	return dict_users


def CUB_non_iid(dataset,k):
	pass
