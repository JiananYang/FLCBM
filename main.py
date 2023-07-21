
import os
import torch
import shutil
import re
from helpers import makedir

from utils.options import args_parser
from utils.sampling import CUB_iid,CUB_non_iid 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from log import create_logger
from preprocess import mean, std

if __name__ == "__main__":
     

	#parse arguments
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(args.gpu))
	from settings import base_architecture, img_size, prototype_shape, num_classes, \
						prototype_activation_function, add_on_layers_type, experiment_run

	base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
	model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
	makedir(model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
	log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

	#training parameters
	from settings import base_architecture, img_size, prototype_shape, num_classes, \
						prototype_activation_function, add_on_layers_type, experiment_run
	#load the data
	from settings import train_dir, test_dir, train_push_dir, \
						train_batch_size, test_batch_size, train_push_batch_size
	normalize = transforms.Normalize(mean=mean,std=std)
	#train set
	train_dataset = datasets.ImageFolder(
		train_dir,
		transforms.Compose([
			transforms.Resize(size=(img_size,img_size)),
			transforms.ToTensor(),
			normalize
		])
	)

	train_dict_users = CUB_iid(train_dataset,args.num_users)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=train_batch_size, shuffle=True,
		num_workers=4, pin_memory=False)
	#arrange local datasets for all clients

	#test set
	# test set
	test_dataset = datasets.ImageFolder(
		test_dir,
		transforms.Compose([
			transforms.Resize(size=(img_size, img_size)),
			transforms.ToTensor(),
			normalize,
		]))
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=test_batch_size, shuffle=False,
		num_workers=4, pin_memory=False)
	#we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
	log('training set size: {0}'.format(len(train_loader.dataset)))
	log('test set size: {0}'.format(len(test_loader.dataset)))
	log('batch size: {0}'.format(train_batch_size))
	#model construction

	#optimizer
	from settings import joint_optimizer_lrs, joint_lr_step_size
	joint_optimizer_specs = \
	[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
	{'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
	{'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
	]
	joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
	joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)


	# number of training epochs, number of warm epochs, push start epoch, push epochs
	from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
	import numpy as np
	import copy
	from FL import fedAvg


