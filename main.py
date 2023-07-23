
import os
import torch
import shutil
import re
import pickle
from helpers import makedir

from utils.options import args_parser,linear_probe_parser
from utils.sampling import CUB_iid,CUB_non_iid 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from log import create_logger
from preprocess import mean, std
from concepts import ConceptBank
from datasets import get_dataset
from models import get_model
from datasets.param import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
from models import PosthocLinearCBM
from training_tools import load_or_compute_projections
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
def run_linear_probe(args, train_data, test_data):
	train_features, train_labels = train_data
	test_features, test_labels = test_data
	
	# We converged to using SGDClassifier. 
	# It's fine to use other modules here, this seemed like the most pedagogical option.
	# We experimented with torch modules etc., and results are mostly parallel.
	classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
							   alpha=args.lam, l1_ratio=args.alpha, verbose=0,
							   penalty="elasticnet", max_iter=10000)
	classifier.fit(train_features, train_labels)

	train_predictions = classifier.predict(train_features)
	train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
	predictions = classifier.predict(test_features)
	test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

	# Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
	cls_acc = {"train": {}, "test": {}}
	for lbl in np.unique(train_labels):
		test_lbl_mask = test_labels == lbl
		train_lbl_mask = train_labels == lbl
		cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
		cls_acc["train"][lbl] = np.mean(
			(train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
		print(f"{lbl}: {cls_acc['test'][lbl]}")

	run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
				"cls_acc": cls_acc,
				}

	# If it's a binary task, we compute auc
	if test_labels.max() == 1:
		run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
		run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
	return run_info, classifier.coef_, classifier.intercept_
if __name__ == "__main__":
	 

	#parse arguments
	args = linear_probe_parser()
	args.device = torch.device('cuda:{}'.format(args.gpu))
	from settings import base_architecture, img_size, prototype_shape, num_classes, \
						prototype_activation_function, add_on_layers_type, experiment_run

	base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
	model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
	makedir(model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
	shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
	# shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
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
	# from settings import joint_optimizer_lrs, joint_lr_step_size
	# joint_optimizer_specs = \
	# [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
	# {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
	# {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
	# ]
	# joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
	# joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)


	# number of training epochs, number of warm epochs, push start epoch, push epochs
	from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
	import numpy as np
	import copy
	from FL import fedAvg

	
	#Concept bank training
	# from concept_learner_fl import concept_learner

	#Linear probe training

	all_concepts = pickle.load(open(os.path.join('saved_models',args.concept_bank), 'rb'))
	all_concept_names = list(all_concepts.keys())
	print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
	concept_bank = ConceptBank(all_concepts, device)

	backbone, preprocess = get_model(args, backbone_name=backbone_name)
	backbone = backbone.to(device)
	backbone.eval()

	train_loader, test_loader, idx_to_class, classes = get_dataset(args,preprocess)
	conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
	num_classes = len(classes)
	
	# Initialize the PCBM module.
	posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
	posthoc_layer = posthoc_layer.to(args.device)

	# We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
	train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
	run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
	
	# Convert from the SGDClassifier module to PCBM module.
	posthoc_layer.set_weights(weights=weights, bias=bias)

	

	model_path = os.path.join(args.out_dir,
							  f"pcbm_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam{args.lam}__alpha{args.alpha}__seed{args.seed}.ckpt")
	torch.save(posthoc_layer, model_path)

	run_info_file = model_path.replace("pcbm", "run_info-pcbm")
	run_info_file = run_info_file.replace(".ckpt", ".pkl")
	# run_info_file = os.path.join(args.out_dir, run_info_file)
	
	with open(run_info_file, "wb") as f:
		pickle.dump(run_info, f)

	
	if num_classes > 1:
		# Prints the Top-5 Concept Weigths for each class.
		print(posthoc_layer.analyze_classifier(k=5))

	print(f"Model saved to : {model_path}")
	print(run_info)

