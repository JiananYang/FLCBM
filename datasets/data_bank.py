import os


from torchvision import transforms
from .cub import CUBDataset
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
def load_cub_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images', resampling=False, resol=299,
			 normalizer=transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2]),
			 n_classes=200):
	"""
	Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
	Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
	NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
	"""
	is_training = any(['train.pkl' in f for f in pkl_paths])
	if is_training:
		transform = transforms.Compose([
			transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
			transforms.RandomResizedCrop(resol),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(), 
			normalizer
			])
	else:
		transform = transforms.Compose([
			transforms.CenterCrop(resol),
			transforms.ToTensor(), 
			normalizer
			])

	dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, n_classes, transform)

	if is_training:
		drop_last = True
		shuffle = True
	else:
		drop_last = False
		shuffle = False
	# if resampling:
	#     sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
	#     loader = DataLoader(dataset, batch_sampler=sampler)
	# else:
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
	return loader
def get_dataset(args,preprocess):

	from .param import CUB_ATTRIBUTE_DIR, CUB_DATA_DIR
	from torchvision import transforms
	num_classes = 200
	TRAIN_PKL = os.path.join(os.path.join("datasets",CUB_ATTRIBUTE_DIR), "train.pkl")
	TEST_PKL = os.path.join(os.path.join("datasets",CUB_ATTRIBUTE_DIR), "test.pkl")
	normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
	train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
		batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
		n_classes=num_classes, resampling=True)

	test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
			batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
			n_classes=num_classes, resampling=True)

	classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()

	classes = [a.split(".")[1].strip() for a in classes]
	idx_to_class = {i: classes[i] for i in range(num_classes)}
	classes = [classes[i] for i in range(num_classes)]
	print(len(classes), "num classes for cub")
	print(len(train_loader.dataset), "training set size")
	print(len(test_loader.dataset), "test set size")

	return train_loader, test_loader, idx_to_class, classes