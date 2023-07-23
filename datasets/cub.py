
import torch
from PIL import Image
import pickle
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from tqdm import tqdm

from .param import CUB_ATTRIBUTE_DIR,CUB_DATA_DIR
from .param import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
from torch.utils.data import Dataset, DataLoader
N_ATTRIBUTES=312
class ResNetBottom(nn.Module):
	def __init__(self, original_model):
		super(ResNetBottom, self).__init__()
		self.features = nn.Sequential(*list(original_model.children())[:-1])
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		return x
	
class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, num_classes, transform=None, pkl_itself=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])        
        if pkl_itself is None:

            for file_path in pkl_file_paths:
                self.data.extend(pickle.load(open(file_path, 'rb')))
        else:
            self.data = pkl_itself
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths

        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
        img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label
def get_model():
	from pytorchcv.model_provider import get_model as ptcv_get_model
	model = ptcv_get_model(backbone_name, pretrained=True, root=out_dir)
	backbone= ResNetBottom(model)
	cub_mean_pxs = np.array([0.5, 0.5, 0.5])
	cub_std_pxs = np.array([2., 2., 2.])
	preprocess = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(cub_mean_pxs, cub_std_pxs)
		])
	return backbone,preprocess
class CUBConceptDataset:
	def __init__(self, images, transform=None):
		self.images = images
		self.transform = transform

	def __len__(self):
		# Return the length of the dataset
		return len(self.images)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_path = self.images[idx]
		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image


def get_concept_dicts(metadata):
	""""
	meta:[{'id':0,'label':21,'attribute_label':[binary classifier of 112 concepts]},{},...]
	"""
	from param import CUB_DATA_DIR
	n_concepts = len(metadata[0]["attribute_label"])
	concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
	for im_data in metadata:
		for c, label in enumerate(im_data["attribute_label"]):
			# print(c)
			img_path = im_data["img_path"]
			# print (img_path)            
			idx = img_path.split('/').index('CUB_200_2011')
			img_path = '/'.join([CUB_DATA_DIR] + img_path.split('/')[idx+1:])
			# print (img_path)
			concept_info[c][label].append(img_path)
	return concept_info

def CUB_iid(dataset,k):
    num_items = int(len(dataset)/k)

    dict_users, all_idxs = {},[i for i in range(len(dataset))]

    for i in range(k):
        dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users
    


def cub_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
	TRAIN_PKL = os.path.join(CUB_ATTRIBUTE_DIR, "train.pkl")
	metadata = pickle.load(open(TRAIN_PKL, "rb"))

	concept_info = get_concept_dicts(metadata=metadata)

	np.random.seed(seed)
	torch.manual_seed(seed)
	concept_loaders = {}
	print (len(concept_info))
	for c_idx, c_data in concept_info.items():
		pos_ims, neg_ims = c_data[1], c_data[0]#pos_ims = list of images that has concept c_idx
		# Sample equal number of positive and negative images
		try:
			pos_concept_ims = np.random.choice(pos_ims, 2*n_samples, replace=False)
			neg_concept_ims = np.random.choice(neg_ims, 2*n_samples, replace=False)
		except Exception as e:
			print(e)
			print(f"{len(pos_ims)} positives, {len(neg_ims)} negatives")
			pos_concept_ims = np.random.choice(pos_ims, 2*n_samples, replace=True)
			neg_concept_ims = np.random.choice(neg_ims, 2*n_samples, replace=True)

		pos_ds = CUBConceptDataset(pos_concept_ims, preprocess)
		neg_ds = CUBConceptDataset(neg_concept_ims, preprocess)
		pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		concept_loaders[c_idx] = {
			"pos": pos_loader,
			"neg": neg_loader
		}
	return concept_loaders


@torch.no_grad()
def get_embeddings(loader, model, device="cuda"):
    """
    Args:
        loader ([torch.utils.data.DataLoader]): Data loader returning only the images
        model ([nn.Module]): Backbone
        n_samples (int, optional): Number of samples to extract the activations
        device (str, optional): Device to use. Defaults to "cuda".

    Returns:
        np.array: Activations as a numpy array.
    """
    activations = None
    for image in tqdm(loader):
        image = image.to(device)
        try:
            batch_act = model(image).squeeze().detach().cpu().numpy()
        except:
            # Then it's a CLIP model. This is a really nasty soln, i should fix this.
            batch_act = model.encode_image(image).squeeze().detach().cpu().numpy()
        if activations is None:
            activations = batch_act
        else:
            activations = np.concatenate([activations, batch_act], axis=0)
    return activations

def get_cavs(X_train, y_train, X_val, y_val, C):
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: activations (numpy arrays) to learn the concepts with.
        C: Regularizer for the SVM. 
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": y_train.sum(),
                   "neg_count": (1-y_train).sum(),
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info


def learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, C, device="cuda"):
    """Learning CAVs and related margin stats.
    Args:
        pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
        neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
        model_bottom (nn.Module): Mode
        n_samples (int): Number of positive samples to use while learning the concept.
        C (float): Regularization parameter for the SVM. Possibly multiple options.
        device (str, optional): Device to use while extracting activations. Defaults to "cuda".

    Returns:
        dict: Concept information, including the CAV and margin stats.
    """
    print("Extracting Embeddings: ")
    pos_act = get_embeddings(pos_loader, backbone, device=device)
    neg_act = get_embeddings(neg_loader, backbone, device=device)
    
    X_train = np.concatenate([pos_act[:n_samples], neg_act[:n_samples]], axis=0)
    X_val = np.concatenate([pos_act[n_samples:], neg_act[n_samples:]], axis=0)
    y_train = np.concatenate([np.ones(pos_act[:n_samples].shape[0]), np.zeros(neg_act[:n_samples].shape[0])], axis=0)
    y_val = np.concatenate([np.ones(pos_act[n_samples:].shape[0]), np.zeros(neg_act[n_samples:].shape[0])], axis=0)
    concept_info = {}
    for c in C:
        concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c)
    return concept_info
if __name__ == "__main__":
	backbone,preprocess = get_model()
	backbone = backbone.to(device)
	backbone = backbone.eval()
	concept_libs = {c: {} for c in C}
	# Get the positive and negative loaders for each concept.
	concept_loaders = cub_concept_loaders(preprocess, n_samples=n_samples_concept, batch_size=concept_batch_size, 
										num_workers=num_workers, seed=seed)
	
	np.random.seed(seed)
	torch.manual_seed(seed)
	for concept_name, loaders in concept_loaders.items():
		pos_loader, neg_loader = loaders['pos'], loaders['neg']
		# Get CAV for each concept using positive/negative image split
		cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples_concept, C, device="cuda")
		
		# Store CAV train acc, val acc, margin info for each regularization parameter and each concept
		for c in C:
			concept_libs[c][concept_name] = cav_info[c]
			print(concept_name, c, cav_info[c][1], cav_info[c][2])

	for key in concept_libs.keys():
		lib_path = os.path.join(out_dir, f"{dataset_name}_{backbone_name}_{key}_{n_samples_concept}.pkl")
		with open(lib_path, "wb") as f:
			pickle.dump(concept_libs[key], f)
		print(f"Saved to: {lib_path}")        
	
		total_concepts = len(concept_libs[key].keys())
		print(f"File: {lib_path}, Total: {total_concepts}")