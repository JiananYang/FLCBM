

# from utils.options import concept_args_parser
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from settings import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept,C
class ResNetBottom(nn.Module):
	def __init__(self, original_model):
		super(ResNetBottom, self).__init__()
		self.features = nn.Sequential(*list(original_model.children())[:-1])
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		return x
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

def get_concept_loaders()

if __name__ == "__main__":
	
	#get model
	from resnet_features import resnet18_features
	features = resnet18_features(True)
	
	#get concept dataset loader
	
	# Bottleneck part of model
	backbone, preprocess = get_model()
	backbone = backbone.to(device)
	backbone = backbone.eval()
	
	concept_libs = {c: {} for c in C}
	# Get the positive and negative loaders for each concept. 
	
	concept_loaders = get_concept_loaders(args.dataset_name, preprocess, n_samples=args.n_samples, batch_size=args.batch_size, 
										  num_workers=args.num_workers, seed=args.seed)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	for concept_name, loaders in concept_loaders.items():
		pos_loader, neg_loader = loaders['pos'], loaders['neg']
		# Get CAV for each concept using positive/negative image split
		cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, args.C, device="cuda")
		
		# Store CAV train acc, val acc, margin info for each regularization parameter and each concept
		for C in args.C:
			concept_libs[C][concept_name] = cav_info[C]
			print(concept_name, C, cav_info[C][1], cav_info[C][2])

	# Save CAV results    
	for C in concept_libs.keys():
		lib_path = os.path.join(args.out_dir, f"{args.dataset_name}_{args.backbone_name}_{C}_{args.n_samples}.pkl")
		with open(lib_path, "wb") as f:
			pickle.dump(concept_libs[C], f)
		print(f"Saved to: {lib_path}")        
	
		total_concepts = len(concept_libs[C].keys())
		print(f"File: {lib_path}, Total: {total_concepts}")
	#train model
	

	