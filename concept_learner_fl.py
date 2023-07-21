

# from utils.options import concept_args_parser

if __name__ == "__main__":
    
	from settings import backbone_name,dataset_name,out_dir,device,seed,num_workers,concept_batch_size,n_samples_concept
	#get model
	from resnet_features import resnet18_features
	features = resnet18_features(True)
	
	#get concept dataset loader

	#train model
	

    