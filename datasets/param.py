

CUB_DATA_DIR = "data/CUB_200_2011"
CUB_ATTRIBUTE_DIR = "class_attr_data_10"

backbone_name = "resnet18_cub"
dataset_name = "cub"
out_dir = "../saved_models"
device = 'cuda'
seed = 1
num_workers = 4
concept_batch_size = 100
C = [0.01,0.1]
n_samples_concept = 50