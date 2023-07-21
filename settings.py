base_architecture = 'vgg19'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

data_path = 'data/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]


#concept learner parameters
backbone_name = "resnet18_cub"
dataset_name = "cub"
out_dir = ""
device = 'cuda'
seed = 1
num_workers = 4
concept_batch_size = 100
C = [0.01,0.1]
n_samples_concept = 50
    # parser.add_argument("--dataset-name", default="cub", type=str)
    # parser.add_argument("--out-dir", required=True, type=str)
    # parser.add_argument("--device", default="cuda", type=str)
    # parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # parser.add_argument("--num-workers", default=4, type=int, help="Number of workers in the data loader.")
    # parser.add_argument("--batch-size", default=100, type=int, help="Batch size in the concept loader.")
    # parser.add_argument("--C", nargs="+", default=[0.01, 0.1], type=float, help="Regularization parameter for SVMs.")
    # parser.add_argument("--n-samples", default=50, type=int, 
    #                     help="Number of positive/negative samples used to learn concepts.")