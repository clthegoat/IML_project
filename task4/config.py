import torch
from torchvision import transforms

# File
# dir_train_all = "/Users/zimengjiang/Downloads/task4_handout/train_triplets.txt" 
# dir_test = "/Users/zimengjiang/Downloads/task4_handout/test_triplets.txt"
# dir_images = "/Users/zimengjiang/Downloads/task4_handout/food/" 
# dir_train = "/Users/zimengjiang/Downloads/task4_handout/train_triplets.npy" 
# dir_val = "/Users/zimengjiang/Downloads/task4_handout/val_triplets.npy" 

dir_train_all = "/content/drive/My Drive/task4_handout/train_triplets.txt"
dir_test = "/content/drive/My Drive/task4_handout/test_triplets.txt"
dir_images = "/content/drive/My Drive/task4_handout/food/food"
dir_train = "/content/drive/My Drive/task4_handout/train_triplets.npy"
dir_val = "/content/drive/My Drive/task4_handout/val_triplets.npy" 



# Training set
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 16
lr = 1e-5
weight_decay = 0.001
num_epochs = 30
log_interval = 100
save_root = "/content/drive/My Drive/Colab Notebooks/task4_source/sgd_checkpoints_lr1e-5"
# save_root = "/content/drive/My Drive/Colab Notebooks/task4_source/checkpoints"
# save_root = "/Users/zimengjiang/code/iml_wxbdl/task4/checkpoints"


TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop((300,400), (0.7,1)),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((300,400)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for network
NUM_CLUSTERS = 16
NUM_ENCODER_DIM = 512

# resume training 
RESUME = False
resume_ckp_location = ' '

# triplet loss margin
MARGIN = 1

# prediction file
# pred_file = '/Users/zimengjiang/code/iml_wxbdl/task4/task4_source/results/pred_'
pred_file = '/content/drive/My Drive/Colab Notebooks/task4_source/results/sgd_pred_alex_soft_1e-5'