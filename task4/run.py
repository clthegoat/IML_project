import torch
import numpy as np 
import pandas as pd 
import config
from food_dataset import FoodDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from network.network import ImageRetrievalModel, TripletNet, EmbeddingNet, vgg_back
from utils import TripletLoss, SoftmaxTripletLoss
from metrics import AverageNonzeroTripletsMetric
from trainer import fit 

np.random.seed(1)

def main():
    print()
    print('***************By Manyeo***************')

    # load data
    train_triplets = np.load(config.dir_train)
    val_triplets = np.load(config.dir_val)
    test_all = np.loadtxt(config.dir_test)
    print("train size:{}".format(train_triplets.shape[0]))
    print("val size:{}".format(val_triplets.shape[0]))
    print("test size:{}".format(test_all.shape[0]))

    # detect device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print('device: ' + str(device) + '\n')

    # init dataset
    train_set = FoodDataset(config.dir_images, train_triplets, train=True, val=False, test=False)
    val_set = FoodDataset(config.dir_images, val_triplets, train=False, val=True, test=False)
    test_set = FoodDataset(config.dir_images, test_all, train=False, val=False, test=True)

    # load data in dataloader
    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers = 16)
    val_loader = DataLoader(val_set, batch_size = 32, shuffle = False, num_workers = 16)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers = 16)
    print("Data Loaded.")

    # set up the model 
    # embedding_net = ImageRetrievalModel(config.NUM_CLUSTERS, config.NUM_ENCODER_DIM)
    # embedding_net = EmbeddingNet()
    embedding_net = vgg_back()
    model = TripletNet(embedding_net)
    model = model.to(device)
    if (config.RESUME):
        model.load_state_dict(torch.load(config.resume_ckp_location, map_location=torch.device(device)))
    # set up loss
    # loss_fn = TripletLoss(config.MARGIN)
    print("lr:{}".format(config.lr))
    loss_fn = SoftmaxTripletLoss()
    my_params = list(model.parameters())
    print("using adamW")
    optimizer = optim.AdamW(params=my_params,
                        lr=config.lr,
                        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    n_epochs = config.num_epochs
    log_interval = config.log_interval
    save_root = config.save_root
    # fit the model
    fit(train_loader, val_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, save_root, metrics = [AverageNonzeroTripletsMetric()])



if __name__ == '__main__':
    main()
