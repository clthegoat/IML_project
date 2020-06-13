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


def main():
    print()
    print('***************By Manyeo***************')
    # load data
    input1 = torch.rand(5,3,300,400)
    input2 = torch.rand(5,3,300,400)
    input3 = torch.rand(5,3,300,400)
    input_all = (input1, input2, input3)
    target = (1,)

    # detect device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print('device: ' + str(device) + '\n')
    # set up the model 
    # embedding_net = ImageRetrievalModel(config.NUM_CLUSTERS, config.NUM_ENCODE R_DIM)
    embedding_net = vgg_back()
    print(embedding_net(input1).shape)
    model = TripletNet(embedding_net)
    model = model.to(device)
    print(model(*input_all))
    # # if (config.RESUME):
    # #     model.load_state_dict(torch.load(config.resume_ckp_location, map_location=torch.device(device)))
    # # # set up loss
    # loss_fn = SoftmaxTripletLoss()
    # # my_params = list(model.parameters())
    # optimizer = optim.AdamW(model.parameters(),
    #                     lr=config.lr,
    #                     weight_decay=config.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    # n_epochs = config.num_epochs
    # # log_interval = config.log_interval
    # # save_root = config.save_root
    # # # fit the model
    # # fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics = [AverageNonzeroTripletsMetric])
    # model.train()
    # losses = []
    # total_loss = 0
    # metrics = [AverageNonzeroTripletsMetric()]
    # for metric in metrics:
    #     metric.reset()
    # for epoch in range(n_epochs):
    #     optimizer.zero_grad()
    #     outputs = model(*input_all)
    #     loss_inputs = outputs

    #     if target is not None:
    #         target = (target,)
    #         loss_inputs += target

    #     loss_outputs = loss_fn(*loss_inputs)
    #     loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
    #     losses.append(loss.item())
    #     total_loss += loss.item()
    #     loss.backward()
    #     optimizer.step()
    #     for metric in metrics:
    #         metric(outputs, target, loss_outputs)
    #         message = 'Epoch:{} \tLoss: {:.6f}'.format(epoch, np.mean(losses))
    #     for metric in metrics:
    #         message += '\t{}: {}'.format(metric.name(), metric.value())
    #     print(message)

        

        

if __name__ == '__main__':
    main()
