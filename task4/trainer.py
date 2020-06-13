import torch
import numpy as np
import os, shutil
from utils import save_checkpoint, predict, write_preds
from itertools import chain
from sklearn.metrics import accuracy_score
import config

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


def fit(train_loader, val_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, save_root, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for epoch in range(start_epoch, n_epochs):
        # scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        # if (epoch+1) % 10 == 0:
        val_acc = val_epoch(val_loader, model, cuda)
        message += '\nEpoch: {}/{}. Validation set: Acc: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                    val_acc)
        # write prediction for every epoch
        prediction = test_epoch(test_loader, model, cuda)
        write_preds(prediction, config.pred_file+str(epoch+1)+'.txt')
            # val_loss, metrics = val_epoch(val_loader, model, cuda)
            # val_loss /= len(val_loader)

            # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
            #                                                                         val_loss)
            # for metric in metrics:
            #     message += '\t{}: {}'.format(metric.name(), metric.value())
        
        if (epoch+1)%5 == 0:
            save_checkpoint(model.state_dict(), False, save_root, str(epoch))
        print(message)
    save_checkpoint(model.state_dict(), False, save_root, str(n_epochs))
    prediction = test_epoch(test_loader, model, cuda)
    write_preds(prediction, config.pred_file+str(epoch+1)+'.txt')
    



def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def val_epoch(val_loader, model, cuda):
    preds = []
    correct = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            pred = predict(*outputs)
            preds.append(pred)

    preds = np.array(list(chain.from_iterable(preds)))
    target = np.zeros(len(preds))+1
    return accuracy_score(target, preds)

def test_epoch(test_loader, model, cuda):
    preds = []
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            pred = predict(*outputs)
            preds.append(pred)

    preds = list(chain.from_iterable(preds))

    return preds

# def val_epoch(val_loader, model, loss_fn, cuda, metrics):
#     with torch.no_grad():
#         for metric in metrics:
#             metric.reset()
#         model.eval()
#         val_loss = 0
#         for batch_idx, (data, target) in enumerate(val_loader):
#             target = target if len(target) > 0 else None
#             if not type(data) in (tuple, list):
#                 data = (data,)
#             if cuda:
#                 data = tuple(d.to(device) for d in data)
#                 if target is not None:
#                     target = target.to(device)

#             outputs = model(*data)

#             if type(outputs) not in (tuple, list):
#                 outputs = (outputs,)
#             loss_inputs = outputs
#             if target is not None:
#                 target = (target,)
#                 loss_inputs += target

#             loss_outputs = loss_fn(*loss_inputs)
#             loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#             val_loss += loss.item()

#             for metric in metrics:
#                 metric(outputs, target, loss_outputs)

#     return val_loss, metrics