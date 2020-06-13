import torch
import shutil, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def save_checkpoint(state,
                    is_best,
                    path,
                    prefix,
                    filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def predict(anchor, place1, place2):
    pred = np.zeros(anchor.shape[0])
    distance_ap1 = (anchor - place1).pow(2).sum(1)  # .pow(.5)
    distance_ap2 = (anchor - place2).pow(2).sum(1)  # .pow(.5)
    mask = distance_ap1 < distance_ap2
    if mask.shape[0] == 1 and mask[0]:
        pred[0] = 1
    else:
        pred[mask.cpu()] = 1
    return pred

def safe_norm(x):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    mask = x_norm == 0.0
    x_norm = x_norm + mask * 1e-16
    return x_norm
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class SoftmaxTripletLoss(nn.Module):
    def __init__(self):
        super(SoftmaxTripletLoss, self).__init__()
        self.eps = 1e-16

    def forward(self, anchor, pos, neg, size_average = True):
        dist1 = safe_norm(anchor - pos)
        dist2 = safe_norm(anchor - neg)
        # for normalizatoin 
        dist_stack = torch.stack((dist1, dist2), dim=1)
        dist_norm = safe_norm(dist_stack)

        dist1 = dist1/dist_norm
        dist2 = dist2/dist_norm

        dist1_z = torch.exp(dist1)/(torch.exp(dist1)+torch.exp(dist2)+self.eps)
        loss = dist1_z**2
        return loss.mean() if size_average else loss.sum()

def write_preds(x, filepath):
    # with open(filepath, 'w') as f:
    #     for i in range(len(x)):
    #         f.writelines(str(int(x[i]))+"\n")
    # print("Prediction written.")
    # f.close()
    np.savetxt(filepath, np.array(x), fmt = '%d')
    print("Prediction written.")
