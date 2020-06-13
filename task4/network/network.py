"""Assemble image retrieval network.
"""
from collections import OrderedDict
from network.netvlad import NetVLAD
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models

class ImageRetrievalModel(nn.Module):
    """Build the image retrieval model with intermediate feature extraction.

    The model is made of a VGG-16 backbone combined with a NetVLAD pooling
    layer.
    """
    def __init__(self, num_clusters: int, encoder_dim: int):
        """Initialize the Image Retrieval Network.

        Args:
            num_clusters: Number of NetVLAD clusters (should match pre-trained)
                weights.
            encoder_dim: NetVLAD encoder dimension.
        """
        super(ImageRetrievalModel, self).__init__()
        self._num_clusters = num_clusters
        self._encoder_dim = encoder_dim
        # self._model = self._build_model()

        # def _build_model(self):
        """ Build image retrieval network and load pre-trained weights.
        """
        # model = nn.Module()

        # Assume a VGG-16 backbone
        encoder = models.vgg16(pretrained=True)
        # layers_all = list(encoder.features.children())
        layers = list(encoder.features.children())[:-2]

        # Assume a NetVLAD pooling layer
        net_vlad = NetVLAD(
            num_clusters=self._num_clusters, dim=self._encoder_dim)
        layers.append(net_vlad)

        encoder = nn.Sequential(*layers)
        self._model = encoder
        # model.add_module('encoder', encoder)

        # # Assume a NetVLAD pooling layer
        # net_vlad = NetVLAD(
        #     num_clusters=self._num_clusters, dim=self._encoder_dim)
        # model.add_module('pool', net_vlad)


    def forward(self, x):
        return self._model(x)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 2), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 2), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 2),nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(128, 256, 2), nn.ReLU(),
                                     nn.MaxPool2d(2, stride=2)
                                    )


        self.fc = nn.Sequential(     nn.Linear(256*8*11,512), nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(512, 256),
                                     nn.Dropout(0.5)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class vgg_back(nn.Module):
    def __init__ (self):
        super(vgg_back, self).__init__()
        # Assume a VGG-16 backbone
        encoder = models.alexnet(pretrained=True)
        # layers_all = list(encoder.features.children())
        layers = list(encoder.features.children())  
        
        encoder = nn.Sequential(*layers)
        self.convnet = encoder
        self.fc = nn.Sequential(     nn.Linear(22528,1024), nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(1024, 512),
                                     nn.Dropout(0.5)
                                    #  nn.Softmax(512) # added 
                                )
        # self.fc = 
    def forward(self,x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output



class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, input1, input2, input3):
        output1 = self.embedding_net(input1)
        output2 = self.embedding_net(input2)
        output3 = self.embedding_net(input3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)