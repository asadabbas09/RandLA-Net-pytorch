import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import SpiralConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from utils_bayes_car import preprocess_spiral

import numpy as np

@variational_estimator
class Net(torch.nn.Module):
    def __init__(self, in_channels, num_classes, indices):
        super(Net, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)


        indices = 0#indices.long()

        self.prior_sigma_1 = 10.0
        self.prior_sigma_2 = 40.0
        # self.conv1 = SpiralConv(16, 32, indices)
        # self.conv2 = SpiralConv(32, 64, indices)
        # self.conv3 = SpiralConv(64, 128, indices)
        # self.conv4 = SpiralConv(128, 256, indices)
        # self.conv1 = SpiralConv(16, 32)
        # self.conv2 = SpiralConv(32, 64)
        # self.conv3 = SpiralConv(64, 128)
        # self.conv4 = SpiralConv(128, 256)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # self.node_classifier = nn.Linear(256, num_classes)
        # self.graph_classifier = nn.Linear(256, 1)

        # self.node_classifier = BayesianLinear(256, num_classes, prior_sigma_1=self.prior_sigma_1, prior_sigma_2=self.prior_sigma_2)
        #self.graph_classifier = BayesianLinear(256, 1, prior_sigma_1=self.prior_sigma_1, prior_sigma_2=self.prior_sigma_2)

        self.reset_parameters()

    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        # nn.init.xavier_uniform_(self.node_classifier.weight, gain=1)
        # nn.init.xavier_uniform_(self.graph_classifier.weight, gain=1)


        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        # nn.init.constant_(self.node_classifier.bias, 0)
        # nn.init.constant_(self.graph_classifier.bias, 0)

    def forward(self, data):
        #data, indices = data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = data.x.float()
        indices = data.indices

        x = F.elu(self.fc0(x)).to(device)

        conv1 = SpiralConv(16, 32, indices).to(device)
        x = F.elu(conv1(x)).to(device)

        conv2 = SpiralConv(32, 64, indices).to(device)
        x = F.elu(conv2(x)).to(device)

        conv3 = SpiralConv(64, 128, indices).to(device)
        x = F.elu(conv3(x)).to(device)

        #conv4 = SpiralConv(128, 256, indices).to(device)
        #x = F.elu(conv4(x)).to(device)

        # x = self.fc2(x).to(device)
        x = self.fc1(x).to(device)
        x = self.fc2(x).to(device)



        # out_node = self.node_classifier(x).to(device)
        #print("forward pass outnode : ", np.shape(out_node))
        return x#torch.tanh(x)
        # return torch.sigmoid(x)#out_node#, out_graph

        #x = self.fc2(x)
        #return x#F.softmax(x, dim=1)
