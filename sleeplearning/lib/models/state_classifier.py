import copy
import os
import sys

import gridfs
from torch.nn import ModuleList

from sleeplearning.lib.models.single_chan_expert import SingleChanExpert
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from pymongo import MongoClient
from torch import nn
from torch.nn.init import xavier_normal_ as xavier_normal
from cfg.config import mongo_url
import torch
import torch.nn.functional as F
import sleeplearning.lib.base

class StateClassifier(nn.Module):
    def __init__(self, ms: dict):
        super(StateClassifier, self).__init__()
        self.dropout = ms['dropout']
        num_classes = ms['nclasses']

        clf = sleeplearning.lib.base.Base()
        clf.restore(ms['expert_model'])
        self.expert_channels = clf.ds['channels']
        for param in clf.model.parameters():
            param.requires_grad = False
        expert = clf

        input_dim_classifier = clf.model.ms["hidden_size"]

        get returnes state prediction from eypert

        self.classifier = nn.Sequential(
             nn.Dropout(p=self.dropout),
             nn.Linear(input_dim_classifier, input_dim_classifier // 2),
             nn.ReLU(),
             nn.Dropout(p=self.dropout),
             nn.Linear(input_dim_classifier // 2, num_classes),
        )

    def train(self, mode=True):
        super(StateClassifier, self).train(mode=mode)

    def forward(self, x):
        use experts
        logits = self.classifier(x)

        result = {'logits': logits}
        return result
