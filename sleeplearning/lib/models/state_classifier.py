import os
import sys

from sleeplearning.lib.models.single_chan_expert import SingleChanExpert
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
from torch import nn
import sleeplearning.lib.base


class StateClassifier(nn.Module):
    def __init__(self, ms: dict):
        super(StateClassifier, self).__init__()
        num_classes = ms['nclasses']

        clf = sleeplearning.lib.base.Base()
        clf.restore(ms['expert_models'][0])
        self.expert_channels = clf.ds['channels']
        for param in clf.model.parameters():
            param.requires_grad = False

        self.expert = clf.model

        input_dim_classifier = clf.model.hidden_size

        self.classifier = nn.Sequential(
             nn.Linear(input_dim_classifier, input_dim_classifier // 2),
             nn.ReLU(),
             nn.Linear(input_dim_classifier // 2, input_dim_classifier // 2),
             nn.ReLU(),
             nn.Linear(input_dim_classifier // 2, num_classes),
        )

    def train(self, mode=True):
        super(StateClassifier, self).train(mode=mode)

    def forward(self, x):
        states = self.expert(x)["states"]  # bs x n_slices x hidden_state
        logits = self.classifier(states)

        result = {'logits': logits}
        return result
