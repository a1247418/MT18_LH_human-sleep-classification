import os
import sys
import numpy as np
from sleeplearning.lib.models.single_chan_expert import SingleChanExpert
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import torch
from torch import nn
import sleeplearning.lib.base


class Ensembler(nn.Module):
    def __init__(self, ms: dict):
        super(Ensembler, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = []

        assert len(ms['expert_models']) > 0
        for exp in ms['expert_models']:
            clf = sleeplearning.lib.base.Base()
            clf.restore(exp)
            self.expert_channels = clf.ds['channels']
            for param in clf.model.parameters():
                param.requires_grad = False

            self.experts.append(clf.model)
        self.experts = nn.ModuleList(self.experts)

    def train(self, mode=True):
        super(Ensembler, self).train(mode=mode)
        self.experts.eval()
        #raise NotImplementedError("The Ensembler is for aggregation of predictions only.")

    def forward(self, x):
        # Majority vote over all experts

        # logits: bs x seq_len x lab
        # print(self.experts[0](x)["logits"].shape)

        result = {'logits': torch.zeros_like(self.experts[0](x)['logits']).float().to(self.device)}
        logits_shape = result['logits'].shape

        for exp in self.experts:
            pred = torch.argmax(exp(x)['logits'], dim=-1)
            for i in range(logits_shape[0]):
                for j in range(logits_shape[1]):
                    result['logits'][i, j, pred[i, j]] += 1 + np.random.rand() * 0.001
        return result