import torch
import torch.nn as nn
import torch.nn.functional as F
from milwsi.model.mil.base_model import BaseModule
from milwsi.model.arch import Attention, Attention_Gated
from milwsi.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ABMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning Model
    args:
        n_classes: number of class
        gate: whether to use gated attention network
        size: config for network size
    """
    def __init__(self, n_classes=2, gate=True, size=None):
        super(ABMIL, self).__init__()

        if size is None:
            size = [1024, 512, 256]

        self.encoder = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())

        if gate:
            self.attention = Attention(L=size[1], D=size[2], K=1)
        else:
            self.attention = Attention_Gated(L=size[1], D=size[2], K=1)

        self.classifier = nn.Linear(size[1], n_classes)

    def forward(self, x):
        h = self.encoder(x)  # N x L
        A = self.attention(h, isNorm=True)  # K x N
        M = torch.mm(A, h)

        logits = self.classifier(M)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        return results_dict


@MODEL_REGISTRY.register()
class ABMILModule(BaseModule):
    def __init__(self, model, optimizer=None, hparams=None):
        super(ABMILModule, self).__init__(model, optimizer, hparams)
