import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        K: output dimension (number of class)
        dropout: whether to use dropout (p = 0.25)
    """

    def __init__(self, L=512, D=256, K=1, dropout=False):
        super(Attention, self).__init__()

        module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            module.append(nn.Dropout(0.25))
        module.append(nn.Linear(D, K))

        self.attention = nn.Sequential(*module)

    def forward(self, x, isNorm=False):  # x: N x L
        A = self.attention(x)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class Attention_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        K: output dimension (number of classes)
        dropout: whether to use dropout (p = 0.25)
    """

    def __init__(self, L=512, D=256, K=1, dropout=False):
        super(Attention_Gated, self).__init__()

        attention_V = [nn.Linear(L, D), nn.Tanh()]
        attention_U = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            attention_V.append(nn.Dropout(0.25))
            attention_U.append(nn.Dropout(0.25))

        self.attention_V = nn.Sequential(*attention_V)
        self.attention_U = nn.Sequential(*attention_U)
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x, isNorm=False):   # x: N x L
        A_V = self.attention_V(x)   # N x D
        A_U = self.attention_U(x)   # N x D
        A = self.attention_weights(A_V * A_U)   # N x K

        A = torch.transpose(A, 1, 0)  # K x N
        if isNorm:
            A = F.softmax(A, dim=1)
        return A
