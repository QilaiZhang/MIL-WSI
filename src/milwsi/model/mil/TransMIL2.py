import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl

from ..optimizer import RAdam
from ..optimizer import Lookahead
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, (7, 7), padding=7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, (5, 5), padding=5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, (3, 3), padding=3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMILModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):
        h = kwargs['data'].float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


class TransMIL(pl.LightningModule):
    # ---->init
    def __init__(self, n_classes=2, lr=0.0002, weight_decay=0.00001):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransMILModel(n_classes=n_classes)
        self.optimizer = Lookahead(RAdam(self.model.parameters(), lr=lr, weight_decay=weight_decay))
        self.loss = nn.CrossEntropyLoss()
        self.n_classes = n_classes

        # ---->acc
        self.data = defaultdict(lambda: [defaultdict(int) for _ in range(self.n_classes)])
        self.results_dict = defaultdict(list)

        # ---->Metrics
        self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes=self.n_classes, average='macro')
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, average='micro')
        kappa = torchmetrics.CohenKappa(task='multiclass', num_classes=self.n_classes)
        f1_score = torchmetrics.F1Score(task='multiclass', num_classes=self.n_classes, average='macro')
        recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=self.n_classes)
        precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=self.n_classes)
        specificity = torchmetrics.Specificity(task='multiclass', average='macro', num_classes=self.n_classes)
        self.metrics = torchmetrics.MetricCollection([accuracy, kappa, f1_score, recall, precision, specificity])

    def configure_optimizers(self):
        return self.optimizer

    def loop_step(self, batch, stage):
        # ---->inference
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        Y = int(label)

        # ---->loss
        loss = self.loss(logits, label)

        self.data[stage][Y]["count"] += 1
        self.data[stage][Y]["correct"] += (Y_hat.item() == Y)

        results_dict['loss'] = loss
        results_dict['label'] = label
        self.results_dict[stage].append(results_dict)

        return loss

    def training_step(self, batch, batch_idx):
        return self.loop_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.loop_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.loop_step(batch, 'test')

    def calculate_metrics(self, stage):
        probs = torch.cat([x['Y_prob'] for x in self.results_dict[stage]], dim=0)
        target = torch.stack([x['label'] for x in self.results_dict[stage]], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in self.results_dict[stage]], dim=0)

        loss = np.mean([x['loss'].item() for x in self.results_dict[stage]])
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.metrics(max_probs.squeeze(), target.squeeze())
        metrics = dict(zip([key.replace('Multiclass', stage + '_') for key in metrics.keys()], metrics.values()))
        return loss, auc, metrics

    def print_accuracy(self, stage):
        for c in range(self.n_classes):
            count = self.data[stage][c]["count"]
            correct = self.data[stage][c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {:.4f}, correct {}/{}'.format(c, acc, correct, count))

    def on_train_epoch_end(self):
        train_loss, train_auc, train_metrics = self.calculate_metrics('train')
        val_loss, val_auc, val_metrics = self.calculate_metrics('val')

        print('Train Set, train_loss: {:.4f}, train_error: {:.4f}, train_auc: {:.4f}'.format(
            train_loss, 1 - train_metrics['train_Accuracy'].item(), train_auc))
        self.print_accuracy('train')
        print('Val Set, val_loss: {:.4f}, val_error: {:.4f}, val_auc: {:.4f}'.format(
            val_loss, 1 - val_metrics['val_Accuracy'].item(), val_auc))
        self.print_accuracy('val')

        self.log('train_loss', train_loss, on_epoch=True, logger=True)
        self.log('train_auc', train_auc, on_epoch=True, logger=True)
        self.log_dict(train_metrics, on_epoch=True, logger=True)
        self.log('val_loss', val_loss, on_epoch=True, logger=True)
        self.log('val_auc', val_auc, on_epoch=True, logger=True)
        self.log_dict(val_metrics, on_epoch=True, logger=True)

        self.data.clear()
        self.results_dict.clear()

    def on_test_epoch_end(self):
        _, test_auc, test_metrics = self.calculate_metrics('test')
        print("Test Results:")
        self.print_accuracy('test')
        self.log('test_auc', test_auc, on_epoch=True, logger=True)
        self.log_dict(test_metrics, on_epoch=True, logger=True)
