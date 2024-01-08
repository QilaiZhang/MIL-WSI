import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

import lightning.pytorch as pl


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A: torch.Tensor, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A: torch.Tensor, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        results_dict = {}
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for _ in
                           range(n_classes)]  # use an independent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        results_dict = {}
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM(pl.LightningModule):
    def __init__(self, n_classes, model_type='clam_sb', size_arg='small', dropout=False, subtyping=False, k_sample=8,
                 bag_loss='ce', inst_loss='ce', opt='adam', bag_weight=0.7, lr=1e-4, reg=1e-5):
        super().__init__()
        self.save_hyperparameters()

        if bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            self.loss_fn = SmoothTop1SVM(n_classes=n_classes)
        elif inst_loss == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
        elif inst_loss == 'ce':
            instance_loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if model_type == 'clam_sb':
            self.model = CLAM_SB(n_classes=n_classes, dropout=dropout, size_arg=size_arg,
                                 subtyping=subtyping, k_sample=k_sample, instance_loss_fn=instance_loss_fn)
        elif model_type == 'clam_mb':
            self.model = CLAM_MB(n_classes=n_classes, dropout=dropout, size_arg=size_arg,
                                 subtyping=subtyping, k_sample=k_sample, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

        if opt == 'adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=reg)
        elif opt == 'sgd':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9,
                                 weight_decay=reg)
        else:
            raise NotImplementedError

        self.bag_weight = bag_weight
        self.data = defaultdict(lambda: [defaultdict(int) for _ in range(self.n_classes)])
        self.results_dict = defaultdict(list)

    def configure_optimizers(self):
        return self.optimizer

    def loop_step(self, batch, stage):
        data, label = batch
        logits, Y_prob, Y_hat, _, instance_dict = self.model(data.squeeze(0), label=label.squeeze(0),
                                                             instance_eval=True)
        loss = self.loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.bag_weight * loss + (1 - self.bag_weight) * instance_loss

        Y = int(label)
        self.data[stage][Y]["count"] += 1
        self.data[stage][Y]["correct"] += (Y_hat.item() == Y)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.loop_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.loop_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.loop_step(batch, 'test')

    def on_train_epoch_end(self):
        # print('Train Set, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
        # for i in range(n_classes):
        #     acc, correct, count = acc_logger.get_summary(i)
        #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        #     if writer:
        #         writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
        #
        # if writer:
        #     writer.add_scalar('train/loss', train_loss, epoch)
        #     writer.add_scalar('train/error', train_error, epoch)
        pass
        # print('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
        # if inst_count > 0:
        #     val_inst_loss /= inst_count
        #     for i in range(2):
        #         acc, correct, count = inst_logger.get_summary(i)
        #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
        # if writer:
        #     writer.add_scalar('val/loss', val_loss, epoch)
        #     writer.add_scalar('val/auc', auc, epoch)
        #     writer.add_scalar('val/error', val_error, epoch)
        #     writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

        # for i in range(n_classes):
        #     acc, correct, count = acc_logger.get_summary(i)
        #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        #
        #     if writer and acc is not None:
        #         writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
