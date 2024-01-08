import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from milwsi.model.arch.attention import Attention, Attention_Gated
from milwsi.thirdparty.topk.svm import SmoothTop1SVM
from milwsi.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CLAM(nn.Module):
    def __init__(self, n_classes=2, subtyping=False, k_sample=8, size=None, dropout=False, gate=True,
                 instance_loss_fn=None, single_branch=True):
        super(CLAM, self).__init__()

        self.n_classes = n_classes
        self.subtyping = subtyping
        self.k_sample = k_sample

        if size is None:
            size = [1024, 512, 256]

        encoder = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            encoder.append(nn.Dropout(0.25))
        self.encoder = nn.Sequential(*encoder)

        if single_branch:
            branch_num = 1
            self.classifier = nn.Linear(size[1], n_classes)
        else:
            branch_num = self.n_classes
            # use an independent linear layer to predict each class
            bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
            self.classifiers = nn.ModuleList(bag_classifiers)

        if gate:
            self.attention = Attention_Gated(L=size[1], D=size[2], K=branch_num, dropout=dropout)
        else:
            self.attention = Attention(L=size[1], D=size[2], K=branch_num, dropout=dropout)

        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        if instance_loss_fn is None:
            self.instance_loss_fn = nn.CrossEntropyLoss
        else:
            self.instance_loss_fn = instance_loss_fn

    # instance-level evaluation for attention branch
    def inst_eval(self, A, h, classifier, in_class=True):
        top_p_ids = torch.topk(A, self.k_sample).indices
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = torch.full((self.k_sample,), 1, device=h.device).long()

        if in_class:
            top_n_ids = torch.topk(-A, self.k_sample).indices
            top_n = torch.index_select(h, dim=0, index=top_n_ids)
            n_targets = torch.full((self.k_sample,), 0, device=h.device).long()
            all_targets = torch.cat([p_targets, n_targets], dim=0)
            all_instances = torch.cat([top_p, top_n], dim=0)
        else:
            all_targets = p_targets
            all_instances = top_p

        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def forward(self, x, instance_eval=False, label=None):
        h = self.encoder(x)  # N x L
        A = self.attention(h, isNorm=True)  # K x N

        results_dict = {}
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            print(inst_labels)
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[0], h, classifier, in_class=True)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[0], h, classifier, in_class=False)
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

        if M.shape[0] == 1:
            logits = self.classifier(M)
        else:
            # multiple attention branch
            logits = torch.empty(1, self.n_classes).float().to(M.device)
            for c in range(self.n_classes):
                logits[0, c] = self.classifiers[c](M[c])

        results_dict['logits'] = logits
        results_dict['Y_hat'] = torch.argmax(logits, dim=1)
        results_dict['Y_prob'] = F.softmax(logits, dim=1)

        return results_dict
