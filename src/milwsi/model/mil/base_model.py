import numpy as np
from collections import defaultdict
import torch
import torchmetrics
import lightning.pytorch as pl
from milwsi.utils.registry import MODEL_REGISTRY
from milwsi.model.optimizer import build_optimizer
from milwsi.utils import logger


def build_model(name, **kwargs):
    return MODEL_REGISTRY.get(name)(**kwargs)


@MODEL_REGISTRY.register()
class BaseModule(pl.LightningModule):
    def __init__(self, model, optimizer=None, hparams=None):
        super(BaseModule, self).__init__()

        self.save_hyperparameters(hparams)

        # create Model
        if type(model) == dict:
            self.model = build_model(**model)
        else:
            self.model = model

        # create Loss
        self.loss = torch.nn.CrossEntropyLoss()

        # create Optimizer
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer is None:
            self.optimizer = torch.optim.Adam(params=trainable_parameters, lr=1e-4, weight_decay=1e-4)
        elif type(optimizer) == dict:
            self.optimizer = build_optimizer(params=trainable_parameters, **optimizer)
        else:
            self.optimizer = optimizer

        # create results dictionary {'train': [], 'val': [], 'test': []}
        self.results_dict = defaultdict(list)

        # create Metrics
        n_classes = model['n_classes']
        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average='micro')
        kappa = torchmetrics.CohenKappa(task='multiclass', num_classes=n_classes)
        f1_score = torchmetrics.F1Score(task='multiclass', num_classes=n_classes, average='macro')
        recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=n_classes)
        precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=n_classes)
        specificity = torchmetrics.Specificity(task='multiclass', average='macro', num_classes=n_classes)
        self.metrics = torchmetrics.MetricCollection([accuracy, kappa, f1_score, recall, precision, specificity])
        self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes=model['n_classes'], average='macro')

    def configure_optimizers(self):
        return self.optimizer

    def loop_step(self, batch, stage):
        # prepare data and label
        data = batch['feature']
        label = torch.tensor([batch['label']]).to(data.device)

        # forward model
        results_dict = self.model(data)

        # calculate loss
        logits = results_dict['logits']
        loss = self.loss(logits, label)

        # save results in dictionary
        results_dict['loss'] = loss
        results_dict['label'] = label
        results_dict = dict((key, value.detach()) for key, value in results_dict.items())
        self.results_dict[stage].append(results_dict)

        return loss

    def training_step(self, batch, batch_idx):
        return self.loop_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.loop_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.loop_step(batch, 'test')

    def on_train_epoch_end(self):
        self.log_results_dict('train')
        self.log_results_dict('val')
        self.results_dict.clear()  # clear results on each epoch end

    def on_test_epoch_end(self):
        self.log_results_dict('test')
        self.results_dict.clear()

    def log_results_dict(self, stage):
        # concat result of each step
        probs = torch.cat([x['Y_prob'] for x in self.results_dict[stage]], dim=0)
        target = torch.stack([x['label'] for x in self.results_dict[stage]], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in self.results_dict[stage]], dim=0)

        # calculate metrics
        loss = np.mean([x['loss'].item() for x in self.results_dict[stage]])
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.metrics(max_probs.squeeze(), target.squeeze())

        # save in log dictionary
        log_dict = dict(zip([key.replace('Multiclass', stage + '_') for key in metrics.keys()], metrics.values()))
        log_dict[f'{stage}_loss'] = loss
        log_dict[f'{stage}_auc'] = auc
        self.log_dict(log_dict)
