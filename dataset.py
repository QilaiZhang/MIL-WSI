import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning.pytorch as pl


def generate_splits(seed, n_splits, class_ids, val_frac, test_frac):
    np.random.seed(seed)

    for k in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        for c in range(len(class_ids)):
            test_num = round(len(class_ids[c]) * test_frac)
            test_ids = np.random.choice(class_ids[c], test_num, replace=False)
            remaining_ids = np.setdiff1d(class_ids[c], test_ids)

            val_num = round(len(class_ids[c]) * val_frac)
            val_ids = np.random.choice(remaining_ids, val_num, replace=False)
            remaining_ids = np.setdiff1d(remaining_ids, val_ids)

            all_test_ids.extend(test_ids)
            all_val_ids.extend(val_ids)
            sampled_train_ids.extend(remaining_ids)

        yield sampled_train_ids, all_val_ids, all_test_ids


class GenericDataset(Dataset):
    def __init__(self, feature_path, csv_path=None, slide_data=None):
        self.seed = 0
        self.feature_path = feature_path
        if csv_path:
            self.slide_data = pd.read_csv(csv_path)
            self.process_label()
        else:
            self.slide_data = slide_data

        self.labels = set(self.slide_data['label'])
        self.n_classes = len(self.labels)
        self.class_ids = [[] for _ in range(self.n_classes)]
        for i, label in enumerate(self.labels):
            self.class_ids[i] = np.where(self.slide_data['label'] == label)[0]
        self.splits = None

    def process_label(self):
        labels = list(set(self.slide_data['label']))
        labels.sort()
        self.slide_data['label'] = [labels.index(label) for label in self.slide_data['label']]
        print("label dictionary:", dict(zip(labels, range(len(labels)))))

    def create_splits(self, n_splits=5, val_frac=0.1, test_frac=0.1):
        self.splits = generate_splits(self.seed, n_splits, self.class_ids, val_frac, test_frac)

    def return_splits(self, csv_path=None):
        if csv_path:
            df = pd.read_csv(csv_path)
            train_data = self.slide_data[self.slide_data['slide'].isin(df['train'])]
            val_data = self.slide_data[self.slide_data['slide'].isin(df['val'])]
            test_data = self.slide_data[self.slide_data['slide'].isin(df['test'])]
        else:
            train_ids, val_ids, test_ids = next(self.splits)
            train_data = self.slide_data.iloc[train_ids]
            val_data = self.slide_data.iloc[val_ids]
            test_data = self.slide_data.iloc[test_ids]

        train_dataset = GenericDataset(self.feature_path, slide_data=train_data.reset_index(drop=True))
        val_dataset = GenericDataset(self.feature_path, slide_data=val_data.reset_index(drop=True))
        test_dataset = GenericDataset(self.feature_path, slide_data=test_data.reset_index(drop=True))

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        full_path = os.path.join(self.feature_path, self.slide_data['slide'][idx] + '.pt')
        feature = torch.load(full_path)
        label = self.slide_data['label'][idx]
        return feature, label


class BinaryClassification(GenericDataset):
    def __init__(self, feature_path, csv_path):
        super().__init__(feature_path, csv_path=csv_path)

    def process_label(self):
        self.slide_data['label'] = [0 if label == 'Benign' else 1 for label in self.slide_data['label']]


class ProstateDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, csv_dir: str, batch_size: int = 1, fold_idx: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.batch_size = batch_size
        self.train_datasets = None
        self.val_datasets = None
        self.test_datasets = None
        self.fold_idx = fold_idx

    def setup(self, stage: str):
        datasets = BinaryClassification(self.data_dir, self.csv_dir)
        self.train_datasets, self.val_datasets, self.test_datasets = datasets.return_splits(
            csv_path="/data_sdb/PRAD/csv/splits/splits_{}.csv".format(self.fold_idx)
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_datasets,
                          batch_size=self.batch_size,
                          num_workers=4,
                          sampler=RandomSampler(self.train_datasets))

    def val_dataloader(self):
        return DataLoader(dataset=self.val_datasets, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_datasets, batch_size=self.batch_size, num_workers=4)
