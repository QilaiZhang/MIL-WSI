import os
import glob
from PIL import Image
import pandas as pd
from typing import Union
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms
import lightning.pytorch as pl

from milwsi.utils.registry import DATASET_REGISTRY
from milwsi.utils import k_fold_stratified_splits


def build_dataset(name, **kwargs):
    return DATASET_REGISTRY.get(name)(**kwargs)


def default_transform():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


@DATASET_REGISTRY.register()
class WSIDataset(data.Dataset):
    """
    A dataset of WSIs. Inherit this class and define the following variables to custom data processing:

        self.wsi_ids (necessary): A list of WSI id.
        self.wsi_paths (optional): A list of WSI folder paths containing patches.
        self.features (optional): A list of WSI feature file paths.
        self.labels (optional): A list of WSI labels.
    """

    # TODO: Define default WSIDataset
    def __init__(self):
        self.wsi_ids = None
        self.wsi_paths = None
        self.features = None
        self.labels = None
        self.patient_ids = None  # used for k-fold cross validation

    def __len__(self):
        return len(self.wsi_ids)

    def __getitem__(self, index):
        output_dict = {'wsi_id': self.wsi_ids[index]}

        if self.wsi_paths:
            output_dict['wsi_path'] = self.wsi_paths[index]

        if self.features:
            feature = torch.load(os.path.join(self.features[index]), map_location=torch.device('cpu'))
            output_dict['feature'] = feature

        if self.labels:
            output_dict['label'] = self.labels[index]

        return output_dict


@DATASET_REGISTRY.register()
class WSIBag(data.Dataset):
    """
    A Bag of Whole Slide Image consist of instances.

    Parameter:
        paths (Union[`str`,`list`]):
            A path of patches or a list of paths for each patch
        ext (`str`, default to "jpg"):
            File extension of patches when paths is `str`.
        transform (torch.Transform, default to None):
            Custom transform of images
    """

    def __init__(self, paths: Union[str, list], ext="jpg", transform=None):
        self.paths = paths if type(paths) == list else glob.glob(os.path.join(paths, '*' + ext))
        self.transform = transform if transform is not None else default_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)


@DATASET_REGISTRY.register()
class CrossValidationModule(pl.LightningDataModule):
    """
    Data Module for cross validation.
    """
    def __init__(
        self,
        dataset,
        split_path,
        n_splits,
        patient_level=False,
        val_size=0.1,
        test_size=0.1,
        seed=None,
        monte_carlo=False,
    ):
        super().__init__()

        if type(dataset) == dict:
            self.dataset = build_dataset(**dataset)
        else:
            self.dataset = dataset

        self.fold = None
        self.split_path = split_path

        os.makedirs(self.split_path, exist_ok=True)
        is_exist = True
        for i in range(n_splits):
            is_exist = is_exist and os.path.exists(os.path.join(self.split_path, f'split_{i}.csv'))

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.train_dataset = self.val_dataset = self.test_dataset = None

        # create splits csv if not exists
        if not is_exist:
            self.create_splits(patient_level, n_splits, val_size, test_size, seed, monte_carlo)
        else:
            subset_ids = dict(zip(self.dataset.wsi_ids, range(len(self.dataset))))
            for i in range(n_splits):
                csv_path = os.path.join(self.split_path, f'split_{i}.csv')
                slide_data = pd.read_csv(csv_path)
                train_indices = [subset_ids[wsi_id] for wsi_id in slide_data['train_ids'].dropna()]
                val_indices = [subset_ids[wsi_id] for wsi_id in slide_data['val_ids'].dropna()]
                test_indices = [subset_ids[wsi_id] for wsi_id in slide_data['test_ids'].dropna()]
                self.train_indices.append(train_indices)
                self.val_indices.append(val_indices)
                self.test_indices.append(test_indices)

    def set_fold(self, fold):
        self.fold = fold

    def create_splits(self, patient_level, n_splits, val_size, test_size, seed, monte_carlo):

        index_dict = defaultdict(list)

        if patient_level:
            # remove duplicate patient ids
            patient_dict = dict(sorted(zip(self.dataset.patient_ids, self.dataset.labels)))
            all_ids = list(patient_dict.keys())
            labels = [str(label) for label in patient_dict.values()]

            for i in range(len(self.dataset)):
                index_dict[all_ids.index(self.dataset.patient_ids[i])].append(i)

        else:
            all_ids = self.dataset.wsi_ids
            labels = [str(label) for label in self.dataset.labels]

        splits_generator = k_fold_stratified_splits(n_splits, all_ids, labels, val_size, test_size, seed, monte_carlo)

        for i in range(n_splits):
            csv_path = os.path.join(self.split_path, f'split_{i}.csv')

            if patient_level:
                patient_indices = next(splits_generator)
                all_stage_indices = [[], [], []]
                for stage, stage_indices in enumerate(patient_indices):
                    for idx in stage_indices:
                        all_stage_indices[stage] += index_dict[idx]
                train_indices, val_indices, test_indices = all_stage_indices
            else:
                train_indices, val_indices, test_indices = next(splits_generator)

            self.train_indices.append(train_indices)
            self.val_indices.append(val_indices)
            self.test_indices.append(test_indices)

            splits = {
                'train_ids': [self.dataset.wsi_ids[idx] for idx in train_indices],
                'train_labels': [self.dataset.labels[idx] for idx in train_indices],
                'val_ids': [self.dataset.wsi_ids[idx] for idx in val_indices],
                'val_labels': [self.dataset.labels[idx] for idx in val_indices],
                'test_ids': [self.dataset.wsi_ids[idx] for idx in test_indices],
                'test_labels': [self.dataset.labels[idx] for idx in test_indices]
            }

            df = pd.DataFrame(pd.DataFrame.from_dict(splits, orient='index').values.T, columns=list(splits.keys()))
            df.to_csv(csv_path, index=False)

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = data.Subset(self.dataset, self.train_indices[self.fold])
            self.val_dataset = data.Subset(self.dataset, self.val_indices[self.fold])
        if stage == 'test':
            self.test_dataset = data.Subset(self.dataset, self.test_indices[self.fold])

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])
