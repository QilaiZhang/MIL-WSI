import os
import yaml
import torch
import torch.distributed as dist
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


def init_dist(backend='nccl'):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def read_yaml(opt_path):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, yaml.Loader)
    return opt


def k_fold_stratified_splits(n_splits, all_ids, labels, val_size=0.1, test_size=0.1, seed=None, monte_carlo=False):
    """
    k-fold cross-validation
    """

    def solve_label(label):
        # To solve error when the number of item for one of labels is less than 2
        # Nothing change when there are at least 2 items for each label
        count_dict = Counter(label)
        max_count_label = list(count_dict.keys())[0]
        return [max_count_label if count_dict[label] < 2 else label for label in label]

    labels = solve_label(labels)

    if monte_carlo:
        k_fold = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    else:
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        test_size = 1 / n_splits

    for train_val_indices, test_indices in k_fold.split(all_ids, labels):
        train_val_labels = [labels[idx] for idx in train_val_indices]
        train_val_labels = solve_label(train_val_labels)

        # split train dataset and val dataset
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size/(1-test_size),
            random_state=seed,
            shuffle=True,
            stratify=train_val_labels
        )

        yield list(train_indices), list(val_indices), list(test_indices)
