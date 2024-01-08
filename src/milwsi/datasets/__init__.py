import os
import importlib
from .base_dataset import build_dataset


data_folder = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [os.path.splitext(v)[0] for v in os.listdir(data_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'milwsi.datasets.{file_name}') for file_name in dataset_filenames]
