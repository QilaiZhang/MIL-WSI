import os
import glob
import pandas as pd
from collections import defaultdict
import torch.utils.data as data
from milwsi.datasets.base_dataset import WSIDataset
from milwsi.utils.registry import DATASET_REGISTRY


def gleason2binary(labels, tumor=None):
    return [1 if pri in tumor or sec in tumor else 0 for pri, sec in labels]


@DATASET_REGISTRY.register()
class SICAPv2Dataset(WSIDataset):
    """
    Dataset of SICAPv2 (Prostate Whole Slide Images with Gleason Grades Annotations)
    https://data.mendeley.com/datasets/9xxm58dvs3/2

    Parameter:
        path(`str`): The root path of 'SICAPv2' and it contains folder 'images', 'masks' and 'partition'.
        feature_path(`str`, default to None): The path of feature files.
        tumor_type(``, default to [3, 4, 5]): To define gleason tumor type as label 1 and others as label 0.
    """
    def __init__(self, path, feature_path=None, tumor_type=None):
        data.Dataset.__init__(self)

        images = glob.glob(os.path.join(path, 'images', '*.jpg'))
        slide_data = pd.read_excel(os.path.join(path, 'wsi_labels.xlsx'), index_col='slide_id')

        wsi_dict = defaultdict(list)
        for image in images:
            wsi_id = os.path.basename(image).split('_')[0]
            wsi_dict[wsi_id].append(image)

        self.wsi_ids = slide_data.index.to_numpy()
        self.patient_ids = slide_data['patient_id'].to_numpy()
        self.wsi_paths = [wsi_dict[wsi_id] for wsi_id in self.wsi_ids]

        if feature_path:
            self.features = [os.path.join(feature_path, wsi_id + '.pt') for wsi_id in self.wsi_ids]

        self.labels = list(zip(slide_data['Gleason_primary'].to_numpy(), slide_data['Gleason_secondary'].to_numpy()))
        if tumor_type is not None:
            self.labels = gleason2binary(self.labels, tumor_type)
