import torch.utils.data as data
import numpy as np
import os, sys
import datasets.data_transforms as data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class UniDCF(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.partial_ima_path = config.PARTIAL_IMA_PATH
        self.complete_ima_path = config.COMPLETE_IMA_PATH

        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)
    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([
                {'callback': 'RandomSamplePoints',
                 'parameters': {'n_points': 14336},
                 'objects': ['partial']},
                {'callback': 'RandomSamplePoints',
                 'parameters': {'n_points': 2048},
                 'objects': ['gt']},
                {'callback': 'TransIma',
                 'parameters': {'size': 224},
                 'objects': ['partialima', 'gtima']},
                {'callback': 'ToTensor',
                 'objects': ['partial', 'gt']}])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 14336
                },
                'objects': ['partial']
            }, {'callback': 'RandomSamplePoints',
                'parameters': {'n_points': 2048},
                'objects': ['gt']},
                {'callback': 'TransIma',
                 'parameters': {'size': 224},
                 'objects': ['partialima', 'gtima']},
                {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']),
                      logger='ZYTOOTHDATASET')
            samples = dc[subset]
            part_path = self.partial_points_path % (subset, dc['taxonomy_id'])
            comp_path = self.complete_points_path % (subset, dc['taxonomy_id'])
            partima_path = self.partial_ima_path % (subset, dc['taxonomy_id'])
            compima_path = self.complete_ima_path % (subset, dc['taxonomy_id'])

            for s in samples:
                file_list.append({'taxonomy_id': dc['taxonomy_id'], 'model_id': s,
                                  'partial_path': os.path.join(part_path, s +".ply"),
                                  'gt_path': os.path.join(comp_path, s +".ply"),
                                  'partialima_path': partima_path,
                                  'gtima_path': compima_path
                                  })
        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='ZYTOOTHDATASET')
        return file_list

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            #x_ima_path, y_ima_path, z_ima_path = "None", "None", "None"
            if type(file_path) == list:
                rand_idx = random.randint(0, len(file_path) - 1) if self.subset == 'train' else 0
                file_path = file_path[rand_idx]
            x_ima_path = os.path.join(sample['%sima_path' % ri], os.path.basename(file_path).split(".ply")[0] + "_x.png")
            y_ima_path = os.path.join(sample['%sima_path' % ri], os.path.basename(file_path).split(".ply")[0] + "_y.png")
            z_ima_path = os.path.join(sample['%sima_path' % ri], os.path.basename(file_path).split(".ply")[0] + "_z.png")
            x_ima = IO.get(x_ima_path).astype(np.float32)
            y_ima = IO.get(y_ima_path).astype(np.float32)
            z_ima = IO.get(z_ima_path).astype(np.float32)

            data[ri] = IO.get(file_path).astype(np.float32)
            
            data[ri + "ima"] = np.stack([x_ima, y_ima, z_ima], axis=-1)

        if self.transforms is not None:
            data = self.transforms(data)
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], data['partialima'], data['gtima'])

    def __len__(self):
        return len(self.file_list)