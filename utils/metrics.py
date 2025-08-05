# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import os
from extensions.emd import emd_module as emd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'eval_key': 'f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Recall',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'eval_key': 'recall',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Precision',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'eval_key': 'precision',
        'is_greater_better': True,
        'init_value': 0
    },
    {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'EMDistance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd.emdModule(),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)
        results_cache = {}

        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device)
            else:
                eval_func = eval(item['eval_func'])

                # 缓存该函数输出避免重复计算
                if item['eval_func'] not in results_cache:
                    results_cache[item['eval_func']] = eval_func(pred, gt)

                result = results_cache[item['eval_func']]
                if isinstance(result, dict):
                    key = item.get('eval_key', None)
                    if key and key in result:
                        _values[i] = result[key]
                    else:
                        logging.warn(f"Key '{key}' not found in result dict of '{item['eval_func']}'")
                        _values[i] = torch.tensor(0.).to(gt.device)
                else:
                    _values[i] = result

        return _values
    
    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """Compute F-score, recall, precision, AUROC, and AUPRC between predicted and GT point clouds."""
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)

        if b != 1:
            metrics_list = []
            for idx in range(b):
                metrics = cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1], th=th)
                metrics_list.append(metrics)

            # 平均batch内每项指标
            averaged = {
                k: torch.stack([m[k] for m in metrics_list]).mean()
                for k in metrics_list[0].keys()
            }
            return averaged

        else:
            pred_o3d = cls._get_open3d_ptcloud(pred)
            gt_o3d = cls._get_open3d_ptcloud(gt)

            dist1 = np.asarray(pred_o3d.compute_point_cloud_distance(gt_o3d))
            dist2 = np.asarray(gt_o3d.compute_point_cloud_distance(pred_o3d))

            # Recall & Precision
            recall = float(np.sum(dist2 < th)) / float(len(dist2))
            precision = float(np.sum(dist1 < th)) / float(len(dist1))
            f_score = 2 * recall * precision / (recall + precision) if recall + precision else 0.0

            result = {
                "f_score": torch.tensor(f_score, device=device),
                "recall": torch.tensor(recall, device=device),
                "precision": torch.tensor(precision, device=device),
            }

            return result

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[3]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[4]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[5]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
