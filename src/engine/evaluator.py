#!/usr/bin/env python3
import numpy as np

from collections import defaultdict
from typing import List, Union

from .eval import multilabel
from .eval import singlelabel
from ..utils import logging
logger = logging.get_logger("visual_prompt")


def dist_acc(dists, thr=0.001):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    # dists = dists.detach().cpu().numpy()
    dist_cal = np.not_equal(dists, 0.0)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

class Evaluator():
    """
    An evaluator with below logics:

    1. find which eval module to use.
    2. store the eval results, pretty print it in log file as well.
    """

    def __init__(
        self,
    ) -> None:
        self.results = defaultdict(dict)
        self.iteration = -1
        self.threshold_end = 0.5

    def update_iteration(self, iteration: int) -> None:
        """update iteration info"""
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        if self.iteration > -1:
            key_name = "epoch_" + str(self.iteration)
        else:
            key_name = "final"
        if isinstance(value, float):
            self.results[key_name].update({metric: value})
        else:
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets, test_data, multilabel=False):
        """
        Evaluate classification result.
        Args:
            probs: np.ndarray for num_data x num_class, predicted probabilities
            targets: np.ndarray for multilabel, list of integers for single label
            test_labels:  map test image ids to a list of class labels
        """
        if not targets:
            raise ValueError(
                "When evaluating classification, need at least give targets")

        if multilabel:
            self._eval_multilabel(probs, targets, test_data)
        else:
            self._eval_singlelabel(probs, targets, test_data)

    def regress(self, pred, target, eval_type, mode = 'pose_estimation'):
        # Return PKA score
        pka =  dist_acc((pred - target)**2) 

        acc_dict = {}
        acc_dict['pka'] = pka
        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict
        self.log_and_update(log_results, save_results, eval_type, mode)

    def _eval_singlelabel(
        self,
        scores: np.ndarray,
        targets: List[int],
        eval_type: str,
        mode = 'classification'
    ) -> None:
        """
        if number of labels > 2:
            top1 and topk (5 by default) accuracy
        if number of labels == 2:
            top1 and rocauc
        """
        acc_dict = singlelabel.compute_acc_auc(scores, targets)

        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict

        self.log_and_update(log_results, save_results, eval_type, mode)

    def _eval_multilabel(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        eval_type: str,
        mode = 'classification'
    ) -> None:
        num_labels = scores.shape[-1]
        targets = multilabel.multihot(targets, num_labels)

        log_results = {}
        ap, ar, mAP, mAR = multilabel.compute_map(scores, targets)
        f1_dict = multilabel.get_best_f1_scores(
            targets, scores, self.threshold_end)

        log_results["mAP"] = np.around(mAP * 100, decimals=2)
        log_results["mAR"] = np.around(mAR * 100, decimals=2)
        log_results.update({
            k: np.around(v * 100, decimals=2) for k, v in f1_dict.items()})
        save_results = {
            "ap": ap, "ar": ar, "mAP": mAP, "mAR": mAR, "f1": f1_dict
        }
        self.log_and_update(log_results, save_results, eval_type, mode)

    def log_and_update(self, log_results, save_results, eval_type, mode):
        log_str = ""
        for k, result in log_results.items():
            if not isinstance(result, np.ndarray):
                log_str += f"{k}: {result:.2f}\t"
            else:
                log_str += f"{k}: {list(result)}\t"
        logger.info(f"{mode} results with {eval_type}: {log_str}")
        # save everything
        self.update_result(mode, {eval_type: save_results})
