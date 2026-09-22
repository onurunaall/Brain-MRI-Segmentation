"""
Per-patient evaluation helpers for test-set benchmarking.
"""

import json
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from medpy.metric.binary import hd95

from utils import dice_similarity_coefficient


class PatientEvaluator:
    """Checkpoint loading fix-up, per-patient metrics, and results export."""

    @staticmethod
    def strip_compile_prefix(state: Dict) -> "OrderedDict":
        """
        Remove the '_orig_mod.' prefix that torch.compile adds to state_dict keys.
        A no-op for checkpoints saved from an uncompiled model.
        """
        cleaned = OrderedDict()
        for key, value in state.items():
            cleaned[key.replace("_orig_mod.", "", 1)] = value
        return cleaned

    @staticmethod
    def raw_dice_per_patient(preds: List[np.ndarray],
                             targets: List[np.ndarray],
                             flat_index: List[Tuple[int, int]],
                             patient_ids: List[str]) -> Dict[str, float]:
        """
        Dice per patient volume after thresholding at 0.5, WITHOUT largest-connected-component filtering.

        :param preds: Flat list of per-slice probability maps, each (1, H, W)
        :param targets: Flat list of per-slice ground-truth masks, each (1, H, W)
        :param flat_index: (patient_idx, slice_idx) mapping from the dataset
        :param patient_ids: Patient identifiers in dataset order
        :return: Dict patient_id -> Dice
        """
        scores: Dict[str, float] = {}
        slices_per_patient = np.bincount([entry[0] for entry in flat_index])

        offset = 0
        for p_idx, n_slices in enumerate(slices_per_patient):
            vol_pred = np.round(np.array(preds[offset: offset + n_slices])).astype(np.int32)
            vol_true = np.round(np.array(targets[offset: offset + n_slices])).astype(np.int32)
            scores[patient_ids[p_idx]] = float(dice_similarity_coefficient(vol_pred, vol_true, apply_lcc=False))
            offset += n_slices

        return scores

    @staticmethod
    def hd95_per_patient(volumes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        95th-percentile Hausdorff distance per patient on the LCC-filtered prediction.

        Units are voxels of the resized 256x256 grid with slice spacing counted as 1,
        NOT millimetres. Returns 0.0 if both masks are empty and NaN if exactly one is empty.

        :param volumes: Dict patient_id -> (input, LCC prediction, target), as built by predict.py
        :return: Dict patient_id -> HD95
        """
        scores: Dict[str, float] = {}
        for pid, (_, pred, gt) in volumes.items():
            pred_bin = np.squeeze(pred, axis=1) > 0.5
            gt_bin = np.squeeze(gt, axis=1) > 0.5

            if not pred_bin.any() and not gt_bin.any():
                scores[pid] = 0.0
            elif not pred_bin.any() or not gt_bin.any():
                scores[pid] = float("nan")
            else:
                scores[pid] = float(hd95(pred_bin, gt_bin))

        return scores

    @staticmethod
    def write_results(path: str,
                      meta: Dict,
                      dice_lcc: Dict[str, float],
                      dice_raw: Dict[str, float],
                      hd95_scores: Dict[str, float]) -> None:
        """
        Write per-patient metrics and run metadata to a JSON file.

        :param path: Output .json path
        :param meta: Run metadata (arch, fold, seed, split, checkpoint)
        :param dice_lcc: Dice after LCC post-processing (the repo's reported metric)
        :param dice_raw: Dice without LCC
        :param hd95_scores: HD95 on LCC prediction (voxel units)
        """
        per_patient = {}
        for pid in dice_lcc:
            per_patient[pid] = {"dice": dice_lcc[pid],
                                "dice_raw": dice_raw[pid],
                                "hd95_vox": hd95_scores[pid]}

        with open(path, "w") as fp:
            json.dump({"meta": meta, "per_patient": per_patient}, fp, indent=2)
