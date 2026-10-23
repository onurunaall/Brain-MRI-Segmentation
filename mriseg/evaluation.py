"""
Per-patient evaluation metrics.

Reported scores (score_patient) are measured in original pixels: the prediction on the preprocessed grid is mapped back
onto the original slices and compared with the *_mask.tif files as they are on disk, every slice included. Training
only uses the cheaper Dice on the grid (raw_dice_per_patient) to pick the best epoch.
"""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from medpy.metric.binary import hd95

from mriseg.utils import (VolumeGeometry, dice_similarity_coefficient, group_slices_by_patient, postprocess_mask,
                          restore_mask_geometry)


def raw_dice_per_patient(preds: List[np.ndarray],
                         targets: List[np.ndarray],
                         flat_index: List[Tuple[int, int]],
                         patient_ids: List[str]) -> Dict[str, float]:
    """
    Dice per patient volume on the preprocessed grid after thresholding at 0.5, WITHOUT post-processing
    (connected-component filtering). Used during training to pick the best epoch.

    :param preds: Flat list of per-slice probability maps, each (1, H, W), in dataset order
    :param targets: Flat list of per-slice ground-truth masks, each (1, H, W), in dataset order
    :param flat_index: (patient_idx, slice_idx) mapping from the dataset
    :param patient_ids: Patient identifiers in dataset order
    :return: Dict patient_id -> Dice
    """
    pred_volumes = group_slices_by_patient(preds, flat_index, patient_ids)
    true_volumes = group_slices_by_patient(targets, flat_index, patient_ids)
    return {pid: dice_similarity_coefficient(pred_volumes[pid], true_volumes[pid]) for pid in pred_volumes}


def hd95_px(prediction: npt.NDArray[np.bool_], ground_truth: npt.NDArray[np.bool_]) -> float:
    """
    95th-percentile Hausdorff distance between two binary volumes of original slices.

    Units are original pixels in-plane, with one slice step counted as 1: the TIFF files carry no pixel size or slice
    thickness, so this is not in millimetres.

    :return: HD95; 0.0 if both masks are empty and NaN if exactly one is (the tumour was missed, or a false alarm)
    """
    if not prediction.any() and not ground_truth.any():
        return 0.0
    if not prediction.any() or not ground_truth.any():
        return float("nan")
    return float(hd95(prediction, ground_truth))


def score_patient(probability: npt.NDArray,
                  geometry: VolumeGeometry,
                  ground_truth: npt.NDArray[np.bool_],
                  postprocess: str) -> Dict[str, float]:
    """
    Metrics of one patient in original pixels.

    :param probability: (Z, R, R) predicted probabilities on the preprocessed grid
    :param geometry: The patient's VolumeGeometry (maps the grid back onto the original slices)
    :param ground_truth: (n_slices, height, width) original masks (dataset.original_masks)
    :param postprocess: One of utils.POSTPROCESS_METHODS
    :return: {'dice': Dice after post-processing, 'dice_raw': Dice without it, 'hd95_px': HD95 after it}
    """
    raw = probability > 0.5
    raw_restored = restore_mask_geometry(raw, geometry)
    post_restored = restore_mask_geometry(postprocess_mask(raw, postprocess), geometry)
    return {"dice": dice_similarity_coefficient(post_restored, ground_truth),
            "dice_raw": dice_similarity_coefficient(raw_restored, ground_truth),
            "hd95_px": hd95_px(post_restored, ground_truth)}


def write_results(path: str, meta: Dict[str, Any], per_patient: Dict[str, Dict[str, float]]) -> None:
    """
    Write per-patient metrics and run metadata to a JSON file.

    :param path: Output .json path
    :param meta: Run metadata (arch, fold, seed, split, model file, post-processing)
    :param per_patient: Dict patient_id -> score_patient's metrics
    """
    with open(path, "w") as fp:
        json.dump({"meta": meta, "per_patient": per_patient}, fp, indent=2)
