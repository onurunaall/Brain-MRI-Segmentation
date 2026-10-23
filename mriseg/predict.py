"""
Inference script for brain MRI FLAIR abnormality segmentation.
Usage: python -m mriseg.predict --model-path ./checkpoints/best_model.pt --split validation

Everything about the model (architecture, image size, context slices, fold and its patients) comes from the record
train.py saves in the checkpoint; exported .onnx / .engine files use the model.pt next to them. The backend follows the
model file: .pt -> PyTorch (optionally torch.compile / fp16 autocast), .onnx -> ONNX Runtime, .engine / .plan ->
TensorRT. --bundle-dir averages all fold models of one architecture.

Scores are measured in original pixels against the *_mask.tif files (evaluation.score_patient). --masks-dir writes the
predicted masks in the size of the input slices; with --split all the patients need no *_mask.tif files (new
patients), and then no metrics are computed.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from mriseg.dataset import MRISegmentationDataset as SegDataset, find_patients
from mriseg.evaluation import score_patient, write_results
from mriseg.inference import COMPILE_MODES, InferenceBackend, create_backend, load_bundle, model_record
from mriseg.utils import (POSTPROCESS_METHODS, VolumeGeometry, dice_similarity_coefficient, draw_contour,
                          grayscale_to_rgb, group_slices_by_patient, postprocess_mask, restore_mask_geometry)

Volumes = Dict[str, np.ndarray]  # patient_id -> (Z, H, W) array on the preprocessed grid


def run_inference(cfg: argparse.Namespace) -> None:
    """
    Run full inference pipeline: predict → postprocess → evaluate → save.

    :param cfg: Parsed command-line arguments
    """
    if cfg.postprocess == "auto" and cfg.split == "all":
        raise SystemExit("--postprocess auto picks the method on the validation patients, which --split all does "
                         f"not have; pass one of {POSTPROCESS_METHODS} (bundles record each member's choice in "
                         "manifest.json)")
    if cfg.masks_dir is not None and _is_inside(cfg.masks_dir, cfg.data_dir):
        raise SystemExit("--masks-dir must be outside --data-dir, or the written masks would be read back as slices")

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    backend, record = _load_model(cfg, device)

    postprocess, validation_dice = cfg.postprocess, None
    if postprocess == "auto":
        postprocess, validation_dice = _choose_postprocess(backend, cfg, record, device)

    dataset = _build_dataset(cfg, record, cfg.split)
    if not dataset.has_masks and (cfg.results_json or cfg.masks_npz):
        raise SystemExit("Patients in --data-dir without *_mask.tif files have no ground truth for --results-json or "
                         "--masks-npz; save their predicted masks with --masks-dir instead")
    if cfg.split == "validation":
        print("[Predict] WARNING: the validation patients chose the checkpoint during training (and, with "
              "--postprocess auto, the post-processing), so their Dice is optimistic. For an unbiased estimate "
              "train with --fold and predict with --split test.")

    flair, probabilities, targets = predict_volumes(backend, dataset, cfg.batch_size, device)
    masks = {pid: postprocess_mask(probability > 0.5, postprocess) for pid, probability in probabilities.items()}

    if cfg.masks_dir is not None:
        _save_original_size_masks(cfg.masks_dir, masks, dataset)
        print(f"[Predict] saved predicted masks of {len(masks)} patients in the size of their input slices "
              f"-> {cfg.masks_dir}")

    if dataset.has_masks:
        scores = {pid: score_patient(probabilities[pid], dataset.geometry[pid], dataset.original_masks(pid),
                                     postprocess)
                  for pid in dataset.patient_ids}
        dice = {pid: patient_scores["dice"] for pid, patient_scores in scores.items()}
        _plot_dice_distribution(dice, cfg.figure_path)
        print(f"[Predict] mean Dice ({postprocess}, original pixels) = {np.mean(list(dice.values())):.4f}")
        if cfg.results_json:
            write_results(cfg.results_json, _results_meta(cfg, record, postprocess, validation_dice), scores)
            print(f"[Predict] per-patient metrics -> {cfg.results_json}")
    else:
        print("[Predict] some patients have no *_mask.tif files: no Dice is computed")

    if cfg.masks_npz:
        _save_masks(cfg.masks_npz, flair, targets, masks, probabilities)
        print(f"[Predict] saved FLAIR, ground truth, predicted masks and probabilities for {len(masks)} patients "
              f"-> {cfg.masks_npz}")

    if not cfg.skip_overlays:
        _save_overlays(cfg.output_dir, flair, masks, targets)


def _load_model(cfg: argparse.Namespace, device: torch.device) -> Tuple[InferenceBackend, Dict[str, Any]]:
    """
    Load the model named by --model-path, or every member of --bundle-dir as one ensemble.

    :return: (backend, record); the record is the checkpoint's training record, or the bundle's manifest (which has
             the architecture, image size and context slices, but no fold, seed or patients)
    """
    if cfg.bundle_dir is None:
        return create_backend(cfg.model_path, device, cfg.compile, cfg.amp), model_record(cfg.model_path)

    if cfg.split != "all":
        raise SystemExit("--bundle-dir averages fold models that were trained on the validation and test patients "
                         "of the other folds. Use it with --split all on new data; evaluate CV folds with "
                         "--model-path instead.")

    backend, manifest = load_bundle(cfg.bundle_dir, cfg.bundle_file, device, cfg.compile, cfg.amp)
    print(f"[Predict] {manifest['arch']} ensemble of {len(manifest['members'])} models ({cfg.bundle_file})")
    return backend, manifest


def _patients(cfg: argparse.Namespace, record: Dict[str, Any], split: str) -> List[str]:
    """
    The patients to predict: the model's own validation or test patients from its record, or with 'all' every
    patient in --data-dir, which must not include the model's training patients.
    """
    if split == "all":
        everyone = sorted(find_patients(cfg.data_dir))
        # Bundle manifests list no training patients (and bundles are only used on new data)
        leaked = sorted(set(everyone) & set(record.get("train_patients", [])))
        if leaked:
            raise SystemExit(f"{len(leaked)} of the {len(everyone)} patients in {cfg.data_dir} were TRAINING "
                             f"patients of {cfg.model_path} (e.g. {leaked[0]}), so their Dice would be inflated")
        return everyone

    patients: List[str] = record["val_patients" if split == "validation" else "test_patients"]
    if not patients:
        raise SystemExit(f"{cfg.model_path} was trained without --fold, so it has no test patients; "
                         f"use --split validation")
    return patients


def _build_dataset(cfg: argparse.Namespace, record: Dict[str, Any], split: str) -> SegDataset:
    """Evaluation dataset of one split (no augmentation), preprocessed as during training."""
    return SegDataset(cfg.data_dir,
                      _patients(cfg, record, split),
                      resolution=record["image_size"],
                      context_slices=record["context_slices"],
                      require_masks=split != "all")


def predict_volumes(backend: InferenceBackend,
                    dataset: SegDataset,
                    batch_size: int,
                    device: torch.device) -> Tuple[Volumes, Volumes, Volumes]:
    """
    Run the backend over a dataset and regroup the slices per patient.

    :return: (FLAIR of the segmented slices, probabilities, grid ground truth), each patient_id -> (Z, H, W)
    """
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=1)
    flair_channel = SegDataset.flair_channel(dataset.context_slices)

    flair: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for images, masks in tqdm(loader, desc="Predicting"):
        probabilities.extend(backend.predict(images.to(device))[:, 0].cpu().numpy())
        targets.extend(masks[:, 0].numpy())
        flair.extend(images[:, flair_channel].numpy())

    def per_patient(slices: List[np.ndarray]) -> Volumes:
        return group_slices_by_patient(slices, dataset.flat_index, dataset.patient_ids)

    return per_patient(flair), per_patient(probabilities), per_patient(targets)


def best_postprocess(probabilities: Volumes,
                     geometry: Dict[str, VolumeGeometry],
                     ground_truth: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, float]]:
    """
    The post-processing with the best mean Dice (in original pixels) over the given patients. Ties go to the earlier
    method in POSTPROCESS_METHODS, i.e. the one that removes less.

    :param probabilities: patient_id -> (Z, H, W) probabilities on the grid
    :param geometry: patient_id -> VolumeGeometry
    :param ground_truth: patient_id -> original masks
    :return: (chosen method, mean Dice per method)
    """
    scores: Dict[str, float] = {}
    for method in POSTPROCESS_METHODS:
        dice = [dice_similarity_coefficient(restore_mask_geometry(postprocess_mask(probability > 0.5, method),
                                                                  geometry[pid]),
                                            ground_truth[pid])
                for pid, probability in probabilities.items()]
        scores[method] = float(np.mean(dice))
    return max(POSTPROCESS_METHODS, key=lambda method: scores[method]), scores


def _choose_postprocess(backend: InferenceBackend,
                        cfg: argparse.Namespace,
                        record: Dict[str, Any],
                        device: torch.device) -> Tuple[str, Dict[str, float]]:
    """
    Pick the post-processing on the model's validation patients (the patients that selected the checkpoint), so the
    test patients play no part in the choice.

    :return: (chosen method, mean validation Dice per method)
    """
    dataset = _build_dataset(cfg, record, "validation")
    _, probabilities, _ = predict_volumes(backend, dataset, cfg.batch_size, device)
    ground_truth = {pid: dataset.original_masks(pid) for pid in dataset.patient_ids}
    chosen, scores = best_postprocess(probabilities, dataset.geometry, ground_truth)
    print("[Predict] validation Dice per post-processing: "
          + ", ".join(f"{method} {score:.4f}" for method, score in scores.items()) + f" -> using '{chosen}'")
    return chosen, scores


def _results_meta(cfg: argparse.Namespace,
                  record: Dict[str, Any],
                  postprocess: str,
                  validation_dice: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Run metadata for the results JSON (bundles have no fold or seed)."""
    return {"arch": record["arch"],
            "split": cfg.split,
            "fold": record.get("fold"),
            "n_folds": record.get("n_folds"),
            "split_seed": record.get("split_seed"),
            "seed": record.get("seed"),
            "model_path": cfg.model_path,
            "bundle_dir": cfg.bundle_dir,
            "bundle_file": cfg.bundle_file if cfg.bundle_dir else None,
            "compile": cfg.compile,
            "amp": cfg.amp,
            "image_size": record["image_size"],
            "context_slices": record["context_slices"],
            "postprocess": postprocess,
            "postprocess_validation_dice": validation_dice}


def _save_original_size_masks(masks_dir: str, masks: Volumes, dataset: SegDataset) -> None:
    """
    Write each patient's post-processed prediction as one uint8 TIFF (0 / 255) per input slice, mapped back onto the
    input slices (same size, same slice numbering): <masks_dir>/<patient>/<image file stem>_pred.tif.
    Slices that preprocessing dropped (first, last and empty ones) get empty masks.

    :param masks_dir: Output folder
    :param masks: patient_id -> (Z, H, W) post-processed prediction on the grid
    :param dataset: The dataset the masks came from (for each patient's file names and geometry)
    """
    for pid, mask in masks.items():
        restored = restore_mask_geometry(mask, dataset.geometry[pid])
        patient_dir = os.path.join(masks_dir, pid)
        os.makedirs(patient_dir, exist_ok=True)
        for image_file, slice_mask in zip(dataset.image_files[pid], restored):
            name = os.path.splitext(image_file)[0] + "_pred.tif"
            imsave(os.path.join(patient_dir, name), slice_mask.astype(np.uint8) * 255, check_contrast=False)


def _is_inside(path: str, root: str) -> bool:
    """Whether `path` is `root` or lies below it."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def _save_masks(path: str, flair: Volumes, targets: Volumes, masks: Volumes, probabilities: Volumes) -> None:
    """
    Store per-patient FLAIR, ground truth, post-processed prediction and probabilities so figures can be redrawn
    without the dataset.

    Keys are '<patient_id>__flair' (uint8, per-patient min-max scaled), '<patient_id>__gt' and '<patient_id>__pred'
    (uint8 0/1) and '<patient_id>__prob' (uint8, probability * 255, before thresholding and post-processing).
    All arrays have shape (Z, H, W) on the preprocessed grid.

    :param path: Output .npz path
    :param flair: patient_id -> FLAIR of the segmented slices
    :param targets: patient_id -> ground truth on the grid
    :param masks: patient_id -> post-processed prediction
    :param probabilities: patient_id -> raw probabilities
    """
    arrays: Dict[str, np.ndarray] = {}
    for pid in masks:
        image = flair[pid].astype(np.float32)
        lo, hi = float(image.min()), float(image.max())
        image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)

        arrays[f"{pid}__flair"] = np.round(image * 255).astype(np.uint8)
        arrays[f"{pid}__gt"] = (targets[pid] > 0.5).astype(np.uint8)
        arrays[f"{pid}__pred"] = (masks[pid] > 0.5).astype(np.uint8)
        arrays[f"{pid}__prob"] = np.round(np.clip(probabilities[pid], 0.0, 1.0) * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_overlays(output_dir: str, flair: Volumes, masks: Volumes, targets: Volumes) -> None:
    """One PNG per slice: FLAIR with the prediction (red) and ground truth (green) contours."""
    for pid in masks:
        for s in range(len(masks[pid])):
            rgb = grayscale_to_rgb(flair[pid][s])
            rgb = draw_contour(rgb, masks[pid][s], color=[255, 0, 0])   # red = prediction
            rgb = draw_contour(rgb, targets[pid][s], color=[0, 255, 0])  # green = ground truth
            imsave(os.path.join(output_dir, f"{pid}-{s:02d}.png"), rgb)


def _plot_dice_distribution(dice_scores: Dict[str, float], path: str) -> None:
    """
    Save a horizontal bar chart of per-patient Dice scores.

    :param dice_scores: Dict from patient_id → Dice coefficient
    :param path: Output image file
    """
    sorted_items = sorted(dice_scores.items(), key=lambda kv: kv[1])
    values = [v for _, v in sorted_items]
    labels = ["_".join(pid.split("_")[1:-1]) for pid, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, align="center", color="skyblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(0.0, 1.0, 0.1))
    ax.set_xlim((0.0, 1.0))

    ax.axvline(float(np.mean(values)), color="tomato", linewidth=2, label="Mean")
    ax.axvline(float(np.median(values)), color="forestgreen", linewidth=2, label="Median")

    ax.set_xlabel("Dice Coefficient", fontsize="x-large")
    ax.xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for brain MRI segmentation")

    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model-path", type=str, default=None,
                              help="Trained model: .pt checkpoint (PyTorch), .onnx (ONNX Runtime) or "
                                   ".engine / .plan (TensorRT) with its model.pt next to it")
    model_source.add_argument("--bundle-dir", type=str, default=None,
                              help="Architecture bundle from export_models.py: averages all fold models. "
                                   "Only with --split all, i.e. on data none of the models was trained on")

    parser.add_argument("--bundle-file", type=str, default="model.pt",
                        help="Model file inside each bundle member, e.g. model.fp16.engine (default: model.pt)")
    parser.add_argument("--compile", type=str, nargs="?", const="default", default="none", choices=COMPILE_MODES,
                        help="torch.compile mode for .pt checkpoints; '--compile' alone means 'default' "
                             "(default: none)")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast for .pt checkpoints on CUDA (default: fp32)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--data-dir", type=str, default="./kaggle_3m", help="Root image directory")
    parser.add_argument("--output-dir", type=str, default="./predictions", help="Directory for overlay images")
    parser.add_argument("--figure-path", type=str, default="./dice_distribution.png", help="Path for Dice chart")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test", "all"],
                        help="Which patients to evaluate: the model's own validation or test patients (from its "
                             "checkpoint), or 'all' = every patient in --data-dir (default: validation)")
    parser.add_argument("--postprocess", type=str, default="auto", choices=["auto"] + POSTPROCESS_METHODS,
                        help="Connected-component filtering of the thresholded prediction: none, drop-small (drop "
                             "components < 10%% of the largest), lcc (largest component only), or auto = the best of "
                             "these on the model's validation patients (default: auto; not with --split all)")
    parser.add_argument("--results-json", type=str, default=None, help="If set, write per-patient metrics to this JSON file")
    parser.add_argument("--masks-npz", type=str, default=None, help="If set, save FLAIR, ground-truth and predicted masks per patient to this .npz (used by compare.py)")
    parser.add_argument("--masks-dir", type=str, default=None,
                        help="If set, write the predicted masks as <masks-dir>/<patient>/<slice>_pred.tif in the size "
                             "of the input slices (must be outside --data-dir)")
    parser.add_argument("--skip-overlays", action="store_true", help="Do not write per-slice overlay PNGs")

    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
