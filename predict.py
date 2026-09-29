"""
Inference script for brain MRI FLAIR abnormality segmentation.
Usage: python predict.py --model-path ./checkpoints/best_model.pt --data-dir ./kaggle_3m

The backend follows the model file: .pt checkpoint -> PyTorch (optionally torch.compile / fp16 autocast),
.onnx -> ONNX Runtime, .engine / .plan -> TensorRT. --bundle-dir averages all fold models of one architecture.
"""

import argparse
import os
from io import BytesIO
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRISegmentationDataset as SegDataset
from network import ModelFactory
from evaluation import PatientEvaluator
from inference import COMPILE_MODES, InferenceBackend, create_backend, load_bundle
from utils import dice_similarity_coefficient, grayscale_to_rgb, draw_contour


def run_inference(cfg: argparse.Namespace) -> None:
    """
    Run full inference pipeline: predict → postprocess → evaluate → save.

    :param cfg: Parsed command-line arguments
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    backend, arch = _create_backend(cfg, device)
    dataset = _build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, drop_last=False, num_workers=1)

    all_inputs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for inputs, targets in tqdm(loader, desc="Predicting"):
        probabilities = backend.predict(inputs.to(device))
        all_preds.extend(probabilities.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_inputs.extend(inputs.numpy())

    patient_volumes = _reassemble_volumes(all_inputs,
                                          all_preds,
                                          all_targets,
                                          dataset.flat_index,
                                          dataset.patient_ids)

    dice_scores = _compute_dice_per_patient(patient_volumes)
    chart_image = _plot_dice_distribution(dice_scores)
    imsave(cfg.figure_path, chart_image)

    if cfg.results_json:
        dice_raw = PatientEvaluator.raw_dice_per_patient(all_preds,
                                                         all_targets,
                                                         dataset.flat_index,
                                                         dataset.patient_ids)
        hd95_scores = PatientEvaluator.hd95_per_patient(patient_volumes)
        meta = {"arch": arch,
                "split": cfg.split,
                "fold": cfg.fold,
                "n_folds": cfg.n_folds,
                "split_seed": cfg.split_seed,
                "seed": cfg.seed,
                "model_path": cfg.model_path,
                "bundle_dir": cfg.bundle_dir,
                "bundle_file": cfg.bundle_file if cfg.bundle_dir else None,
                "compile": cfg.compile,
                "amp": cfg.amp}
        PatientEvaluator.write_results(cfg.results_json, meta, dice_scores, dice_raw, hd95_scores)
        print(f"[Predict] mean Dice (LCC) = {np.mean(list(dice_scores.values())):.4f} -> {cfg.results_json}")

    if cfg.masks_npz:
        probability_volumes = _group_by_patient(all_preds, dataset.flat_index, dataset.patient_ids)
        _save_masks(cfg.masks_npz, patient_volumes, probability_volumes)
        print(f"[Predict] saved FLAIR, ground truth, predicted masks and probabilities for "
              f"{len(patient_volumes)} patients -> {cfg.masks_npz}")

    if cfg.skip_overlays:
        return

    for pid, (vol_in, vol_pred, vol_true) in patient_volumes.items():
        for s in range(vol_in.shape[0]):
            # Use FLAIR channel (index 1) as background
            rgb = grayscale_to_rgb(vol_in[s, 1])
            rgb = draw_contour(rgb, vol_pred[s, 0], color=[255, 0, 0])   # red = prediction
            rgb = draw_contour(rgb, vol_true[s, 0], color=[0, 255, 0])   # green = ground truth

            fname = f"{pid}-{s:02d}.png"
            imsave(os.path.join(cfg.output_dir, fname), rgb)


def _create_backend(cfg: argparse.Namespace, device: torch.device) -> Tuple[InferenceBackend, str]:
    """
    Load the model named by --model-path, or every member of --bundle-dir as one ensemble.

    :param cfg: Parsed arguments
    :param device: Device to run on
    :return: (backend, architecture name)
    """
    if cfg.bundle_dir is None:
        return create_backend(cfg.model_path, cfg.arch, device, cfg.compile, cfg.amp), cfg.arch

    if cfg.split != "all":
        raise SystemExit("--bundle-dir averages fold models that were trained on the validation and test patients "
                         "of the other folds. Use it with --split all on new data; evaluate CV folds with "
                         "--model-path instead.")

    backend, manifest = load_bundle(cfg.bundle_dir, cfg.bundle_file, device, cfg.compile, cfg.amp)
    if manifest["image_size"] != cfg.image_size:
        raise SystemExit(f"The bundle was trained at --image-size {manifest['image_size']}, got {cfg.image_size}")
    print(f"[Predict] {manifest['arch']} ensemble of {len(manifest['members'])} models ({cfg.bundle_file})")
    return backend, str(manifest["arch"])


def _build_dataset(cfg: argparse.Namespace) -> SegDataset:
    """
    Create the evaluation dataset (deterministic, no augmentation).

    :param cfg: Parsed arguments
    :return: Dataset of the requested split
    """
    return SegDataset(data_root=cfg.data_dir,
                      split=cfg.split,
                      resolution=cfg.image_size,
                      n_validation=cfg.n_validation,
                      seed=cfg.split_seed,
                      fold=cfg.fold,
                      n_folds=cfg.n_folds)


def _reassemble_volumes(inputs: List[np.ndarray],
                        preds: List[np.ndarray],
                        targets: List[np.ndarray],
                        flat_index: List[Tuple[int, int]],
                        patient_ids: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Group flat slice lists back into per-patient volumes and apply LCC postprocessing.

    :param inputs: Flat list of input arrays
    :param preds: Flat list of prediction arrays
    :param targets: Flat list of target arrays
    :param flat_index: (patient_idx, slice_idx) mapping
    :param patient_ids: Patient identifier strings
    :return: Dict mapping patient_id → (input_volume, pred_volume, target_volume)
    """
    volumes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    slices_per_patient = np.bincount([entry[0] for entry in flat_index])

    offset = 0
    for p_idx, n_slices in enumerate(slices_per_patient):
        vol_in = np.array(inputs[offset: offset + n_slices])

        # Binarize prediction and retain largest connected component
        vol_pred = np.round(np.array(preds[offset: offset + n_slices])).astype(int)
        if np.any(vol_pred):
            vol_pred = largest_connected_component(vol_pred)

        vol_true = np.array(targets[offset: offset + n_slices])

        volumes[patient_ids[p_idx]] = (vol_in, vol_pred, vol_true)
        offset += n_slices

    return volumes


def _group_by_patient(slices: List[np.ndarray],
                      flat_index: List[Tuple[int, int]],
                      patient_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Stack a flat list of per-slice arrays into one volume per patient.

    :param slices: Flat list of per-slice arrays in dataset order
    :param flat_index: (patient_idx, slice_idx) mapping from the dataset
    :param patient_ids: Patient identifiers in dataset order
    :return: Dict patient_id -> array of shape (Z, ...)
    """
    volumes: Dict[str, np.ndarray] = {}
    slices_per_patient = np.bincount([entry[0] for entry in flat_index])

    offset = 0
    for p_idx, n_slices in enumerate(slices_per_patient):
        volumes[patient_ids[p_idx]] = np.array(slices[offset: offset + n_slices])
        offset += n_slices

    return volumes


def _save_masks(path: str,
                volumes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                probabilities: Optional[Dict[str, np.ndarray]] = None) -> None:
    """
    Store per-patient FLAIR, ground truth and LCC prediction so figures can be redrawn without the dataset.

    Keys are '<patient_id>__flair' (uint8, per-patient min-max scaled), '<patient_id>__gt' and
    '<patient_id>__pred' (uint8 0/1), plus '<patient_id>__prob' (uint8, probability * 255, before thresholding
    and LCC) when probabilities are given. All arrays have shape (Z, H, W) on the preprocessed grid.

    :param path: Output .npz path
    :param volumes: Dict patient_id -> (input, LCC prediction, target), as built by _reassemble_volumes
    :param probabilities: Optional dict patient_id -> raw probabilities of shape (Z, 1, H, W)
    """
    arrays: Dict[str, np.ndarray] = {}
    for pid, (vol_in, vol_pred, vol_true) in volumes.items():
        flair = vol_in[:, 1].astype(np.float32)  # channel 1 = FLAIR, as in the overlay PNGs
        lo, hi = float(flair.min()), float(flair.max())
        flair = (flair - lo) / (hi - lo) if hi > lo else np.zeros_like(flair)

        arrays[f"{pid}__flair"] = np.round(flair * 255).astype(np.uint8)
        arrays[f"{pid}__gt"] = (np.asarray(vol_true)[:, 0] > 0.5).astype(np.uint8)
        arrays[f"{pid}__pred"] = (np.asarray(vol_pred)[:, 0] > 0.5).astype(np.uint8)

        if probabilities is not None:
            probability = np.clip(probabilities[pid][:, 0], 0.0, 1.0)
            arrays[f"{pid}__prob"] = np.round(probability * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _compute_dice_per_patient(volumes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """
    Compute per-patient Dice scores (LCC already applied during reassembly).

    :param volumes: Dict from patient_id → (input, prediction, target)
    :return: Dict from patient_id → Dice score
    """
    scores: Dict[str, float] = {}
    for pid, (_, pred, gt) in volumes.items():
        scores[pid] = dice_similarity_coefficient(pred, gt, apply_lcc=False)
    return scores


def _plot_dice_distribution(dice_scores: Dict[str, float]) -> np.ndarray:
    """
    Create a horizontal bar chart of per-patient Dice scores.

    :param dice_scores: Dict from patient_id → Dice coefficient
    :return: RGBA image array of the rendered figure
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

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf).convert("RGBA")
    return np.array(img)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for brain MRI segmentation")

    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model-path", type=str, default=None,
                              help="Trained model: .pt checkpoint (PyTorch), .onnx (ONNX Runtime) or "
                                   ".engine / .plan (TensorRT)")
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
    parser.add_argument("--image-size", type=int, default=256, help="Target spatial resolution (default: 256)")
    parser.add_argument("--output-dir", type=str, default="./predictions", help="Directory for overlay images")
    parser.add_argument("--figure-path", type=str, default="./dice_distribution.png", help="Path for Dice chart")
    parser.add_argument("--arch", type=str, default="unet", choices=ModelFactory.available(),
                        help="Model architecture (must match a .pt checkpoint; ignored with --bundle-dir)")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test", "all"],
                        help="Which patients to evaluate; 'all' = every patient in --data-dir (default: validation)")
    parser.add_argument("--fold", type=int, default=None, help="Test fold index; required for --split test")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-validation", type=int, default=10, help="Must match the value used in training (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Must match the value used in training (default: 42)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed of this checkpoint; only recorded in the results JSON")
    parser.add_argument("--results-json", type=str, default=None, help="If set, write per-patient metrics to this JSON file")
    parser.add_argument("--masks-npz", type=str, default=None, help="If set, save FLAIR, ground-truth and predicted masks per patient to this .npz (used by compare.py)")
    parser.add_argument("--skip-overlays", action="store_true", help="Do not write per-slice overlay PNGs")

    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
