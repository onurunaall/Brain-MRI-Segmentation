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
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRISegmentationDataset as SegDataset
from network import ModelFactory
from evaluation import PatientEvaluator
from inference import COMPILE_MODES, InferenceBackend, create_backend, load_bundle
from utils import POSTPROCESS_METHODS, dice_similarity_coefficient, draw_contour, grayscale_to_rgb, postprocess_mask


def run_inference(cfg: argparse.Namespace) -> None:
    """
    Run full inference pipeline: predict → postprocess → evaluate → save.

    :param cfg: Parsed command-line arguments
    """
    if cfg.postprocess == "auto" and cfg.split == "all":
        raise SystemExit("--postprocess auto picks the method on the validation patients, which --split all does "
                         f"not have; pass one of {POSTPROCESS_METHODS} (bundles record each member's choice in "
                         "manifest.json)")

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    backend, arch = _create_backend(cfg, device)

    postprocess, validation_dice = cfg.postprocess, None
    if postprocess == "auto":
        postprocess, validation_dice = _choose_postprocess(cfg, backend, device)

    dataset = _build_dataset(cfg)
    all_inputs, all_preds, all_targets = _predict(backend, dataset, cfg.batch_size, device)

    patient_volumes = _reassemble_volumes(all_inputs,
                                          all_preds,
                                          all_targets,
                                          dataset.flat_index,
                                          dataset.patient_ids,
                                          postprocess)

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
                "amp": cfg.amp,
                "context_slices": cfg.context_slices,
                "postprocess": postprocess,
                "postprocess_validation_dice": validation_dice}
        PatientEvaluator.write_results(cfg.results_json, meta, dice_scores, dice_raw, hd95_scores)
        print(f"[Predict] mean Dice ({postprocess}) = {np.mean(list(dice_scores.values())):.4f} -> {cfg.results_json}")

    if cfg.masks_npz:
        probability_volumes = _group_by_patient(all_preds, dataset.flat_index, dataset.patient_ids)
        _save_masks(cfg.masks_npz, patient_volumes, probability_volumes)
        print(f"[Predict] saved FLAIR, ground truth, predicted masks and probabilities for "
              f"{len(patient_volumes)} patients -> {cfg.masks_npz}")

    if cfg.skip_overlays:
        return

    for pid, (vol_in, vol_pred, vol_true) in patient_volumes.items():
        for s in range(vol_in.shape[0]):
            # Use FLAIR channel (index 1 of the segmented slice's channels) as background
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
        return create_backend(cfg.model_path, cfg.arch, device, cfg.compile, cfg.amp, cfg.context_slices), cfg.arch

    if cfg.split != "all":
        raise SystemExit("--bundle-dir averages fold models that were trained on the validation and test patients "
                         "of the other folds. Use it with --split all on new data; evaluate CV folds with "
                         "--model-path instead.")

    backend, manifest = load_bundle(cfg.bundle_dir, cfg.bundle_file, device, cfg.compile, cfg.amp)
    if manifest["image_size"] != cfg.image_size:
        raise SystemExit(f"The bundle was trained at --image-size {manifest['image_size']}, got {cfg.image_size}")
    if manifest.get("context_slices", 0) != cfg.context_slices:
        raise SystemExit(f"The bundle was trained with --context-slices {manifest.get('context_slices', 0)}, "
                         f"got {cfg.context_slices}")
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
                      n_folds=cfg.n_folds,
                      context_slices=cfg.context_slices)


def _predict(backend: InferenceBackend,
             dataset: SegDataset,
             batch_size: int,
             device: torch.device) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run the backend over a dataset in order.

    :return: Flat per-slice lists (inputs, probabilities, targets); inputs keep only the segmented slice's
             channels (not its context neighbours), so FLAIR is channel 1
    """
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=1)
    center_channels = SegDataset.center_slice_channels(dataset.context_slices)

    all_inputs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for inputs, targets in tqdm(loader, desc="Predicting"):
        probabilities = backend.predict(inputs.to(device))
        all_preds.extend(probabilities.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_inputs.extend(inputs[:, center_channels].numpy())

    return all_inputs, all_preds, all_targets


def _choose_postprocess(cfg: argparse.Namespace,
                        backend: InferenceBackend,
                        device: torch.device) -> Tuple[str, Dict[str, float]]:
    """
    Pick the post-processing with the best mean Dice on the validation patients of this fold (the patients that
    selected the checkpoint), so the test patients play no part in the choice. Ties go to the earlier method in
    POSTPROCESS_METHODS, i.e. the one that removes less.

    :return: (chosen method, mean validation Dice per method)
    """
    dataset = _build_dataset(argparse.Namespace(**{**vars(cfg), "split": "validation"}))
    _, preds, targets = _predict(backend, dataset, cfg.batch_size, device)
    pred_volumes = _group_by_patient(preds, dataset.flat_index, dataset.patient_ids)
    true_volumes = _group_by_patient(targets, dataset.flat_index, dataset.patient_ids)

    scores: Dict[str, float] = {}
    for method in POSTPROCESS_METHODS:
        dice = [dice_similarity_coefficient(postprocess_mask(pred_volumes[pid] > 0.5, method).astype(np.float32),
                                            true_volumes[pid],
                                            apply_lcc=False)
                for pid in pred_volumes]
        scores[method] = float(np.mean(dice))

    chosen = max(POSTPROCESS_METHODS, key=lambda method: scores[method])
    print("[Predict] validation Dice per post-processing: "
          + ", ".join(f"{method} {score:.4f}" for method, score in scores.items()) + f" -> using '{chosen}'")
    return chosen, scores


def _reassemble_volumes(inputs: List[np.ndarray],
                        preds: List[np.ndarray],
                        targets: List[np.ndarray],
                        flat_index: List[Tuple[int, int]],
                        patient_ids: List[str],
                        postprocess: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Group flat slice lists back into per-patient volumes, threshold at 0.5 and post-process the prediction.

    :param inputs: Flat list of input arrays
    :param preds: Flat list of prediction arrays
    :param targets: Flat list of target arrays
    :param flat_index: (patient_idx, slice_idx) mapping
    :param patient_ids: Patient identifier strings
    :param postprocess: One of POSTPROCESS_METHODS (see utils.postprocess_mask)
    :return: Dict mapping patient_id → (input_volume, pred_volume, target_volume)
    """
    volumes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    slices_per_patient = np.bincount([entry[0] for entry in flat_index])

    offset = 0
    for p_idx, n_slices in enumerate(slices_per_patient):
        vol_in = np.array(inputs[offset: offset + n_slices])
        vol_pred = postprocess_mask(np.array(preds[offset: offset + n_slices]) > 0.5, postprocess)
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
    Store per-patient FLAIR, ground truth and post-processed prediction so figures can be redrawn without the dataset.

    Keys are '<patient_id>__flair' (uint8, per-patient min-max scaled), '<patient_id>__gt' and
    '<patient_id>__pred' (uint8 0/1), plus '<patient_id>__prob' (uint8, probability * 255, before thresholding
    and post-processing) when probabilities are given. All arrays have shape (Z, H, W) on the preprocessed grid.

    :param path: Output .npz path
    :param volumes: Dict patient_id -> (input, post-processed prediction, target), as built by _reassemble_volumes
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
    Compute per-patient Dice scores (post-processing already applied during reassembly).

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
    parser.add_argument("--context-slices", type=int, default=1,
                        help="Neighbour slices per side the model was trained with (train.py --context-slices; "
                             "0 for models trained before 2.5D input) (default: 1)")
    parser.add_argument("--postprocess", type=str, default="auto", choices=["auto"] + POSTPROCESS_METHODS,
                        help="Connected-component filtering of the thresholded prediction: none, drop-small (drop "
                             "components < 10%% of the largest), lcc (largest component only), or auto = the best of "
                             "these on the validation patients of --fold (default: auto; not with --split all)")
    parser.add_argument("--results-json", type=str, default=None, help="If set, write per-patient metrics to this JSON file")
    parser.add_argument("--masks-npz", type=str, default=None, help="If set, save FLAIR, ground-truth and predicted masks per patient to this .npz (used by compare.py)")
    parser.add_argument("--skip-overlays", action="store_true", help="Do not write per-slice overlay PNGs")

    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
