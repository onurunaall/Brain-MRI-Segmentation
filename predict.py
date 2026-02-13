"""
Inference script for brain MRI FLAIR abnormality segmentation.
Usage: python predict.py --model-path ./checkpoints/best_model.pt --data-dir ./kaggle_3m
"""

import argparse
import os
from io import BytesIO
from typing import Dict, Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRISegmentationDataset as SegDataset
from network import UNetModel
from utils import dice_similarity_coefficient, grayscale_to_rgb, draw_contour


def run_inference(cfg: argparse.Namespace) -> None:
    """
    Run full inference pipeline: predict → postprocess → evaluate → save.

    :param cfg: Parsed command-line arguments
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    val_loader = _build_loader(cfg)

    model = UNetModel(in_channels=SegDataset.num_input_channels,
                      out_channels=SegDataset.num_output_channels)
    
    state = torch.load(cfg.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    all_inputs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Predicting"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            out_np = outputs.cpu().numpy()
            all_preds.extend(out_np[s] for s in range(out_np.shape[0]))

            tgt_np = targets.cpu().numpy()
            all_targets.extend(tgt_np[s] for s in range(tgt_np.shape[0]))

            inp_np = inputs.cpu().numpy()
            all_inputs.extend(inp_np[s] for s in range(inp_np.shape[0]))

    patient_volumes = _reassemble_volumes(all_inputs,
                                          all_preds,
                                          all_targets,
                                          val_loader.dataset.flat_index,
                                          val_loader.dataset.patient_ids)

    dice_scores = _compute_dice_per_patient(patient_volumes)
    chart_image = _plot_dice_distribution(dice_scores)
    imsave(cfg.figure_path, chart_image)

    for pid, (vol_in, vol_pred, vol_true) in patient_volumes.items():
        for s in range(vol_in.shape[0]):
            # Use FLAIR channel (index 1) as background
            rgb = grayscale_to_rgb(vol_in[s, 1])
            rgb = draw_contour(rgb, vol_pred[s, 0], color=[255, 0, 0])   # red = prediction
            rgb = draw_contour(rgb, vol_true[s, 0], color=[0, 255, 0])   # green = ground truth

            fname = f"{pid}-{s:02d}.png"
            imsave(os.path.join(cfg.output_dir, fname), rgb)


def _build_loader(cfg: argparse.Namespace) -> DataLoader:
    """
    Create the validation DataLoader (deterministic, no augmentation).

    :param cfg: Parsed arguments
    :return: DataLoader for validation split
    """
    ds = SegDataset(data_root=cfg.data_dir,
                    split="validation",
                    resolution=cfg.image_size)
    
    return DataLoader(ds, batch_size=cfg.batch_size, drop_last=False, num_workers=1)


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
        vol_pred = largest_connected_component(vol_pred)

        vol_true = np.array(targets[offset: offset + n_slices])

        volumes[patient_ids[p_idx]] = (vol_in, vol_pred, vol_true)
        offset += n_slices

    return volumes


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
    ax.set_xlim([0.0, 1.0])

    ax.axvline(np.mean(values), color="tomato", linewidth=2, label="Mean")
    ax.axvline(np.median(values), color="forestgreen", linewidth=2, label="Median")

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

    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--data-dir", type=str, default="./kaggle_3m", help="Root image directory")
    parser.add_argument("--image-size", type=int, default=256, help="Target spatial resolution (default: 256)")
    parser.add_argument("--output-dir", type=str, default="./predictions", help="Directory for overlay images")
    parser.add_argument("--figure-path", type=str, default="./dice_distribution.png", help="Path for Dice chart")
    
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
