"""
Training script for MRI FLAIR abnormality segmentation using U-Net.
Usage: python train.py --data-dir ./kaggle_3m --epochs 100 --lr 1e-4
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import MRISegmentationDataset as SegDataset
from tb_logger import TensorBoardLogger
from losses import SoftDiceLoss
from augmentations import build_augmentation_pipeline
from network import UNetModel
from utils import compose_visualization, dice_similarity_coefficient


def _create_dataloaders(cfg: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    :param cfg: Parsed arguments
    :return: (train_loader, val_loader) tuple
    """
    augmentation = build_augmentation_pipeline(scale_range=cfg.aug_scale,
                                               rotation_deg=cfg.aug_angle,
                                               flip_probability=0.5)

    train_ds = SegDataset(data_root=cfg.data_dir,
                          split="train",
                          resolution=cfg.image_size,
                          transform=augmentation)
           
    val_ds = SegDataset(data_root=cfg.data_dir,
                        split="validation",
                        resolution=cfg.image_size)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
        np.random.seed(worker_seed)

    # Use WeightedRandomSampler to bias towards slices with more foreground
    train_sampler = WeightedRandomSampler(weights=train_ds.sample_weights,
                                          num_samples=len(train_ds),
                                          replacement=True)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              sampler=train_sampler,
                              drop_last=True,
                              num_workers=cfg.num_workers,
                              worker_init_fn=_seed_worker)
    
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            drop_last=False,
                            num_workers=cfg.num_workers,
                            worker_init_fn=_seed_worker)

    return train_loader, val_loader


def _per_volume_dice(all_preds: List[np.ndarray],
                     all_targets: List[np.ndarray],
                     flat_index: List[Tuple[int, int]]) -> List[float]:
    """
    Compute Dice score per patient volume (not per slice).

    :param all_preds: Flat list of per-slice prediction arrays
    :param all_targets: Flat list of per-slice ground-truth arrays
    :param flat_index: Mapping from flat position → (patient_idx, slice_idx)
    :return: List of Dice scores, one per patient
    """
    scores: List[float] = []
    slices_per_patient = np.bincount([entry[0] for entry in flat_index])
    offset = 0

    for n_slices in slices_per_patient:
        vol_pred = np.array(all_preds[offset: offset + n_slices])
        vol_true = np.array(all_targets[offset: offset + n_slices])
        scores.append(dice_similarity_coefficient(vol_pred, vol_true))
        offset += n_slices

    return scores


def _log_mean_loss(logger: TensorBoardLogger,
                   losses: List[float],
                   step: int,
                   prefix: str = "") -> None:
    """
    Log the mean of accumulated losses and clear the buffer.

    :param logger: TensorBoard logger instance
    :param losses: List of scalar loss values
    :param step: Current training step
    :param prefix: Tag prefix (e.g. 'train/' or 'val/')
    """
    if losses:
        logger.log_scalar(f"{prefix}loss", float(np.mean(losses)), step)


def _ensure_directories(cfg: argparse.Namespace) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)


def _save_config(cfg: argparse.Namespace) -> None:
    """Persist training configuration as JSON for reproducibility."""
    config_path = os.path.join(cfg.log_dir, "config.json")
    with open(config_path, "w") as fp:
        json.dump(vars(cfg), fp, indent=2)


def run_training(cfg: argparse.Namespace) -> None:
    """
    Main training loop with validation and model checkpointing.

    :param cfg: Parsed command-line arguments (see argparse setup below)
    """
    _ensure_directories(cfg)
    _save_config(cfg)

    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    train_loader, val_loader = _create_dataloaders(cfg)
    phase_loaders = {"train": train_loader, "valid": val_loader}

    model = UNetModel(in_channels=SegDataset.num_input_channels,
                      out_channels=SegDataset.num_output_channels)
    model.to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    criterion = SoftDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    best_val_dice = 0.0

    logger = TensorBoardLogger(cfg.log_dir)
    running_train_loss: List[float] = []
    running_val_loss: List[float] = []

    global_step = 0

    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        for phase in ("train", "valid"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            val_predictions: List[np.ndarray] = []
            val_targets: List[np.ndarray] = []

            for batch_idx, (inputs, targets) in enumerate(phase_loaders[phase]):
                if phase == "train":
                    global_step += 1

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets)

                    if phase == "valid":
                        running_val_loss.append(batch_loss.item())

                        # Collect predictions for per-volume Dice evaluation
                        preds_np = outputs.detach().cpu().numpy()
                        val_predictions.extend(preds_np[s] for s in range(preds_np.shape[0]))

                        tgts_np = targets.detach().cpu().numpy()
                        val_targets.extend(tgts_np[s] for s in range(tgts_np.shape[0]))

                        # Visualize sample predictions
                        is_vis_epoch = (epoch % cfg.vis_frequency == 0) or (epoch == cfg.epochs - 1)
                        if is_vis_epoch and batch_idx * cfg.batch_size < cfg.n_vis_images:
                            vis_tag = f"image/{batch_idx}"
                            n_remaining = cfg.n_vis_images - batch_idx * cfg.batch_size
                            
                            logger.log_image_batch(vis_tag,
                                                   compose_visualization(inputs, targets, outputs)[:n_remaining],
                                                   global_step)

                    if phase == "train":
                        running_train_loss.append(batch_loss.item())
                        batch_loss.backward()
                        optimizer.step()

                if phase == "train" and (global_step + 1) % 10 == 0:
                    _log_mean_loss(logger, running_train_loss, global_step, prefix="train/")
                    running_train_loss = []

            if phase == "valid":
                _log_mean_loss(logger, running_val_loss, global_step, prefix="val/")

                vol_dice_scores = _per_volume_dice(val_predictions,
                                                   val_targets,
                                                   val_loader.dataset.flat_index)
                
                mean_dice = float(np.mean(vol_dice_scores))
                logger.log_scalar("val/dice", mean_dice, global_step)

                # Checkpoint if improved
                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), ckpt_path)

                running_val_loss = []

    logger.close()
    print(f"Training complete. Best validation DSC: {best_val_dice:.4f}")

    
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for brain MRI FLAIR segmentation")

    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (default: cuda:0)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count (default: 4)")
    parser.add_argument("--n-vis-images", type=int, default=200, help="Max visualization images per epoch (default: 200)")
    parser.add_argument("--vis-frequency", type=int, default=10, help="Epoch interval for image logging (default: 10)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--log-dir", type=str, default="./tb_logs", help="TensorBoard log directory")
    parser.add_argument("--data-dir", type=str, default="./kaggle_3m", help="Root image directory")
    parser.add_argument("--image-size", type=int, default=256, help="Target spatial resolution (default: 256)")
    parser.add_argument("--aug-scale", type=float, default=0.05, help="Scale augmentation range (default: 0.05)")
    parser.add_argument("--aug-angle", type=float, default=15.0, help="Rotation augmentation range in degrees (default: 15)")
    
    return parser.parse_args()


if __name__ == "__main__":
    run_training(_parse_args())
