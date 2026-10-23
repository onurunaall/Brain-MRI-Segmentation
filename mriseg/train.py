import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from mriseg.dataset import MRISegmentationDataset as SegDataset, find_patients, split_patients
from mriseg.tb_logger import TensorBoardLogger
from mriseg.losses import DiceBCELoss
from mriseg.augmentations import build_augmentation_pipeline
from mriseg.network import available_architectures, create_model
from mriseg.inference import COMPILE_MODES, compile_model, save_checkpoint
from mriseg.evaluation import raw_dice_per_patient
from mriseg.utils import compose_visualization, seed_everything


def _create_dataloaders(cfg: argparse.Namespace, splits: Dict[str, List[str]]) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    :param cfg: Parsed arguments
    :param splits: Patient lists from split_patients
    :return: (train_loader, val_loader) tuple
    """
    augmentation = build_augmentation_pipeline(scale_range=cfg.aug_scale,
                                               rotation_deg=cfg.aug_angle,
                                               flip_probability=0.5,
                                               brightness_range=cfg.aug_brightness,
                                               gamma_range=cfg.aug_gamma)

    train_ds = SegDataset(cfg.data_dir,
                          splits["train"],
                          resolution=cfg.image_size,
                          context_slices=cfg.context_slices,
                          transform=augmentation)

    val_ds = SegDataset(cfg.data_dir,
                        splits["validation"],
                        resolution=cfg.image_size,
                        context_slices=cfg.context_slices)

    # Use WeightedRandomSampler to bias towards slices with more foreground
    # (PyTorch seeds Python, NumPy and torch in every worker, so no worker_init_fn is needed.)
    # persistent_workers keeps the workers between epochs: on Windows / macOS every new worker gets its own copy of
    # the whole preprocessed dataset, which would otherwise be re-sent each epoch.
    train_sampler = WeightedRandomSampler(weights=train_ds.sample_weights,
                                          num_samples=len(train_ds),
                                          replacement=True)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              sampler=train_sampler,
                              drop_last=True,
                              num_workers=cfg.num_workers,
                              persistent_workers=cfg.num_workers > 0,
                              pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            drop_last=False,
                            num_workers=cfg.num_workers,
                            persistent_workers=cfg.num_workers > 0)

    return train_loader, val_loader


def _segmentation_dataset(loader: DataLoader) -> SegDataset:
    """The dataset behind a loader built by _create_dataloaders (typed for patient_ids / flat_index access)."""
    return cast(SegDataset, loader.dataset)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scaler: torch.amp.GradScaler,
                    device: torch.device,
                    logger: TensorBoardLogger,
                    global_step: int,
                    desc: str) -> Tuple[float, int]:
    """
    One pass over the training data (fp16 autocast on CUDA). The mean loss of every 10 steps goes to TensorBoard.

    :param global_step: Optimizer steps taken before this epoch
    :param desc: Progress bar label
    :return: (mean training loss of the epoch, global step after the epoch)
    """
    model.train()
    epoch_losses: List[float] = []
    recent_losses: List[float] = []  # since the last TensorBoard point

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for inputs, targets in pbar:
        global_step += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(loss.item())
        recent_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (global_step + 1) % 10 == 0:
            logger.log_scalar("train/loss", float(np.mean(recent_losses)), global_step)
            recent_losses = []

    return (float(np.mean(epoch_losses)) if epoch_losses else float("nan")), global_step


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             desc: str) -> Tuple[float, float, List[np.ndarray]]:
    """
    Validation loss and Dice. The Dice is per patient volume on the preprocessed grid, thresholded at 0.5 and without
    post-processing: largest-connected-component filtering can swap in a false-positive blob and make the score jump.

    :param desc: Progress bar label
    :return: (mean loss, mean per-patient Dice, per-slice probabilities (1, H, W) in dataset order)
    """
    model.eval()
    losses: List[float] = []
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for inputs, masks in pbar:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(inputs)
            loss = criterion(logits, masks)

        losses.append(loss.item())
        probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy())  # the model returns logits
        targets.extend(masks.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    dataset = _segmentation_dataset(loader)
    dice = raw_dice_per_patient(probabilities, targets, dataset.flat_index, dataset.patient_ids)
    return float(np.mean(losses)), float(np.mean(list(dice.values()))), probabilities


def _log_overlays(logger: TensorBoardLogger,
                  dataset: SegDataset,
                  probabilities: List[np.ndarray],
                  n_images: int,
                  step: int) -> None:
    """TensorBoard images of the first n_images validation slices: prediction (red) and ground truth (green)."""
    flair_channel = SegDataset.flair_channel(dataset.context_slices)
    overlays = []
    for idx in range(min(n_images, len(dataset))):
        image, mask = dataset[idx]
        probability = torch.from_numpy(probabilities[idx])
        overlays.extend(compose_visualization(image[None], mask[None], probability[None], channel_idx=flair_channel))
    logger.log_image_batch("image", overlays, step)


def _ensure_directories(cfg: argparse.Namespace) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)


def _save_config(cfg: argparse.Namespace) -> None:
    """Save the command-line settings as JSON, for people reading the run folder later."""
    config_path = os.path.join(cfg.log_dir, "config.json")
    with open(config_path, "w") as fp:
        json.dump(vars(cfg), fp, indent=2)


def _print_run_header(cfg: argparse.Namespace,
                      device: torch.device,
                      n_params: int,
                      train_loader: DataLoader,
                      val_loader: DataLoader) -> None:
    """Print a one-time summary of what this run trains on and how."""
    if cfg.fold is None:
        split_desc = f"single split (no test fold), split seed {cfg.split_seed}"
    else:
        split_desc = (f"fold {cfg.fold} of {cfg.n_folds} (folds numbered 0-{cfg.n_folds - 1}) "
                      f"held out as test set, split seed {cfg.split_seed}")

    print("\n================ Run configuration ================")
    print(f"  Model        : {cfg.arch} ({n_params / 1e6:.2f} M parameters)")
    print(f"  Input        : {SegDataset.num_input_channels(cfg.context_slices)} channels "
          f"(the slice and {cfg.context_slices} neighbour slice(s) on each side)")
    print(f"  Data split   : {split_desc}")
    print(f"  Train seed   : {cfg.seed}")
    print(f"  Device       : {device}")
    print(f"  torch.compile: {cfg.compile}")
    train_ds = _segmentation_dataset(train_loader)
    val_ds = _segmentation_dataset(val_loader)
    print(f"  Train set    : {len(train_ds.patient_ids)} patients, "
          f"{len(train_ds)} slices, {len(train_loader)} batches/epoch")
    print(f"  Val set      : {len(val_ds.patient_ids)} patients, "
          f"{len(val_ds)} slices, {len(val_loader)} batches/epoch")
    print(f"  Schedule     : {cfg.epochs} epochs, batch size {cfg.batch_size}, "
          f"Adam lr {cfg.lr:g} with cosine decay")
    print(f"  Checkpoints  : {cfg.checkpoint_dir}")
    print(f"  TensorBoard  : {cfg.log_dir}")
    print("===================================================")


def _print_metric_legend() -> None:
    """Explain the columns of the per-epoch summary line."""
    print("\nPer-epoch line columns:")
    print("  train loss : mean soft-Dice loss (1 - Dice) + binary cross-entropy over this epoch's training batches;")
    print("               computed with augmentation on, so not directly comparable to val loss. Lower = better.")
    print("  val loss   : same loss on the validation slices (no augmentation). Lower = better.")
    print("  val Dice   : per-patient 3D Dice on the validation set after thresholding at 0.5, without")
    print("               post-processing (predict.py picks that later), averaged over patients.")
    print("               0 = no overlap, 1 = perfect. Higher = better. Used to pick the checkpoint.")
    print("  best       : highest val Dice so far and the epoch it was reached.")
    print("  NEW BEST   : val Dice improved -> model saved to best_model.pt (the model that gets tested).")
    print("  lr         : learning rate used during this epoch.")
    print("  time       : wall-clock seconds for this epoch (train + validation).\n")


def run_training(cfg: argparse.Namespace) -> None:
    """
    Main training loop with validation and model checkpointing.

    :param cfg: Parsed command-line arguments (see argparse setup below)
    """
    _ensure_directories(cfg)
    _save_config(cfg)
    seed_everything(cfg.seed)

    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    splits = split_patients(sorted(find_patients(cfg.data_dir)), cfg.fold, cfg.n_folds, cfg.n_validation,
                            cfg.split_seed)
    train_loader, val_loader = _create_dataloaders(cfg, splits)

    model = create_model(
        cfg.arch,
        in_channels=SegDataset.num_input_channels(cfg.context_slices),
        out_channels=SegDataset.num_output_channels
    )

    model.to(device)
    base_model = model  # uncompiled handle: its state_dict has no "_orig_mod." prefix
    n_params = sum(p.numel() for p in base_model.parameters())

    _print_run_header(cfg, device, n_params, train_loader, val_loader)
    _print_metric_legend()

    model = compile_model(model, cfg.compile)

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Saved inside best_model.pt: the single record of the model's settings and patients. predict.py and
    # export_models.py read it, so they can never score the model on its own training patients.
    checkpoint_meta = {"arch": cfg.arch,
                       "context_slices": cfg.context_slices,
                       "image_size": cfg.image_size,
                       "fold": cfg.fold,
                       "n_folds": cfg.n_folds,
                       "split_seed": cfg.split_seed,
                       "n_validation": cfg.n_validation,
                       "seed": cfg.seed,
                       "train_patients": splits["train"],
                       "val_patients": splits["validation"],
                       "test_patients": splits["test"]}

    logger = TensorBoardLogger(cfg.log_dir)
    global_step = 0
    best_val_dice = 0.0
    best_epoch = -1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()

    # Per-epoch history for later comparison plots (compare.py); overwritten if the run is restarted
    history_path = os.path.join(cfg.log_dir, "history.csv")
    with open(history_path, "w", newline="") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr", "seconds", "new_best"])

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        epoch_lr = scheduler.get_last_lr()[0]  # LR in effect for this epoch (stepped at epoch end)
        desc = f"Epoch {epoch + 1:03d}/{cfg.epochs:03d}"

        train_loss, global_step = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, logger,
                                                  global_step, f"{desc} Train")
        val_loss, val_dice, probabilities = validate(model, val_loader, criterion, device, f"{desc} Valid")

        logger.log_scalar("val/loss", val_loss, global_step)
        logger.log_scalar("val/dice", val_dice, global_step)
        if epoch % cfg.vis_frequency == 0 or epoch == cfg.epochs - 1:
            _log_overlays(logger, _segmentation_dataset(val_loader), probabilities, cfg.n_vis_images, global_step)

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice, best_epoch = val_dice, epoch
            save_checkpoint(os.path.join(cfg.checkpoint_dir, "best_model.pt"), base_model.state_dict(),
                            checkpoint_meta)

        epoch_secs = time.perf_counter() - epoch_start
        best_desc = f"{best_val_dice:.4f} (ep {best_epoch + 1:03d})" if best_epoch >= 0 else "n/a"
        print(f"{desc} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f} | "
              f"best {best_desc} | lr {epoch_lr:.2e} | time {epoch_secs:.1f}s" + (" (NEW BEST)" if is_best else ""))

        with open(history_path, "a", newline="") as fp:
            csv.writer(fp).writerow([epoch + 1, train_loss, val_loss, val_dice, epoch_lr, epoch_secs, int(is_best)])

        scheduler.step()
        logger.log_scalar("train/lr", float(scheduler.get_last_lr()[0]), global_step)

    logger.close()

    elapsed = time.perf_counter() - train_start
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    summary = {"arch": cfg.arch,
               "fold": cfg.fold,
               "seed": cfg.seed,
               "epochs": cfg.epochs,
               "compile": cfg.compile,
               "best_val_dice": best_val_dice,
               "best_epoch": best_epoch,
               "train_seconds_total": elapsed,
               "seconds_per_epoch": elapsed / cfg.epochs,
               "peak_train_memory_mb": peak_mem_mb,
               "n_params": n_params}

    with open(os.path.join(cfg.log_dir, "train_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    # best_epoch is stored 0-based in train_summary.json; print it 1-based to match the epoch lines
    best_epoch_desc = f"epoch {best_epoch + 1}/{cfg.epochs}" if best_epoch >= 0 else "never improved above 0"
    print(f"Training complete in {elapsed / 60:.1f} min. "
          f"Best validation Dice: {best_val_dice:.4f} ({best_epoch_desc}) -> {cfg.checkpoint_dir}/best_model.pt")


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
    parser.add_argument("--aug-brightness", type=float, default=0.2,
                        help="Per-channel intensity factor range, e.g. 0.2 -> [0.8, 1.2]; 0 disables (default: 0.2)")
    parser.add_argument("--aug-gamma", type=float, default=0.4,
                        help="Per-channel gamma range g: gamma in [1/(1+g), 1+g]; 0 disables (default: 0.4)")
    parser.add_argument("--context-slices", type=int, default=1,
                        help="Neighbour slices on each side given as extra input channels (2.5D); 0 = plain 2D "
                             "(default: 1)")
    parser.add_argument("--arch", type=str, default="unet", choices=available_architectures(), help="Model architecture (default: unet)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed: weight init, sampling, augmentation (default: 0)")
    parser.add_argument("--fold", type=int, default=None, help="Test fold index for K-fold CV (default: None = old 100/10 split)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-validation", type=int, default=10, help="Validation patients for checkpoint selection (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Patient split seed; keep fixed across runs (default: 42)")
    parser.add_argument("--compile", type=str, nargs="?", const="default", default="default", choices=COMPILE_MODES,
                        help="torch.compile mode, or 'none' for eager PyTorch (default: default)")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(_parse_args())
