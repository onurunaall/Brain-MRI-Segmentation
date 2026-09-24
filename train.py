import argparse
import csv
import json
import os
import time
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
from network import ModelFactory
from utils import compose_visualization, dice_similarity_coefficient, Reproducibility


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
                          transform=augmentation,
                          n_validation=cfg.n_validation,
                          seed=cfg.split_seed,
                          fold=cfg.fold,
                          n_folds=cfg.n_folds)

    val_ds = SegDataset(data_root=cfg.data_dir,
                        split="validation",
                        resolution=cfg.image_size,
                        n_validation=cfg.n_validation,
                        seed=cfg.split_seed,
                        fold=cfg.fold,
                        n_folds=cfg.n_folds)

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
                              worker_init_fn=_seed_worker,
                              pin_memory=True)
    
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
    print(f"  Data split   : {split_desc}")
    print(f"  Train seed   : {cfg.seed}")
    print(f"  Device       : {device}")
    print(f"  Train set    : {len(train_loader.dataset.patient_ids)} patients, "
          f"{len(train_loader.dataset)} slices, {len(train_loader)} batches/epoch")
    print(f"  Val set      : {len(val_loader.dataset.patient_ids)} patients, "
          f"{len(val_loader.dataset)} slices, {len(val_loader)} batches/epoch")
    print(f"  Schedule     : {cfg.epochs} epochs, batch size {cfg.batch_size}, "
          f"Adam lr {cfg.lr:g} with cosine decay")
    print(f"  Checkpoints  : {cfg.checkpoint_dir}")
    print(f"  TensorBoard  : {cfg.log_dir}")
    print("===================================================")


def _print_metric_legend() -> None:
    """Explain the columns of the per-epoch summary line."""
    print("\nPer-epoch line columns:")
    print("  train loss : mean soft-Dice loss (1 - Dice) over this epoch's training batches;")
    print("               computed with augmentation on, so not directly comparable to val loss. Lower = better.")
    print("  val loss   : same loss on the validation slices (no augmentation). Lower = better.")
    print("  val Dice   : per-patient 3D Dice on the validation set after thresholding at 0.5")
    print("               and keeping the largest connected component, averaged over patients.")
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
    Reproducibility.seed_everything(cfg.seed)

    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    train_loader, val_loader = _create_dataloaders(cfg)
    phase_loaders = {"train": train_loader, "valid": val_loader}

    model = ModelFactory.create(
        cfg.arch,
        in_channels=SegDataset.num_input_channels,
        out_channels=SegDataset.num_output_channels
    )
    
    model.to(device)
    base_model = model  # uncompiled handle: its state_dict has no "_orig_mod." prefix
    n_params = sum(p.numel() for p in base_model.parameters())

    _print_run_header(cfg, device, n_params, train_loader, val_loader)
    _print_metric_legend()

    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    criterion = SoftDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_val_dice = 0.0

    logger = TensorBoardLogger(cfg.log_dir)
    running_train_loss: List[float] = []
    running_val_loss: List[float] = []

    global_step = 0
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
        epoch_train_losses: List[float] = []   # full-epoch buffer; running_train_loss is reset every 10 steps

        for phase in ("train", "valid"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            val_predictions: List[np.ndarray] = []
            val_targets: List[np.ndarray] = []

            # Wrap the phase DataLoader with tqdm
            pbar = tqdm(
                phase_loaders[phase],
                desc=f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} {phase.capitalize():>5}",
                leave=False,
                dynamic_ncols=True
            )

            for batch_idx, (inputs, targets) in enumerate(pbar):
                if phase == "train":
                    global_step += 1

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.amp.autocast(device_type=device.type,
                                            enabled=(device.type == "cuda")):
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, targets)

                    if phase == "valid":
                        running_val_loss.append(batch_loss.item())

                        preds_np = outputs.detach().cpu().numpy()
                        val_predictions.extend(preds_np[s] for s in range(preds_np.shape[0]))

                        tgts_np = targets.detach().cpu().numpy()
                        val_targets.extend(tgts_np[s] for s in range(tgts_np.shape[0]))

                        is_vis_epoch = (epoch % cfg.vis_frequency == 0) or (epoch == cfg.epochs - 1)
                        if is_vis_epoch and batch_idx * cfg.batch_size < cfg.n_vis_images:
                            vis_tag = f"image/{batch_idx}"
                            n_remaining = cfg.n_vis_images - batch_idx * cfg.batch_size

                            logger.log_image_batch(vis_tag,
                                                   compose_visualization(inputs, targets, outputs)[:n_remaining],
                                                   global_step)

                    if phase == "train":
                        running_train_loss.append(batch_loss.item())
                        epoch_train_losses.append(batch_loss.item())
                        scaler.scale(batch_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Update live progress bar display
                current_loss = batch_loss.item()
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

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
                is_best = ""
                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    best_epoch = epoch
                    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                    torch.save(base_model.state_dict(), ckpt_path)
                    is_best = " (NEW BEST)"

                # Print epoch summary line
                val_loss_avg = float(np.mean(running_val_loss)) if running_val_loss else 0.0
                train_loss_avg = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
                best_desc = f"{best_val_dice:.4f} (ep {best_epoch + 1:03d})" if best_epoch >= 0 else "n/a"
                epoch_secs = time.perf_counter() - epoch_start
                print(f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
                      f"train loss {train_loss_avg:.4f} | val loss {val_loss_avg:.4f} | "
                      f"val Dice {mean_dice:.4f} | best {best_desc} | "
                      f"lr {epoch_lr:.2e} | time {epoch_secs:.1f}s{is_best}")

                with open(history_path, "a", newline="") as fp:
                    csv.writer(fp).writerow([epoch + 1, train_loss_avg, val_loss_avg, mean_dice,
                                             epoch_lr, epoch_secs, int(bool(is_best))])

                running_val_loss = []
        
        scheduler.step()
        logger.log_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    logger.close()

    elapsed = time.perf_counter() - train_start
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    summary = {"arch": cfg.arch,
               "fold": cfg.fold,
               "seed": cfg.seed,
               "epochs": cfg.epochs,
               "best_val_dice": best_val_dice,
               "best_epoch": best_epoch,
               "train_seconds_total": elapsed,
               "seconds_per_epoch": elapsed / cfg.epochs,
               "peak_train_memory_mb": peak_mem_mb,
               "n_params": n_params,
               "train_patients": train_loader.dataset.patient_ids,
               "val_patients": val_loader.dataset.patient_ids}

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
    parser.add_argument("--arch", type=str, default="unet", choices=ModelFactory.available(), help="Model architecture (default: unet)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed: weight init, sampling, augmentation (default: 0)")
    parser.add_argument("--fold", type=int, default=None, help="Test fold index for K-fold CV (default: None = old 100/10 split)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-validation", type=int, default=10, help="Validation patients for checkpoint selection (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Patient split seed; keep fixed across runs (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(_parse_args())
