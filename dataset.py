"""
PyTorch Dataset for brain MRI FLAIR abnormality segmentation.
Reads per-patient TIF slices from disk, applies preprocessing (crop, pad, resize, normalize), and serves (image, mask) tensor pairs.
"""

import os
import random
from typing import Optional, Tuple, List, Callable, Dict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_to_content, pad_to_square, resize_volume, normalize_intensity


class PatientSplitter:
    """
    Deterministic patient-level K-fold split.

    All patient IDs are shuffled once with `seed` and dealt into `n_folds` folds.
    Fold `fold` is the test set. From the remaining patients, `n_validation` are
    drawn (with a fold-specific seed) as the validation set used for checkpoint
    selection; the rest are training patients. The same `seed` gives the same
    folds for every architecture and every training seed.
    """

    @staticmethod
    def kfold(patient_ids: List[str],
              n_folds: int,
              fold: int,
              n_validation: int,
              seed: int) -> Dict[str, List[str]]:
        """
        :param patient_ids: All patient identifiers
        :param n_folds: Number of folds (K)
        :param fold: Index of the test fold, 0 <= fold < n_folds
        :param n_validation: Number of validation patients taken from the non-test patients
        :param seed: Split seed (keep fixed across all runs you want to compare)
        :return: Dict with keys 'train', 'validation', 'test' mapping to sorted patient ID lists
        """
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")

        ids = sorted(patient_ids)
        random.Random(seed).shuffle(ids)

        folds = [ids[i::n_folds] for i in range(n_folds)]
        test_ids = set(folds[fold])
        remaining = [pid for pid in ids if pid not in test_ids]

        if n_validation >= len(remaining):
            raise ValueError(f"n_validation={n_validation} leaves no training patients")

        val_ids = set(random.Random(seed + 1 + fold).sample(remaining, k=n_validation))
        train_ids = [pid for pid in remaining if pid not in val_ids]

        return {"train": sorted(train_ids),
                "validation": sorted(val_ids),
                "test": sorted(test_ids)}


class MRISegmentationDataset(Dataset):
    """
    Dataset for MRI FLAIR abnormality segmentation.

    Loads patient volumes from a directory tree, splits them into
    train / validation subsets, and returns individual 2-D slices
    as (image_tensor, mask_tensor) pairs.
    """

    # Channel configuration for the U-Net
    num_input_channels: int = 3
    num_output_channels: int = 1

    def __init__(self,
                 data_root: str,
                 transform: Optional[Callable] = None,
                 resolution: int = 256,
                 split: str = "train",
                 n_validation: int = 10,
                 seed: int = 42,
                 fold: Optional[int] = None,
                 n_folds: int = 5) -> None:
        """
        :param data_root: Root directory containing per-patient subdirectories with .tif slices
        :param transform: Optional augmentation callable operating on (image, mask) tuples
        :param resolution: Target spatial resolution after preprocessing
        :param split: One of 'all', 'train', 'validation'
        :param n_validation: Number of patients held out for validation
        :param seed: Random seed for reproducible train/val splitting
        :param fold: If given, use K-fold splitting (PatientSplitter) and treat this fold as test
        :param n_folds: Number of folds when fold is given
        """
        assert split in ("all", "train", "validation", "test"), f"Unknown split: {split}"
        if split == "test" and fold is None:
            raise ValueError("split='test' requires fold to be set")

        patient_dirs: dict[str, str] = {}
        
        for dirpath, _, filenames in os.walk(data_root):
            tif_files = [f for f in filenames if ".tif" in f]
            has_images = any(f for f in tif_files if "mask" not in f)
            
            if has_images:
                pid = os.path.basename(dirpath)
                patient_dirs[pid] = dirpath

        all_patient_ids = sorted(patient_dirs)

        if split == "all":
            self.patient_ids = all_patient_ids
        elif fold is not None:
            splits = PatientSplitter.kfold(all_patient_ids, n_folds, fold, n_validation, seed)
            self.patient_ids = splits[split]
        else:
            random.seed(seed)
            val_ids = random.sample(all_patient_ids, k=n_validation)
            
            if split == "validation":
                self.patient_ids = val_ids
            else:
                self.patient_ids = sorted(set(all_patient_ids) - set(val_ids))

        #Load only the relevant patients' slices from disk
        print(f"[Dataset] Reading {split} images ({len(self.patient_ids)} patients) from {data_root}...")
        
        patient_volumes: dict[str, npt.NDArray] = {}
        patient_masks: dict[str, npt.NDArray] = {}

        for pid in self.patient_ids:
            dirpath = patient_dirs[pid]
            filenames = os.listdir(dirpath)

            try:
                tif_files = [name for name in filenames if name.lower().endswith(".tif")]
                tif_files.sort(key=self._slice_index_from_filename)
            except (IndexError, ValueError) as exc:
                print(f"[Dataset] WARNING: Skipping {pid} — could not parse filenames: {exc}")
                self.patient_ids = [p for p in self.patient_ids if p != pid]
                continue

            img_slices, msk_slices = [], []
            for fname in tif_files:
                fpath = os.path.join(dirpath, fname)
                if "mask" in fname:
                    mask = imread(fpath, as_gray=True)
                    if mask.max() > 1:
                        mask = (mask / 255.0).astype(np.float32)
                    msk_slices.append(mask)
                else:
                    img_slices.append(imread(fpath))

            if img_slices:
                # Drop first and last slices
                patient_volumes[pid] = np.array(img_slices[1:-1])
                patient_masks[pid] = np.array(msk_slices[1:-1])

        # Filter out patients that had no loadable slices
        self.patient_ids = [pid for pid in self.patient_ids if pid in patient_volumes]

        print(f"[Dataset] Preprocessing {split} volumes...")
        paired = [(patient_volumes[pid], patient_masks[pid]) for pid in self.patient_ids]
        paired = [crop_to_content(p) for p in paired]
        paired = [pad_to_square(p) for p in paired]

        print(f"[Dataset] Resizing {split} volumes to {resolution}×{resolution}...")
        paired = [resize_volume(p, target_size=resolution) for p in paired]

        paired = [(normalize_intensity(vol), seg) for vol, seg in paired]

        # Slice sampling weights (favour slices with more foreground)
        # These weights are per-slice probabilities suitable for WeightedRandomSampler.
        patient_slice_weights: List[npt.NDArray] = []
        
        for _, seg in paired:
            area_per_slice = seg.sum(axis=-1).sum(axis=-1)
           
            total = area_per_slice.sum()
            floor = total * 0.1 / max(len(area_per_slice), 1)
            weights = (area_per_slice + floor) / max(total * 1.1, 1e-6)
            patient_slice_weights.append(weights)

        self.volumes = [(vol, seg[..., np.newaxis]) for vol, seg in paired]

        print(f"[Dataset] {split} dataset ready — {len(self.patient_ids)} patients")

        # ── Stage 6: Build flat index → (patient_idx, slice_idx) mapping ──
        slices_per_patient = [vol.shape[0] for vol, _ in self.volumes]
        self.flat_index: List[Tuple[int, int]] = []
        
        for p_idx, n_sl in enumerate(slices_per_patient):
            self.flat_index.extend((p_idx, s_idx) for s_idx in range(n_sl))

        self.sample_weights: List[float] = []
        
        for p_idx, n_sl in enumerate(slices_per_patient):
            self.sample_weights.extend(patient_slice_weights[p_idx].tolist())

        self.transform = transform

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_idx, s_idx = self.flat_index[idx]

        vol, seg = self.volumes[p_idx]
        img_slice = vol[s_idx]
        msk_slice = seg[s_idx]

        if self.transform is not None:
            img_slice, msk_slice = self.transform((img_slice, msk_slice))
            
        img_slice = img_slice.transpose(2, 0, 1)
        msk_slice = msk_slice.transpose(2, 0, 1)

        img_tensor = torch.from_numpy(img_slice.astype(np.float32))
        msk_tensor = torch.from_numpy(msk_slice.astype(np.float32))

        return img_tensor, msk_tensor
    
    @staticmethod
    def _slice_index_from_filename(fname: str) -> int:
        try:
            return int(fname.split(".")[-2].split("_")[4])
        except (IndexError, ValueError):
            raise ValueError(f"Cannot parse slice index from '{fname}'")
