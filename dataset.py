"""
PyTorch Dataset for brain MRI FLAIR abnormality segmentation.

Reads per-patient TIF slices from disk, applies preprocessing
(crop, pad, resize, normalize), and serves (image, mask) tensor pairs.
"""

import os
import random
from typing import Optional, Tuple, List, Callable

import numpy as np
import numpy.typing as npt
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_to_content, pad_to_square, resize_volume, normalize_intensity


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
                 seed: int = 42) -> None:
        """
        :param data_root: Root directory containing per-patient subdirectories with .tif slices
        :param transform: Optional augmentation callable operating on (image, mask) tuples
        :param resolution: Target spatial resolution after preprocessing
        :param split: One of 'all', 'train', 'validation'
        :param n_validation: Number of patients held out for validation
        :param seed: Random seed for reproducible train/val splitting
        """
        assert split in ("all", "train", "validation"), f"Unknown split: {split}"

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
                    msk_slices.append(imread(fpath, as_gray=True))
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
           
            floor = area_per_slice.sum() * 0.1 / len(area_per_slice)
            weights = (area_per_slice + floor) / (area_per_slice.sum() * 1.1)
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
    
    def _slice_index_from_filename(fname: str) -> int:
        return int(fname.split(".")[-2].split("_")[4])
