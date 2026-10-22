"""
PyTorch Dataset for brain MRI FLAIR abnormality segmentation.

A patient is a folder of per-slice TIF files (Kaggle LGG layout: <patient>_<n>.tif and <patient>_<n>_mask.tif).
The functions below find patients, split them into train / validation / test, and read and preprocess one patient;
MRISegmentationDataset serves the slices of a given list of patients as (image, mask) tensor pairs.
"""

import os
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from skimage.io import imread
from torch.utils.data import Dataset

from mriseg.utils import VolumeGeometry, content_box, pad_to_square, resize_volume, normalize_intensity

# Slice number and mask marker at the end of a file name: TCGA_CS_4941_19960909_12.tif / ..._12_mask.tif
SLICE_FILE = re.compile(r"_(\d+)(_mask)?\.tif$", re.IGNORECASE)


def find_patients(data_root: str) -> Dict[str, str]:
    """
    Patient folders under data_root: every folder holding image slices, keyed by folder name (the patient ID).

    A second folder with the same name and the same files (e.g. a second copy of the dataset) is ignored; the first
    one in sorted path order is used.

    :param data_root: Root directory, searched recursively
    :return: Dict patient_id -> folder path
    :raises ValueError: if two folders with the same name hold different files
    """
    folders: Dict[str, str] = {}
    for dirpath, _, filenames in sorted(os.walk(data_root)):
        tif_files = sorted(name for name in filenames if name.lower().endswith(".tif"))
        if not any("mask" not in name.lower() for name in tif_files):
            continue
        pid = os.path.basename(dirpath)
        if pid not in folders:
            folders[pid] = dirpath
        elif sorted(name for name in os.listdir(folders[pid]) if name.lower().endswith(".tif")) != tif_files:
            raise ValueError(f"Two different folders are named {pid}: {folders[pid]} and {dirpath}; "
                             f"point --data-dir at one copy of the data")
    return folders


def kfold_split(patient_ids: List[str],
                n_folds: int,
                fold: int,
                n_validation: int,
                seed: int) -> Dict[str, List[str]]:
    """
    Deterministic patient-level K-fold split.

    All patient IDs are shuffled once with `seed` and dealt into `n_folds` folds.
    Fold `fold` is the test set. From the remaining patients, `n_validation` are
    drawn (with a fold-specific seed) as the validation set used for checkpoint
    selection; the rest are training patients. The same `seed` gives the same
    folds for every architecture and every training seed.

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


def split_patients(patient_ids: List[str],
                   fold: Optional[int],
                   n_folds: int,
                   n_validation: int,
                   seed: int) -> Dict[str, List[str]]:
    """
    Patient-level split into 'train', 'validation' and 'test' lists (sorted).

    With `fold` this is kfold_split. Without, it is the single split: `n_validation` random patients for validation,
    the rest for training, and no test patients.

    :param patient_ids: All patient identifiers
    :param fold: Test fold index, or None for the single split
    :param n_folds: Number of folds when fold is given
    :param n_validation: Number of validation patients
    :param seed: Split seed
    :return: Dict with keys 'train', 'validation', 'test'
    """
    if fold is not None:
        return kfold_split(patient_ids, n_folds, fold, n_validation, seed)

    ids = sorted(patient_ids)
    val_ids = random.Random(seed).sample(ids, k=n_validation)
    return {"train": sorted(set(ids) - set(val_ids)), "validation": sorted(val_ids), "test": []}


def slice_files(folder: str) -> Tuple[List[str], List[str]]:
    """
    Image and mask file names of one patient folder, in slice order. Images and masks pair up by slice number rather
    than by position in a list, so a missing file cannot shift the masks onto the wrong slices.

    :param folder: Patient folder
    :return: (image files, mask files); the mask list is empty for a patient without masks (a new patient)
    :raises ValueError: on unparseable names, two files with the same slice number, or images and masks that do not
                        pair up
    """
    images: Dict[int, str] = {}
    masks: Dict[int, str] = {}
    for name in os.listdir(folder):
        if not name.lower().endswith(".tif"):
            continue
        match = SLICE_FILE.search(name)
        if match is None:
            raise ValueError(f"Cannot parse the slice number of '{name}' in {folder}")
        files = masks if match.group(2) else images
        index = int(match.group(1))
        if index in files:
            raise ValueError(f"'{files[index]}' and '{name}' have the same slice number {index}")
        files[index] = name

    if masks and masks.keys() != images.keys():
        missing = sorted(images.keys() - masks.keys())
        extra = sorted(masks.keys() - images.keys())
        raise ValueError(f"{folder}: slices {missing} have no *_mask.tif file and masks {extra} have no image; "
                         f"images and masks must come in pairs")

    order = sorted(images)
    return [images[i] for i in order], [masks[i] for i in order] if masks else []


def read_mask(path: str) -> npt.NDArray[np.float32]:
    """A *_mask.tif as float32 values 0 / 1 (the files store 0 / 255)."""
    mask = imread(path, as_gray=True)
    if mask.max() > 1:
        mask = mask / 255.0
    return np.asarray(mask, dtype=np.float32)


def preprocess(images: npt.NDArray,
               masks: npt.NDArray,
               resolution: int) -> Tuple[npt.NDArray, npt.NDArray, VolumeGeometry]:
    """
    One patient's slices -> model input: drop the first and last slice, crop to the head (content_box), pad to a
    square, resize to resolution x resolution and normalize the intensities.

    :param images: (Z, H, W, 3) slices as read from disk
    :param masks: (Z, H, W) masks of 0 / 1
    :param resolution: Side of the square grid after resizing
    :return: (volume (Z', R, R, 3), mask (Z', R, R), geometry for mapping grid masks back onto the slices)
    """
    vol, seg = images[1:-1], masks[1:-1]
    z, y, x = box = content_box(vol)
    geometry = VolumeGeometry(n_slices=len(images),
                              height=vol.shape[1],
                              width=vol.shape[2],
                              z_start=z.start + 1,  # + 1 for the dropped first slice
                              y_start=y.start,
                              x_start=x.start,
                              crop_slices=z.stop - z.start,
                              crop_height=y.stop - y.start,
                              crop_width=x.stop - x.start,
                              grid_size=resolution)
    vol, seg = resize_volume(pad_to_square((vol[box], seg[box])), target_size=resolution)
    return normalize_intensity(vol), seg, geometry


def slice_sampling_weights(seg: npt.NDArray) -> npt.NDArray:
    """
    Sampling weights of one patient's slices for WeightedRandomSampler; they sum to 1, so every patient is equally
    likely. Half of the weight follows tumour area, the other half is spread evenly over the slices, so tumour-free
    slices (about 60% of the data) are about 30% of training samples. A patient without tumour gets uniform weights.

    :param seg: (Z, H, W) mask volume
    :return: (Z,) weights
    """
    area_per_slice = seg.sum(axis=(1, 2))
    floor = max(area_per_slice.sum(), 1.0) / len(area_per_slice)
    weights: npt.NDArray = (area_per_slice + floor) / (area_per_slice + floor).sum()
    return weights


class MRISegmentationDataset(Dataset):
    """
    Serves the slices of the given patients as (image, mask) tensor pairs. Each image holds the slice itself plus
    `context_slices` neighbouring slices on each side, stacked along the channel axis (2.5D input).
    """

    # Channel configuration for the U-Net: each slice has pre-contrast, FLAIR and post-contrast channels
    channels_per_slice: int = 3
    num_output_channels: int = 1

    @classmethod
    def num_input_channels(cls, context_slices: int) -> int:
        """Model input channels: the slice itself plus `context_slices` neighbours on each side (2.5D input)."""
        return cls.channels_per_slice * (2 * context_slices + 1)

    @classmethod
    def flair_channel(cls, context_slices: int) -> int:
        """Input channel holding the FLAIR of the slice being segmented (the middle channel of its three)."""
        return cls.channels_per_slice * context_slices + 1

    def __init__(self,
                 data_root: str,
                 patient_ids: Sequence[str],
                 resolution: int = 256,
                 context_slices: int = 0,
                 transform: Optional[Callable] = None,
                 require_masks: bool = True) -> None:
        """
        :param data_root: Root directory containing per-patient subdirectories with .tif slices
        :param patient_ids: Patients to serve, in this order (see split_patients and find_patients)
        :param resolution: Target spatial resolution after preprocessing
        :param context_slices: Neighbouring slices added on each side as extra channels (0 = plain 2D input).
                               Neighbours outside the volume are filled with the volume's background value.
        :param transform: Optional augmentation callable operating on (image, mask) tuples
        :param require_masks: If False, patients without *_mask.tif files are allowed (new patients to predict);
                              their masks are all zero and has_masks is False
        """
        if context_slices < 0:
            raise ValueError(f"context_slices must be >= 0, got {context_slices}")

        folders = find_patients(data_root)
        missing = [pid for pid in patient_ids if pid not in folders]
        if missing:
            raise ValueError(f"{len(missing)} of the requested patients are not in {data_root} (e.g. {missing[0]})")

        self.patient_ids = list(patient_ids)
        self.context_slices = context_slices
        self.transform = transform
        self.patient_dirs = {pid: folders[pid] for pid in self.patient_ids}
        # Original slice files of each patient in slice order, and where its crop lies in them (for predict.py)
        self.image_files: Dict[str, List[str]] = {}
        self.mask_files: Dict[str, List[str]] = {}
        self.geometry: Dict[str, VolumeGeometry] = {}
        self.volumes: List[Tuple[npt.NDArray, npt.NDArray]] = []

        print(f"[Dataset] Reading and preprocessing {len(self.patient_ids)} patients from {data_root}...")
        for pid in self.patient_ids:
            folder = self.patient_dirs[pid]
            image_files, mask_files = slice_files(folder)
            if require_masks and not mask_files:
                raise ValueError(f"{pid} has no *_mask.tif files; patients without masks can only be predicted "
                                 f"(predict.py --split all)")

            images = np.array([imread(os.path.join(folder, name)) for name in image_files])
            if mask_files:
                masks = np.array([read_mask(os.path.join(folder, name)) for name in mask_files])
            else:
                masks = np.zeros(images.shape[:3], dtype=np.float32)

            vol, seg, self.geometry[pid] = preprocess(images, masks, resolution)
            self.image_files[pid], self.mask_files[pid] = image_files, mask_files
            self.volumes.append((vol, seg[..., np.newaxis]))

        # False when any patient has no masks: their all-zero masks are placeholders, not ground truth
        self.has_masks = all(self.mask_files.values())
        # Per-channel background value of each volume, used for context slices beyond the volume's ends
        self.backgrounds = [vol.min(axis=(0, 1, 2)) for vol, _ in self.volumes]
        # Flat index -> (patient_idx, slice_idx), and the matching slice sampling weights
        self.flat_index = [(p_idx, s_idx) for p_idx, (vol, _) in enumerate(self.volumes) for s_idx in range(len(vol))]
        self.sample_weights: List[float] = np.concatenate(
            [slice_sampling_weights(seg[..., 0]) for _, seg in self.volumes]).tolist()

        n_tumour_slices = sum(int(seg.reshape(len(seg), -1).any(axis=1).sum()) for _, seg in self.volumes)
        print(f"[Dataset] ready — {len(self.patient_ids)} patients, "
              f"{len(self.flat_index)} slices ({n_tumour_slices} contain tumour)")

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_idx, s_idx = self.flat_index[idx]

        img_slice = self._with_context(p_idx, s_idx)
        msk_slice = self.volumes[p_idx][1][s_idx]

        if self.transform is not None:
            img_slice, msk_slice = self.transform((img_slice, msk_slice))

        img_slice = img_slice.transpose(2, 0, 1)
        msk_slice = msk_slice.transpose(2, 0, 1)

        img_tensor = torch.from_numpy(img_slice.astype(np.float32))
        msk_tensor = torch.from_numpy(msk_slice.astype(np.float32))

        return img_tensor, msk_tensor

    def original_masks(self, pid: str) -> npt.NDArray[np.bool_]:
        """
        A patient's ground truth exactly as on disk: every slice at its original size, before any cropping or
        resizing (for scoring predictions in original pixels).

        :param pid: Patient ID of a patient with masks
        :return: Boolean array of shape (n_slices, height, width)
        """
        folder = self.patient_dirs[pid]
        return np.array([read_mask(os.path.join(folder, name)) for name in self.mask_files[pid]]) > 0.5

    def _with_context(self, p_idx: int, s_idx: int) -> npt.NDArray:
        """
        Slice s_idx of patient p_idx with its neighbours stacked along the channel axis.

        :return: (H, W, channels_per_slice * (2 * context_slices + 1)) array, neighbours in slice order
        """
        vol = self.volumes[p_idx][0]
        if self.context_slices == 0:
            return vol[s_idx]

        stacked = []
        for z in range(s_idx - self.context_slices, s_idx + self.context_slices + 1):
            if 0 <= z < vol.shape[0]:
                stacked.append(vol[z])
            else:
                stacked.append(np.broadcast_to(self.backgrounds[p_idx], vol.shape[1:]))
        return np.concatenate(stacked, axis=-1)
