"""Download the LGG MRI segmentation dataset from Kaggle into ./kaggle_3m."""

import os
import shutil

import kagglehub

DESTINATION = "./kaggle_3m"


def main() -> None:
    if os.path.exists(DESTINATION):
        raise SystemExit(f"{DESTINATION} already exists; delete or rename it yourself if you want a fresh copy")

    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    print(f"Data downloaded to: {path}")
    print(f"Copying data to {DESTINATION}")
    shutil.copytree(path, DESTINATION)


if __name__ == "__main__":
    main()
