import kagglehub
import shutil
import os

path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
print(f"Data downloaded to: {path}")

destination = "./kaggle_3m"
print(f"Moving data to {destination}")

if os.path.exists(destination):
    shutil.rmtree(destination)

shutil.copytree(path, destination, dirs_exist_ok=True)