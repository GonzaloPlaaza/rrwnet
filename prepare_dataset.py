#!/usr/bin/env python3
"""
Prepare Artery/Vein datasets for RRWNet training.

This script:
1. Reads the CSV file with dataset paths.
2. Copies images, arteries, veins, and fov masks to rrwnet/data/
3. Renames files to include dataset/subject prefix to ensure uniqueness.
4. Merges artery and vein masks into RRWNet RGB format:
   - Red channel: arteries
   - Green channel: veins
   - Blue channel: union of arteries and veins
"""

import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from skimage import io

# ------------------------
# CONFIG
# ------------------------
CSV_PATH = Path("../Artery_Vein_Segmentation/data_splits/artery_vein_segmentation/full_tr_av_segmentation.csv")
BASE_DATASET_DIR = Path("../Artery_Vein_Segmentation")
RRWNET_DATA_DIR = Path("./data")  # This will hold rrwnet-ready data

# Create folder structure
(RRWNET_DATA_DIR / "images").mkdir(parents=True, exist_ok=True)
(RRWNET_DATA_DIR / "labels").mkdir(parents=True, exist_ok=True)
(RRWNET_DATA_DIR / "masks").mkdir(parents=True, exist_ok=True)

# READ CSV
df = pd.read_csv(CSV_PATH)

print(f"Found {len(df)} images in CSV.")

# COPY AND PROCESS
for idx, row in df.iterrows():
    subject_id = row['unique_subject_id']  # e.g., datasetname_sub_0001
    dataset = row['dataset']  # e.g., DRIVE, STARE, etc.
    image_path = Path(row['image_path'])
    arteries_path = Path(row['arteries_path'])
    veins_path = Path(row['veins_path'])
    fov_path = Path(row['fov_path']) if 'fov_path' in row else None

    #Generate unique filenames
    img_filename = f"{dataset}_{image_path.name}"
    label_filename = f"{dataset}_{image_path.name}"  # same name for label
    fov_filename = f"{dataset}_{image_path.name}" if fov_path else None

    #Copy image
    src_img = BASE_DATASET_DIR / image_path
    dst_img = RRWNET_DATA_DIR / "images" / img_filename
    shutil.copy(src_img, dst_img)

    # Read artery and vein masks
    arteries = io.imread(BASE_DATASET_DIR / arteries_path)
    veins = io.imread(BASE_DATASET_DIR / veins_path)

    # Make sure masks are binary {0,255}
    arteries = (arteries > 0).astype(np.uint8) * 255
    veins = (veins > 0).astype(np.uint8) * 255

    # Merge into RRWNet format
    rrwnet_label = np.zeros((*arteries.shape[:2], 3), dtype=np.uint8)
    rrwnet_label[:, :, 0] = arteries          # Red = arteries
    rrwnet_label[:, :, 1] = veins             # Green = veins
    #Blue = vessels (union of arteries and veins)
    rrwnet_label[:, :, 2] = arteries | veins

    # Save label
    io.imsave(RRWNET_DATA_DIR / "labels" / label_filename, rrwnet_label)

    # Copy FOV mask if exists
    if fov_path:
        src_fov = BASE_DATASET_DIR / fov_path
        dst_fov = RRWNET_DATA_DIR / "masks" / fov_filename
        shutil.copy(src_fov, dst_fov)

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(df)} images...")

print("Dataset preparation complete!")
