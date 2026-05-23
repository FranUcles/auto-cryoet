#!/usr/bin/env python
import numpy as np
import pandas as pd
import mrcfile
import logging
import coloredlogs
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input", type=str, help="Input cluster to transform in a MRC")
    input_group.add_argument("--inputDir", type=str, default=None, help="Input directory containing all the clusters to be saved together")
    parser.add_argument("--split", action="store_true", help="Save each class in a separate file")
    parser.add_argument("-o", "--outName", type=str, required=True, help="Output MRC filename")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Reference MRC filename")
    parser.add_argument("--outputDir", type=str, default=".", help="Output directory")
    return parser.parse_args()

def configure_logging(debug, quiet, no_logs):
    if no_logs:
        # Disable all logs, even CRITICAL
        logging.disable(logging.CRITICAL)
        return

    if quiet:
        level = logging.ERROR
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    coloredlogs.install(
        level=level,
        logger=logger,
        fmt='%(asctime)s [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
def save_mrc(data, voxel_size, output):
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
    
def open_mrc(input):
    with mrcfile.open(input, permissive=True) as mrc:
        data_original = mrc.data
        shape = data_original.shape
        voxel_size = mrc.voxel_size
    return data_original, shape, voxel_size

def fill_mrc(mrc_data, mrc_shape, coords, value):
    for z, y, x in coords:
        if 0 <= z < mrc_shape[0] and 0 <= y < mrc_shape[1] and 0 <= x < mrc_shape[2]:
            mrc_data[z, y, x] = value
    
def main(input, inputDir, reference, outName, separate=False, outputDir=".", debug=False, quiet=False, no_logs=False):
    configure_logging(debug, quiet, no_logs)
    # Open the reference MRC file to obtain the parameters
    data_original, shape, voxel_size = open_mrc(reference)
    # Create the empty volume for the new MRC
    vols_id = {}
    vols_metadata = {}
    nuevo_vol = np.zeros(shape, dtype=np.float32)
    if inputDir is None:
        logger.info("Processing the file...")
        # Read the input cluster
        df = pd.read_pickle(input)
        # Obtain the coordinates of the data
        coords = df[["Z", "Y", "X"]].astype(int).values
        # Fill the MRC
        fill_mrc(nuevo_vol, shape, coords, 1.0)
        logger.info("File processed!")
    else: 
        # Since the inputDir option is marked, we need to gather all the 
        # clusters together
        cluster_value = 1.0
        id = 1
        logger.info("Processing all files...")
        files = os.listdir(inputDir)
        for file in tqdm(files, total=len(files), desc="Processing files"):
            # Read the input cluster
            file_path = Path(inputDir) / file
            if not file_path.suffix.isin([".clust", ".temb", ".tumap"]):
                continue
            df = pd.read_pickle(file_path)
            # Obtain the coordinates of the data
            coords = df[["Z", "Y", "X"]].astype(int).values
            # Fill the MRC
            fill_mrc(nuevo_vol, shape, coords, cluster_value)
            if separate:
                vols_id[id] = nuevo_vol
                vols_metadata[id] = {"filename": file_path}
                nuevo_vol = np.zeros(shape, dtype=np.float32)
                id += 1
            else:
                cluster_value += 1.0
        if not separate:
            vols_id[0] = nuevo_vol
        logger.info("Files process completed!")
        # Save the MRC
    logger.info("Saving the MRC...")
    os.makedirs(outputDir, exist_ok=True)
    for id, vol in vols_id:
        output = Path(outputDir) / (outName + f"_{id}_segmentation.mrc")
        save_mrc(vol, voxel_size, str(output.resolve()))
    if separate:
        outputJson = Path(outputDir) / (outName + "_manifolds_metadata.json")
        with open(str(outputJson.resolve()), "w", encoding="utf-8") as f:
            json.dump(vols_metadata, f)
    logger.info("MRC saved!")

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
