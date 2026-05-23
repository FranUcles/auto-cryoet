#!/usr/bin/env python
import pandas as pd
import argparse
import logging
import coloredlogs
from PIL import Image
import numpy as np
import os
import matplotlib as mpl
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("--inputDir", type=str, required=True, help="Input directory containing all the manifolds to process")
    parser.add_argument("-r", "--referenceImg", type=str, required=True, help="PNG file with the reference image")
    parser.add_argument("-o", "--outName", type=str, required=True, help="Output PNG filename")
    parser.add_argument("--outputDir", type=str, required=False, default=".",  help="Output directory")
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
    
def generate_colors(color_amount):
    cmap = mpl.colormaps["tab20"].resampled(color_amount)
    colors = [
        tuple(int(255 * c) for c in cmap(i)[:3])
        for i in range(color_amount)
    ]
    return colors

def load_img(imgPath) -> np.ndarray:
    img = Image.open(imgPath).convert("RGB")
    arr = np.array(img)
    return arr

def save_img(img_arr, path: str):
    Image.fromarray(img_arr).save(path)
    
def main(inputDir, referenceImg, outName, outputDir=".", debug=False, quiet=False, no_logs=False):
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    folder = inputDir
    logger.info("Loading the manifolds...")
    # Obtain how many manifolds we have
    dfs_data = [{"df": pd.read_pickle(f), "filename": str(f)} for f in Path(folder).iterdir() if f.is_file() and f.suffix == ".clust"]
    n = len(dfs_data)
    # Generate the colors
    colors = generate_colors(n)
    logger.info("Manifolds loaded!")
    # Load the image
    logger.info("Loading the image...")
    arr = load_img(referenceImg)
    # Flip de Y-axis to match the coordinates
    arr = np.flipud(arr)
    heigh, width = arr.shape[:2]
    logger.info("Image loaded!")
    logger.info("Plotting the manifolds...")
    manifolds_metadata = {}
    id = 1
    for df_data, color in zip(dfs_data, colors):
        df = df_data["df"]
        # Save metadata
        manifolds_metadata[id] = {"filename": df_data["filename"], "color": "#{:02x}{:02x}{:02x}".format(*color)}
        id += 1
        # Obtain all the coordinates
        x = df["x"].to_numpy(dtype=np.int32)
        y = df["y"].to_numpy(dtype=np.int32)
        # Filter those out-bounds points
        mask = (y >= 0) & (y < heigh) & (x >= 0) & (x < width)
        arr[y[mask], x[mask]] = color
    logger.info("Manifolds plotted!")
    outputPath = Path(outputDir) / (outName + ".png")
    outputMetadataPath = Path(outputDir) / (outName + "_pngmetadata.json")
    with open(str(outputMetadataPath.resolve()), "w", encoding="utf-8") as f:
            json.dump(manifolds_metadata, f)
    logger.info("Saving the image...")
    # Revert the Y-axis flip to recover the original points
    arr = np.flipud(arr)
    # Save the image
    save_img(arr, str(outputPath.resolve()))
    logger.info("Image saved!")

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
