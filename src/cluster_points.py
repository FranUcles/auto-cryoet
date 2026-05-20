#!/usr/bin/env python
import numpy as np 
import pandas as pd
import argparse
import logging
import coloredlogs
from pathlib import Path
import os
from tqdm import tqdm
import fitsio
import mrcfile
from PIL import Image
import scipy.ndimage
import pickle
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input UMAP filnename")
    parser.add_argument("-o", "--outName", type=str, required=True, help="Output FITS filename")
    parser.add_argument("--outputDir", type=str, required=False, default=".", help="Output directory")
    parser.add_argument("-s", "--size", type=int, required=True, nargs=2, help="Size of the cube (nx,ny)")
    parser.add_argument("-t", "--threshold", type=int, required=True, help="Threshold value")
    parser.add_argument("-sg", "--sigma", type=int, required=True, help="Gaussian sigma value value")
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

def load_dataframe(input_file: str) -> pd.DataFrame:
    logger.debug(f"Reading {input_file}...")
    try:
        df = pd.read_pickle(input_file)
        logger.debug("File read succesfully!")
        return df
    except FileNotFoundError:
        logger.error(f"File {input_file} not found")
        raise
    except Exception as e:
        logger.error(f"Error while reading {input_file}: {e}")
        raise
        
def compute_boundingbox(df: pd.DataFrame):
    umap_0min, umap_0max = df["umap_0"].min(), df["umap_0"].max()
    umap_1min, umap_1max = df["umap_1"].min(), df["umap_1"].max()
    umap_0min, umap_0max = -20, 15
    umap_1min, umap_1max = -15, 15
    return ((umap_0min, umap_0max), (umap_1min, umap_1max))
    
def compute_clustering(clust_array: np.ndarray, points: pd.DataFrame, x_bb: tuple[float, float], y_bb: tuple[float, float]):
    nx, ny = clust_array.shape
    logger.debug(f"Size: {clust_array.shape}")
    
    # Calculate points box size
    x_size = (x_bb[1] - x_bb[0]) / nx
    y_size = (y_bb[1] - y_bb[0]) / ny
    
    # Vectorized computation instead of row-by-row iteration
    x_off = points["umap_0"].to_numpy() - x_bb[0]
    y_off = points["umap_1"].to_numpy() - y_bb[0]
    
    x_index = np.clip((x_off / x_size).astype(int), 0, nx - 1)
    y_index = np.clip((y_off / y_size).astype(int), 0, ny - 1)
    
    # np.add.at handles duplicate indices correctly (unlike direct indexing)
    np.add.at(clust_array, (x_index, y_index), 1)

    logger.info("Building metadata...")

    df_idx = pd.DataFrame({
        "xi": x_index,
        "yi": y_index,
        "point_idx": points.index
    })
    cell_map = df_idx.groupby(["xi", "yi"])["point_idx"].apply(list).to_dict()

    return cell_map
        
def save_img(filename, outputdir, data:np.ndarray) -> str:
    # Data conversion
    array = data.T.astype(dtype=np.float32)
    # Create the output path
    outputfile = '%s/%s.fits' % (outputdir, filename)

    # Store the result
    try:
        fitsio.write(outputfile, array)
    except:
        logger.error(f"File {outputfile} could not be written")
        raise ValueError(f"File {outputfile} could not be written")  
    return outputfile

def save_png(filename, outputdir, data:np.ndarray) -> str:
    # Normalizar a rango 0–255
    array_uint8 = (255 * data / data.max()).astype(np.uint8)

    # Crear imagen
    img = Image.fromarray(array_uint8)
    # Create the output path
    outputfile = '%s/%s.png' % (outputdir, filename)

    # Guardar como PNG
    img.save(outputfile)
    return outputfile

def build_mask(data:np.ndarray, threshold):
    mask = np.where(data < threshold, 1, 0)
    return mask
    
def apply_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(data, sigma=sigma)

def save_metadata(filename, outputdir, metadata) -> str:

    # Create the output path
    outputfile = '%s/%s.mdata' % (outputdir, filename)
    metadata.to_pickle(outputfile)
    return outputfile

def main(input, outName, outputDir=".", size=(100,100), threshold=50, sigma=2, debug=False, quiet=False, no_logs=False) -> tuple[str, str, str]:

    configure_logging(debug, quiet, no_logs)
    # Read the UMAP file
    points = load_dataframe(input)
    # Create the array
    nx, ny= size
    logger.debug(f"Input size: ({nx}, {ny})")
    clusters = np.zeros([nx, ny]).astype(dtype=int)
    # Compute the boundingbox
    logger.info("Computing the bounding box...")
    umap_0_bb, umap_1_bb= compute_boundingbox(points)
    logger.info(f"Bounding_box: ([{umap_0_bb[0]}, {umap_0_bb[1]}], [{umap_1_bb[0]}, {umap_1_bb[1]}])")
    logger.info("Bounding box computed!")
    metadata = compute_clustering(clusters, points, umap_0_bb, umap_1_bb)
    # Apply gaussian filter
    clusters = apply_filter(clusters, sigma)
    # Save the file
    logger.info("Saving file...")
    output_path = Path(outputDir).resolve()
    os.makedirs(output_path, exist_ok=True)
    img_path = save_img(outName, str(output_path), clusters)
    save_png(outName, str(output_path), clusters)
    mask = build_mask(clusters, threshold)
    save_png(outName + "_mask", str(output_path), mask)
    mask_path = save_img(outName + "_mask", str(output_path), mask)
    metadata_path = save_metadata(outName + "_metadata", str(output_path), metadata)
    logger.info("File saved!")
    return (mask_path, img_path, metadata_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
