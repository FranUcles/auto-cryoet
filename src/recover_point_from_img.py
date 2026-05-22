#!/usr/bin/env python
import pandas as pd
import argparse
import logging
import coloredlogs
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input", type=str, help="Input cluster file")
    input_group.add_argument("--inputDir", type=str, default=None, help="Input directory containing all the clusters to process")
    parser.add_argument("--referenceUMAP", type=str, required=True, help="UMAP reference file (.tumap)")
    parser.add_argument("--referenceEmb", type=str, required=True, help="Embeddings reference file (.temb)")
    parser.add_argument("-s", "--imgXsize", type=int, required=True, help="Pixel size of the X-axis in the image")
    parser.add_argument("-m", "--metadata", type=str, required=True, help="Metadata file")
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
        
def read_points_from_file(fname):
    logger.debug(f"Reading {fname}...")
    try:
        df = pd.read_pickle(fname)
        logger.debug("File read succesfully!")
        return df
    except FileNotFoundError:
        logger.error(f"File {fname} not found")
        raise
    except Exception as e:
        logger.error(f"Error while reading {fname}: {e}")
        raise
        
def save(df: pd.DataFrame, file):
    df.to_pickle(file)


def process_class(_input, xsize, metadata_df, reference_umap_df, reference_temb_df, outputDir):
    logger.info("Converting points...")
    logger.debug("Reading file...")
    df = read_points_from_file(_input)
    logger.debug("File read!")
    pixel_id = df["y"]*xsize + df["x"]
    pixel_id = pixel_id.to_numpy().astype(int)
    subset = metadata_df.loc[metadata_df["pixel_id"].isin(pixel_id), ["X", "Y", "Z"]].astype("float32")
    subset_umap = reference_umap_df.merge(subset, how = "inner", on = ["X", "Y", "Z"])
    subset_temb = reference_temb_df.merge(subset, how = "inner", on = ["X", "Y", "Z"])
    outputfilename = Path(_input).name
    outputfile = Path(outputDir) / outputfilename
    logger.debug(f"Saving file {str(outputfile.resolve())}")
    save(subset_umap, str(outputfile.resolve()))
    save(subset_temb, str(outputfile.with_suffix(".temb").resolve()))
    logger.info("Converted!")

def main(input, imgXsize, inputDir, referenceUMAP, referenceEmb, metadata, outputDir=".", debug=False, quiet=False, no_logs=False):
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    metadata = read_points_from_file(metadata)
    reference_umap = read_points_from_file(referenceUMAP)
    reference_temb = read_points_from_file(referenceEmb)
    if inputDir is None:
        process_class(input, imgXsize, metadata, reference_umap, reference_temb, outputDir)
    else:
        for file in os.listdir(inputDir):
            file_path = Path(inputDir) / file
            process_class(str(file_path.resolve()), imgXsize, metadata, reference_umap, reference_temb, outputDir)

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
