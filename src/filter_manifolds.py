#!/usr/bin/env python
import pandas as pd
import argparse
import logging
import coloredlogs
import os
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("--inputDir", type=str, required=True, default=None, help="Input directory containing all the clusters to process")
    parser.add_argument("-r", "--reference", type=str, required=True, help="DataFrame with all the manifolds to select")
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

def main(inputDir, reference, outputDir=".", debug=False, quiet=False, no_logs=False):
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    ref_df = read_points_from_file(reference)
    files = os.listdir(inputDir)
    indexes = ref_df["original_point_id"].to_list()
    logger.info("Filtering manifolds...")
    filtered = [file for file in files if any(str(index) in file for index in indexes)]
    logger.info("Manifolds filtered!")
    logger.debug(filtered)
    logger.info("Copying manifolds...")
    output_path = Path(outputDir)
    output_path.mkdir(parents=True, exist_ok=True)
    for file in filtered:
        old_file = Path(inputDir) / file
        new_file = output_path / file
        shutil.copy2(old_file, new_file)
    logger.info("All copied!")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
