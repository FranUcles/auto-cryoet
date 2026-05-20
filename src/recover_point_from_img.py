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
    parser.add_argument("-r", "--reference", type=str, required=True, help="UMAP reference file")
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

def process_class(_input, metadata_df, reference_df, outputDir):
    logger.info("Reading files...")
    df = read_points_from_file(_input)
    logger.info("Files read!")
    logger.info("Converting points...")
    keys = list(zip(df["umap_0"], df["umap_1"]))
    indices = [idx for k in keys for idx in metadata_df.get(k, [])]
    subset = reference_df.loc[indices]
    outputfilename = Path(_input).name
    outputfile = Path(outputDir) / outputfilename
    logger.debug("Saving file {str(outputfile.resolve())}")
    save(subset, str(outputfile.resolve()))
    logger.info("Converted!")

def main(input, inputDir, reference, metadata, outputDir=".", debug=False, quiet=False, no_logs=False):
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    _metadata = read_points_from_file(metadata)
    _reference = read_points_from_file(reference)
    if inputDir is None:
        process_class(input, _metadata, _reference, outputDir)
    else:
        for file in os.listdir(inputDir):
            file_path = Path(inputDir) / file
            process_class(str(file_path.resolve()), _metadata, _reference, outputDir)

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
