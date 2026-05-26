#!/usr/bin/env python
import mrcfile
import argparse
import logging
import coloredlogs
import napari
import json
from pathlib import Path
from vispy.color import Colormap, Color

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input tomogram filnename")
    parser.add_argument("--segmentsDir", type=str, required=True, help="Directory containing all the segmentation files")
    parser.add_argument("--filesMetadata", type=str, required=True, help="Metadata of each segmentation file")
    parser.add_argument("--colorsMetadata", type=str, required=True, help="Metadata of the color of each segmentation file")
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

def show_tomogram(tomogram, seg_data, col_data, inputPath):
    logger.info("Loading tomogram...")
    # Tomogram
    with mrcfile.open(tomogram) as mrc:
        tomo = mrc.data
    logger.info("Tomogram loaded!")
    viewer = napari.Viewer()
    # Tomogram image
    viewer.add_image(
        tomo,
        name="tomogram",
        colormap="gray"
    )
    logger.info("Loading colors and filenames...")
    # Load semgmentation files metadata
    with open(seg_data) as f:
        files = json.load(f)
    # Load colors metadata
    with open(col_data) as f:
        colors = json.load(f)
    logger.info("Colors and filenames loaded!")
    logger.info("Creating the layers...")
    inputPath = Path(inputPath)
    # Create the layers
    for id in files:
        fname = inputPath / (f"resultado_{id}_segmentation.mrc")
        fname = str(fname.resolve())
        color = colors[id]["color"]
        with mrcfile.open(fname) as mrc:
            seg = mrc.data

        viewer.add_labels(
            seg.astype(int),
            name=f"segment_{id}",
            colormap={None: "transparent", 1: color},
            opacity=0.5,
            blending="additive"
        )
    logger.info("Layers created!")
    napari.run()
    
def main(input, segmentsDir, filesMetadata, colorsMetadata, debug=False, quiet=False, no_logs=False):
    configure_logging(debug, quiet, no_logs)
    show_tomogram(input, filesMetadata, colorsMetadata, segmentsDir)
    
if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
