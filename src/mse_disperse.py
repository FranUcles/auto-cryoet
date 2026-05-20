#!/usr/bin/env python
import logging
import coloredlogs
import argparse
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input TomoTwin output file")
    parser.add_argument("-o", "--outputDir", type=str, default=None, help="Output directory")
    parser.add_argument("--cut", type=float, required=True, help="Persistence cut")
    parser.add_argument("--manifolds", type=str, required=True, help="Dumpmanifolds values")
    parser.add_argument("--loadMSC", type=str, default=None, help="File containing the MSC to be loaded")
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

def main(input, cut, manifolds, outName, loadMSC = None, outputDir=".", debug=False, quiet=False, no_logs=False) -> tuple[str, str]:
    configure_logging(debug, quiet, no_logs)
    outputDirPath = Path(outputDir)
    outputDirPath.mkdir(parents=True, exist_ok=True)
    # Apply mse
    logger.info("Applying MSE...")
    loadMSC_args = []
    if loadMSC is not None:
        loadMSC_args = ["-loadMSC", loadMSC]
    subprocess.run(["mse", input, "-outName", outName, "-upSkl", "-vertexAsMinima" ,"-cut", str(cut), "-dumpManifolds", manifolds, "-outDir", outputDir] + loadMSC_args, check=True)
    logger.info("MSE applied!")
    cut_suffix = ""
    if cut != 0:
        cut_suffix = f"_c{cut}"
    sklPath = outputDirPath / (outName + f"{cut_suffix}.up.NDskl")
    manifoldsPath = outputDirPath / (outName + f"{cut_suffix}_manifolds_{manifolds}.NDnet")
    return str(sklPath.resolve()), str(manifoldsPath.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
