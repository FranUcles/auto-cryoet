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
    parser.add_argument("-i", "--input", type=str, required=True, help="Input tomogram file")
    
    parser.add_argument("--chunk_size", type=int, required=False, default=4000, help="Chunk size")
    parser.add_argument("--fit_sample_size", type=int, required=False, default=4000, help="Fit sample size")
    parser.add_argument("-o", "--outputDir", type=str, default=None, help="Output directory")
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

def main(input, chunk_size = 4000, fit_sample_size = 4000, outputDir=".", debug=False, quiet=False, no_logs=False) -> str:
    configure_logging(debug, quiet, no_logs)
    outputDirPath = Path(outputDir)
    outputDirPath.mkdir(parents=True, exist_ok=True)
    # Apply umap
    logger.info("Processing the tomogram...")
    subprocess.run(["tomotwin_tools.py", "umap", "-i", input, "-n", "2", "--chunk_size", str(chunk_size), "--fit_sample_size", str(fit_sample_size), "-o", outputDir], check=True)
    logger.info("Tomogram processed!")
    inputfilename = Path(input).stem
    result = outputDirPath / (str(inputfilename) + ".tumap")
    return str(result.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
