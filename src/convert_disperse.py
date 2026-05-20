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
    parser.add_argument("--sklInput", type=str, required=False, help="Input upSkl file")
    parser.add_argument("--manifoldsInput", type=str, required=False, help="Input manifolds file")
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

def main(sklInput, manifoldsInput, outputDir=".", debug=False, quiet=False, no_logs=False) -> tuple[str, str]:
    configure_logging(debug, quiet, no_logs)
    outputDirPath = Path(outputDir)
    outputDirPath.mkdir(parents=True, exist_ok=True)
    logger.info("Starting workflow")
    # Apply skelconv
    logger.info("Applying skelconv...")
    subprocess.run(["skelconv", sklInput, "-to", "vtp", "-outDir", outputDir], check=True)
    logger.info("skelconv applied!")
    sklPath = outputDirPath / (sklInput + ".vtp")
    # Apply netconv
    logger.info("Applying netconv...")
    subprocess.run(["netconv", manifoldsInput, "-to", "vtu", "-outDir", outputDir], check=True)
    logger.info("netconv applied!")
    manifoldsPath = outputDirPath / (manifoldsInput + ".vtu")
    return str(sklPath.resolve()), str(manifoldsPath.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
