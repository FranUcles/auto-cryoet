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
    parser.add_argument("-m", "--model", type=str, required=True, help="TomoTwin model")
    parser.add_argument("-b", "--batch", type=int, required=False, default=256, help="Batch size")
    parser.add_argument("-s", "--stride", type=int, required=False, default=1, help="Stride value")
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

def main(input, model, batch = 256, stride = 256, outputDir=".", debug=False, quiet=False, no_logs=False) -> str:
    configure_logging(debug, quiet, no_logs)
    outputDirPath = Path(outputDir)
    outputDirPath.mkdir(parents=True, exist_ok=True)
    # Apply mse
    logger.info("Processing the tomogram...")
    subprocess.run(["tomotwin_embed.py", "tomogram", "-m", model, "-v", input, "-b", str(batch), "-s", str(stride), "-o", outputDir], check=True)
    logger.info("Tomogram processed!")
    result = outputDirPath / (input + "_embeddings.temb")
    return str(result.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
