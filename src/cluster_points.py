#!/usr/bin/env python
import numpy as np 
import pandas as pd
import argparse
import logging
import coloredlogs
from pathlib import Path
import os
import fitsio
from PIL import Image
import scipy.ndimage
import subprocess
import shutil
import tempfile
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input UMAP filnename")
    parser.add_argument("-o", "--outName", type=str, required=True, help="Output FITS filename")
    parser.add_argument("--outputDir", type=str, required=False, default=".", help="Output directory")
    field_group = parser.add_mutually_exclusive_group(required=True)
    field_group.add_argument("--auto_boundingbox", action="store_true", help="Activa el modo de cálculo automática de la bounding box")
    field_group.add_argument("--interactive_boundingbox", action="store_true", help="Activa el modo interactivo para seleccionar la bounding box")
    field_group.add_argument("--bounding_box", type=float, nargs=4,  help="Define la bounding box a utilizar siendo (umap0_min, umap0_max, umap1_min, umap1_max)")
    parser.add_argument("-s", "--size", type=int, required=True, nargs=2, help="Size of the cube (nx,ny)")
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("-t", "--threshold", type=int, help="Threshold value")
    threshold_group.add_argument("--interactive_threshold", action="store_true", help="Activa el modo interactivo para el umbral de la máscara")
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
    logger.info("Computing bounding box...")
    umap_0min, umap_0max = df["umap_0"].min(), df["umap_0"].max()
    umap_1min, umap_1max = df["umap_1"].min(), df["umap_1"].max()
    logger.info("Bounding box computed!")
    return ((umap_0min, umap_0max), (umap_1min, umap_1max))

def generate_metadata(x_index, y_index, yresolution, points) -> pd.DataFrame:
    logger.info("Building metadata...")
    pixel_id = yresolution*y_index + x_index
    df_new = points[["X", "Y", "Z"]].copy()
    df_new["pixel_id"] = pixel_id
    logger.info("Metadata built!")
    return df_new
    
def compute_clustering(clust_array: np.ndarray, points: pd.DataFrame, x_bb: tuple[float, float], y_bb: tuple[float, float]):
    logger.info("Generating 2D image...")
    ny, nx = clust_array.shape
    logger.debug(f"Size: {clust_array.shape}")
    
    valid = (
        (points["umap_0"] >= x_bb[0]) &
        (points["umap_0"] <  x_bb[1]) &
        (points["umap_1"] >= y_bb[0]) &
        (points["umap_1"] <  y_bb[1])
    )
     
    # Vectorized computation instead of row-by-row iteration
    x_off = points.loc[valid, "umap_0"].to_numpy() - x_bb[0]
    y_off = points.loc[valid, "umap_1"].to_numpy() - y_bb[0]
    
    # Calculate points box size
    x_size = (x_bb[1] - x_bb[0]) / nx
    y_size = (y_bb[1] - y_bb[0]) / ny
    
    x_index = np.clip((x_off / x_size).astype(int), 0, nx - 1)
    y_index = np.clip((y_off / y_size).astype(int), 0, ny - 1)
    # Flip the Y-axis to solve mismatch between UMAP and PNG coordinates
    y_index = ny - 1 - y_index
    
    # np.add.at handles duplicate indices correctly (unlike direct indexing)
    np.add.at(clust_array, (y_index, x_index), 1)
    logger.info("2D image generated!")
    return x_index, y_index, points.loc[valid]

        
def save_img(filename, outputdir, data:np.ndarray) -> str:
    # Data conversion
    array = data.astype(dtype=np.float32)
    # Flip axes
    array = np.flipud(array)
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

def open_image(path: str) -> subprocess.Popen | None:
    if not os.environ.get("DISPLAY"):
        print(f"   Sin display disponible. Imagen en: {path}")
        return None
    
    viewers = ["feh", "eog", "display", "xdg-open"]
    for viewer in viewers:
        if shutil.which(viewer):
            return subprocess.Popen([viewer, path])
    
    raise RuntimeError("No se encontró ningún visor de imágenes instalado")

def build_mask(data:np.ndarray, threshold):
    mask = np.where(data < threshold, 1, 0)
    return mask
    
def apply_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(data, sigma=sigma)

def save_metadata(filename, outputdir, metadata: pd.DataFrame) -> str:

    # Create the output path
    outputfile = '%s/%s.mdata' % (outputdir, filename)
    metadata.to_pickle(outputfile)
    return outputfile

def ask_limits(xlo_prev: float, xhi_prev: float, ylo_prev: float, yhi_prev: float) -> tuple[float, float, float, float]:
    print(
        f"Valores actuales:\n"
        f"  umap_0 = [{xlo_prev:.4f}, {xhi_prev:.4f}]\n"
        f"  umap_1 = [{ylo_prev:.4f}, {yhi_prev:.4f}]"
    )
    print("  (Enter para mantener el valor actual)\n")

    def read_float(label, default):
        while True:
            raw = input(f"  {label} [{default:.4f}]: ").strip()
            if raw == "":
                return default
            try:
                v = float(raw)
                return v
            except ValueError:
                print("  Introduce un número válido")

    xlo = read_float("  Mínimo del umap_0", xlo_prev)
    xhi = read_float("  Máximo del umap_0", xhi_prev)

    if xlo > xhi:
        print("  Mínimo > Máximo, se intercambian automáticamente")
        xlo, xhi = xhi, xlo
    ylo = read_float("  Mínimo del umap_1", ylo_prev)
    yhi = read_float("  Máximo del umap_1", yhi_prev)

    if ylo > yhi:
        print("  Mínimo > Máximo, se intercambian automáticamente")
        ylo, yhi = yhi, ylo

    return xlo, xhi, ylo, yhi

def interactive_bb_loop(clusters, points):

    (xlo, xhi), (ylo, yhi) = compute_boundingbox(points)
    tempdir = tempfile.gettempdir()

    img_path  = os.path.join(tempdir, "boundingbox_preview.png")
    iteration = 0

    while True:
        clusters_aux = clusters.copy()
        iteration += 1
        print(f"\n{'─'*52}")
        print(f"  Iteración {iteration}")
        if iteration > 1:
            xlo, xhi, ylo, yhi = ask_limits(xlo, xhi, ylo, yhi)
        else:
            print(" Usando bounding box calculada de forma automática para la primera iteración")
        x, y, selected_points = compute_clustering(clusters_aux, points, (xlo, xhi), (ylo, yhi))
        save_png("boundingbox_preview", tempdir, clusters_aux)
        open_image(img_path)
        resp = input("\n  ¿Es el resultado esperado? [s/N]: ").strip().lower()
        if resp in ("s", "si", "sí", "y", "yes"):
            clusters[:] = clusters_aux
            break
        
    print(f"\n{'═'*52}")
    print(
        f"  Bounding box final :  \n"
        f"    umap_0 = [{xlo:.4f}, {xhi:.4f}]\n"
        f"    umap_1 = [{ylo:.4f}, {yhi:.4f}]"
        )
    print(f"{'═'*52}\n")

    return x, y, selected_points

def ask_threshold(threshold_curr: float) -> float:
    print(
        f"Valor actual:{threshold_curr}"
    )
    print("  (Enter para mantener el valor actual)\n")

    def read_float(label, default):
        while True:
            raw = input(f"  {label} [{default:.4f}]: ").strip()
            if raw == "":
                return default
            try:
                v = float(raw)
                if v >= 0:
                    return v
                print("  Debe ser mayor o igual que 0")
            except ValueError:
                print("  Introduce un número válido")


    return read_float("  Threshold de máscara", threshold_curr)

def interactive_threshold_loop(clusters):
    tempdir = tempfile.gettempdir()

    threshold = 0.0
    img_path  = os.path.join(tempdir, "threshold_preview.png")
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'─'*52}")
        print(f"  Iteración {iteration}")
        threshold = ask_threshold(threshold)
        mask = build_mask(clusters, threshold)
        save_png("threshold_preview", tempdir, mask)
        open_image(img_path)
        resp = input("\n  ¿Es el resultado esperado? [s/N]: ").strip().lower()
        if resp in ("s", "si", "sí", "y", "yes"):
            result = mask
            break
        
    print(f"\n{'═'*52}")
    print(f"  Threshold final :  {threshold:.4f}")
    print(f"{'═'*52}\n")

    return result

def main(input, outName, auto_boundingbox, interactive_boundingbox, interactive_threshold, bounding_box, outputDir=".", size=(100,100), threshold=50, sigma=2, debug=False, quiet=False, no_logs=False) -> tuple[str, str, str]:

    configure_logging(debug, quiet, no_logs)
    # Read the UMAP file
    points = load_dataframe(input)
    # Create the array
    nx, ny = size
    logger.debug(f"Input size: ({nx}, {ny})")
    clusters = np.zeros([ny, nx]).astype(dtype=int)
    if interactive_boundingbox:
        x, y, selected_points = interactive_bb_loop(clusters, points)
    else: 
        if auto_boundingbox:
            # Compute the boundingbox
            logger.info("Computing the bounding box...")
            umap_0_bb, umap_1_bb = compute_boundingbox(points)
            logger.info(f"Bounding_box: ([{umap_0_bb[0]}, {umap_0_bb[1]}], [{umap_1_bb[0]}, {umap_1_bb[1]}])")
            logger.info("Bounding box computed!")
        else:
            umap_0_bb, umap_1_bb = (bounding_box[0], bounding_box[1]), (bounding_box[0], bounding_box[1])
        x, y, selected_points = compute_clustering(clusters, points, umap_0_bb, umap_1_bb)
    
    metadata = generate_metadata(x, y, ny, selected_points)    
    # Apply gaussian filter
    clusters = apply_filter(clusters, sigma)
    # Save the file
    logger.info("Saving file...")
    if interactive_threshold:
        mask = interactive_threshold_loop(clusters)
    else:
        mask = build_mask(clusters, threshold)
    output_path = Path(outputDir).resolve()
    os.makedirs(output_path, exist_ok=True)
    img_path = save_img(outName, str(output_path), clusters)
    save_png(outName, str(output_path), clusters)
    save_png(outName + "_mask", str(output_path), mask)
    mask_path = save_img(outName + "_mask", str(output_path), mask)
    metadata_path = save_metadata(outName + "_metadata", str(output_path), metadata)
    logger.info("File saved!")
    return (mask_path, img_path, metadata_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
