#!/usr/bin/env python
import vtk
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
    parser.add_argument("-i", "--input", type=str, required=True, help="Input VTU file")
    parser.add_argument("-o", "--outputDir", type=str, required=True, help="Output directory")
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
        
def load_VTU(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    mesh = reader.GetOutput()
    return mesh

def main(input, outputDir=".", debug=False, quiet=False, no_logs=False):
    configure_logging(debug, quiet, no_logs)
    # Class attribute name
    attr_name = "source_index"
    logger.info("Loading the VTU file...")
    # We read the VTU file
    mesh = load_VTU(input)
    cell_data = mesh.GetCellData().GetArray(attr_name)
    if cell_data is None:
        logger.error("The VTU does not contain the proper format")
        raise ValueError(f"No '{attr_name}' attribute.") 
    logger.info("VTU file loaded!")
    # Construir lista de celdas que cumplan la condición
    logger.info("Clustering the points...")
    points = mesh.GetPoints()
    class_data = dict()
    for cell_id in range(mesh.GetNumberOfCells()):
        class_id = cell_data.GetValue(cell_id)
        cell = mesh.GetCell(cell_id)
        point_ids = cell.GetPointIds()
        cell_points = []
        # Each cell is a triangle. Thus, we iterate over the vertices
        for j in range(point_ids.GetNumberOfIds()):
            pid = point_ids.GetId(j)
            point = points.GetPoint(pid)
            cell_points.append(point)
            # Add the point of the triangle to the class
            if not(class_id in class_data):
                class_data[class_id] = list()
            class_data[class_id].append({"x": point[0], "y": point[1]})

    logger.info("Points clustered correctly!")
    # Make sure the output directory exists
    os.makedirs(Path(outputDir).resolve(), exist_ok=True)
    # Convert every class to a dataframe
    logger.info("Saving all the points...")
    input_filename = Path(input).with_suffix("").stem
    for class_id, points in class_data.items():
        df = pd.DataFrame(points)
        df = df.drop_duplicates()
        logger.debug(f"Class: {class_id} with data: {df}")
        # Save the dataframe
        output_name = (Path(outputDir) / (str(input_filename) + "_" + str(int(class_id)) + ".clust")).resolve()
        df.to_pickle(output_name)
    logger.info("Saved all the classes!")

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
