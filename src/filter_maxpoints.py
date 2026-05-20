#!/usr/bin/env python
import vtk
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
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
    parser.add_argument("-o", "--outputName", type=str, required=True, help="Output filename")
    parser.add_argument("--field-limits", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Rango de field_value a filtrar")
    parser.add_argument("--outputDir", type=str, default=".", help="Output directory")
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

def filter_points(points_path, field_filter: tuple[int, int] | None) -> pd.DataFrame:

    # --- Leer archivo VTP ---
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(points_path)
    reader.Update()

    polydata = reader.GetOutput()

    id_filter =  vtk.vtkGenerateIds()
    id_filter.SetInputData(polydata)
    id_filter.PointIdsOn()
    id_filter.CellIdsOn()
    id_filter.SetPointIdsArrayName("original_point_id")
    id_filter.SetCellIdsArrayName("original_cell_id")
    id_filter.Update()

    polydata_with_ids = id_filter.GetOutput()

    # --- Eliminar los filamentos ---
    threshold_points = vtk.vtkThreshold()
    threshold_points.SetInputData(polydata_with_ids)

    threshold_points.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
        "type"
    )
    # Como los filamentos son el type = 1, solo tenemos que tomar el resto que no sean 1
    # para quedarnos con los puntos
    threshold_points.SetLowerThreshold(-1)
    threshold_points.SetUpperThreshold(0.5)
    threshold_points.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)

    # --- Seleccionar solo los máximos ---
    threshold_max = vtk.vtkThreshold()
    threshold_max.SetInputConnection(threshold_points.GetOutputPort())

    threshold_max.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "critical_index"
    )
    # Cómo hemos utilizado el -vertexAsMinima entonces los máximos tienen el 
    # critical_index = 2 por lo que seleccionar estos sería suficiente
    threshold_max.SetLowerThreshold(1.5)
    threshold_max.SetUpperThreshold(2)
    threshold_max.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)

    # --- Seleccionar puntos críticos con un cierto valor  ---
    if (field_filter is not None):
        threshold = vtk.vtkThreshold()
        threshold.SetInputConnection(threshold_max.GetOutputPort())

        threshold.SetInputArrayToProcess(
            0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
            "field_value"
        )

        threshold.SetLowerThreshold(field_filter[0])
        threshold.SetUpperThreshold(field_filter[1])
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    else:
        threshold = threshold_max

    threshold.Update()

    # --- Convertir a PolyData ---
    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputConnection(threshold.GetOutputPort())
    geom_filter.Update()

    filtered_polydata = geom_filter.GetOutput()

    # --- Extraer puntos que han quedado filtrados ---
    points_vtk = filtered_polydata.GetPoints()

    if points_vtk is None:
        raise ValueError("No hay puntos tras aplicar los thresholds")

    points = vtk_to_numpy(points_vtk.GetData())

    df = pd.DataFrame(points, columns=["x", "y", "z"])

    # --- Extraer los atributos de la malla ---
    point_data = filtered_polydata.GetPointData()

    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
    
        if array is None:
            continue

        name = array.GetName() or f"array_{i}"
        np_array = vtk_to_numpy(array)

        if np_array.ndim > 1:
            for j in range(np_array.shape[1]):
                df[f"{name}_{j}"] = np_array[:, j]
        else:
            df[name] = np_array
    return df

def save(df: pd.DataFrame, file):
    df.to_pickle(file)

def main(input, outputName, field_limits=None, outputDir=".", debug=False, quiet=False, no_logs=False) -> str:
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    if field_limits is None:
        field_filter = None
    else:
        field_filter = (field_limits[0], field_limits[1])
    logger.info("Filtering the points...")
    df = filter_points(input, field_filter)
    logger.info("Points filtered!")
    out_filename = Path(outputDir) / (outputName + ".slt")
    save(df, out_filename.resolve())
    return str(out_filename.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

