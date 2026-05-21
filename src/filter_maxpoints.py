#!/usr/bin/env python
import vtk
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
import argparse
import logging
import coloredlogs
import os
from pathlib import Path
import tempfile
import subprocess
import shutil

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process cloud points")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only errors")
    parser.add_argument("--no-logs", action="store_true", help="Disable all logs")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input VTU file")
    parser.add_argument("-o", "--outputName", type=str, required=True, help="Output filename")
    field_group = parser.add_mutually_exclusive_group(required=True)
    field_group.add_argument("--interactive", action="store_true", help="Activa el modo interactivo para seleccionar el filtro por valor")
    field_group.add_argument("--field-limits", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Rango de field_value a filtrar")
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
    
def open_image(path: str) -> subprocess.Popen | None:
    if not os.environ.get("DISPLAY"):
        print(f"   Sin display disponible. Imagen en: {path}")
        return None
    
    viewers = ["feh", "eog", "display", "xdg-open"]
    for viewer in viewers:
        if shutil.which(viewer):
            return subprocess.Popen([viewer, path])
    
    raise RuntimeError("No se encontró ningún visor de imágenes instalado")
    
def render_screenshot(filtered_polydata, output_path: str,
                      lo: float, hi: float, fmin: float, fmax: float):
    """Renderiza filtered_polydata en offscreen y guarda PNG."""

    # Lookup table por field_value
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(fmin, fmax)
    lut.SetHueRange(0.667, 0.0)   # azul → rojo
    lut.Build()

    # Colorear por field_value
    arr = filtered_polydata.GetPointData().GetArray("field_value")

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(filtered_polydata)
    if arr:
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("field_value")
        mapper.SetScalarRange(fmin, fmax)
        mapper.SetLookupTable(lut)
    else:
        mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    actor.GetProperty().SetRepresentationToPoints()

    # Barra de color
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetTitle("field_value")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetPosition(0.88, 0.1)
    scalar_bar.SetWidth(0.08)
    scalar_bar.SetHeight(0.7)

    # Texto con los límites actuales
    n = filtered_polydata.GetNumberOfPoints()
    text = vtk.vtkTextActor()
    text.SetInput(f"Min: {lo:.4f}   Max: {hi:.4f}   Puntos: {n}")
    text.SetPosition(10, 10)
    text.GetTextProperty().SetFontSize(18)
    text.GetTextProperty().SetColor(1, 1, 1)

    # Renderer con vista 2D cenital
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.15, 0.15, 0.15)
    renderer.AddViewProp(actor)      
    renderer.AddViewProp(scalar_bar)
    renderer.AddViewProp(text)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.ParallelProjectionOn()
    bounds = filtered_polydata.GetBounds()
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cam.SetPosition(cx, cy, 1)
    cam.SetFocalPoint(cx, cy, 0)
    cam.SetViewUp(0, 1, 0)
    renderer.ResetCamera()
    
    # Render offscreen
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(1280, 960)
    render_window.AddRenderer(renderer)
    render_window.Render()
    
    # Guardar PNG
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    
    render_window.Finalize()
    
def filter_max(points_path):
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
    threshold_max.Update()
    return threshold_max

def apply_field_filter(threshold_max, lo: float, hi: float):
    """Aplica threshold + geom_filter y devuelve (filtered_polydata, n_points)."""
    threshold = vtk.vtkThreshold()
    threshold.SetInputConnection(threshold_max.GetOutputPort())
    threshold.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "field_value"
    )
    threshold.SetLowerThreshold(lo)
    threshold.SetUpperThreshold(hi)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    
    return threshold

def convert_polydata(threshold):
    # --- Convertir a PolyData ---
    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputConnection(threshold.GetOutputPort())
    geom_filter.Update()

    filtered_polydata = geom_filter.GetOutput()
    return filtered_polydata, filtered_polydata.GetNumberOfPoints()

def ask_limits(fmin: float, fmax: float, lo_prev: float, hi_prev: float) -> tuple[float, float]:
    print(f"\n  Rango disponible : [{fmin:.4f},  {fmax:.4f}]")
    print(f"  Valores actuales : [{lo_prev:.4f}, {hi_prev:.4f}]")
    print("  (Enter para mantener el valor actual)\n")

    def read_float(label, default):
        while True:
            raw = input(f"  {label} [{default:.4f}]: ").strip()
            if raw == "":
                return default
            try:
                v = float(raw)
                if fmin <= v <= fmax:
                    return v
                print(f"  Debe estar entre {fmin:.4f} y {fmax:.4f}")
            except ValueError:
                print("  Introduce un número válido")

    lo = read_float("  Mínimo field_value", lo_prev)
    hi = read_float("  Máximo field_value", hi_prev)

    if lo > hi:
        print("  Mínimo > Máximo, se intercambian automáticamente")
        lo, hi = hi, lo

    return lo, hi

def interactive_loop(threshold_max, lo_init: float, hi_init: float):

    fmin, fmax    = get_field_range(threshold_max)
    lo = max(fmin, lo_init)
    hi = min(fmax, hi_init)

    img_path  = os.path.join(tempfile.gettempdir(), "field_filter_preview.png")
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'─'*52}")
        print(f"  Iteración {iteration}")

        if iteration > 1:
            lo, hi = ask_limits(fmin, fmax, lo, hi)

        threshold = apply_field_filter(threshold_max, lo, hi)
        filtered_polydata, n = convert_polydata(threshold)
        render_screenshot(filtered_polydata, img_path, lo, hi, fmin, fmax)

        print(f"\n    Puntos visibles : {n}")
        print(f"    Preview         : {img_path}")
        open_image(img_path)

        resp = input("\n  ¿Es el resultado esperado? [s/N]: ").strip().lower()
        if resp in ("s", "si", "sí", "y", "yes"):
            break

    print(f"\n{'═'*52}")
    print(f"  Límites finales  : --field-limits {lo:.4f} {hi:.4f}")
    print(f"  Puntos           : {n}")
    print(f"{'═'*52}\n")
    return filtered_polydata, threshold

def get_field_range(threshold) -> tuple[float, float]:
    arr = threshold.GetOutput().GetPointData().GetArray("field_value")
    return arr.GetRange() if arr else (0.0, 1000.0)

def filter_points(points_path, interactive: bool, field_filter: tuple[int, int] | None) -> pd.DataFrame:
    
    # Filtrar los puntos máximos
    threshold_max = filter_max(points_path)
    # Obtener el rango de field_value
    low, high = get_field_range(threshold_max)
    
    # Si estamos en el modo interactivo, preguntamos los límites
    if interactive:
        filtered_polydata, threshold = interactive_loop(threshold_max, low, high)
    # Sino, aplicamos el filtro si es necesario
    else: 
        if (field_filter is not None):
            threshold = apply_field_filter(threshold_max, field_filter[0], field_filter[1])
        else: 
            threshold = threshold_max
        filtered_polydata, n = convert_polydata(threshold)
    threshold.Update()

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

def main(input, outputName, interactive=False, field_limits=None, outputDir=".", debug=False, quiet=False, no_logs=False) -> str:
    os.makedirs(outputDir, exist_ok=True)
    configure_logging(debug, quiet, no_logs)
    logger.info("Filtering the points...")
    df = filter_points(input, interactive, field_limits)
    logger.info("Points filtered!")
    out_filename = Path(outputDir) / (outputName + ".slt")
    save(df, out_filename.resolve())
    return str(out_filename.resolve())

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))

