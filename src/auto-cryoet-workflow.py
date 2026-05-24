#!/usr/bin/env python3
"""
╔══════════════════════════════════════════╗
║   CRYO-ET TOMOGRAM ANALYSIS WORKFLOW     ║
║   CLI interactivo con fases modulares    ║
╚══════════════════════════════════════════╝
Hay que tener instalado el entorno y la suite de TomoTwin y DESPUÉS instalar también:
Dependencias: conda install questionary coloredlogs vtk fitsio conda-forge::disperse
"""

import time
import sys
from pathlib import Path
from datetime import datetime
import cluster_points as cloudpoints
import filter_maxpoints as maxpoints
import filter_manifolds as filtermanifolds
import recover_point_from_img as recover
import obtain_classes as createmanifolds
import create_mrc as createmrc
import mse_disperse as mse
import convert_disperse as convertvtk
import tomotwin_analysis as tomotwin
import tomotwin_umap as umap
import show_manifolds as showmanifolds
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.rule import Rule
    from rich.padding import Padding
    from rich.live import Live
    from rich.align import Align
    import questionary
    from questionary import Style as QStyle
except ImportError:
    print("\n[ERROR] Faltan dependencias. Instálalas con:\n")
    print("  pip install rich questionary\n")
    sys.exit(1)


# ─────────────────────────────────────────
#  CONFIGURACIÓN VISUAL
# ─────────────────────────────────────────

console = Console()

CUSTOM_STYLE = QStyle([
    ("qmark",        "fg:#00d7af bold"),
    ("question",     "fg:#ffffff bold"),
    ("answer",       "fg:#00d7af bold"),
    ("pointer",      "fg:#00d7af bold"),
    ("highlighted",  "fg:#ffffff bg:#005f5f bold"),
    ("selected",     "fg:#00d7af"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888 italic"),
    ("text",         "fg:#cccccc"),
    ("disabled",     "fg:#555555 italic"),
])

PALETTE = {
    "primary":   "#00d7af",
    "secondary": "#0087af",
    "accent":    "#ffd700",
    "warning":   "#ff8700",
    "error":     "#ff005f",
    "muted":     "#555555",
    "success":   "#00af5f",
}

# ─────────────────────────────────────────
#  PARÁMETROS GLOBALES
#  → Se preguntan una sola vez y se inyectan
#    en todas las fases que los declaren con
#    depends_on: {"phase": "__global__", "key": "..."}
# ─────────────────────────────────────────
 
GLOBAL_PARAMS = [
    {
        "key":     "outputDir",
        "label":   "Directorio de salida base para todas las fases",
        "type":    "path",
        "default": "./output",
    },
    {
        "key":     "outName",
        "label":   "Nombre del archivo de salida (sin extensión)",
        "type":    "path",
        "default": "resultado",
    },
]

# ─────────────────────────────────────────
#  DEFINICIÓN DE FASES
#  → Adapta esta lista a tus herramientas
# ─────────────────────────────────────────

PHASES = [
    {
        "id":    "01",
        "name":  "Generación el espacio latente",
        "desc":  "Genera el espacio latente a partir del tomograma usando el modelo de TomoTwin.",
        "group": "Análisis del tomograma",
        "params": [
            {"key": "input", "label": "Fichero MRC del tomograma (.mrc)",         "type": "path",   "default": "tomograma.mrc"},
            {"key": "model", "label": "Modelo de TomoTwin (.pth)",         "type": "path",   "default": "model.pth"},
            {"key": "batch", "label": "Batch de la red neuronal",         "type": "number",   "default": 256},
            {"key": "stride", "label": "Stride de la red neuronal",         "type": "number",   "default": 1},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
            ],
    },
    {
        "id":    "02",
        "name":  "Reducción dimensional",
        "desc":  "Reduce la dimensión del espacio latente de 32 a 2 dimensiones mediante UMAP.",
        "group": "Análisis del tomograma",
        "params": [
            {"key": "input", "label": "Fichero del espacio latente del tomograma (.temb)",         "type": "path",   "default": "tomograma.temb", "depends_on": {"phase": "01", "key": "outputDir"}},
            {"key": "chunk_size", "label": "Tamaño del chunk para la transformada",         "type": "number",   "default": 40000},
            {"key": "fit_sample_size", "label": "Tamaño del sample para UMAP",         "type": "number",   "default": 40000},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "03",
        "name":  "Generación de imagen 2D, máscara y los metadatos",
        "desc":  "Genera la imagen 2D y la máscara del espacio latente y también los metadatos para reconstruir los manifolds.",
        "group": "Pre-process",
        "params": [
            {"key": "input", "label": "Fichero UMAP de entrada (.tumap)",         "type": "path",   "default": "tomograma.tumap", "depends_on": {"phase": "02", "key": "outputDir"}},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)",  "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
            {"key": "auto_bounding",   "label": "¿Seleccionar de forma automática la bounding box?",   "type": "bool",   "default": False},
            {"key": "interactive_bb",   "label": "¿Seleccionar de forma interactiva la bounding box?",   "type": "bool",   "default": False, "show_if": {"key": "auto_bounding", "equals": False}},
            {"key": "xlow_bb",  "label": "Limite inferior para la bounding box del eje X", "type": "number", "default": 0, "min": -9999, "max": 9999, "show_if": {"key": "interactive_bb", "equals": False}},
            {"key": "xhi_bb",  "label": "Limite superior para la bounding box del eje X", "type": "number", "default": 0, "min": -9999, "max": 9999, "show_if": {"key": "interactive_bb", "equals": False}},
            {"key": "ylow_bb",  "label": "Limite inferior para la bounding box del eje Y", "type": "number", "default": 0, "min": -9999, "max": 9999, "show_if": {"key": "interactive_bb", "equals": False}},
            {"key": "yhi_bb",  "label": "Limite superior para la bounding box del eje Y", "type": "number", "default": 0, "min": -9999, "max": 9999, "show_if": {"key": "interactive_bb", "equals": False}},
            {"key": "size_x", "label": "Tamaño del cubo — nx", "type": "number", "default": 2000, "min": 1, "max": 9999},
            {"key": "size_y", "label": "Tamaño del cubo — ny", "type": "number", "default": 2000, "min": 1, "max": 9999},
            {"key": "interactive_threshold",   "label": "¿Seleccionar de forma interactiva el umbral para la máscara?",   "type": "bool",   "default": False},
            {"key": "threshold",  "label": "Umbral para máscara", "type": "number", "default": 30,  "min": 0, "max": 9999,  "show_if": {"key": "interactive_threshold", "equals": False}},
            {"key": "sigma", "label": "Sigma del filtro gaussiano", "type": "number", "default": 5,   "min": 0, "max": 100},
        ],
    },
    {
        "id":    "04",
        "name":  "Simplificación de Morse",
        "desc":  "Realiza el análsis de Morse mediante DisPerSE",
        "group": "Análisis topológico",
        "params": [
            {"key": "input",      "label": "Imagen del espacio latente (.fits)",         "type": "path",   "default": "espacio_latente.fits", "depends_on": {"phase": "03", "key": "input"}},
            {"key": "mask",      "label": "Mascara del espacio latente (.fits)",         "type": "path",   "default": "mascara.fits", "depends_on": {"phase": "03", "key": "input"}},
            {"key": "cut",      "label": "Umbral de persistencia",         "type": "number",   "default": "0"},
            {"key": "manifolds",      "label": "Expresión de DisPerSE sobre los manifolds a exportar",        "type": "path",   "default": "J2d"},
            {"key": "loadMSC", "label": "Fichero del complejo previamente generado (.MSC)", "type": "path",   "default": ""},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)", "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "05",
        "name":  "Exportación de puntos críticos y manifolds",
        "desc":  "Exporta tantos los puntos críticos como los manifolds a fotmatos .vtp y .vtu",
        "group": "Análisis topológico",
        "params": [
            {"key": "sklInput",      "label": "Fichero con los puntos críticos (.up.NDskl)",         "type": "path",   "default": "criticos.NDkel", "depends_on": {"phase": "04", "key": "input"}},
            {"key": "manifoldsInput",      "label": "Fichero con los manifolds (.NDnet)",         "type": "path",   "default": "manifolds.vtu", "depends_on": {"phase": "04", "key": "input"}},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "06",
        "name":  "Separación de los diferentes manifolds",
        "desc":  "Separa los descending manifolds en diferentes ficheros .clust.",
        "group": "Post-process",
        "params": [
            {"key": "input",      "label": "Fichero VTU de entrada (.vtu)",         "type": "path",   "default": "input.vtu", "depends_on": {"phase": "05", "key": "outputDir"}},
            {"key": "outputDir",  "label": "Directorio de salida",                   "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "07",
        "name":  "Filtrado de los puntos críticos relevantes",
        "desc":  "Selecciona los máximos relevantes en un fichero .slt.",
        "group": "Post-process",
        "params": [
            {"key": "input",      "label": "Fichero VTP de entrada (.vtp)", "type": "path",   "default": "input.vtp", "depends_on": {"phase": "05", "key": "outputDir"}},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)", "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida", "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
            {"key": "interactive",   "label": "¿Seleccionar de forma interactiva el filtro de valor?",   "type": "bool",   "default": False},
            {"key": "low_field_limits",  "label": "Limite inferior para el valor del punto crítico (-1 es para no aplicar filtro)", "type": "number", "default": -1, "min": -1, "max": 9999, "show_if": {"key": "interactive", "equals": False}},
            {"key": "upper_field_limits",  "label": "Limite superior para el valor del punto crítico (-1 es para no aplicar filtro)", "type": "number", "default": -1, "min": -1, "max": 9999, "show_if": {"key": "interactive", "equals": False}}
        ],
    },
    {
        "id":    "08",
        "name":  "Clasificación de los manifolds asociados a los puntos críticos seleccionados",
        "desc":  "Filtra los manifolds para quedarse solamente con aquellos de los puntos críticos seleccionados.",
        "group": "Post-process",
        "params": [
            {"key": "inputDir",      "label": "Directorio de entrada donde se encuentran los manifolds", "type": "path",   "default": ".", "depends_on": {"phase": "06", "key": "outputDir"}},
            {"key": "reference",      "label": "Fichero de referencia con los puntos críticos seleccionados (.slt)", "type": "path",   "default": "maxpoints.slt", "depends_on": {"phase": "07", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida", "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "09",
        "name":  "Transformación de los manifolds al espacio latente",
        "desc":  "Transforma los manifolds de los ficheros .clust en otros manifolds relativos al espacio latente en el formato de embeddings (.temb)",
        "group": "Post-process",
        "params": [
            {"key": "inputDir",      "label": "Directorio de entrada donde se encuentran los manifolds", "type": "path",   "default": ".", "depends_on": {"phase": "08", "key": "outputDir"}},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)", "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "sizeXimg", "label": "Tamaño del espacio latente en el eje X", "type": "number", "default": 2000, "min": 1, "max": 9999, "depends_on": {"phase": "03", "key": "size_x"}},
            {"key": "sizeYimg", "label": "Tamaño del espacio latente en el eje Y", "type": "number", "default": 2000, "min": 1, "max": 9999, "depends_on": {"phase": "03", "key": "size_y"}},
            {"key": "referenceEmb",      "label": "Fichero de embeddings del tomograma de referencia (.temb)", "type": "path",   "default": "tomograma.temb", "depends_on": {"phase": "02", "key": "input"}},
            {"key": "metadata",      "label": "Fichero de metadatos (.mdata)", "type": "path",   "default": "maxpoints.mdata", "depends_on": {"phase": "03", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida", "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "10",
        "name":  "Creación de una imagen PNG con los manifolds sobre el espacio latente",
        "desc":  "Crea una imagen PNG que contiene los manifolds sobre el espacio latente.",
        "group": "Post-process",
        "params": [
            {"key": "inputDir",      "label": "Directorio de entrada donde se encuentran los manifolds con coordenadas de la imagen del espacio latente", "type": "path",   "default": ".", "depends_on": {"phase": "08", "key": "outputDir"}},
            {"key": "referenceImg",      "label": "Imagen PNG del espacio latente (.png)", "type": "path",   "default": "espacio_latente.png", "depends_on": {"phase": "03", "key": "input"}},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)", "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "outputDir",  "label": "Directorio de salida", "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
    {
        "id":    "11",
        "name":  "Creación del archivo MRC con la segmentación",
        "desc":  "Crea un archivo MRC igual que el del tomograma de referencia con la segmentación.",
        "group": "Post-process",
        "params": [
            {"key": "inputDir",      "label": "Directorio de entrada donde se encuentran los manifolds en coordenadas del espacio latente", "type": "path",   "default": ".", "depends_on": {"phase": "09", "key": "outputDir"}},
            {"key": "reference",      "label": "Fichero MRC del tomograma de referencia (.mrc)", "type": "path",   "default": "tomograma.mrc", "depends_on": {"phase": "01", "key": "input"}},
            {"key": "outName",    "label": "Nombre base de salida (sin extensión)", "type": "text",   "default": "resultado", "depends_on": {"phase": "__global__", "key": "outName"}},
            {"key": "separate",   "label": "¿Separar cada manifold en un fichero MRC distinto?",   "type": "bool",   "default": False},
            {"key": "outputDir",  "label": "Directorio de salida", "type": "path",   "default": ".", "depends_on": {"phase": "__global__", "key": "outputDir"}},
        ],
    },
]


# ─────────────────────────────────────────
#  UTILIDADES DE PRESENTACIÓN
# ─────────────────────────────────────────

def clear():
    console.clear()

def banner():
    """Cabecera principal del workflow runner."""
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    title = Text()
    title.append(" CRYO-ET TOMOGRAM ANALYSIS WORKFLOW  ", style=f"bold {PALETTE['primary']}")

    subtitle = Text(f"  {now}", style=f"dim {PALETTE['muted']}")

    console.print()
    console.print(Panel(
        Align.left(title + "\n" + subtitle),
        border_style=PALETTE["secondary"],
        padding=(0, 1),
        box=box.HEAVY,
    ))

def phase_table(phases, selected_ids=None):
    """Tabla visual de todas las fases del workflow."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=PALETTE["muted"],
        header_style=f"bold {PALETTE['secondary']}",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("#",     style="dim",                         width=4)
    table.add_column("Fase",  style=f"bold",                      min_width=26)
    table.add_column("Etapa", style=f"dim {PALETTE['muted']}",    width=16)
    table.add_column("Params",style=f"dim {PALETTE['muted']}",    width=7)
    table.add_column("",                                           width=4)
 
    groups_seen = {}
    for p in phases:
        # Marca de selección
        if selected_ids is None:
            mark = f"[{PALETTE['primary']}]●[/]"
        elif p["id"] in selected_ids:
            mark = f"[{PALETTE['success']}]✔[/]"
        else:
            mark = f"[{PALETTE['muted']}]○[/]"
 
        n_params = len(p["params"])
        params_text = (
            f"[{PALETTE['accent']}]{n_params}[/]"
            if n_params > 0
            else f"[{PALETTE['muted']}]—[/]"
        )
 
        # Separador de grupo
        if p["group"] not in groups_seen:
            groups_seen[p["group"]] = True
            table.add_section()
 
        table.add_row(
            p["id"],
            p["name"],
            p["group"],
            params_text,
            mark,
        )
 
    console.print(table)

def ask_params(phase, all_params=None, selected_ids=None):
    """Solicita al usuario los parámetros de una fase.
 
    Si un parámetro tiene 'depends_on' y la fase de origen ya fue
    configurada, el valor se hereda automáticamente sin preguntar.
    """
    if not phase["params"]:
        return {}
 
    all_params   = all_params   or {}
    selected_ids = selected_ids or set()
 
    # Separar parámetros heredados de los que hay que preguntar
    inherited = {}
    to_ask    = []
    for param in phase["params"]:
        dep = param.get("depends_on")
        if dep:
            src_phase = dep["phase"]
            src_key   = dep["key"]
            print()
            if src_phase == "__global__" and src_key in all_params.get("__global__", {}):
                inherited[param["key"]] = all_params["__global__"][src_key]
                continue
            if src_phase in selected_ids and src_key in all_params.get(src_phase, {}):
                inherited[param["key"]] = all_params[src_phase][src_key]
                continue
        to_ask.append(param)
 
    # Cabecera solo si hay algo que mostrar
    n_heredados = len(inherited)
    n_preguntas = len(to_ask)
    if n_heredados + n_preguntas == 0:
        return {}
 
    values = {}
    
    # Cargar heredados 
    for key, val in inherited.items():
        values[key] = val

    if to_ask:
        console.print(f"\n  [bold {PALETTE['accent']}]Parámetros — {phase['name']}[/]")
        console.print()
 
    # Preguntar el resto
    for param in to_ask:
        show_if = param.get("show_if")
        if show_if:
            show_key = show_if["key"]
            show_val = show_if["equals"]
            # El valor puede estar ya resuelto en esta misma fase
            if values.get(show_key, show_val) != show_val:
                # No se cumple la condición → saltar con el default
                values[param["key"]] = param["default"]
                continue
        key     = param["key"]
        label   = param["label"]
        ptype   = param["type"]
        default = param["default"]
 
        if ptype == "bool":
            val = questionary.confirm(
                f"  {label}",
                default=default,
                style=CUSTOM_STYLE,
            ).ask()
 
        elif ptype == "path":
            raw = questionary.path(
                f"  {label}",
                default=str(default),
                style=CUSTOM_STYLE,
            ).ask()
            val = raw if raw is not None else str(default)
 
        elif ptype == "number":
            min_v = param.get("min", None)
            max_v = param.get("max", None)
            hint  = f"[{min_v}–{max_v}]" if (min_v is not None and max_v is not None) else ""
            while True:
                raw = questionary.text(
                    f"  {label} {hint}",
                    default=str(default),
                    style=CUSTOM_STYLE,
                ).ask()
                try:
                    val = float(raw) if "." in str(raw) else int(raw)
                    if min_v is not None and val < min_v:
                        console.print(f"  [{PALETTE['warning']}]  Valor mínimo: {min_v}[/]")
                        continue
                    if max_v is not None and val > max_v:
                        console.print(f"  [{PALETTE['warning']}]  Valor máximo: {max_v}[/]")
                        continue
                    break
                except ValueError:
                    console.print(f"  [{PALETTE['error']}]  Introduce un número válido.[/]")
 
        else:  # text
            val = questionary.text(
                f"  {label}",
                default=str(default),
                style=CUSTOM_STYLE,
            ).ask()
 
        values[key] = val

    return values


# ─────────────────────────────────────────
#  EJECUCIÓN DE FASES
# ─────────────────────────────────────────

def run_phase(phase, all_params, params, index, total):
    """Simula la ejecución de una fase con barra de progreso."""

    console.print()
    console.print(Rule(
        f"[bold {PALETTE['primary']}][{index}/{total}]  {phase['name']}[/]",
        style=PALETTE["muted"],
    ))
    console.print(f"  [dim]{phase['desc']}[/]\n")
    
    if phase["id"] == "01":   # TomoTwin analysis
        result = tomotwin.main(
            input     = params["input"],
            model   = params["model"],
            outputDir = params["outputDir"],
            stride      = params["stride"],
            batch = params["batch"]
        )
        params["temb"] = result
        
    elif phase["id"] == "02":   # UMAP reduction
        params_01 = all_params.get("01", None)
        if params_01 is not None:
            params["input"] = params_01["temb"]
        result_umap, result_emb = umap.main(
            input     = params["input"],
            chunk_size   = params["chunk_size"],
            fit_sample_size = params["fit_sample_size"],
            outputDir = params["outputDir"],
        )
        params["tumap"] = result_umap
        params["temb"] = result_emb

    elif phase["id"] == "03":   # Cloud Points
        params_02 = all_params.get("02", None)
        if params_02 is not None:
            params["input"] = params_02["tumap"]
        mask, img, metadata = cloudpoints.main(
            input     = params["input"],
            outName   = params["outName"],
            outputDir = params["outputDir"],
            auto_boundingbox = params["auto_bounding"],
            interactive_boundingbox = params["interactive_bb"],
            bounding_box = (params["xlow_bb"], params["xhi_bb"], params["ylow_bb"], params["yhi_bb"]),
            interactive_threshold = params["interactive_threshold"],
            size      = (int(params["size_x"]), int(params["size_y"])),
            threshold = int(params["threshold"]),
            sigma     = int(params["sigma"]),
        )
        params["mask"] = mask
        params["img"] = img 
        params["metadata"] = metadata
    elif phase["id"] == "04":   # MSE analysis
        params_03 = all_params.get("03", None)
        if params_03 is not None:
            params["input"] = params_03["img"]
            params["mask"] = params_03["mask"]
        skl, manifolds = mse.main(
            input = params["input"],
            mask = params["mask"],
            cut = params["cut"],
            manifolds = params["manifolds"],
            loadMSC = None if (params["loadMSC"] == "") else params["loadMSC"],
            outName = params["outName"],
            outputDir = params["outputDir"],
        )
        params["skl"] = skl
        params["manifolds"] = manifolds
    elif phase["id"] == "05":   # Convert to vtp and vtu
        params_04 = all_params.get("04", None)
        if params_04 is not None:
            params["sklInput"] = params_04["skl"]
            params["manifoldsInput"] = params_04["manifolds"]
        
        skl, manifolds = convertvtk.main(
            sklInput     = params["sklInput"],
            manifoldsInput     = params["manifoldsInput"],
            outputDir = params["outputDir"],
        )
        params["vtp"] = skl
        params["vtu"] = manifolds
    elif phase["id"] == "06":   # Cluster manifolds
        outputDirPath = Path(params["outputDir"]) / ("manifolds")
        params["outputDir"] = str(outputDirPath)
        params_05 = all_params.get("05", None)
        if params_05 is not None:
            params["input"] = params_05["vtu"]

        createmanifolds.main(
            input     = params["input"],
            outputDir = params["outputDir"],
        )
    elif phase["id"] == "07":   # Filter max
        if params["low_field_limits"] == -1 or params["upper_field_limits"] == -1:
            field_limits = None
        else:
            field_limits = (params["low_field_limits"], params["upper_field_limits"])
        params_05 = all_params.get("05", None)
        if params_05 is not None:
            params["input"] = params_05["vtp"]
        result = maxpoints.main(
            input     = params["input"],
            outputName   = params["outName"],
            outputDir = params["outputDir"],
            interactive = params["interactive"],
            field_limits = field_limits
        )
        params["result"] = result
    elif phase["id"] == "08":   # Filter manifolds
        outputDirPath = Path(params["outputDir"]) / ("manifolds_filtered")
        params["outputDir"] = str(outputDirPath)
        params_06 = all_params.get("06", None)
        if params_06 is not None:
            params["inputDir"] = params_06["outputDir"]
        params_07 = all_params.get("07", None)
        if params_07 is not None:
            params["reference"] = params_07["result"]
        filtermanifolds.main(
            inputDir     = params["inputDir"],
            reference   = params["reference"],
            outputDir = params["outputDir"],
        )
    elif phase["id"] == "09":   # Transform manifolds
        outputDirPath = Path(params["outputDir"]) / ("manifolds_filtered_emb")
        params["outputDir"] = str(outputDirPath)
        params_08 = all_params.get("08", None)
        if params_08 is not None:
            params["inputDir"] = params_08["outputDir"]
        params_02 = all_params.get("02", None)
        if params_02 is not None:
            params["referenceUMAP"] = params_02["tumap"]
            params["referenceEmb"] = params_02["temb"]
        params_03 = all_params.get("03", None)
        if params_03 is not None:
            params["metadata"] = params_03["metadata"]
        recover.main(
            input = None,
            inputDir     = params["inputDir"],
            imgXsize = params["sizeXimg"],
            referenceEmb   = params["referenceEmb"],
            metadata = params["metadata"],
            outputDir = params["outputDir"],
        )
    elif phase["id"] == "10":   # Create PNG
        params_08 = all_params.get("08", None)
        if params_08 is not None:
            params["inputDir"] = params_08["outputDir"]
        params_03 = all_params.get("03", None)
        if params_03 is not None:
            params["referenceImg"] = str(Path(params_03["img"]).with_suffix(".png"))
        print(params)
        showmanifolds.main(
            inputDir     = params["inputDir"],
            referenceImg   = params["referenceImg"],
            outName = params["outName"] + "_manifolds",
            outputDir = params["outputDir"],
        )
    elif phase["id"] == "11":   # Create MRC
        outputDirPath = Path(params["outputDir"]) / ("manifolds_mrc")
        params["outputDir"] = str(outputDirPath)
        params_09 = all_params.get("09", None)
        if params_09 is not None:
            params["inputDir"] = params_09["outputDir"]
        params_02 = all_params.get("02", None)
        if params_02 is not None:
            params["reference"] = params_02["tumap"]
        createmrc.main(
            input       = None,
            inputDir     = params["inputDir"],
            reference   = params["reference"],
            separate = params["separate"],
            outName = params["outName"],
            outputDir = params["outputDir"],
        )
    else:
        raise ValueError(f"Error en la fase {id}, no está registrada")
    console.print(f"\n  [{PALETTE['success']}]✔  Fase completada correctamente[/]")


# ─────────────────────────────────────────
#  RESUMEN FINAL
# ─────────────────────────────────────────

def summary(results):
    """Muestra tabla resumen al finalizar el workflow."""
    console.print()
    console.print(Rule(f"[bold {PALETTE['accent']}]RESUMEN DE EJECUCIÓN[/]", style=PALETTE["muted"]))
    console.print()
 
    table = Table(
        box=box.MINIMAL_DOUBLE_HEAD,
        border_style=PALETTE["muted"],
        header_style=f"bold {PALETTE['secondary']}",
        padding=(0, 2),
    )
    table.add_column("Fase",   min_width=28)
    table.add_column("Estado", width=14)
    table.add_column("Params", width=8)
 
    for r in results:
        estado = f"[{PALETTE['success']}]✔  OK[/]" if r["ok"] else f"[{PALETTE['error']}]✖  Error[/]"
        n_params = str(len(r["params"])) if r["params"] else "—"
        table.add_row(
            r['name'],
            estado,
            n_params,
        )
 
    console.print(table)
    total_ok = sum(1 for r in results if r["ok"])
    console.print(f"\n  [bold {PALETTE['primary']}]{total_ok}/{len(results)} fases completadas[/]  ·  "
                  f"[dim]{datetime.now().strftime('%H:%M:%S')}[/]")
    console.print()



# ─────────────────────────────────────────
#  FLUJO PRINCIPAL
# ─────────────────────────────────────────

def main():
    while True:
        clear()
        banner()
 
        # ── 1. Selección de modo ────────────────
        console.print()
        mode = questionary.select(
            "  ¿Cómo quieres ejecutar el workflow?",
            choices=[
                questionary.Choice("Ejecutar el workflow completo",     value="all"),
                questionary.Choice("Seleccionar etapas manualmente",            value="groups"),
                questionary.Choice("Seleccionar fases manualmente",          value="pick"),
                questionary.Choice("Ver fases disponibles (solo info)",      value="info"),
                questionary.Choice("Salir",                                  value="exit"),
            ],
            style=CUSTOM_STYLE,
            use_indicator=True,
        ).ask()
 
        if mode is None or mode == "exit":
            console.print(f"\n  [dim {PALETTE['muted']}]Hasta pronto.[/]\n")
            sys.exit(0)
 
        # ── 2. Vista informativa → vuelve al menú ──
        if mode == "info":
            console.print()
            console.print(Rule(f"[bold {PALETTE['secondary']}]FASES DISPONIBLES[/]", style=PALETTE["muted"]))
            console.print()
            phase_table(PHASES)
            console.print()
            questionary.press_any_key_to_continue(
                "  Pulsa cualquier tecla para volver al menú…",
                style=CUSTOM_STYLE,
            ).ask()
            continue
 
        # ── 3. Elegir fases ─────────────────────
        if mode == "all":
            selected = [p["id"] for p in PHASES]
 
        elif mode == "groups":
            # Obtener grupos únicos preservando orden de aparición
            seen_groups = {}
            for p in PHASES:
                g = p["group"]
                if g not in seen_groups:
                    seen_groups[g] = []
                seen_groups[g].append(p["id"])
 
            group_choices = [
                questionary.Choice(
                    title=f"{g}  ({len(ids)} fase(s): {', '.join(ids)})",
                    value=g,
                )
                for g, ids in seen_groups.items()
            ]
            selected_groups = questionary.checkbox(
                "  Selecciona los grupos a ejecutar (espacio = marcar, enter = confirmar):",
                choices=group_choices,
                style=CUSTOM_STYLE,
            ).ask()
 
            if not selected_groups:
                console.print(f"\n  [{PALETTE['warning']}]No se seleccionó ningún grupo. Volviendo al menú.[/]")
                time.sleep(1.2)
                continue
 
            # Expandir grupos a fases, respetando el orden original
            selected = [p["id"] for p in PHASES if p["group"] in selected_groups]
 
            # Mostrar resumen de lo que se va a ejecutar
            console.print()
            for g in selected_groups:
                fases = [p["name"] for p in PHASES if p["group"] == g]
                console.print(f"  [{PALETTE['primary']}]{g}[/]  →  [dim]{'  ·  '.join(fases)}[/]")
 
        else:  # pick
            choices = [
                questionary.Choice(
                    title=f"[{p['id']}]  {p['name']}  ({p['group']})",
                    value=p["id"],
                )
                for p in PHASES
            ]
            selected = questionary.checkbox(
                "  Selecciona las fases a ejecutar (espacio = marcar, enter = confirmar):",
                choices=choices,
                style=CUSTOM_STYLE,
            ).ask()
 
            if not selected:
                console.print(f"\n  [{PALETTE['warning']}]No se seleccionó ninguna fase. Volviendo al menú.[/]")
                time.sleep(1.2)
                continue
 
        # ── 4. Confirmar selección ───────────────
        selected_phases = [p for p in PHASES if p["id"] in selected]
 
        console.print()
        console.print(Rule(f"[bold {PALETTE['secondary']}]FASES SELECCIONADAS[/]", style=PALETTE["muted"]))
        console.print()
        phase_table(PHASES, selected_ids=selected)
 
        console.print()
        ok = questionary.confirm(
            f"  Se ejecutarán {len(selected_phases)} fase(s). ¿Continuar?",
            default=True,
            style=CUSTOM_STYLE,
        ).ask()
 
        if not ok:
            console.print(f"\n  [{PALETTE['warning']}]Cancelado. Volviendo al menú.[/]")
            time.sleep(1.0)
            continue   # vuelve al menú en vez de salir
 
        # ── 5. Recopilar parámetros ──────────────
        console.print()
        console.print(Rule(f"[bold {PALETTE['accent']}]CONFIGURACIÓN DE PARÁMETROS[/]", style=PALETTE["muted"]))
 
        all_params   = {}
        selected_ids = {p["id"] for p in selected_phases}
 
        # Parámetros globales: se preguntan una sola vez
        if GLOBAL_PARAMS:
            console.print(f"\n  [bold {PALETTE['secondary']}]Parámetros globales[/]\n")
            global_values = {}
            for param in GLOBAL_PARAMS:
                ptype   = param["type"]
                default = param["default"]
                label   = param["label"]
                if ptype == "path":
                    raw = questionary.path(
                        f"  {label}",
                        default=str(default),
                        style=CUSTOM_STYLE,
                    ).ask()
                    val = raw if raw is not None else str(default)
                elif ptype == "bool":
                    val = questionary.confirm(f"  {label}", default=bool(default), style=CUSTOM_STYLE).ask()
                elif ptype == "number":
                    raw = questionary.text(f"  {label}", default=str(default), style=CUSTOM_STYLE).ask()
                    val = int(raw) if raw.isdigit() else float(raw)
                else:
                    val = questionary.text(f"  {label}", default=str(default), style=CUSTOM_STYLE).ask()
                global_values[param["key"]] = val
            all_params["__global__"] = global_values
            console.print()
 
        # Parámetros por fase
        for phase in selected_phases:
            if phase["params"]:
                all_params[phase["id"]] = ask_params(phase, all_params, selected_ids)
            else:
                all_params[phase["id"]] = {}
 
        # ── 6. Confirmación final ────────────────
        console.print()
        console.print(Rule(style=PALETTE["muted"]))
        go = questionary.confirm(
            "  ¿Lanzar el workflow ahora?",
            default=True,
            style=CUSTOM_STYLE,
        ).ask()
 
        if not go:
            console.print(f"\n  [{PALETTE['warning']}]Cancelado.[/]\n")
            sys.exit(0)
 
        # ── 7. Ejecución ─────────────────────────
        console.print()
        console.print(Rule(f"[bold {PALETTE['primary']}]EJECUTANDO WORKFLOW[/]", style=PALETTE["muted"]))
 
        results = []
        for i, phase in enumerate(selected_phases, 1):
            params = all_params.get(phase["id"], {})
            try:
                run_phase(phase, all_params, params, i, len(selected_phases))
                results.append({**phase, "ok": True, "params": params})
            except Exception as e:
                console.print(f"\n  [{PALETTE['error']}]✖  Error en fase {phase['name']}: {e}[/]")
                results.append({**phase, "ok": False, "params": params})
 
                abort = questionary.confirm(
                    "  ¿Abortar el workflow?",
                    default=False,
                    style=CUSTOM_STYLE,
                ).ask()
                if abort:
                    break
 
        # ── 8. Resumen ───────────────────────────
        summary(results)
        break  # workflow completado, salir del bucle
 
 
# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(f"\n\n  [dim {PALETTE['muted']}]Interrumpido por el usuario.[/]\n")
        sys.exit(0)
