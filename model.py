from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mining_matplotlib_cache")

import gempy as gp
import gempy_viewer as gpv
import numpy as np
import pandas as pd


PROJECT_NAME = "Murchison_Gold_Model"
SURFACE_POINTS_PATH = Path("gempy_surface_points.csv")
ORIENTATIONS_PATH = Path("gempy_orientations.csv")
OUTPUT_DIR = Path("model_outputs")
MODEL_PATH = OUTPUT_DIR / "murchison_gold_model.gempy"
GRID_RESOLUTION = [30, 30, 30]
PADDING_FRACTION = 0.15
MIN_PADDING_METRES = 10.0


def _require_columns(table: pd.DataFrame, required_columns: set[str], table_name: str) -> None:
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{table_name} is missing required column(s): {missing}")


def _load_csv_templates() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SURFACE_POINTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SURFACE_POINTS_PATH}. Run surfaces.py to create the GemPy templates first."
        )
    if not ORIENTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ORIENTATIONS_PATH}. Run surfaces.py to create the GemPy templates first."
        )

    surface_points = pd.read_csv(SURFACE_POINTS_PATH)
    orientations = pd.read_csv(ORIENTATIONS_PATH)

    _require_columns(surface_points, {"X", "Y", "Z", "surface"}, SURFACE_POINTS_PATH.name)
    _require_columns(
        orientations,
        {"X", "Y", "Z", "surface", "dip", "dip_direction"},
        ORIENTATIONS_PATH.name,
    )

    for table in (surface_points, orientations):
        for column in ("X", "Y", "Z"):
            table[column] = pd.to_numeric(table[column], errors="raise")
        table["surface"] = table["surface"].astype("string").str.strip()

    orientations["dip"] = pd.to_numeric(orientations["dip"], errors="raise")
    orientations["dip_direction"] = pd.to_numeric(
        orientations["dip_direction"], errors="raise"
    )
    orientations["polarity"] = 1.0

    surface_points = surface_points.dropna(subset=["X", "Y", "Z", "surface"])
    orientations = orientations.dropna(
        subset=["X", "Y", "Z", "surface", "dip", "dip_direction"]
    )

    if surface_points.empty:
        raise ValueError(f"{SURFACE_POINTS_PATH} does not contain any surface points.")
    if orientations.empty:
        raise ValueError(f"{ORIENTATIONS_PATH} does not contain any orientations.")

    return surface_points, orientations


def _calculate_extent(surface_points: pd.DataFrame) -> list[float]:
    coords = surface_points[["X", "Y", "Z"]]
    mins = coords.min()
    maxs = coords.max()
    padding = (maxs - mins).mul(PADDING_FRACTION).clip(lower=MIN_PADDING_METRES)

    return [
        mins["X"] - padding["X"],
        maxs["X"] + padding["X"],
        mins["Y"] - padding["Y"],
        maxs["Y"] + padding["Y"],
        mins["Z"] - padding["Z"],
        maxs["Z"] + padding["Z"],
    ]


def _orientation_to_pole_vector(orientations: pd.DataFrame) -> np.ndarray:
    dip = np.deg2rad(orientations["dip"].to_numpy(dtype=float))
    dip_direction = np.deg2rad(orientations["dip_direction"].to_numpy(dtype=float))
    polarity = orientations["polarity"].to_numpy(dtype=float)

    return np.column_stack(
        [
            np.sin(dip) * np.sin(dip_direction) * polarity,
            np.sin(dip) * np.cos(dip_direction) * polarity,
            np.cos(dip) * polarity,
        ]
    )


def _build_structural_frame(
    surface_points: pd.DataFrame, orientations: pd.DataFrame
) -> gp.data.StructuralFrame:
    surface_points_table = gp.data.SurfacePointsTable.from_arrays(
        x=surface_points["X"].to_numpy(dtype=float),
        y=surface_points["Y"].to_numpy(dtype=float),
        z=surface_points["Z"].to_numpy(dtype=float),
        names=surface_points["surface"].to_numpy(dtype=str),
    )

    pole_vectors = _orientation_to_pole_vector(orientations)
    orientations_table = gp.data.OrientationsTable.from_arrays(
        x=orientations["X"].to_numpy(dtype=float),
        y=orientations["Y"].to_numpy(dtype=float),
        z=orientations["Z"].to_numpy(dtype=float),
        G_x=pole_vectors[:, 0],
        G_y=pole_vectors[:, 1],
        G_z=pole_vectors[:, 2],
        names=orientations["surface"].to_numpy(dtype=str),
        name_id_map=surface_points_table.name_id_map,
    )

    return gp.data.StructuralFrame.from_data_tables(
        surface_points_table, orientations_table
    )


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def _export_meshes(geo_model: gp.data.GeoModel, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []

    if geo_model.solutions is None or geo_model.solutions.dc_meshes is None:
        return exported_paths

    structural_elements = geo_model.structural_frame.structural_elements[:-1]
    for element, mesh in zip(structural_elements, geo_model.solutions.dc_meshes):
        if mesh is None:
            continue

        base_name = _safe_filename(element.name)
        vertices_path = output_dir / f"{base_name}_vertices.csv"
        faces_path = output_dir / f"{base_name}_faces.csv"

        pd.DataFrame(element.vertices, columns=["X", "Y", "Z"]).to_csv(
            vertices_path, index=False
        )
        pd.DataFrame(
            element.edges, columns=["vertex_0", "vertex_1", "vertex_2"]
        ).to_csv(faces_path, index=False)

        exported_paths.extend([vertices_path, faces_path])

    return exported_paths


def build_model() -> gp.data.GeoModel:
    surface_points, orientations = _load_csv_templates()
    structural_frame = _build_structural_frame(surface_points, orientations)

    geo_model = gp.create_geomodel(
        project_name=PROJECT_NAME,
        extent=_calculate_extent(surface_points),
        resolution=GRID_RESOLUTION,
        structural_frame=structural_frame,
    )

    gp.compute_model(geo_model)
    return geo_model


def draw_model(geo_model: gp.data.GeoModel, show: bool = True) -> gpv.GemPyToVista:
    return gpv.plot_3d(
        geo_model,
        show_lith=True,
        show_boundaries=True,
        show_data=True,
        show_topography=False,
        show=show,
        kwargs_plotter={"window_size": [1200, 900]},
        kwargs_plot_data={"arrow_size": 8},
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    saved_model_path = gp.save_model(
        model=model,
        path=str(MODEL_PATH),
        validate_serialization=False,
    )
    exported_mesh_paths = _export_meshes(model, OUTPUT_DIR)
    draw_model(model)

    print(f"- Saved GemPy model: {saved_model_path}")
    for path in exported_mesh_paths:
        print(f"- Saved mesh table: {path}")
    print("GemPy 3D model successfully built and drawn with GemPy Viewer.")
