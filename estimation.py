import json
from pathlib import Path

import gempy as gp
import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from scipy.spatial import cKDTree

from model import (
    GRID_RESOLUTION,
    OUTPUT_DIR,
    _calculate_extent,
    _load_csv_templates,
    build_model as build_geological_model,
)


COLLARS_PATH = Path("data/raw_collars.csv")
ASSAYS_PATH = Path("data/raw_assays.csv")
GRADE_BLOCK_PATH = OUTPUT_DIR / "grade_block_model.csv"
VARIOGRAM_PATH = OUTPUT_DIR / "grade_variogram.json"
VARIOGRAM_LAGS = 5
MIN_DIRECTIONAL_BINS = 3
MAX_ANISOTROPY_RATIO = 5.0
NEIGHBORS_PER_BLOCK = 6
MIN_NEIGHBORS = 3
SEARCH_MULTIPLIERS = (1.0, 2.0, 3.0)


def _load_grade_samples() -> pd.DataFrame:
    if not COLLARS_PATH.exists() or not ASSAYS_PATH.exists():
        raise FileNotFoundError(
            f"Expected collar and assay CSVs at {COLLARS_PATH} and {ASSAYS_PATH}."
        )

    collars = pd.read_csv(COLLARS_PATH, dtype={"hole_id": "string"})
    assays = pd.read_csv(ASSAYS_PATH, dtype={"hole_id": "string"})
    required_collars = {
        "hole_id", "easting", "northing", "elevation", "dip", "azimuth"
    }
    required_assays = {"hole_id", "from_m", "to_m", "au_gpt"}
    missing_collars = required_collars - set(collars.columns)
    missing_assays = required_assays - set(assays.columns)
    if missing_collars or missing_assays:
        missing = []
        if missing_collars:
            missing.append(f"collars: {', '.join(sorted(missing_collars))}")
        if missing_assays:
            missing.append(f"assays: {', '.join(sorted(missing_assays))}")
        raise ValueError("Missing required column(s): " + "; ".join(missing))

    collars["hole_id"] = collars["hole_id"].str.strip()
    assays["hole_id"] = assays["hole_id"].str.strip()

    numeric_columns = {
        "collars": (collars, ["easting", "northing", "elevation", "dip", "azimuth"]),
        "assays": (assays, ["from_m", "to_m", "au_gpt"]),
    }
    for table_name, (table, columns) in numeric_columns.items():
        for column in columns:
            try:
                table[column] = pd.to_numeric(table[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{table_name} column {column!r} must contain numeric values."
                ) from error

    if collars["hole_id"].duplicated().any():
        raise ValueError("Each collar row must have a unique hole_id.")

    unmatched_holes = sorted(set(assays["hole_id"].dropna()) - set(collars["hole_id"].dropna()))
    if unmatched_holes:
        raise ValueError(
            "Assay hole_id values have no matching collar: "
            + ", ".join(unmatched_holes)
        )

    samples = assays.merge(
        collars,
        on="hole_id",
        how="inner",
        validate="many_to_one",
    )
    samples = samples.dropna(
        subset=["from_m", "to_m", "au_gpt", "easting", "northing", "elevation", "dip", "azimuth"]
    ).copy()
    if (samples["to_m"] <= samples["from_m"]).any():
        raise ValueError("Every assay interval must have to_m greater than from_m.")

    midpoint = (samples["from_m"] + samples["to_m"]) / 2.0
    dip = np.deg2rad(samples["dip"].to_numpy(dtype=float))
    azimuth = np.deg2rad(samples["azimuth"].to_numpy(dtype=float))
    horizontal_distance = midpoint.to_numpy(dtype=float) * np.cos(dip)
    samples["X"] = samples["easting"] + horizontal_distance * np.sin(azimuth)
    samples["Y"] = samples["northing"] + horizontal_distance * np.cos(azimuth)
    samples["Z"] = samples["elevation"] + midpoint.to_numpy(dtype=float) * np.sin(dip)

    # Average coincident samples so the kriging system has unique coordinates.
    samples = (
        samples.groupby(["X", "Y", "Z"], as_index=False)
        .agg(au_gpt=("au_gpt", "mean"), sample_count=("au_gpt", "size"))
    )
    if len(samples) < MIN_NEIGHBORS:
        raise ValueError(
            f"At least {MIN_NEIGHBORS} distinct assay locations are required; "
            f"found {len(samples)}."
        )
    return samples


def _fit_variogram(samples: pd.DataFrame) -> tuple[dict[str, float | bool], dict[float, float]]:
    try:
        from skgstat import DirectionalVariogram, Variogram
    except ImportError as error:
        raise ImportError(
            "Grade estimation requires scikit-gstat. Install it in this environment "
            "with `pip install scikit-gstat`."
        ) from error

    coordinates = samples[["X", "Y", "Z"]].to_numpy(dtype=float)
    values = samples["au_gpt"].to_numpy(dtype=float)
    n_lags = min(VARIOGRAM_LAGS, max(3, len(samples) // 2))
    variogram = Variogram(
        coordinates,
        values,
        estimator="cressie",
        model="spherical",
        use_nugget=True,
        n_lags=n_lags,
        fit_method="trf",
    )
    omni_parameters = np.asarray(variogram.parameters, dtype=float)
    if len(omni_parameters) < 3 or not np.isfinite(omni_parameters[:3]).all():
        raise ValueError("scikit-gstat could not fit a finite spherical variogram.")

    omni_range, partial_sill, nugget = (float(value) for value in omni_parameters[:3])
    if omni_range <= 0 or partial_sill <= 0 or nugget < 0:
        raise ValueError("The fitted spherical variogram has invalid parameters.")

    directional_ranges: dict[float, float] = {}
    directional_coordinates = coordinates[:, :2]
    for azimuth in range(0, 180, 15):
        try:
            directional = DirectionalVariogram(
                directional_coordinates,
                values,
                estimator="cressie",
                model="spherical",
                use_nugget=True,
                n_lags=n_lags,
                fit_method="trf",
                azimuth=float(azimuth),
                tolerance=22.5,
                directional_model="triangle",
            )
            supported_bins = np.isfinite(
                np.asarray(directional.experimental, dtype=float)
            ).sum()
            if supported_bins < MIN_DIRECTIONAL_BINS:
                continue
            parameters = np.asarray(directional.parameters, dtype=float)
            if (
                len(parameters) >= 3
                and np.isfinite(parameters[:3]).all()
                and parameters[0] > 0
                and parameters[1] > 0
            ):
                directional_ranges[float(azimuth)] = float(parameters[0])
        except (ValueError, RuntimeError, FloatingPointError):
            continue

    ellipsoid = {
        "major_range_m": omni_range,
        "minor_range_m": omni_range,
        "vertical_range_m": omni_range,
        "azimuth_deg": 0.0,
        "anisotropy_fitted": False,
    }

    if directional_ranges:
        major_azimuth = max(directional_ranges, key=directional_ranges.get)
        perpendicular = (major_azimuth + 90.0) % 180.0
        perpendicular_azimuth = min(
            directional_ranges,
            key=lambda angle: min(
                abs(angle - perpendicular), 180.0 - abs(angle - perpendicular)
            ),
        )
        angular_difference = min(
            abs(perpendicular_azimuth - perpendicular),
            180.0 - abs(perpendicular_azimuth - perpendicular),
        )
        if angular_difference <= 15.0:
            major_range = directional_ranges[major_azimuth]
            minor_range = directional_ranges[perpendicular_azimuth]
            if minor_range > major_range:
                major_range, minor_range = minor_range, major_range
                major_azimuth = perpendicular_azimuth

            if major_range / minor_range >= 1.15:
                minor_range = max(minor_range, major_range / MAX_ANISOTROPY_RATIO)
                vertical_range = min(
                    max(minor_range, omni_range),
                    major_range * MAX_ANISOTROPY_RATIO,
                )
                ellipsoid = {
                    "major_range_m": float(major_range),
                    "minor_range_m": float(minor_range),
                    "vertical_range_m": float(vertical_range),
                    "azimuth_deg": float(major_azimuth),
                    "anisotropy_fitted": True,
                }

    metadata: dict[str, float | bool] = {
        "effective_range_m": omni_range,
        "kriging_range_m": float(ellipsoid["major_range_m"]),
        "partial_sill_gpt2": partial_sill,
        "nugget_gpt2": nugget,
        **ellipsoid,
    }
    return metadata, directional_ranges


def _rotate_and_scale(
    coordinates: np.ndarray,
    center: np.ndarray,
    azimuth_deg: float,
    scale_y: float,
    scale_z: float,
) -> np.ndarray:
    offsets = np.asarray(coordinates, dtype=float) - center
    angle = np.deg2rad(azimuth_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated_x = cosine * offsets[:, 0] - sine * offsets[:, 1]
    rotated_y = sine * offsets[:, 0] + cosine * offsets[:, 1]
    return np.column_stack(
        [rotated_x, rotated_y * scale_y, offsets[:, 2] * scale_z]
    )


def build_grade_block() -> gp.data.GeoModel:
    """Krig the assay grades onto the GemPy model grid and return its GeoModel.

    The block estimates and variogram/search parameters are written beneath
    model_outputs/. Three search passes enlarge the ellipsoid without changing
    the variogram. Cells with insufficient local support remain unestimated.
    """
    samples = _load_grade_samples()
    variogram, directional_ranges = _fit_variogram(samples)
    geological_model = build_geological_model()
    surface_points, _ = _load_csv_templates()
    extent = _calculate_extent(surface_points)

    major_range = float(variogram["major_range_m"])
    minor_range = float(variogram["minor_range_m"])
    vertical_range = float(variogram["vertical_range_m"])
    azimuth = float(variogram["azimuth_deg"])
    scale_y = major_range / minor_range
    scale_z = major_range / vertical_range

    coordinates = samples[["X", "Y", "Z"]].to_numpy(dtype=float)
    grades = samples["au_gpt"].to_numpy(dtype=float)
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2.0
    kriging = OrdinaryKriging3D(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        grades,
        variogram_model="spherical",
        variogram_parameters={
            "psill": float(variogram["partial_sill_gpt2"]),
            "range": float(variogram["kriging_range_m"]),
            "nugget": float(variogram["nugget_gpt2"]),
        },
        anisotropy_scaling_y=scale_y,
        anisotropy_scaling_z=scale_z,
        anisotropy_angle_z=-azimuth,
        pseudo_inv=True,
        pseudo_inv_type="pinv",
    )

    nx, ny, nz = (int(value) for value in GRID_RESOLUTION)
    x_centers = np.linspace(extent[0], extent[1], nx + 1)
    y_centers = np.linspace(extent[2], extent[3], ny + 1)
    z_centers = np.linspace(extent[4], extent[5], nz + 1)
    x_centers = (x_centers[:-1] + x_centers[1:]) / 2.0
    y_centers = (y_centers[:-1] + y_centers[1:]) / 2.0
    z_centers = (z_centers[:-1] + z_centers[1:]) / 2.0
    grid_x, grid_y, grid_z = np.meshgrid(
        x_centers, y_centers, z_centers, indexing="ij"
    )
    block_coordinates = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
    )

    transformed_samples = _rotate_and_scale(
        coordinates, center, azimuth, scale_y, scale_z
    )
    transformed_blocks = _rotate_and_scale(
        block_coordinates, center, azimuth, scale_y, scale_z
    )
    search_tree = cKDTree(transformed_samples)
    neighbor_counts = np.zeros(len(block_coordinates), dtype=np.int32)
    neighbors_used = np.zeros(len(block_coordinates), dtype=np.int32)
    search_pass = np.zeros(len(block_coordinates), dtype=np.int32)
    search_multiplier = np.full(len(block_coordinates), np.nan)
    neighbor_limit = min(NEIGHBORS_PER_BLOCK, len(samples))
    estimates = np.full(len(block_coordinates), np.nan, dtype=float)
    kriging_variance = np.full(len(block_coordinates), np.nan, dtype=float)
    for pass_number, multiplier in enumerate(SEARCH_MULTIPLIERS, start=1):
        pending = np.flatnonzero(search_pass == 0)
        if pending.size == 0:
            break
        counts = search_tree.query_ball_point(
            transformed_blocks[pending], r=major_range * multiplier,
            return_length=True,
        )
        neighbor_counts[pending] = counts
        # Group by neighborhood size so every PyKrige neighbor stays inside
        # the current ellipsoid, including cells with fewer than six samples.
        local_sizes = np.minimum(counts, neighbor_limit)
        for size in range(MIN_NEIGHBORS, neighbor_limit + 1):
            indices = pending[local_sizes == size]
            if indices.size == 0:
                continue
            estimated_values, estimated_variance = kriging.execute(
                "points",
                block_coordinates[indices, 0],
                block_coordinates[indices, 1],
                block_coordinates[indices, 2],
                backend="loop",
                n_closest_points=size,
            )
            estimates[indices] = np.asarray(estimated_values, dtype=float).reshape(-1)
            kriging_variance[indices] = np.asarray(
                estimated_variance, dtype=float
            ).reshape(-1)
            neighbors_used[indices] = size
            search_pass[indices] = pass_number
            search_multiplier[indices] = multiplier

    estimated_mask = search_pass > 0
    if not estimated_mask.any():
        raise ValueError(
            f"No block centers have {MIN_NEIGHBORS} samples within the largest "
            "search ellipsoid. Review the sample/grid coordinates and search settings."
        )

    cell_size = [
        (extent[1] - extent[0]) / nx,
        (extent[3] - extent[2]) / ny,
        (extent[5] - extent[4]) / nz,
    ]
    block_model = pd.DataFrame(
        {
            "block_id": np.arange(len(block_coordinates), dtype=np.int64),
            "X": block_coordinates[:, 0],
            "Y": block_coordinates[:, 1],
            "Z": block_coordinates[:, 2],
            "block_dx_m": cell_size[0],
            "block_dy_m": cell_size[1],
            "block_dz_m": cell_size[2],
            "au_gpt_est": estimates,
            "kriging_variance": kriging_variance,
            "neighbors_within_search": neighbor_counts,
            "neighbors_used": neighbors_used,
            "search_pass": search_pass,
            "search_multiplier": search_multiplier,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    block_model.to_csv(GRADE_BLOCK_PATH, index=False)
    metadata = {
        "variogram_model": "spherical",
        "estimator": "Cressie-Hawkins",
        "sample_count": int(len(samples)),
        "grid_resolution": [nx, ny, nz],
        "partial_sill_gpt2": float(variogram["partial_sill_gpt2"]),
        "nugget_gpt2": float(variogram["nugget_gpt2"]),
        "effective_range_m": float(variogram["effective_range_m"]),
        "kriging_range_m": float(variogram["kriging_range_m"]),
        "search_ellipsoid": {
            "major_range_m": major_range,
            "minor_range_m": minor_range,
            "vertical_range_m": vertical_range,
            "azimuth_deg_clockwise_from_east": azimuth,
            "anisotropy_fitted": bool(variogram["anisotropy_fitted"]),
            "directional_ranges_m": {
                str(int(angle)): float(value)
                for angle, value in directional_ranges.items()
            },
        },
        "neighbor_limit": int(neighbor_limit),
        "minimum_neighbors": MIN_NEIGHBORS,
        "search_multipliers": list(SEARCH_MULTIPLIERS),
        "blocks_per_search_pass": {
            str(number): int((search_pass == number).sum())
            for number in range(1, len(SEARCH_MULTIPLIERS) + 1)
        },
        "estimated_block_count": int(estimated_mask.sum()),
        "unestimated_block_count": int((~estimated_mask).sum()),
        "density_t_m3":2.8
    }
    VARIOGRAM_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"- Saved grade block model: {GRADE_BLOCK_PATH}")
    print(f"- Saved variogram and search ellipsoid: {VARIOGRAM_PATH}")
    if not variogram["anisotropy_fitted"]:
        print(
            "- Directional variograms did not support a stable horizontal anisotropy fit; "
            "using the omnidirectional range."
        )
    return geological_model


def draw_grade_blocks(
    csv_path: Path = GRADE_BLOCK_PATH,
    show: bool = True,
    off_screen: bool = False,
):
    """Display estimated voxel cells with Au grades and selectable labels."""
    import pyvista as pv

    blocks = pd.read_csv(csv_path)
    required = {
        "block_id", "X", "Y", "Z", "block_dx_m", "block_dy_m",
        "block_dz_m", "au_gpt_est", "kriging_variance",
    }
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Block CSV is missing columns: {', '.join(sorted(missing))}")

    axes = [np.sort(blocks[axis].unique()) for axis in ("X", "Y", "Z")]
    spacing = blocks[["block_dx_m", "block_dy_m", "block_dz_m"]].iloc[0].to_numpy(float)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Block dimensions must be finite and positive.")
    shape = np.array([len(axis) for axis in axes])
    origin = np.array([axis[0] for axis in axes]) - spacing / 2.0
    grid = pv.ImageData(dimensions=shape + 1, spacing=spacing, origin=origin)
    indices = [np.searchsorted(axis, blocks[column]) for axis, column in zip(axes, ("X", "Y", "Z"))]
    cell_ids = indices[0] + shape[0] * (indices[1] + shape[1] * indices[2])
    for name in ("au_gpt_est", "kriging_variance", "block_id"):
        values = np.full(grid.n_cells, np.nan)
        values[cell_ids] = blocks[name].to_numpy(float)
        grid.cell_data[name] = values
    valid = np.isfinite(grid.cell_data["au_gpt_est"])
    if not valid.any():
        raise ValueError("The block model contains no estimated grades to display.")
    voxels = grid.extract_cells(valid)
    voxels.cell_data["block_id"] = voxels.cell_data["block_id"].astype(np.int64)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voxels.save(OUTPUT_DIR / "grade_blocks.vtu")

    plotter = pv.Plotter(off_screen=off_screen, window_size=[1200, 900])
    plotter.set_background("white")
    plotter.add_mesh(
        voxels, scalars="au_gpt_est", preference="cell", cmap="YlOrRd",
        show_edges=True, edge_color="#666666", line_width=0.3,
        scalar_bar_args={"title": "Au grade (g/t)", "fmt": "%.2f", "color": "black"},
        name="gold_blocks",
    )
    plotter.show_bounds(
        grid="back", location="outer", color="black",
        xtitle="Easting (m)", ytitle="Northing (m)", ztitle="Elevation (m)",
    )
    plotter.add_axes(color="black")
    plotter.add_text("Gold grade block model", position="upper_left", color="black", font_size=16)
    centers = voxels.cell_centers().points

    def show_selected_grade(point, picker):
        cell_id = picker.GetCellId()
        if picker.GetDataSet() != voxels or cell_id < 0 or cell_id >= voxels.n_cells:
            return
        grade = voxels.cell_data["au_gpt_est"][cell_id]
        block_id = voxels.cell_data["block_id"][cell_id]
        plotter.add_point_labels(
            [point], [f"Block {block_id}: {grade:.3f} g/t Au"],
            name="selected_grade", font_size=16, text_color="black",
            shape_color="white", always_visible=True, show_points=False,
        )

    plotter.enable_surface_point_picking(
        callback=show_selected_grade, use_picker=True, left_clicking=True,
        show_message=False, show_point=False,
    )

    def toggle_labels(enabled):
        if enabled:
            plotter.add_point_labels(
                centers, [f"{grade:.2f}" for grade in voxels.cell_data["au_gpt_est"]],
                name="all_grades", font_size=10, text_color="black",
                shape_color="white", show_points=False, always_visible=False,
            )
        else:
            plotter.remove_actor("all_grades")

    plotter.add_checkbox_button_widget(toggle_labels, value=False, position=(15, 15), size=25)
    plotter.add_text("Grade labels (g/t Au)", position=(50, 18), font_size=10, color="black")
    plotter.view_isometric()
    plotter.reset_camera()
    if show:
        plotter.show()
    return plotter


def calculate() -> float:
    """
    According to the grade block model, calculate the total ounce of resources.
    :return: the total ounce
    """
    if not GRADE_BLOCK_PATH.exists():
        raise FileNotFoundError(f"Missing grade block model: {GRADE_BLOCK_PATH}")
    blocks = pd.read_csv(GRADE_BLOCK_PATH)
    dimensions = ["block_dx_m", "block_dy_m", "block_dz_m"]
    required = {"au_gpt_est", *dimensions}
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Block model is missing columns: {', '.join(sorted(missing))}")
    if "block_id" in blocks and blocks["block_id"].duplicated().any():
        raise ValueError("Duplicate block_id values would double-count contained gold.")

    grades = pd.to_numeric(blocks["au_gpt_est"], errors="raise")
    estimated = grades.notna()
    if not estimated.any():
        return 0.0
    grades = grades.loc[estimated]
    if not np.isfinite(grades).all() or (grades < 0).any():
        raise ValueError("Estimated gold grades must be finite and nonnegative.")
    sizes = blocks.loc[estimated, dimensions].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(sizes).all().all() or (sizes <= 0).any().any():
        raise ValueError("Estimated blocks must have finite, positive dimensions in metres.")

    if "density_t_m3" in blocks:
        density = pd.to_numeric(blocks.loc[estimated, "density_t_m3"], errors="raise")
    else:
        if not VARIOGRAM_PATH.exists():
            raise FileNotFoundError(f"Missing density metadata: {VARIOGRAM_PATH}")
        metadata = json.loads(VARIOGRAM_PATH.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or "density_t_m3" not in metadata:
            raise ValueError(
                "Provide density_t_m3 in the block CSV or the grade variogram metadata."
            )
        density = pd.Series(float(metadata["density_t_m3"]), index=grades.index)
    if not np.isfinite(density).all() or (density <= 0).any():
        raise ValueError("Bulk density must be finite and positive, in t/m3.")

    # Volume (m3) * density (t/m3) * grade (g/t) gives contained gold grams.
    ounces = sizes.prod(axis=1) * density * grades / 31.1034768
    total_ounces = float(ounces.sum())
    if not np.isfinite(ounces).all() or not np.isfinite(total_ounces):
        raise ValueError("Contained gold calculation produced a nonfinite result.")
    return total_ounces



if __name__ == "__main__":
    build_grade_block()
    # result = calculate()
    # print(result)