import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 1. Load the raw data extracted from JORC Table 1
    collars = pd.read_csv("data/raw_collars.csv")
    assays = pd.read_csv("data/raw_assays.csv")

    # 2. Deconstruct trajectories (Assuming straight-line projection for simplified initial models)
    # Converting geological survey angles to radians for calculation
    dip_rad = np.radians(collars["dip"])
    az_rad = np.radians(collars["azimuth"])

    # Merge collar locations into assay intervals
    drill_db = pd.merge(assays, collars, on="hole_id", suffixes=("", "_collar"))

    # 3. Calculate 3D XYZ Coordinates for the top of the mineralised intercept ("From" depth)
    # Length along hole to target depth
    depth = drill_db["from_m"]

    # Calculate vector offsets
    delta_z = depth * np.sin(dip_rad)
    horizontal_displacement = depth * np.cos(dip_rad)
    delta_x = horizontal_displacement * np.sin(az_rad)
    delta_y = horizontal_displacement * np.cos(az_rad)

    # Apply vector offsets directly to the original surface collar coordinates
    drill_db["X"] = drill_db["easting"] + delta_x
    drill_db["Y"] = drill_db["northing"] + delta_y
    drill_db["Z"] = drill_db["elevation"] + delta_z

    # 4. Generate the GemPy 'Surface Points' Template
    # Filter by a target grade boundary (e.g., modeling a 1.0 g/t or 2.0 g/t Au envelope shell)
    grade_threshold = 1.0
    surface_points = drill_db[drill_db["au_gpt"] >= grade_threshold].copy()

    # Map to defined surfaces (e.g., marking the Reedy or Boomerang structural trends)
    surface_points["surface"] = "Gold_Enriched_Zone"

    gempy_points = surface_points[["hole_id","X", "Y", "Z", "surface"]]
    gempy_points.to_csv("gempy_surface_points.csv", index=False)
    print("- Saved: gempy_surface_points.csv")

    # 5. Generate the GemPy 'Orientations' Template
    # GemPy maps dip direction instead of standard strike azimuth.
    # Dip Direction = Azimuth (if dipping forward). GemPy expects dip to be absolute/positive (0-90)
    orientations = collars.copy()
    orientations["dip"] = orientations["dip"].abs()
    orientations["dip_direction"] = orientations["azimuth"]

    # We use the computed intercept position to lock down where that local gradient field anchor lies
    # Merging calculated X, Y, Z from our target points to map the orientation anchors
    orientation_anchors = pd.merge(
        orientations, gempy_points, left_on="hole_id", right_index=True, how="inner"
    )
    orientation_anchors["surface"] = "Gold_Enriched_Zone"

    gempy_orientations = orientation_anchors[
        ["X", "Y", "Z", "surface", "dip", "dip_direction"]
    ]
    gempy_orientations.to_csv("gempy_orientations.csv", index=False)
    print("- Saved: gempy_orientations.csv")

    print("Data successfully formatted for GemPy import!")