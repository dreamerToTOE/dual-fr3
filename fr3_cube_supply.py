"""Sequential supply arrivals with separate dynamic actors and optical identities."""
from __future__ import annotations


def add_cube_marker(stage, cube_path, cube_size_m, marker_size_m, marker_id):
    import cv2
    from pxr import Gf, UsdGeom

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if not isinstance(marker_id, int) or not 0 <= marker_id < len(dictionary.bytesList):
        raise ValueError("Cube marker ID is outside DICT_4X4_50")
    pixels = (cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
              if hasattr(cv2.aruco, "generateImageMarker")
              else cv2.aruco.drawMarker(dictionary, marker_id, 6))
    if cube_size_m <= 0 or not 0 < marker_size_m < cube_size_m / 1.24:
        raise ValueError("Cube marker and its quiet zone must fit on the top face")
    root = f"{cube_path}/ArUcoMarker"
    UsdGeom.Xform.Define(stage, root)
    backing = UsdGeom.Cube.Define(stage, f"{root}/white_backing")
    backing.CreateSizeAttr(1.)
    backing_xf = UsdGeom.Xformable(backing.GetPrim())
    backing_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, .5 + .0004 / cube_size_m))
    backing_xf.AddScaleOp().Set(Gf.Vec3f(
        1.24 * marker_size_m / cube_size_m, 1.24 * marker_size_m / cube_size_m, .001 / cube_size_m))
    backing.CreateDisplayColorAttr([Gf.Vec3f(.98)])
    cell_size = marker_size_m / 6 / cube_size_m
    for row in range(6):
        for column in range(6):
            cell = UsdGeom.Cube.Define(stage, f"{root}/cell_{row}_{column}")
            cell.CreateSizeAttr(1.)
            xf = UsdGeom.Xformable(cell.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(
                (column - 2.5) * cell_size, (2.5 - row) * cell_size, .5 + .008 / cube_size_m))
            xf.AddScaleOp().Set(Gf.Vec3f(cell_size, cell_size, .001 / cube_size_m))
            cell.CreateDisplayColorAttr([Gf.Vec3f(.98 if pixels[row, column] else .02)])


def spawn_supply_cube(stage, world, config, cube_index, contact_monitor, *, add_marker=True):
    """Create one supply arrival; call initialize after the next controlled physics step."""
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, UsdGeom
    from fr3_ab_visual_runner import _box

    sequence = config["cube_sequence"]
    if not isinstance(cube_index, int) or not 0 <= cube_index < len(sequence):
        raise ValueError("Supply index must identify an entry in cube_sequence")
    entry = sequence[cube_index]
    path = f"/World/Cube_{entry['cube_id']}"
    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"Supply actor already exists: {path}")
    # DynamicCuboid silently adds a different friction material; use the first Cube's constructor.
    prim = _box(stage, path, config["supply_cube_center_m"], [config["cube_size_m"]] * 3,
                collision=True, dynamic=True, mass=config["physics"]["cube_mass_kg"])
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*config["row_colors_rgb"][cube_index])])
    body = world.scene.add(SingleRigidPrim(path, name=f"supply_cube_{entry['cube_id']}"))
    if add_marker:
        add_cube_marker(stage, path, config["cube_size_m"], config["vision"]["aruco_marker_size_m"], entry["marker_id"])
    contact_monitor.add_cube_reporter(prim)
    contact_monitor.set_active_cube(path)
    return body
