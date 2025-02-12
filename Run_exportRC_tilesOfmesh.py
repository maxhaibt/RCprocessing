import os
import subprocess
import numpy as np
import time
import math
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import open3d as o3d
#from RClib import read_rcbox, write_rcbox

class RCCommandBuilder:
    TEMP_RCCMD_PATH = "E:/WES_L18_Boat/exportTestd/WES_L18_Boat_tiledexport/temp.rccmd"
    RC_EXECUTABLE = "\"C:/Program Files/Capturing Reality/RealityCapture/RealityCapture.exe\""

    def __init__(self):
        self.commands = []
    
    def add_command(self, command, *args):
        """Add a command with optional arguments."""
        cmd = f"-{command}"
        if args:
            cmd += " " + " ".join(map(str, args))
        self.commands.append(cmd)
        return self
    
    def write_rccmd_file(self):
        """Write the command list to a temporary .rccmd file."""
        with open(self.TEMP_RCCMD_PATH, 'w') as file:
            for cmd in self.commands:
                file.write(cmd + "\n")
        return self.TEMP_RCCMD_PATH
    
    def execute(self):
        """Execute the built command in RealityCapture using the .rccmd file with -delegate *."""
        rccmd_file = self.write_rccmd_file()
        command = f"{self.RC_EXECUTABLE} -delegateTo * -execRCCMD \"{rccmd_file}\""
        print("Executing command:", command)
        
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Executed RealityCapture command successfully: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing RealityCapture command: {e.stderr}")
            raise SystemExit(e.returncode)


def check_dtype(name, array):
    """
    Checks and prints the dtype of a given NumPy array.
    Alerts if there's an unintended conversion between float32 and float64.
    """
    dtype = array.dtype
    print(f"[CHECK] {name} dtype: {dtype}")
    if dtype == np.float32:
        print(f"[WARNING] {name} is float32! Potential precision loss.")
    elif dtype == np.float64:
        print(f"[INFO] {name} is float64.")

def transform_to_local(rcbox):
    """Transforms the RCBox to local space using its stored transformations."""
    translation = rcbox['transform_to_local']['translation_matrix']
    rotation = rcbox['transform_to_local']['rotation_matrix']
    
    # Use translation matrix as the implicit rotation center
    rcbox['geometry'].rotate(rotation, center=-translation)
    rcbox['geometry'].translate(translation, relative=True)

    return rcbox


def transform_to_global(rcbox):
    """Transforms the RCBox back to global space."""
    translation = rcbox['transform_to_local']['translation_matrix']
    rotation = rcbox['transform_to_local']['rotation_matrix']
    
    # Apply inverse translation first
    rcbox['geometry'].translate(-translation, relative=True)
    
    # Apply inverse rotation around the same implicit center
    rcbox['geometry'].rotate(rotation.T, center=-translation)

    return rcbox



def persistent_translate(rcbox, x, y, z):
    translation_matrix = np.array([x, y, z], dtype=np.float64)
    check_dtype("Translation matrix", translation_matrix)

    rcbox['geometry'].translate(translation_matrix, relative=True)
    rcbox['transform_to_local']['translation_matrix'] = rcbox['transform_to_local']['translation_matrix'] - translation_matrix

    check_dtype("Updated Translation Matrix", rcbox['transform_to_local']['translation_matrix'])
    return rcbox

def localspace_rotation(rcbox, yaw, pitch, roll):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    R = o3d.geometry.Geometry3D.get_rotation_matrix_from_yxz([yaw_rad, pitch_rad, roll_rad])
    check_dtype("Rotation Matrix R", np.array(R))

    center = rcbox['geometry'].get_center()
    print('this is the center where the rotation is applied:', center)

    rcbox['geometry'].rotate(R, center=center)
    rcbox['transform_to_local']['rotation_matrix'] = rcbox['transform_to_local']['rotation_matrix'] @ R.T

    check_dtype("Updated Rotation Matrix", rcbox['transform_to_local']['rotation_matrix'])
    return rcbox



def wait_for_file(file_path, timeout=30, check_interval=0.5):
    """
    Waits for a file to be fully written by checking its existence and stability.
    
    :param file_path: Path to the file.
    :param timeout: Maximum time to wait in seconds.
    :param check_interval: Time between file existence checks.
    :return: True if file is ready, raises an error otherwise.
    """
    start_time = time.time()
    last_size = -1
    
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            if current_size > 0 and current_size == last_size:
                print(f"File '{file_path}' is stable and ready for reading.")
                return True
            last_size = current_size
        time.sleep(check_interval)
    
    raise TimeoutError(f"File '{file_path}' did not become available within {timeout} seconds.")

import time
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d

def read_rcbox(rcbox_path):
    if not rcbox_path.is_file():
        raise FileNotFoundError(f"[ERROR] File not found: {rcbox_path}")

    # Wait for file stability before reading
    while True:
        try:
            with open(rcbox_path, 'r') as f:
                pass  # Just check if it can be opened
            break
        except PermissionError:
            print(f"[WAIT] Waiting for {rcbox_path} to be accessible...")
            time.sleep(1)

    # Parse XML
    tree = ET.parse(rcbox_path)
    root = tree.getroot()
    print("[DEBUG] Parsed XML content:\n", ET.tostring(root, encoding="unicode"))

    # Extract width, height, depth from the element
    width_height_depth_element = root.find("widthHeightDepth")
    if width_height_depth_element is None or width_height_depth_element.text is None:
        raise ValueError(f"[ERROR] 'widthHeightDepth' element is missing in file: {rcbox_path}")

    depth_RC, width_RC, height_RC = [float(val) for val in width_height_depth_element.text.split()]

    # Extract center coordinates
    centre_element = root.find("CentreEuclid/centre")
    if centre_element is None or centre_element.text is None:
        raise ValueError(f"[ERROR] 'CentreEuclid/centre' element is missing in file: {rcbox_path}")

    x, y, z = [float(val) for val in centre_element.text.split()]

    # Extract yaw, pitch, roll
    yaw_pitch_roll_element = root.find("yawPitchRoll")
    if yaw_pitch_roll_element is None or yaw_pitch_roll_element.text is None:
        raise ValueError(f"[ERROR] 'yawPitchRoll' element is missing in file: {rcbox_path}")

    yaw, pitch, roll = [float(val) for val in yaw_pitch_roll_element.text.split()]

    global_coord_system = root.get("globalCoordinateSystem")
    global_coord_system_name = root.get("globalCoordinateSystemName")

    # Create the box
    cube_legacy = o3d.geometry.TriangleMesh.create_box(
        width=depth_RC, height=width_RC, depth=height_RC
    )
    cube_legacy.translate(-cube_legacy.get_center())  # Center at origin

    rcbox = {
        "name": rcbox_path.stem,
        "geometry": cube_legacy,
        "transform_to_local": {
            "translation_matrix": np.array([0, 0, 0], dtype=np.float64),
            "rotation_matrix": np.identity(3, dtype=np.float64),
            "scale_matrix": np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64),
        },
        "globalCoordinateSystem": global_coord_system,
        "globalCoordinateSystemName": global_coord_system_name,
    }

    print(f"[INFO] Initial translate to global: {x}, {y}, {z}")
    rcbox = persistent_translate(rcbox, x, y, z)
    rcbox = localspace_rotation(rcbox, yaw, pitch, roll)

    return rcbox






def rotation_matrix_to_euler_angles(R):
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = math.atan2(R[2, 1], R[2, 2])
    return [yaw, pitch, roll]

def write_rcbox(rcbox):
    x, y, z = rcbox['geometry'].get_center()

    rcbox_local = transform_to_local(rcbox)
    print('This should be 0,0,0:', rcbox_local['geometry'].get_center())

    bounds = rcbox_local['geometry'].get_axis_aligned_bounding_box().get_extent()
    depth, width, height = np.array(bounds, dtype=np.float64)
    check_dtype("Bounding Box Dimensions", np.array([depth, width, height]))

    rotation_matrix = rcbox['transform_to_local']['rotation_matrix']
    yaw_rad, pitch_rad, roll_rad = rotation_matrix_to_euler_angles(rotation_matrix)
    yaw, pitch, roll = [math.degrees(angle) for angle in [yaw_rad, pitch_rad, roll_rad]]

    root = ET.Element('ReconstructionRegion', {
        'globalCoordinateSystem': rcbox['globalCoordinateSystem'],
        'globalCoordinateSystemWkt': "",
        'globalCoordinateSystemName': rcbox['globalCoordinateSystemName'],
        'isGeoreferenced': "1",
        'isLatLon': "0"
    })

    ET.SubElement(root, "yawPitchRoll").text = f"{yaw} {pitch} {roll}"
    ET.SubElement(root, "widthHeightDepth").text = f"{depth} {width} {height}"
    ET.SubElement(root, "Header", {'magic': "5395016", 'version': "2"})

    centre_euclid = ET.SubElement(root, "CentreEuclid")
    print('writing center:', x, y, z)
    ET.SubElement(centre_euclid, "centre").text = f"{x} {y} {z}"

    ET.SubElement(root, "Residual", {
        'R': "1 0 0 0 1 0 0 0 1",
        't': "0 0 0",
        's': "1",
        'ownerId': "{2B36705F-74C9-4270-BED3-074F279D427B}"
    })

    tree = ET.ElementTree(root)
    return tree

    

def get_user_input():
    """Prompt user for model details and grid size."""
    total_triangles = int(input("Enter total number of triangles in basemodel: "))
    total_textures = int(input("Enter total number of 8k textures in basemodel: "))
    
    min_parts_tri = np.ceil(total_triangles / 40_000_000)
    min_parts_tex = np.ceil(total_textures / 64)
    min_parts = int(max(min_parts_tri, min_parts_tex))
    
    print(f"Minimum required parts: {min_parts}")
    grid_input = input("Enter tiling grid as NxMxL (e.g., 6x2x1): ")
    grid = tuple(map(int, grid_input.split('x')))
    
    return total_triangles, total_textures, min_parts, grid

def export_rcbox(output_path):
    """Export the reconstruction region from RealityCapture using RCCommandBuilder."""
    builder = RCCommandBuilder()
    builder.add_command("exportReconstructionRegion", output_path)
    builder.execute()
    print(f"Exported ReconstructionRegion to: {output_path}")
    print(f"Exported ReconstructionRegion to: {output_path}")

def clip_mesh_legacy_support(mesh_legacy, point, normal):
    """
    Clips a legacy Open3D mesh using clip_plane(), converting to tensor format only during the operation.
    """
    #print the incoming datatypes
    print('This is the point datatype:', type(point), point)
    print('This is the normal datatype:', type(normal), normal)
    print('This is the mesh datatype:', type(mesh_legacy))
    # Convert legacy mesh to tensor mesh with explicit dtype specifications
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    point_tensor = o3d.core.Tensor(point)
    normal_tensor = o3d.core.Tensor(normal)

    # Apply the clipping
    clipped_mesh_tensor = mesh_tensor.clip_plane(point_tensor, normal_tensor)
    print('This is the clipped mesh tensor:', clipped_mesh_tensor)

    # Convert back to legacy
    clipped_mesh_legacy = clipped_mesh_tensor.to_legacy()

    return clipped_mesh_legacy






def split_mesh_by_plane(mesh, point, normal):
    """
    Splits a mesh into two parts using Open3D's clip_plane().
    The first part is the side **in the direction of the normal**.
    The second part is the **opposite side** (inverted normal).
    """
    part1 = clip_mesh_legacy_support(mesh, point, normal)
    part2 = clip_mesh_legacy_support(mesh, point, -np.array(normal))
    return [part1, part2]

def cut_mesh_along_axis(mesh_list, axis, steps):
    """
    Recursively slices a list of meshes along a given axis (X, Y, or Z) into `steps` parts.
    Returns a **list of smaller meshes** after cutting along this axis.
    """
    axis_normals = {
        0: [1, 0, 0],  # X-axis
        1: [0, 1, 0],  # Y-axis
        2: [0, 0, 1]   # Z-axis
    }
    
    normal = axis_normals[axis]
    min_bound = mesh_list[0].get_min_bound()[axis]
    max_bound = mesh_list[0].get_max_bound()[axis]
    step_size = (max_bound - min_bound) / steps

    result_meshes = []
    
    for step in range(1, steps):  # Start at 1, as 0 is the original bound
        cutting_plane = min_bound + step * step_size
        print('This is the cutting plane:',cutting_plane)

        front_parts = []
        for mesh in mesh_list:
            cut_results = split_mesh_by_plane(mesh, [cutting_plane if i == axis else 0 for i in range(3)], normal)
            front_parts.append(cut_results[0])  # Keep the positive side
            result_meshes.append(cut_results[1])  # Store the negative side for later
        
        mesh_list = front_parts  # Process the next slices only on the remaining part
    
    result_meshes.extend(mesh_list)  # Add remaining parts
    return result_meshes

def generate_tiled_rcboxes(rcbox, grid, output_folder):
    """
    Generates **tiled RCBoxes** by slicing an input mesh along all three axes.
    Saves the results as individual .rcbox files with structured filenames.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)



    # Transform the mesh to local space
    rcbox_local = transform_to_local(rcbox)
    print('This is the transformed local RCBox:', rcbox_local)


    mesh_list = [rcbox_local['geometry']]  # Start with one full box
    #Allaxis_tiles = []

    tilesOfaxis0 = cut_mesh_along_axis(mesh_list, 0, grid[0])
    print('This is the number of tiles of axis 0:', len(tilesOfaxis0))
    tilesOfaxis1 = cut_mesh_along_axis(tilesOfaxis0, 1, grid[1])
    print('This is the number of tiles of axis 1:', len(tilesOfaxis1))
    tilesOfbox2 = cut_mesh_along_axis(tilesOfaxis1, 2, grid[2])
    print('This is the number of tiles of axis 2:', len(tilesOfbox2))


    scandf = pd.DataFrame(columns=["tile_id", "rcbox_path", "processed"])

    for idx, mesh in enumerate(tilesOfbox2):
        i, j, k = idx % grid[0], (idx // grid[0]) % grid[1], (idx // (grid[0] * grid[1])) % grid[2]

        tile_name = f"tile_x{str(i).zfill(3)}_y{str(j).zfill(3)}_z{str(k).zfill(3)}.rcbox"
        output_path = os.path.join(output_folder, tile_name)

        # Prepare new RCBox metadata as a copy of the original
        tile_rcbox = rcbox.copy()
        tile_rcbox['geometry'] = mesh
        tile_rcbox = transform_to_global(tile_rcbox)


        # Write to file
        tree = write_rcbox(tile_rcbox)
        tree.write(output_path)
        print(f"Created: {output_path}")

        scandf = scandf.append({"tile_id": tile_name, "rcbox_path": output_path, "processed": False}, ignore_index=True)

    #scandf.to_csv(os.path.join(output_folder, "processing_log.csv"), index=False)



def process_tiles(processing_log):
    """Process each tile in RealityCapture using RCCommandBuilder."""
    scandf = pd.read_csv(processing_log)
    
    for index, tile in scandf.iterrows():
        if not tile["processed"]:
            rcbox_path = tile["rcbox_path"]
            builder = RCCommandBuilder()
            builder.add_command("importReconstructionRegion", rcbox_path)
            builder.add_command("selectLargestModelComponent")
            builder.add_command("simplify", "1000000")
            builder.add_command("unwrap")
            builder.add_command("calculateTexture")
            builder.add_command("exportModel", f"{rcbox_path.replace('.rcbox', '.obj')}")
            builder.add_command("save")
            
            command = ["RealityCapture.exe"] + builder.build().split()
            subprocess.run(' '.join(command), check=True, shell=True)
            scandf.at[index, "processed"] = True
            scandf.to_csv(processing_log, index=False)

def main():
    """Main function to execute the RCBox tiling process."""

    output_folder = 'E:/WES_L18_Boat/exportTestd/WES_L18_Boat_tiledexport/'
    processing_log = os.path.join(output_folder, "processing_log.csv")
    rcbox_path = os.path.join(output_folder, "WES_L18_Boat.rcbox")

    # Step 1: Export the RCBox from RealityCapture
    export_rcbox(rcbox_path)

    # Step 2: Read the exported RCBox
    new_rcbox = read_rcbox(Path(rcbox_path))
    print('This is the o3d box:', new_rcbox)

    # Step 3: Write the unmodified RCBox to verify integrity
    #output_path = os.path.join(output_folder, 'test.rcbox')
    #tree = write_rcbox(new_rcbox)
    #tree.write(output_path)

    # Step 4: Generate tiled RCBoxes using the clipping pipeline
    grid = (3, 2, 1)  # Set the tiling grid
    generate_tiled_rcboxes(new_rcbox, grid, output_folder)

    # Step 5: Process the tiled RCBoxes in RealityCapture
    #process_tiles(processing_log)

if __name__ == "__main__":
    main()
