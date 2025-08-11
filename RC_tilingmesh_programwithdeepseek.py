import os
import numpy as np
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import csv

class RCCommandBuilder:
    TEMP_RCCMD_PATH = "E:/WES_L18_Boat/temp.rccmd"
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

def export_rcbox(outputfolder, modelname):
    """Export the reconstruction region from RealityCapture using RCCommandBuilder."""
    output_path = os.path.join(outputfolder, f"{modelname}.rcbox")
    builder = RCCommandBuilder()
    builder.add_command("renameSelectedModel", modelname)
    builder.add_command("exportReconstructionRegion", output_path)
    builder.execute()
    # Wait for the file to be written
    wait_for_file(output_path)
    print(f"Exported ReconstructionRegion to: {output_path}")
    return output_path


def get_user_input():
    """Prompt user for model details and grid size."""
    modelname = str(input("Enter official name of model: "))
    total_triangles = int(input("Enter total number of triangles in basemodel: "))
    total_textures = int(input("Enter total number of 8k textures in basemodel: "))
    buffer = 0.01  # 1 cm buffer added to each side of each tile box
    BENCHMARK_LOG_PATH = Path("benchmark_log.csv")
    
    min_parts_tri = np.ceil(total_triangles / 40000000)
    min_parts_tex = np.ceil(total_textures / 64)
    min_parts = int(max(min_parts_tri, min_parts_tex))
    
    print(f"Minimum required parts: {min_parts}")
    grid_input = input("Enter tiling grid as NxMxL (e.g., 6x2x1): ")
    grid = tuple(map(int, grid_input.split('x')))
    
    return modelname, total_triangles, total_textures, min_parts, grid, buffer, BENCHMARK_LOG_PATH


def log_benchmark(label, elapsed, BENCHMARK_LOG_PATH):
    write_header = not BENCHMARK_LOG_PATH.exists()
    with open(BENCHMARK_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if write_header:
            writer.writerow(["Step", "Elapsed_Time_sec"])
        writer.writerow([label, f"{elapsed:.3f}"])

def read_rcbox(rcbox_path):
    """
    Reads the rcbox XML file and extracts the center, extents, and rotation.
    """
    if not Path(rcbox_path).is_file():
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
    width_height_depth = root.find("widthHeightDepth")
    if width_height_depth is None or width_height_depth.text is None:
        width_height_depth = root.get("widthHeightDepth")
        if width_height_depth is None:
            raise ValueError(f"[ERROR] 'widthHeightDepth' attribute or element is missing in file: {rcbox_path}")
        else:
            print('this is the width_height_depth from attribute:', width_height_depth)
            width_RC, height_RC, depth_RC = [float(val) for val in width_height_depth.split()]

    else:
        print('this is the width_height_depth from element:', width_height_depth.text)
        width_RC, height_RC, depth_RC = [float(val) for val in width_height_depth.text.split()]

    # Extract center coordinates
    centre_element = root.find("CentreEuclid/centre")
    if centre_element is None or centre_element.text is None:
        raise ValueError(f"[ERROR] 'CentreEuclid/centre' element is missing in file: {rcbox_path}")

    center = [float(val) for val in centre_element.text.split()]

    # Extract yaw, pitch, roll
    yaw_pitch_roll_element = root.find("yawPitchRoll")
    if yaw_pitch_roll_element is None or yaw_pitch_roll_element.text is None:
        raise ValueError(f"[ERROR] 'yawPitchRoll' element is missing in file: {rcbox_path}")

    yaw, pitch, roll = [float(val) for val in yaw_pitch_roll_element.text.split()]

    global_coord_system = root.get("globalCoordinateSystem")
    global_coord_system_name = root.get("globalCoordinateSystemName")

    return center, width_RC, height_RC, depth_RC, yaw, pitch, roll

def rotation_matrix(yaw, pitch, roll):
    """
    Creates a 3D rotation matrix from yaw, pitch, and roll angles (in degrees) for a left-handed coordinate system.
    """
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Yaw (Y-axis)
    Ry = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Pitch (X-axis)
    Rx = np.array([
        [1, 0,         0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Roll (Z-axis)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1]
    ])

    # Combined rotation matrix
    return Rz @ Rx @ Ry

def calculate_rotation_from_matrix(R):
    """
    Extracts yaw, pitch, and roll angles (in degrees) from a 3D rotation matrix for a left-handed coordinate system.
    """
    # Extract yaw (rotation around Y-axis)
    yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))  # Yaw: atan2(R13, R33)

    # Extract pitch (rotation around X-axis)
    pitch = np.degrees(np.arctan2(-R[1, 2], np.sqrt(R[1, 0]**2 + R[1, 1]**2)))  # Pitch: atan2(-R23, sqrt(R21^2 + R22^2))

    # Extract roll (rotation around Z-axis)
    roll = np.degrees(np.arctan2(R[1, 0], R[1, 1]))  # Roll: atan2(R21, R22)

    return yaw, pitch, roll

def create_local_axes_for_subbox(subbox_width, subbox_height, subbox_depth, local_center):
    """
    Creates the local axes (width, height, depth) for a subbox in the local coordinate system of the original box.
    The local_center is the center of the subbox relative to the original box's local coordinate system.
    """
    # Width axis (X-axis)
    width_axis = (
        local_center + np.array([-subbox_width / 2, 0, 0]),
        local_center + np.array([subbox_width / 2, 0, 0])
    )

    # Height axis (Y-axis)
    height_axis = (
        local_center + np.array([0, -subbox_height / 2, 0]),
        local_center + np.array([0, subbox_height / 2, 0])
    )

    # Depth axis (Z-axis)
    depth_axis = (
        local_center + np.array([0, 0, -subbox_depth / 2]),
        local_center + np.array([0, 0, subbox_depth / 2])
    )

    print("[DEBUG] Local axes for subbox:")
    print("Local center:", local_center)
    print("Width axis:", width_axis)
    print("Height axis:", height_axis)
    print("Depth axis:", depth_axis)

    return width_axis, height_axis, depth_axis

def transform_subboxes_to_global(subbox, R, original_center):
    """
    Transforms the local axes and center of a subbox to global coordinates.
    Rotates the axes and center around the origin, then translates by the original box's global center.
    Updates the subbox dictionary with the transformed axes and global center.
    """
    # Extract local axes and center
    width_axis = subbox["width_axis"]
    height_axis = subbox["height_axis"]
    depth_axis = subbox["depth_axis"]
    local_center = subbox["local_center"]

    

    # Rotate the local center around the origin
    rotated_center = R.T @ local_center
    print("[DEBUG] Rotation matrix R:", R.T)
    print("[DEBUG] calculate_rotation_from_matrix(R):", calculate_rotation_from_matrix(R))
    print("[DEBUG] Local center:", local_center)
    print("[DEBUG] Rotated center:", rotated_center)

    # Translate to global coordinates using the original box's global center
    print("[DEBUG] Original center:", original_center)
    print("[DEBUG] Rotated center:", rotated_center)
    global_center = original_center + rotated_center

    # Transform width axis
    p1_width_rotated = R @ width_axis[0]
    p2_width_rotated = R @ width_axis[1]
    p1_width_global = original_center + p1_width_rotated
    p2_width_global = original_center + p2_width_rotated
    transformed_width_axis = (p1_width_global, p2_width_global)

    # Transform height axis
    p1_height_rotated = R @ height_axis[0]
    p2_height_rotated = R @ height_axis[1]
    p1_height_global = original_center + p1_height_rotated
    p2_height_global = original_center + p2_height_rotated
    transformed_height_axis = (p1_height_global, p2_height_global)

    # Transform depth axis
    p1_depth_rotated = R @ depth_axis[0]
    p2_depth_rotated = R @ depth_axis[1]
    p1_depth_global = original_center + p1_depth_rotated
    p2_depth_global = original_center + p2_depth_rotated
    transformed_depth_axis = (p1_depth_global, p2_depth_global)

    # Update the subbox dictionary with transformed axes and global center
    subbox["width_axis"] = transformed_width_axis
    subbox["height_axis"] = transformed_height_axis
    subbox["depth_axis"] = transformed_depth_axis
    subbox["center"] = global_center

    print("[DEBUG] Transformed subbox:")
    print("Global center:", global_center)
    print("Transformed width axis:", transformed_width_axis)
    print("Transformed height axis:", transformed_height_axis)
    print("Transformed depth axis:", transformed_depth_axis)
    #check wether the middle of each axis is the same as the global center
    print("Middle of width axis:", (transformed_width_axis[0] + transformed_width_axis[1]) / 2)
    print("Middle of height axis:", (transformed_height_axis[0] + transformed_height_axis[1]) / 2)
    print("Middle of depth axis:", (transformed_depth_axis[0] + transformed_depth_axis[1]) / 2)


    return subbox

def calculate_rotation_from_axes(subbox):
    """
    Calculates the yaw, pitch, and roll angles for a subbox based on its local axes.
    Updates the subbox dictionary with the calculated rotation.
    """
    # Extract global center of the subbox
    global_center = subbox["center"]

    # Extract global axes
    width_axis = subbox["width_axis"]
    height_axis = subbox["height_axis"]
    depth_axis = subbox["depth_axis"]

    # Translate the points in the axes to the subbox's local coordinate system (subtract global center)
    width_axis_local = [point - global_center for point in width_axis]  # Translate width axis points
    height_axis_local = [point - global_center for point in height_axis]  # Translate height axis points
    depth_axis_local = [point - global_center for point in depth_axis]  # Translate depth axis points

    # Calculate direction vectors from the translated points
    width_dir = width_axis_local[1] - width_axis_local[0]  # Vector along the width axis (X-axis)
    height_dir = height_axis_local[1] - height_axis_local[0]  # Vector along the height axis (Y-axis)
    depth_dir = depth_axis_local[1] - depth_axis_local[0]  # Vector along the depth axis (Z-axis)

    # Normalize direction vectors
    width_dir = width_dir / np.linalg.norm(width_dir)  # Normalize width vector
    height_dir = height_dir / np.linalg.norm(height_dir)  # Normalize height vector
    depth_dir = depth_dir / np.linalg.norm(depth_dir)  # Normalize depth vector

    # Construct the rotation matrix for the subbox
    R = np.column_stack((width_dir, height_dir, depth_dir))

    # Extract yaw, pitch, and roll from the rotation matrix
    yaw, pitch, roll = calculate_rotation_from_matrix(R)

    # Update the subbox dictionary with rotation
    subbox["rotation"] = (yaw, pitch, roll)
    print("[DEBUG] Calculated rotation for subbox:", (yaw, pitch, roll))

    return subbox
def divide_rcbox(center, width, height, depth, yaw, pitch, roll, grid):
    """
    Divides the rcbox into subboxes based on the grid division.
    Returns a list of subboxes, each with its center, extents, and rotation.
    Includes buffer logic to expand tile boxes slightly.
    """
    start = time.perf_counter()

    # Subbox extents with buffer
    subbox_width = (width / grid[0]) + buffer
    subbox_height = (height / grid[1]) + buffer
    subbox_depth = (depth / grid[2]) + buffer

    # Rotation matrix for the original box
    R = rotation_matrix(yaw, pitch, roll)

    # Calculate subbox centers
    subboxes = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                # Local center in the unrotated coordinate system of the original box
                local_center = np.array([
                    (i + 0.5) * (width / grid[0]) - width / 2,  # without buffer offset
                    (j + 0.5) * (height / grid[1]) - height / 2,
                    (k + 0.5) * (depth / grid[2]) - depth / 2
                ])
                print("[DEBUG] Local center subbox:", local_center)

                # Create local axes for the subbox
                width_axis, height_axis, depth_axis = create_local_axes_for_subbox(
                    subbox_width, subbox_height, subbox_depth, local_center
                )

                # Create a dictionary for the subbox
                subbox = {
                    "part_id": f"tile_x{str(i).zfill(3)}_y{str(j).zfill(3)}_z{str(k).zfill(3)}",
                    "local_center": local_center,
                    "width_axis": width_axis,
                    "height_axis": height_axis,
                    "depth_axis": depth_axis,
                    "extents": (subbox_width, subbox_height, subbox_depth)
                }

                # Transform subbox to global coordinates
                subbox = transform_subboxes_to_global(subbox, R, center)

                # Determine the rotation of the subbox
                subbox = calculate_rotation_from_axes(subbox)

                # Add the subbox to the list
                subboxes.append(subbox)

    elapsed = time.perf_counter() - start
    print(f"[BENCHMARK] Dividing RCBox into subboxes took {elapsed:.3f} seconds")
    log_benchmark("Divide_RCBox", elapsed)

    return subboxes




def create_subbox_xml(subbox):
    """
    Creates an XML structure for a subbox.
    """
    center = subbox["center"]
    extents = subbox["extents"]
    yaw, pitch, roll = subbox["rotation"]

    xml_content = f"""
    <ReconstructionRegion globalCoordinateSystem="+proj=utm +zone=38 +datum=WGS84 +units=m +no_defs"
    globalCoordinateSystemName="epsg:32638 - WGS 84 / UTM zone 38N" isGeoreferenced="1"
    isLatLon="0">
    <globalCoordinateSystemWkt>PROJCS["WGS_1984_UTM_Zone_38N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]</globalCoordinateSystemWkt>
    <yawPitchRoll>{yaw} {pitch} {roll}</yawPitchRoll>
    <widthHeightDepth>{extents[0]} {extents[1]} {extents[2]}</widthHeightDepth>
    <Header magic="5395016" version="2"/>
    <CentreEuclid>
        <centre>{center[0]} {center[1]} {center[2]}</centre>
    </CentreEuclid>
    <Residual s="0.208341582931947" ownerId="{{26593CA4-DA98-4826-B551-5FA0FAAED06A}}">
        <R>-0.963369294999673 -0.268055914009874 0.00810113671996866 0.268119866970172 -0.963350403621457 0.00823023561143743 0.00559806999963212 0.0101008319783295 0.999933315179311</R>
        <t>-0.127648353924026 0.125738531285549 -0.0326934514458055</t>
    </Residual>
    </ReconstructionRegion>
    """
    return xml_content

def write_subbox_to_file(subbox):
    """
    Writes the XML content of a subbox to a file.
    """
    xml_content = create_subbox_xml(subbox)
    filename = subbox['rcbox_path']
    with open(filename, 'w') as f:
        f.write(xml_content)


def RCexport_tiles(listofrcboxpaths, outputfolder, modelname):
    builder = RCCommandBuilder()
    for tile in listofrcboxpaths:
        tile_label = f"Export_{tile['part_id']}"
        start = time.perf_counter()

        rcbox_path = tile["rcbox_path"]
        print(f"Processing tile: {rcbox_path}")
        obj_path = os.path.join(outputfolder, f"{modelname}_{tile['part_id']}.obj")

        builder.add_command("selectModel", modelname)
        builder.add_command("setReconstructionRegion", rcbox_path)
        builder.add_command("cutByBox", "outer", "false")
        builder.add_command("cleanModel")
        builder.add_command("renameSelectedModel", f"{modelname}_{tile['part_id']}")
        builder.add_command("unwrap")
        builder.add_command("reprojectTexture", modelname, f"{modelname}_{tile['part_id']}")
        builder.add_command("exportModel", f"{modelname}_{tile['part_id']}", obj_path, "C:/Users/tronc/Documents/GitHub/RCprocessing/export_param_visualtile.xml")

        subfolder = os.path.join(outputfolder, f"{modelname}_{tile['part_id']}_editsubtile")
        Path(subfolder).mkdir(parents=True, exist_ok=True)
        editsubtile_obj_path = os.path.join(subfolder, f"{modelname}_{tile['part_id']}.obj")
        builder.add_command("exportModel", f"{modelname}_{tile['part_id']}", editsubtile_obj_path, "C:/Users/tronc/Documents/GitHub/RCprocessing/export_param_editsubtile.xml")

        elapsed = time.perf_counter() - start
        print(f"[BENCHMARK] {tile_label} completed in {elapsed:.3f} seconds")
        log_benchmark(tile_label, elapsed)

    builder.execute()



def main():
    metafolder = 'E:/WES_L18_Boat/'
    modelname, total_triangles, total_textures, min_parts, grid = get_user_input()
    output_folder = os.path.join(metafolder, modelname)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    rcbox_path = export_rcbox(output_folder, modelname)
    print(f"Exported rcbox to: {rcbox_path}")
    # Path to the rcbox XML file1
    #rcbox_path = Path("E:/WES_L18_Boat/exportTestd/WES_L18_Boat_tiledexport/deepseek_code/originalbox.rcbox")

    # Read the rcbox
    center, width, height, depth, yaw, pitch, roll = read_rcbox(rcbox_path)

    # Divide the rcbox into subboxes
    subboxes = divide_rcbox(center, width, height, depth, yaw, pitch, roll, grid)
    exported_subboxes = []

    # Generate XML for each subbox
    for idx, subbox in enumerate(subboxes):
        
        subbox['rcbox_path'] = os.path.join(output_folder, subbox['part_id'] +'.rcbox')
        
        write_subbox_to_file(subbox)
        exported_subboxes.append(subbox)
    
    RCexport_tiles(exported_subboxes, output_folder, modelname)

if __name__ == "__main__":
    main()