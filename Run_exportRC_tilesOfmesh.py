import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import open3d as o3d
from RClib import read_rcbox, persistent_translate, localspace_rotation, localspace_scale, tranformmatrix_to_local, transformmatrix_to_global, write_rcbox

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

def split_mesh_by_plane(mesh, point, normal):
    """
    Splits a mesh into two parts using Open3D's clip_plane().
    The first part is the side **in the direction of the normal**.
    The second part is the **opposite side** (inverted normal).
    """
    part1 = mesh.clip_plane(point=o3d.core.Tensor(point, dtype=o3d.core.float32),
                            normal=o3d.core.Tensor(normal, dtype=o3d.core.float32))
    
    part2 = mesh.clip_plane(point=o3d.core.Tensor(point, dtype=o3d.core.float32),
                            normal=o3d.core.Tensor(-np.array(normal), dtype=o3d.core.float32)) # Inverted normal
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
    print("Global vertices of the box:\n", rcbox['geometry'].vertex.positions)

    #transform the mesh to local space
    to_local_matrix = tranformmatrix_to_local(rcbox)
    print('This is the to_local_matrix:',to_local_matrix)
    localgeom = rcbox['geometry'].transform(to_local_matrix)
    vertices_tensor = localgeom.vertex.positions
    print("Vertices as Tensor:\n", vertices_tensor)



    mesh_list = [localgeom]  # Start with one full box
    
    for axis, steps in enumerate(grid):
        mesh_list = cut_mesh_along_axis(mesh_list, axis, steps)

    # Small buffer to prevent gaps
    buffer_size = 1e-6
    for mesh in mesh_list:
        mesh.scale(1 - buffer_size, mesh.get_center())  # Shrink slightly

    scandf = pd.DataFrame(columns=["tile_id", "rcbox_path", "processed"])

    for idx, mesh in enumerate(mesh_list):
        i, j, k = idx % grid[0], (idx // grid[0]) % grid[1], (idx // (grid[0] * grid[1])) % grid[2]

        tile_name = f"tile_x{str(i).zfill(3)}_y{str(j).zfill(3)}_z{str(k).zfill(3)}.rcbox"
        output_path = os.path.join(output_folder, tile_name)

        # Prepare new RCBox metadata
        new_rcbox = {
            'name': tile_name,
            'geometry': mesh,
            'transform_to_local': rcbox['transform_to_local'],  # Keep original transform
            'globalCoordinateSystem': rcbox['globalCoordinateSystem'],
            'globalCoordinateSystemName': rcbox['globalCoordinateSystemName']
        }

        # Write to file
        tree = write_rcbox(new_rcbox)
        tree.write(output_path)
        print(f"Created: {output_path}")

        scandf = scandf.append({"tile_id": tile_name, "rcbox_path": output_path, "processed": False}, ignore_index=True)

    scandf.to_csv(os.path.join(output_folder, "processing_log.csv"), index=False)



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
    
    export_rcbox(rcbox_path)
    #total_triangles, total_textures, min_parts, grid = get_user_input()
    grid = (3, 2, 1)
    new_rcbox = read_rcbox(Path(rcbox_path))
    print('This is the o3d box:',new_rcbox)
    tree = write_rcbox(new_rcbox)
    output_path = os.path.join(output_folder, 'test.rcbox')
    tree.write(output_path)

    #generate_tiled_rcboxes(rcbox, grid, output_folder)
    #process_tiles(processing_log)

if __name__ == "__main__":
    main()
