import unreal
import shutil
import subprocess
from pathlib import Path
import numpy as np
import os
import pandas as pd

import RClib as RClib





def provide_meshdf(folder_path):
    """
    Scan a specified folder for .obj files and gather the associated files into a pandas DataFrame.
    
    Parameters:
    - folder_path: The path to the folder to scan.

    Returns:
    - A pandas DataFrame with each row being a 3D model and columns for the associated files.
    """

    # List all files in the directory
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Create a dictionary to hold the data
    data = {}

    # Texture file extensions for identification
    texture_extensions = ['.jpg', '.jpeg', '.png', '.tga', '.bmp']

    # For each file in the directory
    for file in all_files:
        # Check if the file is an .obj file
        if file.endswith('.obj'):
            stem_name = os.path.splitext(file)[0]
            
            # Initialize file categories
            obj_file = os.path.join(folder_path, file)
            mtl_file = None
            texture_files = []
            other_files = []

            # Find associated files with the same stem name
            associated_files = [f for f in all_files if f.startswith(stem_name)]
            for assoc_file in associated_files:
                if assoc_file.endswith('.mtl'):
                    mtl_file = os.path.join(folder_path, assoc_file)
                elif any(assoc_file.endswith(ext) for ext in texture_extensions):
                    texture_files.append(os.path.join(folder_path, assoc_file))
                else:
                    other_files.append(os.path.join(folder_path, assoc_file))

            data[stem_name] = [obj_file, mtl_file, texture_files, other_files]

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=['OBJ File', 'MTL File', 'Texture Files', 'Other Files'])

    return df


def parse_mtl(file_path):
    """
    Parse a .mtl file and represent it as a pandas DataFrame.
    
    Parameters:
    - file_path: The path to the .mtl file.

    Returns:
    - A pandas DataFrame with each row being a material definition.
    """

    # List to hold the material data
    materials_data = []

    # Dictionary to hold the current material's properties
    current_material_data = {}

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check if the line defines a new material
            if line.startswith('newmtl'):
                # If there's already a material being processed, add it to the list
                if current_material_data:
                    materials_data.append(current_material_data)
                    current_material_data = {}
                
                current_material_data['Material Name'] = line.split()[1]
            
            # Check if the line defines a texture
            elif line.startswith('map_Kd'):
                current_material_data['Texture Path'] = line.split()[1]

    # Add the last material data if it exists
    if current_material_data:
        materials_data.append(current_material_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(materials_data)

    return df
def modify_mtl(mtl_df):
    """
    Modify the parsed .mtl DataFrame to remove all but the first material definition 
    and also remove the texture reference from the first material.
    
    Parameters:
    - mtl_df: The original .mtl DataFrame.

    Returns:
    - A modified .mtl DataFrame.
    """

    # Keep only the first row and remove the texture reference
    modified_df = mtl_df.iloc[:1].copy()
    #print(modified_df)
    #modified_df['Texture Path'] = modify_mtl['Texture Path']

    return modified_df

def write_mtl(modified_mtl_df, original_file_path):
    """
    Write a modified .mtl DataFrame to a new .mtl file, replacing the original.
    
    Parameters:
    - modified_mtl_df: The modified .mtl DataFrame.
    - original_file_path: The path to the original .mtl file.

    Returns:
    - None.
    """

    with open(original_file_path, 'w') as file:
        for _, row in modified_mtl_df.iterrows():
            file.write(f"newmtl {row['Material Name']}\n")
            file.write("Ka 1 1 1\n")
            file.write("Kd 1 1 1\n")
            file.write("d 1\n")
            #file.write("Ns 0\n")
            file.write("illum 1\n")
            if row['Texture Path']:
                file.write(f"map_Kd {row['Texture Path']}\n")
            file.write("\n")  # Separate materials with a newline

    print(f"Modified .mtl file saved to {original_file_path}")



def build_import_options(texturestemname: str):
    options = unreal.FbxImportUI()
    #print(options.get_editor_property('static_mesh_import_data'))
    # unreal.FbxImportUI
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("is_obj_import", True)
    options.set_editor_property("import_textures", True)
    options.set_editor_property("import_materials", True)
    #options.import_as_skeletal = True
    #unreal.FbxMeshImportData
    #unreal.log(options.get_editor_property("is_obj_import"))
    options.static_mesh_import_data.set_editor_properties({'import_uniform_scale': 100.0})
    options.static_mesh_import_data.set_editor_properties({'build_nanite': True}) 
    # Texture options
    options.texture_import_data.set_editor_properties({'base_material_name': unreal.SoftObjectPath('/Game/WES_paleoenvi/cores/sedimentcores_base_material')})     
    options.texture_import_data.set_editor_properties({'material_search_location': unreal.MaterialSearchLocation.LOCAL})
    print(texturestemname)
    options.texture_import_data.set_editor_properties({'base_diffuse_texture_name' : texturestemname})                                        
    return options

def build_fbxfactory(task):
    fbxfactory = unreal.FbxFactory()
    fbxfactory.set_editor_property("asset_import_task", task)
    fbxfactory.set_editor_property('editor_import', True)
    #fbxfactory.set_editor_property('formats',[]
    return fbxfactory

def build_texture_factory():
    texture_factory = unreal.TextureFactory()
    # Set any properties for the texture factory if needed
    return texture_factory

def build_import_tasks(filename: str, texture_files: list, destination_path: str, options):
    tasks = []
    
    # Setup the OBJ import task as you did
    obj_task = unreal.AssetImportTask()
    obj_task.set_editor_property("automated", True)
    obj_task.set_editor_property("destination_path", destination_path)
    obj_task.set_editor_property("filename", filename)
    obj_task.set_editor_property("replace_existing", True)
    obj_task.set_editor_property("replace_existing_settings", True)
    obj_task.set_editor_property("options", options)
    fbxfactory = build_fbxfactory(obj_task)
    obj_task.set_editor_property("factory", fbxfactory)
    

    # Identify UDIM-texture files and create tasks for each
    texture_factory = build_texture_factory()
    
    for file in texture_files:
        
        texture_task = unreal.AssetImportTask()
        texture_task.set_editor_property("automated", True)
        texture_task.set_editor_property("destination_path", destination_path)
        texture_task.set_editor_property("filename", file)
        texture_task.set_editor_property("replace_existing", True)
        texture_task.set_editor_property("replace_existing_settings", True)
        texture_task.set_editor_property("factory", texture_factory)
        tasks.append(texture_task)
    tasks.append(obj_task)
    return tasks

def import_static_mesh(tasks):
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset_tools.import_asset_tasks(tasks)

def get_texturestemname(texture_files: list):
    directory, obj_filename = os.path.split(texture_files[0])
    stem_name = os.path.splitext(obj_filename)[0]
    texturestemname = stem_name.split('.')[0]
    return texturestemname

def rename_created_material_instance(destination_path: str, old_material_name: str, new_material_name: str):
    old_material_instance_path = destination_path + "/" + old_material_name
    new_material_instance_path = destination_path + "/" + new_material_name

    # Check if the old material instance exists
    if unreal.EditorAssetLibrary.does_asset_exist(old_material_instance_path):
        # Rename the material instance
        result = unreal.EditorAssetLibrary.rename_asset(old_material_instance_path, new_material_instance_path)
        if not result:
            unreal.log_error(f"Failed to rename material instance {old_material_name} to {new_material_name}")
    else:
        unreal.log_error(f"Material instance {old_material_name} does not exist in path {destination_path}")

def set_udim_texture_to_material_instance(material_instance_name: str, texture_name: str, parameter_name: str):
    # Load the material instance and the texture
    material_instance = unreal.EditorAssetLibrary.load_asset(material_instance_name)
    texture = unreal.EditorAssetLibrary.load_asset(texture_name)
    
    # Check if both assets are loaded properly
    if not material_instance or not texture:
        unreal.log_error("Unable to load assets!")
        return

    if isinstance(material_instance, unreal.MaterialInstanceConstant):
        # Access the list of texture parameter values
        texture_params = material_instance.texture_parameter_values

        # Check if the parameter exists and set its value
        param_exists = False
        for param in texture_params:
            if param.parameter_info.name == parameter_name:
                param.parameter_value = texture
                param_exists = True
                break
        
        # If the parameter does not exist, add a new one
        if not param_exists:
            new_param = unreal.TextureParameterValue()
            new_param.parameter_info.name = parameter_name
            new_param.parameter_value = texture
            texture_params.append(new_param)
        
        # Set the modified texture parameter list back to the material instance
        #material_instance.texture_parameter_values = texture_params
    else:
        unreal.log_error("The loaded asset is not a MaterialInstanceConstant!")


def post_import_process(mesh_import_path: str, texture_stem_name: str, destination_path: str):
    # Names based on your provided paths and names
    material_instance_name = mesh_import_path + "_material"
    oldname =  "defaultMat_ncl1_1"
    rename_created_material_instance("/Game/WES_paleoenvi/cores/",oldname, material_instance_name)
    texture_name = "/Game/WES_paleoenvi/cores/" + texture_stem_name  # Change the path if it's different
    game_path = destination_path + "/" + material_instance_name

    # Assuming 'BaseTexture' is the parameter name in your material instance where you want to set the UDIM texture.
    # Change 'BaseTexture' to your actual parameter name if it's different.
    set_udim_texture_to_material_instance(game_path, texture_name, 'diffuse')
