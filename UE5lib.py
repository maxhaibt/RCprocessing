import unreal as ue
import json
import requests
import shutil
import subprocess
from pathlib import Path
import numpy as np
import os
import re
import pandas as pd
import math
from collections import defaultdict
#import open3d as o3d

#import RClib as RClib



##### request from idaifield #####

def loadconfigs(configpath):
    #loads a json file from configpath as dict
    try:
        with open(configpath) as configfile:
            config = json.load(configfile)
        return config
    except:
        print('Filepath or JSON invalid.')

    

def couchDB_APIs(config, db_name = None):
    #creates the urls of the couchdb-api endpoints needed to access the data
    if not db_name:
        db_name = config['db_name']
    api = {}
    try:
        api['db_name'] = db_name
        api['auth'] = eval(config['auth'])
        api['find'] = config['db_url'] + '/' + db_name + '/_find'
        api['base'] = config['db_url'] + '/' + db_name 
        api['bulk'] = config['db_url'] + '/' + db_name + '/_bulk_docs'
        api['all_docs'] = config['db_url'] + '/' + db_name + '/_all_docs'
        api['all_dbs'] = config['db_url'] + '/' + '_all_dbs'
    except KeyError:
        print('Necessarry keys and values are missing in the config.json file')
    
    return api


def getAllDocs(api):
    try:
        response = requests.get(api['base'] , auth=api['auth'])    
        result = json.loads(response.text)
        print('The database ' + api['db_name'] + ' contains ' + str(result['doc_count']) + ' docs.',)
    except:
        print(api['base'])
        print('Cannot connect to database, is it on?')
    if result['doc_count'] > 10000:
        collect = {"total_rows":0,"rows":[], "offset": 0}
        limit = math.ceil(result['doc_count'] / 10000)
        for i in range(limit):    
            response = requests.get(api['all_docs'], auth=api['auth'], params={'limit':10000, 'include_docs':True, 'skip': i * 10000})
            i = i + 1
            result = json.loads(response.text)
            print('This is round ' + str(i) + 'offset :', str(result['offset']) )
            collect['total_rows'] = collect['total_rows'] + result['total_rows']
            collect['offset'] = result['offset']
            collect['rows'] = collect['rows'] + result['rows']
    else:
        response = requests.get(api['all_docs'], auth=api['auth'],params = {'include_docs':True})
        collect = json.loads(response.text)
    return collect

def create_regex_for_string(search_string):
    """
    Create a regex pattern that matches variations of a search string.
    It matches different cases and allows for spaces, hyphens, or underscores between characters.
    """
    # Remove any non-alphanumeric characters
    sanitized_string = re.sub(r'[\s\-_]', '', search_string)

    # Build a regex pattern to match variations (case-insensitive, allowing spaces, hyphens, and underscores)
    pattern = '.*' + ''.join(
        f'[{char.upper()}{char.lower()}][\\s\\-_]*' if char.isalpha() else char
        for char in sanitized_string
    ) + '.*'

    return pattern

def getDocsIfContainStringInIdentifier(api, search_string):
    """
    Queries CouchDB for documents whose 'resource.identifier' field matches
    a regex pattern based on the search_string (ignoring case, spaces, hyphens, etc.).
    """
    try:
        # Build the regex pattern
        regex_pattern = create_regex_for_string(search_string)
        
        # Build the payload for the _find query
        payload = {
            "selector": {
                "resource.identifier": {
                    "$regex": regex_pattern
                }
            },
            "limit": 100  # Adjust limit as needed
        }
        
        # Send the query to CouchDB
        headers = {'Content-Type': 'application/json'}
        response = requests.post(api['find'], auth=api['auth'], headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            if 'docs' in result:
                return result['docs']
            else:
                return []
        else:
            print(f"Error occurred: {response.status_code} - {response.text}")
            return []
    
    except Exception as e:
        print(f"Exception during querying: {str(e)}")
        return []


def create_actors_in_content_browser(resources, base_path="Content/idaifield_resources"):
    """
    Create actors in the Unreal Engine content browser based on the provided resources.

    Args:
        resources (list): A list of resources containing data to create actors.
        base_path (str): The path in the content browser where the actors will be stored.
    """
    for resource in resources:
        # Construct the name for the actor based on the resource identifier
        actor_name = resource['resource'].get('identifier', 'DefaultActorName')

        # Create a new actor (assumed to be a custom actor class)
        actor_class = ue.EditorAssetLibrary.load_asset("/Game/PathToYourCustomActor")  # Adjust the path as necessary
        if actor_class is None:
            ue.log_error("Actor class could not be loaded!")
            continue

        # Spawn the actor in the content browser
        asset_tools = ue.AssetToolsHelpers.get_asset_tools()
        task = ue.AssetImportTask()
        task.set_editor_property("automated", True)
        task.set_editor_property("destination_path", base_path)
        task.set_editor_property("filename", f"{base_path}/{actor_name}")
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("options", None)  # Provide options if needed
        task.set_editor_property("factory", actor_class)

        # Perform the import
        asset_tools.import_asset_tasks([task])

        ue.log(f"Created actor: {actor_name} at {base_path}")



def filter_docs_by_category(docs, search_category):
    """
    Filter documents based on whether resource.category contains the search_category string.
    
    Args:
        docs (list): A list of documents from CouchDB.
        search_category (str): The category string to search for.
    
    Returns:
        list: A list of filtered documents where resource.category contains the search_category.
    """
    filtered_docs = []
    
    for doc in docs:
        # Check if the document has 'resource' and 'category' fields
        if 'resource' in doc and 'category' in doc['resource']:
            # Check if the category contains the search string
            if search_category.lower() in doc['resource']['category'].lower():
                filtered_docs.append(doc)
    
    return filtered_docs

def get_related_docs_by_liesWithin(api, filtered_resources):
    """
    Query CouchDB for all documents that have the resource's ID in their 'liesWithin' relation.

    Args:
        api (dict): A dictionary containing the CouchDB API endpoints and authentication.
        filtered_resources (list): A list of filtered resources with 'id' fields.

    Returns:
        list: A list of documents where the 'liesWithin' relation contains the resource's 'id'.
    """
    related_docs = []

    for resource in filtered_resources:
        # Extract the ID of the resource
        resource_id = resource['resource']['id']
        
        # Define the CouchDB Mango query to search for documents where 'liesWithin' contains the resource_id
        payload = {
            "selector": {
                "resource.relations.liesWithin": {
                    "$in": [resource_id]
                }
            },
            "limit": 100  # Limit the number of results returned to avoid overloading
        }
        
        # Make a POST request to the CouchDB _find endpoint
        try:
            response = requests.post(api['find'], auth=api['auth'], json=payload)
            response.raise_for_status()  # Check for HTTP request errors
            result = response.json()

            # Append the matching documents to the related_docs list
            if 'docs' in result:
                related_docs.extend(result['docs'])

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    
    return related_docs


def extract_polygon_geometries(resource):
    """
    Extract polygon geometries from the resources.

    Args:
        resource (dict): A resource containing geometry information.

    Returns:
        tuple: A list of Y-coordinates from all polygon vertices and a flag indicating if vertical extent extraction is needed.
    """
    y_coordinates = []
    needs_vertical_extent_extraction = False

    if isinstance(resource, dict) and 'resource' in resource:
        ue.log('Resource is a dict')
        try:
            if isinstance(resource['resource'], dict):
                # Check for geometry first
                if 'geometry' in resource['resource']:
                    ue.log('Geometry exists')
                    geometry = resource['resource']['geometry']
                    if geometry['type'] == 'Polygon':
                        coordinates = geometry['coordinates'][0]  # Get the first ring of the polygon
                        if coordinates:  # Check if coordinates are not empty
                            for vertex in coordinates:
                                y_coordinates.append(vertex[1])  # Append the Y-coordinate
                        else:
                            ue.log_warning("Coordinates are empty, vertical extent extraction needed.")
                            needs_vertical_extent_extraction = True
                    else:
                        ue.log_warning("Geometry type is not Polygon, vertical extent extraction needed.")
                        needs_vertical_extent_extraction = True
                else:
                    ue.log_warning("No geometry field found, vertical extent extraction needed.")
                    needs_vertical_extent_extraction = True

        except KeyError as e:
            ue.log_error(f"KeyError: {str(e)}")
        except Exception as e:
            ue.log_error(f"An error occurred: {str(e)}")

    return y_coordinates, needs_vertical_extent_extraction


def extract_verticalextent(resource):
    """
    Extract vertical extent values from the resource.

    Args:
        resource (dict): A resource containing vertical extent information.

    Returns:
        tuple: A tuple containing the min and max values of the vertical extent,
               or (None, None) if not found.
    """
    if isinstance(resource, dict) and 'resource' in resource:
        ue.log('Resource is a dict')
        try:
            if isinstance(resource['resource'], dict):
                # Check for 'dimensionVerticalExtent' field
                if 'dimensionVerticalExtent' in resource['resource']:
                    ue.log('Vertical extent exists')
                    vertical_extent_list = resource['resource']['dimensionVerticalExtent']

                    # Ensure it's a list and has at least one element
                    if isinstance(vertical_extent_list, list) and vertical_extent_list:
                        vertical_extent = vertical_extent_list[0]  # Assuming we want the first entry
                        
                        # Extract the min and max values
                        top_value = vertical_extent.get('inputValue')
                        bottom_value = vertical_extent.get('inputRangeEndValue')

                        return top_value, bottom_value
                    else:
                        ue.log_warning("Vertical extent is not a valid list or is empty")
                
                # Check for spatialLocation as a fallback
                elif 'spatialLocation' in resource['resource']:
                    ue.log('Using spatialLocation for vertical extent')
                    spatial_location = resource['resource']['spatialLocation']
                    top, bottom = map(extract_number, spatial_location.split('-'))
                    return top, bottom
                else:
                    ue.log_warning("No vertical extent or spatialLocation field found in resource")
        except KeyError as e:
            ue.log_error(f"KeyError: {str(e)}")
        except Exception as e:
            ue.log_error(f"An error occurred: {str(e)}")

    return None, None  # Return None for both values if not found

def extract_number(value):
    """
    Extracts the first numeric value from a string, ignoring any letters.

    Args:
        value (str): The string from which to extract a number.

    Returns:
        float: The extracted number as a float, or None if no valid number is found.
    """
    if isinstance(value, str):
        # Use regex to find the first numeric part in the string
        match = re.search(r'[\d.]+', value)  # This matches integers and decimals
        if match:
            return float(match.group(0))  # Convert to float
    return None  # Return None if not a valid string

def calculate_min_max_y(y_coordinates):
    """
    Calculate the minimum and maximum Y values from the given list.

    Args:
        y_coordinates (list): A list of Y-coordinates.

    Returns:
        tuple: Minimum and maximum Y values.
    """
    if not y_coordinates:
        return None, None  # Return None if the list is empty

    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    return min_y, max_y

def height_translate_UTM2UrukVR(min_y, max_y):
    """
    Translate Y values for ue Engine based on specified parameters.

    Args:
        min_y (float): The minimum Y value.
        max_y (float): The maximum Y value.

    Returns:
        tuple: Transformed min and max Y values for ue Engine.
    """
    if min_y is None or max_y is None:
        return None, None  # Return None if inputs are invalid

    bottom_value= (min_y - 12.2076) * 100  # Convert to cm
    top_value = (max_y - 12.2076) * 100  # Convert to cm

    return top_value, bottom_value



def calculate_cylinder_position(top_center_position, top_value, bottom_value):
    """
    Calculate the adjusted positions for creating a cylinder based on the top center position and vertical extent values.

    Args:
        top_center_position (Vector): The world position of the top center.
        min_y (float): The minimum Y value.
        max_y (float): The maximum Y value.

    Returns:
        tuple: The adjusted min and max Y positions for the cylinder.
    """
    if top_center_position is None or top_value is None or bottom_value is None:
        ue.log_error("Invalid inputs for calculating cylinder position.")
        return None, None

    # Calculate the actual positions by subtracting the vertical extent from the top center Z value
    actual_top_value = top_center_position.z - top_value
    actual_bottom_value = top_center_position.z - bottom_value

    ue.log(f"Adjusted cylinder positions - Min Y: {actual_top_value}, Max Y: {actual_bottom_value}")

    return actual_top_value, actual_bottom_value



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
            associated_files = []
            for f in all_files:
                print(f.split('.')[0])
                print(stem_name)
                if f.startswith(stem_name) or stem_name.startswith(f.split('.')[0]):
                    associated_files.append(f)
              
           
            for assoc_file in associated_files:
                print(assoc_file)
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


def parse_mesh_andsubtiles_directory(mesh_folder):
    """
    Parses a directory containing multiple parts of a 3D mesh and organizes the files into a structured list of dictionaries.
    Ensures .mtl files are correctly assigned to the corresponding .obj files and supports multiple parts.
    Additionally, it collects 'editsubtile' subfolders and organizes them under their respective tiles, 
    ensuring OBJ-MTL pairs are correctly mapped within them.

    Parameters:
    - mesh_folder (str): Path to the directory containing the mesh files.

    Returns:
    - list: A list of dictionaries, each representing a tile with its OBJ tiles, MTL files, textures, and editsubtiles.
    """
    texture_extensions = {".jpg", ".jpeg", ".png", ".tga", ".bmp"}
    tile_pattern = re.compile(r'(\w+_tile_x\d+_y\d+_z\d+)')
    
    mesh_data = defaultdict(lambda: {
        "partname": None,
        "OBJAndMtl": [],
        "textures": [],
        "editsubtiles": []
    })
    
    obj_mtl_mapping = defaultdict(lambda: {"OBJtile": None, "Mtl": None})
    
    file_list = [os.path.normpath(os.path.join(mesh_folder, f)) for f in os.listdir(mesh_folder) if os.path.isfile(os.path.join(mesh_folder, f))]
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
        
        match = tile_pattern.search(filename)
        if not match:
            continue
        
        tile_name = match.group(1)
        
        if mesh_data[tile_name]["partname"] is None:
            mesh_data[tile_name]["partname"] = tile_name
        
        if ext == ".obj":
            obj_mtl_mapping[base_name]["OBJtile"] = file_path
        elif ext == ".mtl":
            obj_mtl_mapping[base_name]["Mtl"] = file_path
        elif ext in texture_extensions:
            mesh_data[tile_name]["textures"].append(file_path)
    
    for base_name, obj_mtl in obj_mtl_mapping.items():
        match = tile_pattern.search(base_name)
        if match:
            tile_name = match.group(1)
            mesh_data[tile_name]["OBJAndMtl"].append(obj_mtl)
    
    for folder in os.listdir(mesh_folder):
        folder_path = os.path.join(mesh_folder, folder)
        if os.path.isdir(folder_path) and folder.endswith("_editsubtile"):
            tile_name = folder.replace("_editsubtile", "")
            
            editsubtile_mapping = defaultdict(lambda: {"OBJtile": None, "Mtl": None})
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.normpath(os.path.join(root, file))
                    filename = os.path.basename(file_path)
                    base_name, ext = os.path.splitext(filename)
                    
                    if ext == ".obj":
                        editsubtile_mapping[base_name]["OBJtile"] = file_path
                    elif ext == ".mtl":
                        editsubtile_mapping[base_name]["Mtl"] = file_path
            
            mesh_data[tile_name]["editsubtile"] = list(editsubtile_mapping.values())
    
    return list(mesh_data.values())




def parse_mesh_directory(mesh_folder):
    """
    Parses a directory containing multiple parts of a 3D mesh and organizes the files into a structured list of dictionaries.
    Ensures .mtl files are correctly assigned to the corresponding .obj files and supports multiple parts.

    Parameters:
    - mesh_folder (str): Path to the directory containing the mesh files.

    Returns:
    - list: A list of dictionaries, each representing a part with its OBJ tiles, corresponding MTL files, and textures.
    """
    
    # Supported texture file extensions
    texture_extensions = {".jpg", ".jpeg", ".png", ".tga", ".bmp"}

    # Regex pattern to extract part identifier from filenames (e.g., parts_0000002_0000001.obj)
    part_pattern = re.compile(r'parts_(\d+)_')  # Matches the integer after 'parts_'

    # Data structure to store parsed information
    mesh_parts = defaultdict(lambda: {"partname": None, "OBJtilesAndMtls": [], "textures": []})

    # Temporary dictionary to store OBJ-MTL pairs
    obj_mtl_mapping = defaultdict(lambda: {"OBJtile": None, "Mtl": None})

    # List all files in the directory
    file_list = [os.path.normpath(os.path.join(mesh_folder, f)) for f in os.listdir(mesh_folder) if os.path.isfile(os.path.join(mesh_folder, f))]


    # First pass: Collect all OBJ and MTL files
    for file_path in file_list:
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)  # Extract base name and extension

        # Extract part number
        match = part_pattern.search(filename)
        if not match:
            continue  # Skip files that don't match the expected pattern

        part_id = int(match.group(1))
        part_name = f"part_{part_id}"

        # Assign part name if not already set
        if mesh_parts[part_id]["partname"] is None:
            mesh_parts[part_id]["partname"] = part_name

        # Store OBJ and MTL files together
        if ext == ".obj":
            obj_mtl_mapping[base_name]["OBJtile"] = file_path
        elif ext == ".mtl":
            obj_mtl_mapping[base_name]["Mtl"] = file_path

        # Identify texture files
        elif ext in texture_extensions:
            mesh_parts[part_id]["textures"].append(file_path)

    # Second pass: Add paired OBJ-MTL data to the final structure
    for base_name, obj_mtl in obj_mtl_mapping.items():
        if obj_mtl["OBJtile"]:  # Ensure only valid pairs are added
            part_id = int(part_pattern.search(base_name).group(1))  # Extract part ID from base name
            mesh_parts[part_id]["OBJtilesAndMtls"].append(obj_mtl)

    return list(mesh_parts.values())



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
            elif line.startswith('Kd') or line.startswith('map_Kd'):
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



def build_import_options(texturestemname: str, basematerial):
    options = ue.FbxImportUI()
    #print(options.get_editor_property('static_mesh_import_data'))
    # ue.FbxImportUI
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("is_obj_import", True)
    options.set_editor_property("import_textures", True)
    options.set_editor_property("import_materials", True)
    #options.import_as_skeletal = True
    #ue.FbxMeshImportData
    #ue.log(options.get_editor_property("is_obj_import"))
    options.static_mesh_import_data.set_editor_properties({'import_uniform_scale': 100.0})
    options.static_mesh_import_data.set_editor_properties({'build_nanite': True}) 
    # Texture options
    options.texture_import_data.set_editor_properties({'base_material_name': ue.SoftObjectPath(basematerial)})     
    options.texture_import_data.set_editor_properties({'material_search_location': ue.MaterialSearchLocation.LOCAL})
    print(texturestemname)
    options.texture_import_data.set_editor_properties({'base_diffuse_texture_name' : texturestemname})                                        
    return options

def build_meshtile_import_options():
    options = ue.FbxImportUI()
    #print(options.get_editor_property('static_mesh_import_data'))
    # ue.FbxImportUI
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("is_obj_import", True)
    options.set_editor_property("import_textures", False)
    options.set_editor_property("import_materials", False)
    #options.import_as_skeletal = True
    #ue.FbxMeshImportData
    #ue.log(options.get_editor_property("is_obj_import"))
    options.static_mesh_import_data.set_editor_properties({'import_uniform_scale': 100.0})
    options.static_mesh_import_data.set_editor_properties({'build_nanite': True}) 
    # Texture options
                                     
    return options

def build_fbxfactory(task):
    fbxfactory = ue.FbxFactory()
    fbxfactory.set_editor_property("asset_import_task", task)
    fbxfactory.set_editor_property('editor_import', True)
    fbxfactory.set_editor_property('formats',[])
    return fbxfactory

def build_texture_factory():
    texture_factory = ue.TextureFactory()
    # Set any properties for the texture factory if needed
    return texture_factory

def build_import_tasks(filename: str, texture_files: list, destination_path: str, options):
    tasks = []
    
    # Setup the OBJ import task as you did
    obj_task = ue.AssetImportTask()
    obj_task.set_editor_property("automated", True)
    obj_task.set_editor_property("destination_path", destination_path)
    obj_task.set_editor_property("filename", filename)
    obj_task.set_editor_property("replace_existing", True)
    obj_task.set_editor_property("replace_existing_settings", True)
    obj_task.set_editor_property("options", options)
    fbxfactory = build_fbxfactory(obj_task)
    obj_task.set_editor_property("factory", fbxfactory)
    obj_task.set_editor_property("save", True)
    

    # Identify UDIM-texture files and create tasks for the first one in list which takes the rest of the UDIM files in the folder as dependencies
    #texture_factory = build_texture_factory()
    
    for file in [texture_files[0]]:
        
       texture_task = ue.AssetImportTask()
       texture_task.set_editor_property("automated", True)
       texture_task.set_editor_property("destination_path", destination_path)
       texture_task.set_editor_property("filename", file)
       texture_task.set_editor_property("replace_existing", True)
       texture_task.set_editor_property("replace_existing_settings", True)
       texture_task.set_editor_property("factory", build_texture_factory())
       tasks.append(texture_task)
    tasks.append(obj_task)
    return tasks


def build_import_texture_tasks(texture_files: list, destination_path: str):
    tasks = []
  

    # Identify UDIM-texture files and create tasks for the first one in list which takes the rest of the UDIM files in the folder as dependencies
    #texture_factory = build_texture_factory()
    
    for file in [texture_files[0]]:
        
       texture_task = ue.AssetImportTask()
       texture_task.set_editor_property("automated", True)
       texture_task.set_editor_property("destination_path", destination_path)
       texture_task.set_editor_property("filename", file)
       texture_task.set_editor_property("replace_existing", False)
       texture_task.set_editor_property("replace_existing_settings", True)
       texture_task.set_editor_property("factory", build_texture_factory())
       texture_task.set_editor_property("save", True)
       tasks.append(texture_task)
    tasks.append(texture_task)
    return tasks

def build_meshtile_import_tasks(filename: str, destination_path: str, options):
    tasks = []
    
    # Setup the OBJ import task as you did
    obj_task = ue.AssetImportTask()
    obj_task.set_editor_property("automated", True)
    obj_task.set_editor_property("destination_path", destination_path)
    obj_task.set_editor_property("filename", filename)
    obj_task.set_editor_property("replace_existing", True)
    obj_task.set_editor_property("replace_existing_settings", True)
    obj_task.set_editor_property("options", options)
    fbxfactory = build_fbxfactory(obj_task)
    obj_task.set_editor_property("factory", fbxfactory)
    obj_task.set_editor_property("save", True)
    tasks.append(obj_task)
    return tasks

def import_asset(tasks):
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()
    asset_tools.import_asset_tasks(tasks)
    ue.log("Imported static mesh and texture assets.")

def get_texturestemname(texture_files: list):
    directory, obj_filename = os.path.split(texture_files[0])
    stem_name = os.path.splitext(obj_filename)[0]
    texturestemname = stem_name.split('.')[0]
    return texturestemname

def rename_created_material_instance(destination_path: str, old_material_name: str, new_material_name: str):
    old_material_instance_path = destination_path + "/" + old_material_name
    new_material_instance_path = destination_path + "/" + new_material_name

    # Check if the old material instance exists
    if ue.EditorAssetLibrary.does_asset_exist(old_material_instance_path):
        # Rename the material instance
        result = ue.EditorAssetLibrary.rename_asset(old_material_instance_path, new_material_instance_path)
        if not result:
            ue.log_error(f"Failed to rename material instance {old_material_name} to {new_material_name}")
    else:
        ue.log_error(f"Material instance {old_material_name} does not exist in path {destination_path}")

def set_udim_texture_to_material_instance(mesh_import_path: str, material_instance_name: str, texture_name: str, parameter_name: str):
    # Load the material instance and the texture
    material_instance = ue.EditorAssetLibrary.load_asset(material_instance_name)
    texture = ue.EditorAssetLibrary.load_asset(texture_name)
    
    # Check if both assets are loaded properly
    if not material_instance or not texture:
        ue.log_error("Unable to load assets!")
        return

    if isinstance(material_instance, ue.MaterialInstanceConstant):
        # Access the list of texture parameter values
        texture_params = list(material_instance.texture_parameter_values)  # Make a copy

        # Check if the parameter exists and set its value
        param_exists = False
        for param in texture_params:
            if param.parameter_info.name == parameter_name:
                print(param.parameter_info.name, 'is already in the list.')
                param.parameter_value = texture
                param_exists = True
                break
        
        # If the parameter does not exist, add a new one
        if not param_exists:
            new_param = ue.TextureParameterValue()
            new_param.parameter_info.name = parameter_name
            new_param.parameter_value = texture
            print('This is the new parameter:', new_param)
            texture_params.append(new_param)  # Append to the copied list
        
        # Use the set_editor_property method
        material_instance.set_editor_property('texture_parameter_values', texture_params)
        
        mesh = ue.EditorAssetLibrary.load_asset(mesh_import_path)

        if isinstance(mesh, ue.StaticMesh):
        # Create a new StaticMaterial object
            new_static_material = ue.StaticMaterial()
            new_static_material.material_interface = material_instance

            # Assign to the first slot (or whichever slot you want to assign to)
            mesh.static_materials[0] = new_static_material
            ue.EditorAssetLibrary.save_asset(mesh_import_path, only_if_is_dirty=False)

        #ue.EditorAssetLibrary.save_asset(mesh)
        # Save the modified material instance
        ue.EditorAssetLibrary.save_asset(material_instance_name, only_if_is_dirty=False)
        ue.EditorAssetLibrary.save_asset(texture_name, only_if_is_dirty=False)

    else:
        ue.log_error("The loaded asset is not a MaterialInstanceConstant!")

def create_constant_material_instance(base_material_path: str, UEdestination: str, material_instance_name: str, texturepath: str):

    # Get the asset tools interface
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()

    # Define the base material (Replace with your actual material path)
    #material_path = "/Game/Materials/BaseMaterial"  # Change to your material
    base_material = ue.EditorAssetLibrary.load_asset(base_material_path)

    # Define where to save the material instance
    #instance_name = "MyMaterialInstance"
    instance_path = UEdestination + "/" + material_instance_name
    # Ensure the material exists
    if not base_material:
        ue.log_error(f"Material not found: {base_material_path}")
    else:
        # Create a MaterialInstanceConstantFactoryNew instance
        factory = ue.MaterialInstanceConstantFactoryNew()
        # Create the Material Instance Constant asset
        material_instance = asset_tools.create_asset(material_instance_name, UEdestination, ue.MaterialInstanceConstant, factory)

        if material_instance:
            # Set the parent material
            material_instance.set_editor_property("parent", base_material)
            texture = ue.EditorAssetLibrary.load_asset(texturepath)
            new_param = ue.TextureParameterValue()
            new_param.parameter_info.name = "emmisive"
            new_param.parameter_value = texture
            print('This is the new parameter:', new_param)
            material_instance.set_editor_property('texture_parameter_values', [new_param])

            # Save the new Material Instance to the Content Browser
            ue.EditorAssetLibrary.save_asset(instance_path)

            ue.log(f"Material Instance created successfully: {instance_path}")
            return material_instance
        else:
            ue.log_error("Failed to create Material Instance")
            return None






def post_import_process(mesh_import_path: str, texture_stem_name: str, destination_path: str):
    # Names based on your provided paths and names
    material_instance_name = mesh_import_path + "_material"
    oldname =  "defaultMat_ncl1_1"
    rename_created_material_instance(destination_path,oldname, material_instance_name)
    texture_name = destination_path + "/" + texture_stem_name  # Change the path if it's different
    material_path = destination_path + "/" + material_instance_name
    mesh_path = destination_path + "/" + mesh_import_path

    set_udim_texture_to_material_instance(mesh_path, material_path, texture_name, 'diffuse')


# Function to find all Static Mesh Actors in the current level
def get_all_static_mesh_actors():
    # Use the Editor Level Library to get all actors of a specific type
    static_mesh_actors = ue.EditorLevelLibrary.get_all_level_actors()
    
    # Filter for Static Mesh Actors
    static_mesh_actors = [actor for actor in static_mesh_actors if isinstance(actor, ue.StaticMeshActor)]
    
    return static_mesh_actors

def list_static_mesh_actors():
    static_mesh_actors = get_all_static_mesh_actors()
    if static_mesh_actors:
        ue.log("Found {} static mesh actors:".format(len(static_mesh_actors)))
        for actor in static_mesh_actors:
            ue.log(actor.get_name())
    else:
        ue.log("No static mesh actors found in the level.")

def get_static_mesh_actors_by_name(search_string):
    # Get the current level's actors
    level_actors = ue.EditorLevelLibrary.get_all_level_actors()
    
    # List to hold the matching actors
    matching_actors = []
    
    # Iterate through all the actors
    for actor in level_actors:
        # Check if the actor is a StaticMeshActor
        if isinstance(actor, ue.StaticMeshActor):
            # Get the actor's display name
            actor_name = actor.get_actor_label()
            
            # Check if the display name contains the search string
            if search_string.lower() in actor_name.lower():
                matching_actors.append(actor)

    return matching_actors

def get_top_center_of_0to1m_mesh(actors_list):
    """
    Find the top-center world position of the StaticMeshActor whose name contains '0to1m'.

    Args:
        actors_list (list): A list of StaticMeshActor objects to search within.

    Returns:
        Vector: The world position of the top center, or None if no match is found.
    """
    # Iterate through the list of provided actors
    for actor in actors_list:
        # Get the actor's display name
        actor_name = actor.get_actor_label()
        # get from actor_name the substring until the first _ character
        actor_name_short = actor_name.split('_')[0]
        
        # Check if the actor's name contains "0to1m"
        if "0to1m" in actor_name.lower():
            ue.log(f"0to1: {actor_name}")
            # Get the static mesh component
            static_mesh_component = actor.get_component_by_class(ue.StaticMeshComponent)
            
            if static_mesh_component:
                # Get the bounding box of the static mesh in local space
                bounds_origin, bounds_box_extent = static_mesh_component.get_local_bounds()
                
                # Calculate the relative extent by subtracting the origin from the extent
                relative_extent = ue.Vector(
                    bounds_box_extent.x - bounds_origin.x,
                    bounds_box_extent.y - bounds_origin.y,
                    bounds_box_extent.z - bounds_origin.z
                )

                # Calculate the full height based on the bounding box extent
                full_height = relative_extent.z * 2  # Full height in Unreal's units

                # Adjust the top-center position based on the bounds origin
                top_center_local = ue.Vector(bounds_origin.x, bounds_origin.y, bounds_origin.z + relative_extent.z)
                ue.log(f"Bounds Origin: {bounds_origin}, Extent: {bounds_box_extent}, Relative Extent: {relative_extent}, Top Center Local: {top_center_local}")

                # Convert local top-center position to world space by applying the world transform
                top_center_world = static_mesh_component.get_world_transform().transform_location(top_center_local)
                ue.log(f"Top Center World: {top_center_world}")

                # Create a white cone at the top center position
                create_cone_at_position(top_center_world, height=100, radius=20, resource_identifier=actor_name_short)

                # Return the top-center world position
                return top_center_world  # Return the world position instead of bounds_origin
    
    # If no actor with "0to1m" is found, return None
    return None
    
def create_cone_at_position(top_center_position, height, radius, resource_identifier):
    """
    Create a cone mesh using Geometry Script at a specified world location.

    Args:
        top_center_position (Vector): The world position where the cone will be created (at the base).
        height (float): The height of the cone.
        radius (float): The radius of the cone at the base.
        resource_identifier (str): The identifier string to set as the display name for the cone.

    Returns:
        StaticMeshAsset: The created static mesh asset for the cone.
    """
    # Log the creation parameters
    ue.log(f"Creating cone with height: {height}, radius: {radius}")

    # Create a DynamicMesh object to hold the cone geometry
    target_mesh = ue.DynamicMesh()

    # Create a primitive options object for the cone
    primitive_options = ue.GeometryScriptPrimitiveOptions()

    # Create a transform for the cone at the desired world location
    transform = ue.Transform()
    offset = ue.Vector(6, 3, 0)
    # Adjust the Z-value to center the cylinder correctly
    transform.translation = ue.Vector(top_center_position.x, top_center_position.y,top_center_position.y) + offset
    # Adjust the Z-value to set the base of the cone at the specified position
    #transform.translation = ue.Vector(top_center_position.x, top_center_position.y, top_center_position.z)

    # Append the cone to the target mesh
    ue.GeometryScript_Primitives.append_cone(
        target_mesh,
        primitive_options,
        transform,
        base_radius=20, 
        top_radius=0.1000000,
        height=height,
        radial_steps=12,  # Adjust as needed
        height_steps=0,
        capped=True,
        origin=ue.GeometryScriptPrimitiveOriginMode.BASE,  # Set the origin to the base
    )

    # Create a Static Mesh Asset from the Dynamic Mesh
    # Replace spaces with underscores in the resource identifier for the asset name
    static_mesh_name = resource_identifier.replace(' ', '_')
    static_mesh_path = f"/Game/idaifield_resources/"  # Define the asset path
    # Create AssetTools
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()

    # Create a new static mesh asset
    static_mesh_asset = asset_tools.create_asset(static_mesh_name, static_mesh_path, ue.StaticMesh.static_class(), None)

    # Check if the asset was created successfully
    if static_mesh_asset is None:
        ue.log_error(f"Failed to create static mesh asset: {static_mesh_name}")
        return None

    # Create a new Static Mesh asset
    options = create_copy_mesh_options()
    lod = ue.GeometryScriptMeshWriteLOD(False, 0)
    outcome = ue.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(target_mesh, static_mesh_asset, options=options, target_lod=lod)
    ue.log(f"Copy mesh outcome: {outcome}")

    return static_mesh_asset


def create_dynamic_cylinder_and_save(top_center_position, top_value, bottom_value, resource_identifier, radius=3.0):
    """
    Create a dynamic cylinder mesh using Geometry Script at a specified world location and save it as an asset.

    Args:
        top_center_position (Vector): The world position where the cylinder will be created.
        top_value (float): The maximum Y value to set as the cylinder's top.
        bottom_value (float): The minimum Y value to set as the cylinder's base.
        resource_identifier (str): The identifier string to set as the display name for the cylinder.
        radius (float): The radius of the cylinder. Default is 20 cm.
    
    Returns:
        None
    """
    # Calculate the height of the cylinder
    height = top_value - bottom_value

    # Log the creation parameters
    ue.log(f"Creating dynamic cylinder with height: {height}, radius: {radius}")

    # Create a DynamicMesh object to hold the cylinder geometry
    target_mesh = ue.DynamicMesh()

    # Create a primitive options object for the cylinder
    primitive_options = ue.GeometryScriptPrimitiveOptions()
    
    # Create a transform for the cylinder at the desired world location
    transform = ue.Transform()
    # Offsetvector
    offset = ue.Vector(6, 3, 0)
    # Adjust the Z-value to center the cylinder correctly
    transform.translation = ue.Vector(top_center_position.x, top_center_position.y, bottom_value + (height / 2.0)) + offset

    # Append the cylinder to the target mesh
    ue.GeometryScript_Primitives.append_cylinder(
        target_mesh,
        primitive_options,
        transform,
        radius=radius,
        height=height,
        radial_steps=60,  # Adjust as needed
        height_steps=0,
        capped=True,
        origin=ue.GeometryScriptPrimitiveOriginMode.CENTER,
    )

    # Create a Static Mesh Asset from the Dynamic Mesh
    # Replace spaces with underscores in the resource identifier for the asset name
    static_mesh_name = resource_identifier.replace(' ', '_')
    static_mesh_path = f"/Game/idaifield_resources/"  # Define the asset path
    # Create AssetTools
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()

    # Create a new static mesh asset
    static_mesh_asset = asset_tools.create_asset(static_mesh_name, static_mesh_path, ue.StaticMesh.static_class(), None)

    # Check if the asset was created successfully
    if static_mesh_asset is None:
        ue.log_error(f"Failed to create static mesh asset: {static_mesh_name}")
        return None

    # Create a new Static Mesh asset
    options = create_copy_mesh_options()
    lod = ue.GeometryScriptMeshWriteLOD(False, 0)
    outcome = ue.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(target_mesh, static_mesh_asset, options=options, target_lod=lod)
    ue.log(f"Copy mesh outcome: {outcome}")
    return static_mesh_asset

 



def create_copy_mesh_options():
    # Create an instance of GeometryScriptCopyMeshToAssetOptions
    copy_options = ue.GeometryScriptCopyMeshToAssetOptions()

    # Set the properties as needed
    copy_options.enable_recompute_normals = True        # Enable recomputation of normals
    copy_options.enable_recompute_tangents = True       # Enable recomputation of tangents
    copy_options.enable_remove_degenerates = True       # Enable removal of degenerate triangles
    copy_options.replace_materials = False               # Do not replace materials
    copy_options.new_materials = []                      # No new materials
    copy_options.new_material_slot_names = []            # No new material slot names
    copy_options.apply_nanite_settings = False           # Do not apply Nanite settings
    copy_options.nanite_settings = ue.GeometryScriptNaniteOptions()  # Default Nanite options
    copy_options.new_nanite_settings = ue.MeshNaniteSettings()        # Default new Nanite settings
    copy_options.emit_transaction = False                # Do not emit a transaction
    copy_options.defer_mesh_post_edit_change = False    # Do not defer post-edit change

    return copy_options


def create_static_mesh_with_options(asset_name, asset_path):
    # Create an instance of InterchangeStaticMeshFactoryNode
    factory_node = ue.InterchangeStaticMeshFactoryNode()

    # Initialize the static mesh node with unique ID, display label, and asset class
    unique_id = "my_unique_id"  # Change this to a unique identifier
    display_label = asset_name
    asset_class = "StaticMesh"  # Type of asset to create

    factory_node.initialize_static_mesh_node(unique_id, display_label, asset_class)

    # Set some basic properties (customize these as needed)
    factory_node.set_custom_build_nanite(True)  # Enable Nanite
    factory_node.set_custom_generate_lightmap_u_vs(True)  # Enable lightmap UV generation
    factory_node.set_custom_max_lumen_mesh_cards(12)  # Set max Lumen mesh cards

    # Now, create the static mesh asset
    # Assume you have a valid Dynamic Mesh instance `dynamic_mesh`
    dynamic_mesh = ue.DynamicMesh()  # Replace with your dynamic mesh creation logic

    # Create a new static mesh asset
    static_mesh_asset = ue.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name, asset_path, ue.StaticMesh, factory_node
    )

    # Copy the mesh data from the dynamic mesh to the static mesh
    outcome = ue.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(dynamic_mesh, static_mesh_asset, factory_node, 0)

    # Check if the operation was successful
    if outcome == ue.GeometryScriptOutcomePins.SUCCESS:
        ue.log(f"Static mesh '{asset_name}' created successfully in '{asset_path}'.")
    else:
        ue.log_error(f"Failed to create static mesh '{asset_name}'. Outcome: {outcome}")

def create_dynamic_material_instance(static_mesh_actor, material_path):
    """
    Create and assign a dynamic material instance to the static mesh component of the given actor.

    Args:
        static_mesh_actor: The actor to which the material will be assigned.
        material_path: The path of the base material to create the dynamic instance from.
    """
    # Load the base material
    base_material = ue.EditorAssetLibrary.load_asset(material_path)
    
    if base_material is None:
        ue.log_error(f"Failed to load the material from {material_path}")
        return

    # Get the static mesh component of the actor
    mesh_component = static_mesh_actor.get_component_by_class(ue.StaticMeshComponent)

    if mesh_component is not None:
        # Create a dynamic material instance
        dynamic_material_instance = ue.MaterialLibrary.create_dynamic_material_instance(
            static_mesh_actor,  # world_context_object
            base_material  # parent material
        )

        if dynamic_material_instance is None:
            ue.log_error("Failed to create dynamic material instance")
            return
        
        # Assign the dynamic material instance to the first material slot
        mesh_component.set_material(0, dynamic_material_instance)
        ue.log(f"Dynamic material instance assigned to {static_mesh_actor.get_actor_label()}")
    else:
        ue.log_error("StaticMeshComponent not found in the actor")



def spawn_resource_actor(static_mesh_asset, doc):
    """
    Create ResourceActor instances next to the static meshes and populate their properties with JSON data.

    Args:
        static_mesh_asset: The created static mesh asset.
        resource: The resource JSON data to populate the actor.
    """
    resource = doc['resource']
    # Load the ResourceActor class
    resource_actor_asset = ue.EditorAssetLibrary.load_asset("/Game/idaifield_resources/ResourceActor")  # Adjust the path as necessary
    if resource_actor_asset is None:
        ue.log_error("ResourceActor asset could not be loaded!")
        return
    # Get the class from the asset
    resource_actor_class = resource_actor_asset.generated_class()
    if resource_actor_class is None:
        ue.log_error("ResourceActor class could not be loaded!")
        return
    ue.log(f"ResourceActor class loaded: {resource_actor_class}")



    # Spawn the ResourceActor next to the static mesh
    spawn_location = ue.Vector(0, 0, 0)  # You can adjust the spawn location as needed
    resource_actor = ue.EditorLevelLibrary.spawn_actor_from_class(resource_actor_class, spawn_location)
    resource_actor.set_actor_label(resource['identifier'])  # Set the actor label to the identifier
    if resource_actor is None:
        ue.log_error("Failed to spawn ResourceActor!")
        return

    # Assign the static mesh to the ResourceActor
    mesh_component = resource_actor.get_component_by_class(ue.StaticMeshComponent)
    mesh_component.set_static_mesh(static_mesh_asset)

    # Populate the ResourceActor properties with JSON data
    parse_resource(resource, resource_actor)

    # Create and assign dynamic material instance
    material_path = "/Game/idaifield_resources/MA_ResourceActor"  # Adjust to your material path
    create_dynamic_material_instance(resource_actor, material_path)

    # Log the creation of the ResourceActor
    ue.log(f"Created ResourceActor: {resource_actor.get_actor_label()} with static mesh: {static_mesh_asset.get_name()}")
    # Position the text components based on the static mesh bounds
    resource_actor.call_method("PositionTextComponents")


def parse_resource(resource_json, resource_actor):
    """
    Parse resource JSON data and populate the ResourceActor instance.

    Args:
        resource_json (dict): The JSON data for the resource.
        resource_actor (AResourceActor): The ResourceActor instance to populate.
    """
    # Ensure the resource_actor is valid
    if resource_actor is None:
        ue.log_error("ResourceActor reference is None!")
        return

    # Parse common attributes
    if 'identifier' in resource_json:
        resource_actor.set_editor_property("Identifier", resource_json['identifier'])
    
    if 'category' in resource_json:
        resource_actor.set_editor_property("Category", resource_json['category'])
    
    if 'type' in resource_json:
        resource_actor.set_editor_property("Category", resource_json['type'])
    
    if 'sampleType' in resource_json:
        resource_actor.set_editor_property("SampleType", resource_json['sampleType'])
    
        
    if 'shortDescription' in resource_json and resource_json['shortDescription']:
        short_description = resource_json['shortDescription']
    else:
        short_description = ""  # Default to an empty string if not present

    if 'description' in resource_json and resource_json['description']:
        normal_description = resource_json['description']
        resource_actor.set_editor_property("Description", normal_description)  # Keep the original description
    else:
        normal_description = ""  # Default to an empty string if not present

    # Combine shortDescription and description for ShortDescription
    combined_description = f"{short_description} {normal_description}".strip()
    resource_actor.set_editor_property("ShortDescription", combined_description)
        

def shift_sediment_cores_to_player(search_string, player_actor_label="CameraActor_compareProfile"):
    """
    Temporarily shift sediment cores in X and Y axes to place them next to the player.

    Args:
        search_string (str): String to search for sediment core static meshes.
        player_actor_label (str): Label of the player actor. Default is "Player".
    """
    try:
        # Step 1: Search for the sediment core actors
        sediment_core_actors = get_static_mesh_actors_by_name(search_string)
        if not sediment_core_actors:
            ue.log_error(f"No actors found with search string: {search_string}")
            return

        # Step 2: Find the top-center position of the core series
        top_center_position = get_top_center_of_0to1m_mesh(sediment_core_actors)
        if not top_center_position:
            ue.log_error("Failed to find top-center position for the cores.")
            return

        # Step 3: Find the player's location
        all_actors = ue.EditorLevelLibrary.get_all_level_actors()
        player_actor = next((actor for actor in all_actors if actor.get_actor_label() == player_actor_label), None)
        if not player_actor:
            ue.log_error(f"Player actor with label '{player_actor_label}' not found.")
            return

        player_location = player_actor.get_actor_location()

        # Step 4: Calculate the transformation vector for X and Y
        shift_vector = ue.Vector(
            player_location.x - top_center_position.x,
            player_location.y - top_center_position.y,
            0  # Only shifting in X and Y, so Z remains unchanged
        )
        ue.log(f"Calculated shift vector: {shift_vector}")

        # Step 5: Apply the shift to each sediment core
        for actor in sediment_core_actors:
            current_location = actor.get_actor_location()
            new_location = ue.Vector(
                current_location.x + shift_vector.x,
                current_location.y + shift_vector.y,
                current_location.z  # Keep Z position the same
            )
            actor.set_actor_location(new_location, sweep=False, teleport=True)
            ue.log(f"Shifted actor {actor.get_actor_label()} to {new_location}")

        ue.log("All sediment cores successfully shifted to the player's location.")

    except Exception as e:
        ue.log_error(f"An error occurred while shifting sediment cores: {str(e)}")

def assign_dynamic_material_to_dynamic_mesh(dynamic_mesh_actor, material_path):
    """
    Create and assign a dynamic material instance to the dynamic mesh component of the given actor.

    Args:
        dynamic_mesh_actor: The actor to which the material will be assigned.
        material_path: The path of the base material to create the dynamic instance from.
    """
    # Load the base material
    base_material = ue.EditorAssetLibrary.load_asset(material_path)
    
    if base_material is None:
        ue.log_error(f"Failed to load the material from {material_path}")
        return

    # Get the dynamic mesh component of the actor
    mesh_component = dynamic_mesh_actor.get_dynamic_mesh_component()

    if mesh_component is not None:
        # Create a dynamic material instance
        dynamic_material_instance = ue.MaterialLibrary.create_dynamic_material_instance(
            dynamic_mesh_actor,  # world_context_object
            base_material  # parent material
        )

        if dynamic_material_instance is None:
            ue.log_error("Failed to create dynamic material instance")
            return
        
        # Assign the dynamic material instance to the first material slot
        mesh_component.set_material(0, dynamic_material_instance)
        ue.log(f"Dynamic material instance assigned to {dynamic_mesh_actor.get_actor_label()}")
    else:
        ue.log_error("DynamicMeshComponent not found in the actor")



def assign_texture_to_constant_material(material_instance, texture_path: str):
    """
    Assigns a texture to the BaseColor parameter of the given material instance.

    :param material_instance: The material instance reference
    :param texture_path: Path to the texture asset
    """

    if not material_instance:
        ue.log_error("Invalid material instance provided.")
        return
    
    # Load texture
    texture = ue.EditorAssetLibrary.load_asset(texture_path)
    if not texture:
        ue.log_error(f"Texture not found: {texture_path}")
        return
    
    # Set the texture parameter
    material_instance.set_editor_property("texture_parameter_values", [{"ParameterInfo": {"Name": "BaseColor"}, "ParameterValue": texture}])

    # Save the modified material instance
    ue.EditorAssetLibrary.save_asset(material_instance.get_path_name())

    ue.log(f"Texture assigned successfully to material instance: {material_instance.get_path_name()}")
    ue.EditorLoadingAndSavingUtils.unload_unused_assets()





