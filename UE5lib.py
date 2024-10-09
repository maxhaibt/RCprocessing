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
        resources (list): A list of resources containing geometry information.

    Returns:
        list: A list of Y-coordinates from all polygon vertices.
    """
    y_coordinates = []


    if isinstance(resource, dict) and 'resource' in resource:
        ue.log('Resource Is a dict')
        try:
            if isinstance(resource['resource'], dict) and 'geometry' in resource['resource']:
                ue.log('Geometry exists')
                geometry = resource['resource']['geometry']
                if geometry['type'] == 'Polygon':
                    coordinates = geometry['coordinates'][0]  # Get the first ring of the polygon
                    for vertex in coordinates:
                        y_coordinates.append(vertex[1])  # Append the Y-coordinate
        except KeyError as e:
            ue.log_error(f"KeyError: {str(e)}")
        except Exception as e:
            ue.log_error(f"An error occurred: {str(e)}")

    return y_coordinates

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

    translated_min_y = (min_y - 12.2076) * 100  # Convert to cm
    translated_max_y = (max_y - 12.2076) * 100  # Convert to cm

    return translated_min_y, translated_max_y

def create_cylinder_at_position(top_center_position, translated_min_y, translated_max_y, radius=20.0):
    """
    Create a cylinder actor in Unreal Engine at the specified position, with height defined by translated Y values.

    Args:
        top_center_position (Vector): The world position where the cylinder will be created.
        translated_min_y (float): The minimum Y value to set as the cylinder's base.
        translated_max_y (float): The maximum Y value to set as the cylinder's top.
        radius (float): The radius of the cylinder. Default is 20 cm.
    """
    # Calculate the height of the cylinder
    height = translated_max_y - translated_min_y

    # Log the creation parameters
    ue.log(f"Creating cylinder at position: {top_center_position}, height: {height}, radius: {radius}")

    # Create a cylinder actor
    cylinder_actor = ue.EditorLevelLibrary.spawn_actor_from_class(ue.StaticMeshActor, top_center_position)

    # Create a cylinder mesh
    cylinder = ue.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cylinder")

    # Access the static mesh component directly
    static_mesh_component = cylinder_actor.static_mesh_component

    # Set the static mesh to the cylinder
    static_mesh_component.set_static_mesh(cylinder)

    # Set the actor location at the desired height
    new_location = ue.Vector(top_center_position.x, top_center_position.y, translated_min_y + (height / 2.0))
    cylinder_actor.set_actor_location(new_location, False, None)

    # Set the scale of the cylinder to match the radius and height
    scale = ue.Vector(radius / 100.0, radius / 100.0, height / 100.0)  # Convert cm to Unreal scale
    static_mesh_component.set_relative_scale3d(scale)






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

def build_fbxfactory(task):
    fbxfactory = ue.FbxFactory()
    fbxfactory.set_editor_property("asset_import_task", task)
    fbxfactory.set_editor_property('editor_import', True)
    #fbxfactory.set_editor_property('formats',[]
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
    texture_factory = build_texture_factory()
    
    for file in [texture_files[0]]:
        
        texture_task = ue.AssetImportTask()
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
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()
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
        tuple: The actor's name and the top-center world position, or None if no match is found.
    """
    # Iterate through the list of provided actors
    for actor in actors_list:
        # Get the actor's display name
        actor_name = actor.get_actor_label()
        
        # Check if the actor's name contains "0to1m"
        if "0to1m" in actor_name.lower():
            # Get the static mesh component
            static_mesh_component = actor.get_component_by_class(ue.StaticMeshComponent)
            
            if static_mesh_component:
                # Get the bounding box of the static mesh in local space
                bounds_origin, bounds_box_extent = static_mesh_component.get_local_bounds()
                
                # Calculate the local top-center position
                top_center_local = ue.Vector(bounds_origin.x, bounds_origin.y, bounds_origin.z + bounds_box_extent.z)
                ue.log(top_center_local)
                # Convert local top-center position to world space by applying the world transform
                top_center_world = static_mesh_component.get_world_transform().transform_location(top_center_local)
                ue.log(top_center_world)
                # Return the actor's name and its top-center world position
                return top_center_world
    
    # If no actor with "0to1m" is found, return None
    return None