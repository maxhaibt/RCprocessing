import shutil
import subprocess
from pathlib import Path
import numpy as np
import os
import json
import itertools
import pandas as pd
import exifread
import pathconfig
from datetime import datetime
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.affinity import rotate
import xml.etree.ElementTree as ET
import re
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d
#from geopandas.tools import sjoin
#from shapely.geometry import Polygon
#import pickle


import os
import exifread
from datetime import datetime, timedelta
from shutil import move


def locateColorPatches():
    # Load the image
    img = io.imread('image.jpg')

    # Convert the image to grayscale
    gray = color.rgb2gray(img)

    # Detect edges using the Canny algorithm
    edges = feature.canny(gray)

    # Find contours in the image
    contours = feature.corner_peaks(feature.corner_harris(edges), min_distance=5)

    # Sort the contours by x and y coordinates
    contours = sorted(contours, key=lambda x: (x[0], x[1]))

    # Calculate the patch size
    patch_size = int(np.mean(np.diff(contours[:, 0])))

    # Define the patch locations
    patch_locs = []
    for y in range(patch_size//2, 4*patch_size, patch_size):
        for x in range(patch_size//2, 6*patch_size, patch_size):
            patch_locs.append((x, y))

    print(patch_locs)




def sort_image_series(folderpath):
    # List all files in folderpath
    files = os.listdir(folderpath)
    # Filter only JPG and DNG files
    files = [f for f in files if f.endswith('.JPG') or f.endswith('.dng')]

    # Read image metadata using exifread
    image_data = []
    for f in files:
        path = os.path.join(folderpath, f)
        with open(path, 'rb') as file:
            tags = exifread.process_file(file, details=False)
            datetime_str = str(tags.get('EXIF DateTimeOriginal'))
            datetime_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
            image_data.append({'filename': f, 'datetime': datetime_obj})

    # Sort image data by datetime
    image_data.sort(key=lambda x: x['datetime'])

    # Cluster images by datetime
    clusters = []
    cluster_start = None
    for data in image_data:
        if cluster_start is None:
            cluster_start = data['datetime']
            clusters.append([data['filename']])
        else:
            time_diff = data['datetime'] - cluster_start
            if time_diff <= timedelta(seconds=60):
                clusters[-1].append(data['filename'])
                cluster_start = data['datetime']
            else:
                cluster_start = data['datetime']
                clusters.append([data['filename']])
                cluster_start = data['datetime']

    # Create new folders and move images into them
    for i, cluster in enumerate(clusters):
        new_folder = os.path.join(folderpath, f'cluster_{i+1}')
        os.makedirs(new_folder)
        for filename in cluster:
            src_path = os.path.join(folderpath, filename)
            dst_path = os.path.join(new_folder, filename)
            shutil.move(src_path, dst_path)

def loadconfigs(configpath):
    with open(configpath) as configfile:
        config = json.load(configfile)
    return config
<<<<<<< HEAD
config = loadconfigs('C:/Users/gilgamesh/Documents/GitHub/RCprocessing/config_sedimentcores.json')
=======
config = loadconfigs('E:/GitHub/RCprocessing/config_scanner.json')
>>>>>>> 949a27ffe2db70d9fac21a54b1c4817b2c11908e


def sort_image_series(folderpath, timedelta=60):
    # List all files in folderpath
    files = os.listdir(folderpath)
    # Filter only JPG and DNG files
    files = [f for f in files if f.endswith('.JPG') or f.endswith('.dng')]

    # Read image metadata using exifread
    image_data = []
    for f in files:
        path = os.path.join(folderpath, f)
        with open(path, 'rb') as file:
            tags = exifread.process_file(file, details=False)
            datetime_str = str(tags.get('EXIF DateTimeOriginal'))
            datetime_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
            image_data.append({'filename': f, 'datetime': datetime_obj})

    # Sort image data by datetime
    image_data.sort(key=lambda x: x['datetime'])

    # Cluster images by datetime
    clusters = []
    cluster_start = None
    for data in image_data:
        if cluster_start is None:
            cluster_start = data['datetime']
            clusters.append([data['filename']])
        else:
            time_diff = data['datetime'] - cluster_start
            if time_diff.total_seconds() <= timedelta:
                clusters[-1].append(data['filename'])
                cluster_start = data['datetime']
            else:
                cluster_start = data['datetime']
                clusters.append([data['filename']])
                cluster_start = data['datetime']

    # Create new folders and move images into them
    for i, cluster in enumerate(clusters):
        new_folder = os.path.join(folderpath, f'cluster_{i+1}')
        os.makedirs(new_folder)
        for filename in cluster:
            src_path = os.path.join(folderpath, filename)
            dst_path = os.path.join(new_folder, filename)
            shutil.move(src_path, dst_path)



def provide_scandf(inputdirectory: str, imageformat = '*.dng') ->pd.DataFrame:
    scandf = []
    for scan_id in Path(inputdirectory).iterdir():
        if scan_id.is_dir() and not scan_id.stem in config['excludescanids']:
            scan = {}
            scan['id']= scan_id.stem
            scan['processingstate'] = pd.DataFrame({'command': 'provide_scandf', 'datetime':datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'success': True}, index=['datetime'])
            #pd.concat([scan['processingstate'],
            scan['scan_dir'] = Path(os.path.join(inputdirectory, scan_id))
            scan['pp3file'] = [file for file in scan['scan_dir'].rglob("*.pp3")]
            if 'constantpp3file' in config.keys() and Path(config['constantpp3file']).is_file():
                scan['pp3file'] = [config['constantpp3file']]
            scan['gcpsfile'] = [file for file in scan['scan_dir'].rglob("*rcgcps.csv")]
            scan['orthoboxfile'] = [file for file in scan['scan_dir'].rglob("*.rcortho")]
            scan['boxfile'] = [Path(file) for file in scan['scan_dir'].rglob("*base.rcbox")]
            scan['camregistrationpath'] = [Path(file) for file in scan['scan_dir'].rglob("*camregistration.csv")]
            scan['scannerlogfile'] = [file for file in scan['scan_dir'].rglob("00-*")]
            imagelist = []
            for file in scan['scan_dir'].rglob(imageformat):
                if not file.stem.endswith('.mask') or not file.stem.startswith('URUK'):
                    image_dict = {}
                    image_dict['rawimg_path']= Path(file)
                    mask = image_dict['rawimg_path'].with_name(image_dict['rawimg_path'].name + '.mask.png')

                    if mask.is_file():
                        image_dict['maskimg_path'] = mask
                    imagelist.append(image_dict.copy())
            scan['imagedf'] = pd.DataFrame(imagelist)
            scandf.append(scan.copy())
    return pd.DataFrame(scandf)

def visualize_rcbox(rcbox):
    print("Let's define some primitives")


    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    mesh_box = rcbox['geometry']
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    #print(mesh_box)
    #mesh_box.compute_vertex_normals()
    #mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries([mesh_box])

def create_scaling_matrix(sx, sy, sz):
    """
    Create a 3D scaling matrix.
    """
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def create_object_coordinate_frame(x, y, z, x_length, y_length, z_length):
    # Create a coordinate frame with unit size
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[x, y, z])

    return coord_frame




def tranformmatrix_to_local(rcbox):
    # Extract the transformations
    translation = rcbox['transform_to_local']['translation_matrix']
    rotation = rcbox['transform_to_local']['rotation_matrix']

    print(rotation)

    # Create a 4x4 identity matrix
    to_local_matrix = np.eye(4)

    # Set the rotation
    to_local_matrix[:3, :3] = rotation  # Transpose of the rotation matrix

    # Set the translation (inverse)
    to_local_matrix[:3, 3] = translation

    return to_local_matrix

def transformmatrix_to_global(rcbox):
    # Extract the transformations
    translation = rcbox['transform_to_local']['translation_matrix']
    rotation = rcbox['transform_to_local']['rotation_matrix']

    # Create a 4x4 identity matrix
    to_global_matrix = np.eye(4)

    # Set the rotation
    to_global_matrix[:3, :3] = rotation.T

    # Set the translation
    to_global_matrix[:3, 3] = -translation

    return to_global_matrix

def persistent_translate(rcbox, x, y, z):
    #create translation matrix
    translation_matrix = np.array([x, y, z])
    rcbox['geometry'].translate(translation_matrix, relative=True)
    #document inverted translation_matrix
    rcbox['transform_to_local']['translation_matrix'] = rcbox['transform_to_local']['translation_matrix'] -translation_matrix
    return rcbox

def localspace_rotation(rcbox, yaw, pitch, roll):
    # transform the rcbox to the local space
    to_local_matrix = tranformmatrix_to_local(rcbox)
    localgeom = rcbox['geometry'].transform(to_local_matrix)
    # convert to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    # Create a rotation matrix
    R = o3d.geometry.Geometry3D.get_rotation_matrix_from_yxz([yaw_rad, pitch_rad, roll_rad])
    # Apply the rotation
    print('Back to local center should be 000:',localgeom.get_center())
    locallymodified = localgeom.rotate(R, center=[0,0,0])
    # transform the locally modified geometry of the rcbox to the global space
    to_global_matrix = transformmatrix_to_global(rcbox)
    print('back matrix is: ', to_global_matrix)
    # Apply the global transformation to the locally modified geometry
    rcbox['geometry'] = locallymodified.transform(to_global_matrix)
    #rcbox['geometry'] = locallymodified
    rcbox['transform_to_local']['rotation_matrix'] = rcbox['transform_to_local']['rotation_matrix'] @ R.T
    return rcbox


def localspace_scale(rcbox, scale_x, scale_y, scale_z, save=True):
    # Transform the rcbox to the local space
    to_local_matrix = tranformmatrix_to_local(rcbox)
    localgeom = rcbox['geometry'].transform(to_local_matrix)

    # Create a scaling matrix
    S = np.array([
        [scale_x, 0, 0, 0],
        [0, scale_y, 0, 0],
        [0, 0, scale_z, 0],
        [0, 0, 0, 1]
    ])
    print('S: ', S)

    # Apply the scaling in local space
    locally_scaled = localgeom.transform(S)

    # Transform the locally scaled geometry of the rcbox back to the global space
    to_global_matrix = transformmatrix_to_global(rcbox)

    # Apply the global transformation to the locally scaled geometry
    rcbox['geometry'] = locally_scaled.transform(to_global_matrix)

    # Update the scaling matrix in the rcbox's transform_to_local
    if save:
        rcbox['transform_to_local']['scale_matrix'] = rcbox['transform_to_local']['scale_matrix'] @ S

    return rcbox

def read_rcbox(rcbox_path):
    if rcbox_path.is_file():
        # Parse the XML file
        tree = ET.parse(rcbox_path)
        root = tree.getroot()
        print(root.text)
        # Extract the necessary information from the XML
        depth_RC, width_RC, height_RC = [float(val) for val in root.find('widthHeightDepth').text.split()]
        x, y, z = [float(val) for val in root.find('CentreEuclid/centre').text.split()]
        yaw, pitch, roll = [float(val) for val in root.find('yawPitchRoll').text.split()]

        global_coord_system = root.get('globalCoordinateSystem')
        global_coord_system_name = root.get('globalCoordinateSystemName')

        # Create a cube (box)
        #WARNING open3D width,height,depth are switched compared to reality capture
        #width in open3d is x-directional length is depth in RC
        #height in open3d is y-directional length is width in RC
        #depth in open3d is  z-directional length is height in RC
        cube = o3d.geometry.TriangleMesh.create_box(width=depth_RC, height=width_RC, depth=height_RC)
        cube = cube.translate([0,0,0], relative=False)
        rcbox = {
            'name': rcbox_path.stem,
            'geometry': cube,
            'transform_to_local': {'translation_matrix': np.array([0, 0, 0]),
                                   'rotation_matrix': np.identity(3),
                                   'scale_matrix': np.array([
                                        [1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]),
            'globalCoordinateSystem': '',
            'globalCoordinateSystemName': ''
        }}

        # apply the global transformations from the rcbox file
        rcbox = persistent_translate(rcbox, x, y, z)
        rcbox = localspace_rotation(rcbox, yaw, pitch, roll)
        rcbox['globalCoordinateSystem'] = global_coord_system
        rcbox['globalCoordinateSystemName'] = global_coord_system_name
        print(rcbox)
    else:
        raise FileNotFoundError(f"File not found: {rcbox_path}")
    return rcbox



def rotation_matrix_to_euler_angles(R):
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = math.atan2(R[2, 1], R[2, 2])

    return [yaw, pitch, roll]

def write_rcbox(rcbox):
    # Extract the dimensions from the geometry
    bounds = rcbox['geometry'].get_axis_aligned_bounding_box().get_extent()
    depth, width, height = bounds

    # Extract the rotation matrix and convert it to Euler angles
    rotation_matrix = rcbox['transform_to_local']['rotation_matrix']
    yaw_rad, pitch_rad, roll_rad = rotation_matrix_to_euler_angles(rotation_matrix)

    # Convert the Euler angles to degrees
    yaw, pitch, roll = [math.degrees(angle) for angle in [yaw_rad, pitch_rad, roll_rad]]

    # Create the XML root element with its attributes
    root = ET.Element('ReconstructionRegion', {
        'globalCoordinateSystem': rcbox['globalCoordinateSystem'],
        'globalCoordinateSystemWkt': "",
        'globalCoordinateSystemName': rcbox['globalCoordinateSystemName'],
        'isGeoreferenced': "1",
        'isLatLon': "0"
    })

    # Add the yawPitchRoll element
    ET.SubElement(root, "yawPitchRoll").text = f"{yaw} {pitch} {roll}"

    # Add the widthHeightDepth element
    ET.SubElement(root, "widthHeightDepth").text = f"{depth} {width} {height}"

    # Add the Header element (keeping it same as the provided XML)
    ET.SubElement(root, "Header", {'magic': "5395016", 'version': "2"})

    # Add the CentreEuclid element
    centre_euclid = ET.SubElement(root, "CentreEuclid")
    x, y, z = rcbox['geometry'].get_center()
    ET.SubElement(centre_euclid, "centre").text = f"{x} {y} {z}"

    # Add the Residual element (keeping it same as the provided XML)
    ET.SubElement(root, "Residual", {
        'R': "1 0 0 0 1 0 0 0 1",
        't': "0 0 0",
        's': "1",
        'ownerId': "{2B36705F-74C9-4270-BED3-074F279D427B}"
    })

    # Serialize the XML tree to a file
    tree = ET.ElementTree(root)
    return tree

def generate_rcortho(rcbox, width, height, colorType, modelName, boxSideCornerIndices):
    # Generate the root XML element for the rcortho file
    attributes = {
        'width': str(width),
        'height': str(height),
        'name': "Ortho projection 2",
        'modelName': modelName,
        'modelGuid': "{4C1E40FE-7A70-4837-A884-187882800364}",
        'colorType': colorType,
        'boxSideConerIndex': str(boxSideCornerIndices),
        'bEmpty': "0",
        'backFaceColorType': "1",
        'backFaceColor': "2139127936",
        'projectionType': "0",
        'bShowOrthoProjection': "1"
    }

    # Create an empty root
    root = ET.Element(None)

    # Create OrthoProjection element and add to root
    ortho_projection = ET.SubElement(root, 'OrthoProjection', attributes)
    ET.SubElement(ortho_projection, "Header", {'magic': "5787472", 'version': "2"})

    # Generate the rcbox XML tree and add its root to our main root
    rcbox_tree = write_rcbox(rcbox)
    root.append(rcbox_tree.getroot())

    return root



def createGCPfile_forsedimentcores(scan) -> None:
    # Load the GCPs from the CSV file
    gcps_df = pd.read_csv(config['basegcps'])

    # Load the GeoPackage file
    gcp_startpoints_gdf = gpd.read_file(config["GCPstartpointfile"])

    # Extract the first part of scan['id'] (e.g., URUK27)
    first_part = scan['id'].split('_')[0].replace("URUK", "URUK ").replace("Uruk", "URUK ")  # add space

    # Find the corresponding point in the GeoPackage file
    start_point = gcp_startpoints_gdf[gcp_startpoints_gdf['Name'] == first_part]
    if len(start_point) == 0:
        raise ValueError(f"No point found in GeoPackage file with name '{first_part}'")

    # Extract the XYZ coordinates of the start point
    start_point_coords = start_point.iloc[0]['geometry'].coords[0]

    # Extract the second part of scan['id'] (e.g., 0to1m) and determine the Z translation
    second_part = scan['id'].split('_')[1]
    z_translation = -int(second_part.split('to')[0])

    # Apply the affine transformation to the GCPs
    gcps_df['X'] += start_point_coords[0]
    gcps_df['Y'] += start_point_coords[1]
    gcps_df['Z'] += start_point_coords[2] + z_translation
    # read the start_rcbox
    rcbox = read_rcbox(Path(config['start_rcbox']))
    # Apply the affine transformation to the rcbox
    rcbox = persistent_translate(rcbox, start_point_coords[0], start_point_coords[1], start_point_coords[2] + z_translation)
    #print(rcbox['geometry'].get_center())

    # Write the modified GCPs to a new CSV file
    output_path = Path(scan['scan_dir']) / "modified_rcgcps.csv"
    gcps_df.to_csv(output_path, index=False)
    scan['gcpsfile'] = [file for file in scan['scan_dir'].rglob("*rcgcps.csv")]


    # Write the modified XML content to a new rcbox-file
    scan['rcbox'] = rcbox
    output_path_rcbox = Path(scan['scan_dir']) / "base.rcbox"
    print(output_path_rcbox)
    tree = write_rcbox(rcbox)
    tree.write(output_path_rcbox)
    return scan

def loadCameraPoses(scan):
    print(scan['camregistrationpath'])
    camera_poses_df = pd.read_csv(scan['camregistrationpath'][0], skiprows=1)

    # Extract the filename from the 'name' column
    camera_poses_df['filename'] = camera_poses_df['#name'].apply(lambda x: Path(x).name)

    # Before merging, check for duplicate columns (excluding 'filename') and drop them from scan['imagedf']
    duplicate_columns = camera_poses_df.columns.intersection(scan['imagedf'].columns)
    duplicate_columns = duplicate_columns.drop('filename') if 'filename' in duplicate_columns else duplicate_columns

    if not duplicate_columns.empty:
        scan['imagedf'] = scan['imagedf'].drop(columns=duplicate_columns)

    merged_df = pd.merge(scan['imagedf'], camera_poses_df, on='filename', how='inner')
    scan['imagedf'] = merged_df
    return scan

def generate_point_cloud_with_cameras(scan_df):
    # Extract xyz coordinates from the dataframe
    points = scan_df[['x', 'y', 'z']].values

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)


    # Create a list of camera names corresponding to each point
    #camera_names = scan_df['camera_name'].tolist()  # Replace 'camera_name' with the appropriate column name

    return pcd

def get_frontal_cameras(scan):
    # Step 1: Scale the rcbox by 30 along the local y-axis
    modified_rcbox = localspace_scale(scan['rcbox'], 1, 30, 1,save=False)
    print('Thisbox',modified_rcbox)

    # Step 2: Create an oriented bounding box around the modified rcbox
    obb = modified_rcbox['geometry'].get_oriented_bounding_box()

    # Step 3: Generate a point cloud for cameras
    pcd_cameras = generate_point_cloud_with_cameras(scan['imagedf'])

    # Step 4: Get the indices of cameras inside the bounding box
    indices = obb.get_point_indices_within_bounding_box(pcd_cameras.points)
    print(indices)

    # Step 5: Filter out the points (cameras) in scan['imagedf'] that are not inside the bounding box
    scan['imagedf'] = scan['imagedf'].iloc[indices]
    print(f"Number of cameras inside the bounding box: {len(scan['imagedf'])}")

    return scan



def provide_imageinfo_scanner(series):
    #print('provide_imageinfo_scanner: ', series['rawimg_path'].stem)
    cam, cam2 , objectidfile, roundnumber, imgnumber = series['rawimg_path'].stem.split('_')
    series['cam_id'] = cam + '_'+ cam2 if cam + '_'+ cam2 in config['expected_cam_ids'] else None
    if series['cam_id'] is None:
        print('cam_id not in expected_cam_ids: ', series['rawimg_path'].stem)
    #image_dict['objectid'] = scan_id.split('-')[2]
    series['roundnumber'] = roundnumber
    imgnumber = imgnumber.replace('.jpg','')
    series['imgnumber'] = int(imgnumber.replace('test',''))
    return series


def importRegisteredParameters(scan, searchfilename='*camparam.csv'):

    #find the registered parameters csv-file use glob
    camparam_path = [file for file in scan['scan_dir'].rglob(searchfilename)]
    print('importRegisteredParameters: ', scan['id'], camparam_path)
    if len(camparam_path) == 0:
        print('No camparam.csv file found in ', scan['scan_dir'])

    if len(camparam_path) >= 1:

        #create a new column 'name' in the scan['imagedf'], which is a dataframe, that contains the image stem not the Path-object contained in the column 'rawimg_path'
        scan['imagedf']['#name'] = scan['imagedf']['rawimg_path'].apply(lambda x: x.name)
        #read the csv-file as a dataframe and join it with the scan['image_df'] based on the image name. The csv-file has column's names written in the first row
        imported = pd.read_csv(camparam_path[0], sep=',', header=0)
        #print(imported['#name'])
        #print(scan['imagedf']['#name'])
        merged = pd.merge(scan['imagedf'],imported, on='#name', how='left')
        scan['imagedf'] = merged
        return scan



def plotParametersStats(allscan):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Merging the scan['imagedf'] from each scan in allscan
    merged_df = pd.concat([scan['imagedf'] for index, scan in allscan.iterrows()])
    print(len(merged_df[merged_df['f'].notnull()]), len(merged_df))

    resultparams = pd.DataFrame()

    # Pre-fill resultparams with expected cameras
    expected_cams_df = pd.DataFrame(config['expected_cam_ids'], columns=['cam_id'])
    resultparams = pd.concat([resultparams, expected_cams_df], ignore_index=True)

    # From the merged df, plot the histogram of the parameters for each camera
    unique_cam_ids = merged_df['cam_id'].unique()
    num_cameras = len(unique_cam_ids)

    for parameter in ['f', 'px', 'py']:

        # Define the number of rows and columns for the subplots
        ncols = 3
        nrows = (num_cameras + ncols - 1) // ncols

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows))
        axes = axes.ravel()

        for index, cam_id in enumerate(unique_cam_ids):
            cam_df = merged_df[merged_df['cam_id'] == cam_id]

            Q1 = cam_df[parameter].quantile(0.25)
            Q3 = cam_df[parameter].quantile(0.75)
            IQR = Q3 - Q1

            # Filter out outliers using the IQR method
            filtered_df = cam_df[(cam_df[parameter] >= (Q1 - 1.5 * IQR)) & (cam_df[parameter] <= (Q3 + 1.5 * IQR))]

            mean_val = filtered_df[parameter].mean()
            std_val = filtered_df[parameter].std()


            resultparams.loc[resultparams['cam_id'] == cam_id, parameter + '_mean'] = mean_val
            resultparams.loc[resultparams['cam_id'] == cam_id, parameter + '_std'] = std_val

            # Plotting histograms
            axes[index].hist(filtered_df[parameter], bins=40)
            axes[index].axvline(mean_val, color='r', linestyle='-', linewidth=1, label=f'Mean: {mean_val:.2f}')
            axes[index].axvline(mean_val + std_val, color='g', linestyle='--', linewidth=1, label=f'Std: {std_val:.2f}')
            axes[index].axvline(mean_val - std_val, color='g', linestyle='--', linewidth=1)

            axes[index].set_xlabel(parameter)
            axes[index].set_ylabel('Amount of camera poses')
            axes[index].set_title(f'{parameter} distribution for {cam_id} (ex. outliers)')

            # Set the x-axis limits based on the specified minimum and maximum values
            if parameter == 'f':
                x_min = 25
                x_max = 35
            if parameter == 'px' or parameter == 'py':
                x_min = -0.1
                x_max = 0.1
            axes[index].set_xlim(x_min, x_max)
            axes[index].legend()

        # Remove any unused subplots
        for index in range(num_cameras, nrows * ncols):
            fig.delaxes(axes[index])

        plt.tight_layout()
        plt.show()


    resultparams.reset_index(drop=True, inplace=True)
    return resultparams







def extract_timestamps(series):
    timestamp_pattern = r"(finish|start)_timestamp:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{6})"
    timestamps = {}
    if 'scannerlogfile' in series.keys() and series['scannerlogfile'] and Path(series['scannerlogfile'][0]).is_file():
        #print('logfile: ', series['scannerlogfile'][0])

        with open(series['scannerlogfile'][0], 'r') as file:
            content = file.read()
            matches = re.findall(timestamp_pattern, content)

            for match in matches:
                key, value = match
                #rint(key, value)
                timestamps[key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
        series['timestamps'] = timestamps
        series['duration'] = timestamps['finish'] - timestamps['start']
    return series

def extract_creation_datetime(series):
    creation_time_key = 'DateTimeOriginal'

    # Open the image
    with Image.open(series['rawimg_path']) as img:
        # Get the EXIF data
        exif_data = img._getexif()

        if exif_data:
            # Convert the EXIF tag ID to tag names
            exif = {TAGS[k]: v for k, v in exif_data.items() if k in TAGS}

            if creation_time_key in exif:
                # Extract the creation datetime as a string
                creation_time_str = exif[creation_time_key]

                # Convert the string to a datetime object
                creation_time = datetime.strptime(creation_time_str, '%Y:%m:%d %H:%M:%S')
                series['creation_datetime'] = creation_time
        else:
            print("No EXIF data found.")
    return series

def calculate_duration(series):
    min_datetime = series.min()
    max_datetime = series.max()
    duration = max_datetime - min_datetime
    return duration



def plot_duration_histogram(df, num_bins=120):
    # Convert the durations to a common unit (e.g., minutes) for better visualization
    durations_in_minutes = df['duration'].dt.total_seconds()

    # Specify the figure size (width, height) in inches
    plt.figure(figsize=(12, 6))

    # Create a histogram with suitable labels and increased number of bins
    plt.hist(durations_in_minutes, bins=num_bins)
    plt.xlabel('Scanning duration per object (seconds)')
    plt.ylabel('Amount of objects')
    plt.title('Histogram of Scanning Durations')

    # Set the x-axis limits based on the specified minimum and maximum values
    x_min = 60
    x_max = 300
    plt.xlim(x_min, x_max)

    plt.savefig(Path(config['workspace']) / 'scanduration_histogram.png')





### This is more complicated then I thought.
def add_shotnumber(df: pd.DataFrame) -> pd.DataFrame:
    #print(df.columns)

    df['shotnumber'] = np.nan
    newdf = pd.DataFrame(columns=(df.columns))
    for name, group in df.groupby('roundnumber'):
        maxi = int(len(group) / 12)
        group = group.sort_values(by='imgnumber').reset_index(drop=True)
        for i in range(1, maxi):
            #print('i: ', i)
            shotdf = pd.DataFrame(columns=(df.columns))
            # exclude from group the rows which are already in newdf
            groupmod = group[~group['rawimg_path'].isin(newdf['rawimg_path'].to_list())]
            # sort shotdf['cam_id'].to_list() and config['expected_cam_ids'] and compare
            t = 0
            for index, row in groupmod.iterrows():
                if not sorted(shotdf['cam_id'].to_list()) == sorted(config['expected_cam_ids']) :

                    if row['cam_id'] not in shotdf['cam_id'].to_list() :
                        if int(row['imgnumber']) > shotdf['imgnumber'].max() + 6:
                            #print('skip due to high imgnumber')
                            break

                        #print('Cam_id: ', row['cam_id'], ' is added')
                        #print(shotdf['cam_id'].to_list() )
                        row['shotnumber'] = i
                        shotdf = pd.concat([shotdf, row.to_frame().transpose()])
                        #print(row['rawimg_path'].stem, ' is added')
                    else :
                        #print( 'skip cam_id: ', row['cam_id'])
                        t = t + 1

                        if t>5:
                            #print('t>5')
                            #shotdf = pd.concat([shotdf, row.to_frame().transpose()])
                            break
                else:
                    break
            #if sorted(shotdf['cam_id'].to_list()) == sorted(config['expected_cam_ids']) :
            #print('shotdf complete ', len(shotdf),'. Range of imgnumbers: ', shotdf['imgnumber'].min(), ' - ', shotdf['imgnumber'].max())
            newdf = pd.concat([newdf, shotdf])
            i += 1



    return newdf







    # Return the modified DataFrame with the new 'shotnumber' column
    return grouped


def baseimageIsDevimage(series):
    series['dev-img_path']=series['rawimg_path']
    return series

def defineRawTherapeeOutput(series, foldername=''):
    series['RToutputfolder'] = Path(config['workspace']) / series['id'] / foldername
    series['RToutputfolder'].mkdir(exist_ok=True)
    return series

def defineRealityCaptureOutput(series, foldername=''):
    series['RCoutputfolder'] = Path(config['workspace']) / series['id'] / foldername
    series['RCoutputfolder'].mkdir(exist_ok=True)
    if config['overwrite_all_rcproj'] :
        shutil.rmtree(series['RCoutputfolder'])
        series['RCoutputfolder'].mkdir(exist_ok=True)
    return series

def defineResultOutput(series, foldername=''):
    series['Resultoutputfolder'] = Path(config['workspace']) / series['id'] / foldername
    series['Resultoutputfolder'].mkdir(exist_ok=True)
    return series

def developwithRawTherapee(imageseries, pp3filepath , outputfolderpath, inputrawimagepathfield = 'rawimg_path', outputdevimagepathfield='dev-img_path'):
    outfile = outputfolderpath / Path(str(imageseries[inputrawimagepathfield].stem) + config['devimage_format'])
    if not outfile.is_file() or outfile.is_file() and config['overwrite_dev-img'] :
        if imageseries['rawimg_path'].is_file():
            print('Inputfile: ', imageseries['rawimg_path'])
            print('Expect file: ', outfile)
        subprocess.check_output( '"' + str(config['RTpath']) + '"' \
             + ' -o ' + '"' + str(outfile.as_posix()) + '"'   + config['devimage_param']+ ' -q ' +  ' -Y ' \
            + '-p ' + '"' + str(Path(pp3filepath).as_posix()) + '"' + \
             ' -c '  +  '"' + str(imageseries['rawimg_path']) +  '"' \
             )
    if outfile.is_file():
        imageseries[outputdevimagepathfield]= outfile
        #print(imageseries[outputdevimagepathfield])
    else: print(outfile, ' was not developed.')
    return imageseries
def makeImagelist(scan, imagelistname, imagefield='dev-img_path'):
    #takes a dataframe as input and expects imagepaths in the specified field
    # The name of the resulting imagelist-file must be specified and will be stored in the df
    imagelistname = imagelistname + '.imagelist'
    imagelistpath = scan['RCoutputfolder']/ imagelistname
    #print(scan[imagefield])
    #imagelistpath.parent.mkdir(parents=True, exist_ok=True)
    if not imagelistpath.is_file() or imagelistpath.is_file() and config['overwrite_imagelist'] :
        if imagelistpath.is_file():
            imagelistpath.unlink()
        imagelistpath.touch(exist_ok=True)
        with imagelistpath.open('a') as imagelistfile:
            for index, image in scan['imagedf'].iterrows():
            # write each item on a new line
                imagelistfile.write("%s\n" % image[imagefield])
    scan['list_' + imagefield]= imagelistpath
    return scan

def createRCproject(scan):
    rcproj = scan['id']+'_project.rcproj'
    rcproj_path = scan['RCoutputfolder'] / rcproj
    if not rcproj_path.is_file() or rcproj_path.is_file() and config['overwrite_all_rcproj'] :
        subprocess.check_output( '"' + str(Path(config['RCpath']).as_posix()) + '"' \
        + ' -headless' + ' -newScene' \
        + ' -save ' + '"' + str(rcproj_path.as_posix()) + '"' \
        + ' -quit')
    if rcproj_path.is_file():
        scan['rcproj_path']=rcproj_path
    return scan

def covertRCsettingsDFToRCCMD(series, outputfile):
    with outputfile.open('a') as rccmdsettings:
        rccmdsettings.write('-set "' + str(series['Key']) + '=' + str(series['Default value']) + '"' + "\n")
    return series


def missingInMaster(all, master):
    merged = pd.merge(all, master, on=['imgpath', 'x', 'y'], how='left', indicator='exists')
    merged['exists'] = np.where(merged.exists == 'both', True, False)
    return merged

def writeProcessingstateFile(scan):
    ProcessingstateFile = 'Processingstate' + '_' + scan['id'] +'.json'
    ProcessingstateFilepath = scan['RCoutputfolder'] / ProcessingstateFile
    if ProcessingstateFilepath.is_file():
        ProcessingstateFilepath.unlink()
    scan['processingstate'].to_json(ProcessingstateFilepath , orient="records")

def readProcessingstateFile(scan):
    ProcessingstateFile = 'Processingstate' + '_' + scan['id'] +'.json'
    ProcessingstateFilepath = scan['RCoutputfolder'] / ProcessingstateFile
    if ProcessingstateFilepath.is_file():
        scan['processingstate'] = pd.read_json(ProcessingstateFilepath, orient='records')
    return scan

def checkProcessingstate(scan, command):
    if len(scan['processingstate'][scan['processingstate']['command']== command ]) == 1:
        return True
    else:
        return False

def resumeProcessing_collective(scandf):
    previousscandf_path = Path(config['workspace']) / 'rcprocessingdf.pkl'
    if previousscandf_path.is_file() and config['resume_processing']:
        previousscandf = pd.read_pickle(str(Path(config['workspace']) / 'rcprocessingdf.pkl'))
        previousscandf = previousscandf.set_index('id',drop=False)
        scandf = scandf.set_index('id',drop=False)
        print(previousscandf.columns)
        scandf  = pd.concat([scandf , previousscandf[previousscandf.columns.difference(scandf.columns)]], join='outer', axis=1)
        scandf.update(previousscandf)
    return scandf

def makeRCCMDfromListfield(scan, commandlistfield, rccmdpathfield='rccmdpath'):
    rccmdname = commandlistfield + '_' + scan['id'] +'.rccmd'
    rccmdpath = scan['RCoutputfolder'] / rccmdname
    if rccmdpath.is_file():
        rccmdpath.unlink()
    rccmdpath.touch(exist_ok=True)
    with rccmdpath.open('a') as rccmds:
        for rccmd in scan[commandlistfield]:
        # write each item on a new line
            rccmds.write("%s\n" % rccmd)
    if rccmdpath.is_file():
        print(rccmdpath)
        scan[rccmdpathfield]=rccmdpath
    return scan
def executeRCCMDuseRCproject(scan, rccmdpathfield='rccmdpath', instanceName = 'default', headless=True):
    if headless:
        try:
            subprocess.check_output('"' + str(Path(config['RCpath']).as_posix()) + '"' \
            + ' -headless' + ' -setInstanceName ' + instanceName + ' -load ' \
            + str(scan['rcproj_path']) + ' -execRCCMD ' + '"' + str(scan[rccmdpathfield]) + '"' )
            scan['processingstate']=pd.concat([scan['processingstate'], pd.DataFrame({'command': rccmdpathfield, 'datetime':datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'success': True}, index=['datetime'])])
            return scan
        except subprocess.CalledProcessError as e:
            print(e.output)
            scan['processingstate']=pd.concat([scan['processingstate'], pd.DataFrame({'command': rccmdpathfield, 'datetime':datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'success': False}, index=['datetime'])])

            return scan
    if not headless:
        try:
            subprocess.check_output('"' + str(Path(config['RCpath']).as_posix()) + '"' \
            + ' -setInstanceName ' + instanceName + ' -load ' \
            + str(scan['rcproj_path']) + ' -execRCCMD ' + '"' + str(scan[rccmdpathfield]) + '"' )
            scan['processingstate']=pd.concat([scan['processingstate'], pd.DataFrame({'command': rccmdpathfield, 'datetime':datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'success': True}, index=['datetime'])])
            return scan
        except subprocess.CalledProcessError as e:
            print(e.output)
            scan['processingstate']=pd.concat([scan['processingstate'], pd.DataFrame({'command': rccmdpathfield, 'datetime':datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'success': False}, index=['datetime'])])
            return scan
def rccmdExportControlPoints(commandlist, cpmFileName):
    command = '-exportControlPointsMeasurements ' + cpmFileName
    commandlist.append(command)
    return commandlist


def writeImagelist(series, sourceimagefolder, outputfolder):
    with open(str(rccmdpath), "w") as outfile:
        outfile.write("\n".join(commandlist))
    series['rcimagelistoutpath'] = os.path.join(outputfolder, 'tile3D_' + str(series['GridId']) +'.imagelist')
    with open(series['rcimagelistoutpath'], 'w') as f:
        for camera in series['bestfp_cameralist']:
            session,FP,name = str(camera).split('_')
            name, fileext = name.split('.')
            geometryimage = Path(sourceimagefolder + FP + '/' + camera)
            texture014 = Path(sourceimagefolder + FP + '/' + session + '_' + FP + '_' + name + '.texture014.png')

            f.write("%s\n" % geometryimage)
            f.write("%s\n" % texture014)
    return series


def load_xml(name):
    tree = ET.parse(name)
    root = tree.getroot()
    return tree, root

def getLengthWidth(box):
    xmin,ymin,xmax,ymax = box.bounds
    xdist = xmax - xmin
    ydist = ymax - ymin
    return xdist, ydist

def read_rcorthobox(series, rcorthofield='orthoboxfile', outfield='orthobox'):
    boxlist = []
    #print(series[rcorthofield])
    for item in series[rcorthofield]:
        if item.is_file:
            #print(item)
            with open(item, 'r') as f:
                contents = f.read()
            xmls = contents.split('</OrthoProjection>')
            ortho_xml = xmls[0] + '</OrthoProjection>'
            recon_xml = xmls[1].lstrip('<')
            #print(recon_xml)
            # Read the reconstruction region box coordinates from the second xml
            reconstruction_region = ET.fromstring(recon_xml)
            try:
                x,y,z = tuple(map(float,reconstruction_region.find('CentreEuclid').attrib['centre'].split()))
            except:
                x,y,z = tuple(map(float, reconstruction_region.find('CentreEuclid').find('centre').text.split()))
            #print(center_elem )
            #center_point = tuple(map(float, center_elem.split()))
            try:
                width, height, depth = tuple(map(float, reconstruction_region.attrib['widthHeightDepth'].split()))
            except:
                width, height, depth = tuple(map(float, reconstruction_region.find('widthHeightDepth').text.split()))
            #print(width, height, depth )
            # Create the 3D box geometry as a Shapely Polygon
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2
            coordinates = [
                (x - half_width, y - half_height),
                (x - half_width, y + half_height),
                (x + half_width, y + half_height),
                (x + half_width, y - half_height)
            ]
            box1 = Polygon(coordinates)

            # Rotate the box to match the yawPitchRoll rotation in the XML file
            try:
                yaw, pitch, roll = tuple(map(float, reconstruction_region.attrib['yawPitchRoll'].split()))
            except:
                yaw, pitch, roll = tuple(map(float, reconstruction_region.find('yawPitchRoll').text.split()))
            print(yaw,pitch,roll)
            box_3d = rotate(box1, 180 - roll, origin=(x,y))  # Rotate around the z-axis
            print(box_3d)
            box1 = {}
            box1['geometry'] = box_3d
            box1['name'] = item.stem
            box1['orthoprojection'] = ortho_xml
            boxlist.append(box1.copy())
    # Create a GeoDataFrame with the box geometry
    series[outfield] = gpd.GeoDataFrame(boxlist, geometry='geometry')

    # Add CRS information if available
    if len(boxlist) > 0 and 'globalCoordinateSystem' in reconstruction_region.attrib:
        crs = reconstruction_region.attrib['globalCoordinateSystem']
        series[outfield].crs = crs

    return series



def write_rcbox_wide(series, length, width, depth, reprojbuffer, tree, root):
    series['rcboxwideoutpath'] = os.path.join(outputfolder, 'tile3D_' + str(series['GridId']) + '_wide.rcbox')
    centro_x,centro_y = series['geometry'].centroid.xy
    for centre in root.iter('centre'):
        centre.text = str(centro_x[0]) + ' ' + str(centro_y[0]) + ' ' + str(15)
    for widthHeightDepth in root.iter('widthHeightDepth'):
        widthHeightDepth.text = str(width + 2*reprojbuffer) + ' ' + str(length + 2*reprojbuffer) + ' ' + str(depth + 2*reprojbuffer)
    tree.write(series['rcboxwideoutpath'])
    return series

def write_rcbox_tight(series,  depth, overlap, tree, root):
    series['rcboxtightoutpath'] = os.path.join(outputfolder, 'tile3D_' + str(series['GridId']) + '_tight.rcbox')
    centro_x,centro_y = series['geometry'].centroid.xy
    for centre in root.iter('centre'):
        centre.text = str(centro_x[0]) + ' ' + str(centro_y[0]) + ' ' + str(15)
    for widthHeightDepth in root.iter('widthHeightDepth'):
        widthHeightDepth.text = str(width + 2*overlap) + ' ' + str(length + 2*overlap) + ' ' + str(depth + 2*overlap)
    tree.write(series['rcboxtightoutpath'])
    return series

def write_rcbox_makrotight(series, depth, overlap, tree, root):
    series['rcboxtightoutpath'] = os.path.join(outputfolder, 'makrotile3D_' + str(series['Name']) + '_tight.rcbox')
    xdist, ydist = getLengthWidth(series['geometry'])
    centro_x,centro_y = series['geometry'].centroid.xy
    for centre in root.iter('centre'):
        centre.text = str(centro_x[0]) + ' ' + str(centro_y[0]) + ' ' + str(15)
    for widthHeightDepth in root.iter('widthHeightDepth'):
        widthHeightDepth.text = str(xdist + 2*overlap) + ' ' + str(ydist + 2*overlap) + ' ' + str(depth + 2*overlap)
    tree.write(series['rcboxtightoutpath'])
    return series

def write_rcbox_makrowide(series, length, width, depth, reprojbuffer, tree, root):
    series['rcboxwideoutpath'] = os.path.join(outputfolder, 'makrotile3D_' + str(series['GridId']) + '_wide.rcbox')
    centro_x,centro_y = series['geometry'].centroid.xy
    for centre in root.iter('centre'):
        centre.text = str(centro_x[0]) + ' ' + str(centro_y[0]) + ' ' + str(15)
    for widthHeightDepth in root.iter('widthHeightDepth'):
        widthHeightDepth.text = str(width + 2*reprojbuffer) + ' ' + str(length + 2*reprojbuffer) + ' ' + str(depth + 2*reprojbuffer)
    tree.write(series['rcboxwideoutpath'])
    return series

def write_rcorthobox(series, height, width, resolution, overlap, depth,tree, root):
    series['rcorthoboxoutpath'] = os.path.join(outputfolder, 'makrotile3D_' + str(series['GridId']) + '_overlap.rcortho')
    centro_x,centro_y = series['geometry'].centroid.xy
    sh = tree.find('OrthoProjection')
    sh.set('width', str(int((width + overlap)//resolution)))
    sh.set('height', str(int((height + overlap)//resolution)))
    sh.set('modelName','makrotile3D_' + str(series['GridId']) + '_highpoly')
    sh.set('name','makrotile2D_' + str(series['GridId']))
    for centre in root.iter('centre'):
        centre.text = str(centro_x[0]) + ' ' + str(centro_y[0]) + ' ' + str(15)
    for widthHeightDepth in root.iter('widthHeightDepth'):
        print( widthHeightDepth)
        print(str(width + overlap ) + ' ' + str(length + overlap ) + ' ' + str(depth ))
        widthHeightDepth.text = str(width + overlap ) + ' ' + str(length + overlap ) + ' ' + str(depth )
    tree.write(series['rcorthoboxoutpath'])
    with open(series['rcorthoboxoutpath']) as input_file:
        text = input_file.read()
    text = text.replace('<Documents>','')
    text = text.replace('</Documents>','')
    with open(series['rcorthoboxoutpath'], 'w') as output_file:
        output_file.write(text)

    return series

def write_rcimagelist(series, sourceimagefolder, outputfolder):
    series['rcimagelistoutpath'] = os.path.join(outputfolder, 'tile3D_' + str(series['GridId']) +'.imagelist')
    with open(series['rcimagelistoutpath'], 'w') as f:
        for camera in series['bestfp_cameralist']:
            session,FP,name = str(camera).split('_')
            name, fileext = name.split('.')
            geometryimage = Path(sourceimagefolder + FP + '/' + camera)
            texture014 = Path(sourceimagefolder + FP + '/' + session + '_' + FP + '_' + name + '.texture014.png')

            f.write("%s\n" % geometryimage)
            f.write("%s\n" % texture014)
    return series

def write_makrorcimagelist(series, sourceimagefolder, outputfolder):
    series['rcimagelistoutpath'] = os.path.join(outputfolder, 'makrotile3D_' + str(series['Name']) +'.imagelist')
    with open(series['rcimagelistoutpath'], 'w') as f:
        for camera in series['bestfp_cameralist']:
            print(str(camera))
            session,FP,name = str(camera).split('_')
            name, fileext = name.split('.')
            geometryimage = Path(sourceimagefolder + FP + '/' + camera)
            texture014 = Path(sourceimagefolder + FP + '/' + session + '_' + FP + '_' + name + '.texture014.png')

            f.write("%s\n" % geometryimage)
            f.write("%s\n" % texture014)
    return series

def write_neighbourlist(series,outputfolder):
    series['neighbourlistoutpath'] = os.path.join(outputfolder, 'tile3D_' + str(series['GridId']) + '_neighbours.txt')
    with open(series['neighbourlistoutpath'], 'w') as f:
        for neighbour in series['neighbours']:
            f.write("%s\n" % neighbour)

    return series

def createTileFrame(geodf):
    geo = geodf.geometry
    #geo_clean = geo[~(geo.is_empty | geo.isna())]
    geo_union = geo.unary_union
    geo_envelope = geo_union.envelope
    return geo_envelope

def createBuffer(grid_df, camerabuffer):
    grid_geom = grid_df.geometry
    grid_df['buffer'] = grid_geom.buffer(camerabuffer)
    return grid_df
def removeEmptyCameralist(df, column, limit):
    cleandictlist = []
    for index, row in df.iterrows():
        if len(row[column]) >= limit:
            cleandict = row
            cleandictlist.append(cleandict)
    return gpd.GeoDataFrame(cleandictlist)


def filterCameralistFP(series):
    camdictlist = []
    for camera in series['cameralist']:
        #print(camera)
        camdict = {}
        session,FP,name = str(camera).split('_')

        camdict['camera'] = camera
        camdict['FP'] = FP
        camdict['session'] = session
        camdict['name'] = name
        camdictlist.append(camdict)
    dataset = pd.DataFrame(camdictlist)

    statsdf = dataset.groupby('FP').session.agg('count').to_frame('count').reset_index()
    winner = statsdf['FP'][statsdf['count']==statsdf['count'].max()]
    #print(winner)
    series['bestfp_cameralist'] = list(dataset['camera'][dataset['FP']==list(winner)[0]])
    print(series['bestfp_cameralist'])
    print ('NEXT GRID')
    return series



def getCameralist(series, geodf):
    cameraswithinlist = geodf.geometry.within(series['buffer'])
    cameraswithin = geodf.loc[cameraswithinlist]
    series['cameralist']=list(cameraswithin['name'])
    return series

def takeoverCameralistFP(series, grid_df):
    takeoverCameralist = []
    squareswithinlist = grid_df.geometry.within(series['geometry'])
    squares = grid_df.loc[squareswithinlist]
    for index,square in squares.iterrows():
        takeoverCameralist = takeoverCameralist + square['bestfp_cameralist']
    series['bestfp_cameralist'] = takeoverCameralist
    return series

def getNeighbourlist(series, grid_df):
    neighbourlist = grid_df.geometry.intersects(series.geometry)
    neighbour = grid_df.loc[neighbourlist]
    series['neighbours'] = list(neighbour['GridId'])
    print(len(series['neighbours']))
    return series

def createGrid(geodf_envelope, length, width):
    xmin,ymin,xmax,ymax = geodf_envelope.total_bounds
    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), width))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), length))
    #rows.reverse()
    grid_df = gpd.GeoDataFrame()
    for ix,x in enumerate(cols):
        for iy,y in enumerate(rows):
            grid={}
            grid['GridId']= str(int(ix)) + '_' + str(int(iy))
            grid['geometry'] = Polygon([(x,y), (x+width, y), (x+width, y+length), (x, y+length)])
            grid_df = grid_df.append(grid, True)
    return grid_df

def geomNesting(series, grid_df):
    squareswithinlist = grid_df.geometry.within(series['geometry'])
    nestedgrids = grid_df.loc[squareswithinlist]
    print('Makro',series['GridId'])
    print(nestedgrids['GridId'])
    series['mikrogrid']=nestedgrids.to_dict('records')
    #print(series['mikrogrid'])
    return series




def  detect_tiepoints(groups):
    #[groups.get_group(x) for x in groups.groups]
    #print(groups['scan_id'])
    scan_id = pd.unique(groups['scan_id'])
    imgnumber = pd.unique(groups['imgnumber'])
    #print(type(scan_id))
    print(scan_id[0])
    filename_imagelist = str(scan_id[0]) + '_' + str(imgnumber[0]) + '.imagelist'
    RCimagelist_path = workingdirectory + filename_imagelist
    filename_tiepoints = str(scan_id[0]) + '_' + str(imgnumber[0]) + '_tiepoints.csv'
    RCtiepoints_path = workingdirectory + filename_tiepoints
    #print(pd.unique(groups['scan_id'][0]))
    groups['img_path'].replace('/','\\').to_csv(RCimagelist_path, header=False, index=False)
    #groups.apply(produce_RCimagelist)
    check_output('"C:\\Program Files\\Capturing Reality\\RealityCapture\\RealityCapture.exe" -silent ' + workingdirectory + ' -set "appQuitOnError=true" -add ' + RCimagelist_path + ' -detectMarkers -exportControlPointsMeasurements ' + RCtiepoints_path + ' -quit' , shell=True)

def read_tiepoints(imagelist):
    tiepoints_all = pd.DataFrame()
    for RCtiepoints in os.listdir(workingdirectory):
        if RCtiepoints.endswith(("_tiepoints.csv")):
            colnames=['img_path', 'tiepointid_raw', 'X', 'Y']
            tiepoints_df = pd.read_csv(workingdirectory + RCtiepoints, names=colnames, header=None)
            idadd = RCtiepoints.replace('_tiepoints.csv','')
            print(tiepoints_df['tiepointid_raw'])
            tiepoints_df['tiepointid'] = idadd + '_' + tiepoints_df['tiepointid_raw']
            tiepoints_all = tiepoints_all.append(tiepoints_df)

    imagetiepointlist_unfolded = pd.merge(imagelist,tiepoints_all, on ='img_path')
    return imagetiepointlist_unfolded

def create_RCtiepoints(groups):
    scan_id = pd.unique(groups['scan_id'])
    RCtiepoints_path = workingdirectory + scan_id[0] + '_alltiepoints.csv'
    groups[['img_path','tiepointid','X','Y']].to_csv(RCtiepoints_path, header=False, index=False)





def createStartCommand(grid_df, RCpath, instanceName, messagepath, RCbaseProject, RCCMD):

    subprocess.run('"'+ RCpath + '"' + ' -setInstanceName ' + instanceName + ' -silent ' + messagepath + ' -set "appQuitOnError=true" -load ' + RCbaseProject + ' -execRCCMD ' + RCCMD + ' -quit')


def createTileCommand( grid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, unwrapparamspath, reprojectparams, exportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in grid_df.iterrows():
        modeloutpath = os.path.join(outputfolder, 'tile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'tile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'tile2D_' + str(row['GridId']) + '_dem.tiff')
        tilecommand = '-selectModel ' + BaseHighpolymodel + ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_highpoly' + ' -unwrap ' + unwrapparamspath + ' -calculateTexture' + ' -selectModel ' + BaseLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_lowpoly' + ' -unwrap ' + unwrapparamspath + ' -reprojectTexture ' + 'tile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'tile3D_' + str(row['GridId']) + '_lowpoly' + ' ' + reprojectparams + ' -selectModel ' +  'tile3D_' + str(row['GridId']) + '_lowpoly'  + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_lowpolyTight' + ' -exportModel ' + 'tile3D_' + str(row['GridId']) + '_lowpolyTight' + ' ' + modeloutpath + ' ' + exportparams + ' -calculateOrthoProjection ' + row['rcorthoboxoutpath'] + ' -selectOrthoProjection ' + 'tile2D_' + str(row['GridId']) + ' -exportOrthoProjection ' + orthodiffuseoutpath + ' ' + exportorthodiffuseparams + ' -exportOrthoProjection ' + orthodemoutpath + ' ' + exportorthodemparams
        commandlist.append(tilecommand)

    return commandlist

def createMakrotileCommand( grid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, unwrapparamspath, unwrapparamspathMakrotile,  reprojectparams, exportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in grid_df.iterrows():
        modeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        #orthodiffuseoutpath = os.path.join(outputfolder, 'tile2D_' + str(row['GridId']) + '_diffuse.tiff')
        #orthodemoutpath = os.path.join(outputfolder, 'tile2D_' + str(row['GridId']) + '_dem.tiff')
        tilecommand = '-selectModel ' + BaseHighpolymodel + ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_highpoly' + ' -unwrap ' + unwrapparamspath + ' -calculateTexture' + ' -selectModel ' + BaseLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_lowpoly' + ' -unwrap ' + unwrapparamspathMakrotile + ' -reprojectTexture ' + 'tile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'tile3D_' + str(row['GridId']) + '_lowpoly' + ' ' + reprojectparams + ' -selectModel ' +  'tile3D_' + str(row['GridId']) + '_lowpoly'  + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(row['GridId']) + '_lowpolyTight' + ' -exportModel ' + 'tile3D_' + str(row['GridId']) + '_lowpolyTight' + ' ' + modeloutpath + ' ' + exportparams
        commandlist.append(tilecommand)

    return commandlist

def createMakroAndDetailCommand( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')
        makrotilecommand = '-selectModel ' + BaseHighpolymodel + \
          ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + \
            ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + \
            ' -unwrap ' + unwrapparamspath +' -calculateTexture' + \
            ' -selectModel ' + BaseLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + \
            ' -unwrap ' + unwrapparamspath + ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + ' ' + makroreprojectparams + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + ' ' + delightmakromodeloutpath + ' ' + highmakroexportparams + \
            ' -calculateOrthoProjection ' + row['rcorthoboxoutpath'] + ' -selectOrthoProjection ' + 'makrotile2D_' + str(row['GridId']) + ' -exportOrthoProjection ' + orthodiffuseoutpath + ' ' + exportorthodiffuseparams + ' -exportOrthoProjection ' + orthodemoutpath + ' ' + exportorthodemparams + \
            ' -selectModel ' + BaseMakroLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_lowpoly' + \
            ' -unwrap ' + makrounwrapparamspath + ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres ' +         'makrotile3D_' + str(row['GridId']) + '_lowpoly' + ' ' + makroreprojectparams + \
            ' -selectModel ' +  'makrotile3D_' + str(row['GridId']) + '_lowpoly'  + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_lowpolyTight' + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_lowpolyTight' + ' ' + makromodeloutpath + ' ' + makroexportparams

        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])
        for mikrorow in mikrogrid:
            print('Mikro', mikrorow['GridId'])
            mikromodeloutpath = os.path.join(outputfolder, 'mikrotile3D_' + str(mikrorow['GridId']) + '.fbx')
            mikrotilecommand = ' -selectModel ' + BaseLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + mikrorow['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + \
                ' -unwrap ' + unwrapparamspath + \
                ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' +  ' ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + ' ' + makroreprojectparams + \
                ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + ' ' + reprojectparams + \
                ' -selectModel ' +  'tile3D_' + str(mikrorow['GridId']) + '_lowpoly'  + ' -deselectModelTriangles  -setReconstructionRegion ' + mikrorow['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpolyTight' + \
                ' -exportModel ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpolyTight' + ' ' + mikromodeloutpath + ' ' + exportparams


            commandlist.append(mikrotilecommand)

    return commandlist


def createPreMakro( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')
        makrotilecommand = '-selectModel ' + BaseHighpolymodel + \
            ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + \
            ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + \
            ' -unwrap ' + unwrapparamspath +' -calculateTexture' + \
            ' -selectModel ' + BaseLowpolymodel + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + \
            ' -unwrap ' + unwrapparamspath + ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + ' ' + makroreprojectparams + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_mikrotileres' + ' ' + delightmakromodeloutpath + ' ' + highmakroexportparams + \
            ' -calculateOrthoProjection ' + row['rcorthoboxoutpath'] + ' -selectOrthoProjection ' + 'makrotile2D_' + str(row['GridId']) + ' -exportOrthoProjection ' + orthodiffuseoutpath + ' ' + exportorthodiffuseparams + ' -exportOrthoProjection ' + orthodemoutpath + ' ' + exportorthodemparams

        commandlist.append(makrotilecommand)

    return commandlist

def ProduceLandscape( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')
        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' -calculateOrthoProjection ' + row['rcorthoboxoutpath'] + ' -selectOrthoProjection ' + 'makrotile2D_' + str(row['GridId']) + ' -exportOrthoProjection ' + orthodiffuseoutpath + ' ' + exportorthodiffuseparams + ' -exportOrthoProjection ' + orthodemoutpath + ' ' + exportorthodemparams

        commandlist.append(makrotilecommand)

    return commandlist

def ProduceMikro3DTiles( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')
        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly'

        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])
        for mikrorow in mikrogrid:
            print('Mikro', mikrorow['GridId'])
            mikromodeloutpath = os.path.join(outputfolder, 'mikrotile3D_' + str(mikrorow['GridId']) + '.fbx')
            mikrotilecommand = ' -selectModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + mikrorow['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + ' -unwrap ' + unwrapparamspath + \
                ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + ' ' + reprojectparams + \
                ' -exportModel ' + 'tile3D_' + str(mikrorow['GridId']) + '_lowpoly' + ' ' + mikromodeloutpath + ' ' + exportparams + ' -deleteSelectedModel'



            commandlist.append(mikrotilecommand)

    return commandlist


def ProduceLODTiles( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + \
            row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight' + \
             ' -unwrap '+ unwrapparamspath + ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + reprojectparams + \
            ' -exportLod ' + makromodeloutpath + ' ' + makrolodexportparams
        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])



    return commandlist

def ProduceMakro( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight' + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + makromodeloutpath + ' ' + exportparams
        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])

def ProduceLODTilesandOriginal( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + \
            row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight' + \
             ' -unwrap '+ unwrapparamspath + ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + reprojectparams + \
            ' -exportLod ' + makromodeloutpath + ' ' + makrolodexportparams + \
                ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + makromodeloutpath + ' ' + exportparams
        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])

def ProduceLODTilesandOriginalandRetexture( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

        makrotilecommand = '-selectModel ' + BaseHighpolymodel + \
            ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + \
            ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxwideoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highestpoly' + \
            ' -unwrap ' + unwrapparamspath +' -calculateTexture' + \
        ' -importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + \
            row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight' + \
             ' -unwrap '+ unwrapparamspath + \
                 ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highestpoly' +  ' ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + reprojectparams + \
                ' -exportLod ' + makromodeloutpath + ' ' + makrolodexportparams + \
                ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + makromodeloutpath + ' ' + exportparams
        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])
    return commandlist

def ProduceLODTilesandOriginalandRetextureExport( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
        commandlist = []

        for index, row in makrogrid_df.iterrows():
            delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
            makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
            orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
            orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

            makrotilecommand = '-selectModel makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + \
                    ' -exportLod ' + makromodeloutpath + ' ' + makrolodexportparams + \
                    ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + makromodeloutpath + ' ' + exportparams
            commandlist.append(makrotilecommand)
            mikrogrid = row['mikrogrid']
            #print(type(mikrogrid))
            #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
            print('Makrogrid',row['GridId'])
        return commandlist

def ProduceMakroWithNormals( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')

        makrotilecommand = '-importModel ' + delightmakromodeloutpath + ' -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + ' -deselectModelTriangles  -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -deselectModelTriangles -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight' + \
            ' -unwrap '+ unwrapparamspath + \
        ' -reprojectTexture ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' +  ' ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + reprojectparams + \
            ' -reprojectTexture ' + 'BaseHighpolymodel' +  ' ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + normalsreprojectparams + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpolyTight ' + makromodeloutpath + ' ' + normalsexportparams
        commandlist.append(makrotilecommand)
        mikrogrid = row['mikrogrid']
        #print(type(mikrogrid))
        #mikrogrid_df = gpd.GeoDataFrame(mikrogrid)
        print('Makrogrid',row['GridId'])
    return commandlist

def createNormaldetailMakro( makrogrid_df, RCpath, instanceName, BaseHighpolymodel, BaseLowpolymodel, BaseMakroLowpolymodel, unwrapparamspath, makrounwrapparamspath,  reprojectparams, makroreprojectparams, exportparams, makroexportparams, outputfolder,  exportorthodiffuseparams,  exportorthodemparams ):
    commandlist = []

    for index, row in makrogrid_df.iterrows():
        delightmakromodeloutpath = os.path.join(outputfolder, 'delightmakrotile3D_' + str(row['GridId']) + '.fbx')
        makromodeloutpath = os.path.join(outputfolder, 'makrotile3D_' + str(row['GridId']) + '.fbx')
        orthodiffuseoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_diffuse.tiff')
        orthodemoutpath = os.path.join(outputfolder, 'makrotile2D_' + str(row['GridId']) + '_dem.tiff')
        makrotilecommand = '-selectModel ' + 'highpolyx9_nonoise' + \
            ' -selectAllImages -enableTexturingAndColoring false -enableInComponent false -deselectAllImages -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableTexturingAndColoring true -enableInComponent true' + \
            ' -deselectModelTriangles' + ' -setReconstructionRegion ' + row['rcboxtightoutpath'] + ' -selectTrianglesInsideReconReg -invertTrianglesSelection -removeSelectedTriangles -renameSelectedModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly' + \
            ' -unwrap ' + unwrapparamspath +' -calculateTexture' + \
            ' -exportModel ' + 'makrotile3D_' + str(row['GridId']) + '_highpoly ' + makromodeloutpath + ' ' + exportparams


        commandlist.append(makrotilecommand)

    return commandlist
