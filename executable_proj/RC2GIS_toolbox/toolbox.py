
import pyproj
pyproj.datadir.set_data_dir('C:/Users/mhaibt/Anaconda3/envs/orthobox_env/Library/share/proj/')
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Polygon
from shapely.affinity import rotate
import xml.etree.ElementTree as ET
#from osgeo import gdal, gdal_array
import gdal
from shapely.geometry import LineString
from tkinter import filedialog, messagebox
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine
import threading
import tkinter as tk
from tkinter import ttk
from rasterio.transform import from_origin
from rasterio.env import Env







def process_files(file_paths, progress, status, should_continue, root):
    epsg_code = ask_for_epsg(root)
    num_files = len(file_paths)
    progress["maximum"] = num_files

    for i, file_path in enumerate(file_paths):
        if not should_continue.get():
            status.set("Aborted")
            break

        # Update the status
        status.set(f"Processing file {i+1} of {num_files}")

        output_path = Path(file_path).with_stem(Path(file_path).stem + '_warp').with_suffix('.tif')

        output_gpkg_path = Path(file_path).with_suffix('.gpkg')

        # Reproject and warp
        with rasterio.open(file_path) as src:
            transform, width, height = calculate_default_transform(src.crs, epsg_code, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': epsg_code,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for j in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, j),
                        destination=rasterio.band(dst, j),  # Changed i to j
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=epsg_code)

        with Env(GDAL_PAM_ENABLED='YES'):
            # Translate to GeoPackage
            gdal.Translate(str(output_gpkg_path), str(output_path), format='GPKG', creationOptions=['TILE_FORMAT=PNG', 'NODATA_VALUE=0'])

            # Open the dataset
            ds = gdal.Open(str(output_gpkg_path), gdal.GA_Update)

            # Create a new transparency band where we set the black and near black pixels as transparent
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                band_array = band.ReadAsArray()
                band_array[band_array < 10] = 0
                band.SetNoDataValue(0)  # Set NoData value to 0
                band.WriteArray(band_array)

            # Generate overviews
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'JPEG')
            ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64, 128])

        # Update the progress bar
        progress["value"] = i + 1

    if should_continue.get():
        status.set("Operation completed successfully!")
        messagebox.showinfo("Info", "Operation completed successfully!")

def ask_for_epsg(root):
    def on_submit():
        epsg_code.set(epsg_entry.get())
        popup.destroy()

    popup = tk.Toplevel(root)
    popup.title("Enter EPSG Code")
    
    tk.Label(popup, text="Enter EPSG Code:").pack(padx=20, pady=5)
    
    epsg_code = tk.StringVar()
    epsg_entry = tk.Entry(popup, textvariable=epsg_code)
    epsg_entry.pack(padx=20, pady=5)
    
    submit_button = tk.Button(popup, text="Submit", command=on_submit)
    submit_button.pack(pady=10)
    
    root.wait_window(popup)
    
    return epsg_code.get()

def profilemappping_to_gpkg(raster_files, output_gpkg, root):
    epsg_code = ask_for_epsg(root)
    data = []
    lines = []
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            bounds = src.bounds
            geometry = box(*bounds)
            data.append({
                'path': str(raster_file),
                'stem': Path(raster_file).stem,
                'geometry': geometry,
            })
            
            # Create LineString for upper line of rectangle
            upper_line = LineString([(bounds.left, bounds.top), (bounds.right, bounds.top)])
            lines.append({
                'path': str(raster_file),
                'stem': Path(raster_file).stem,
                'geometry': upper_line,
            })
            
    # Save rectangles layer
    gdf = gpd.GeoDataFrame(data, crs=epsg_code, geometry='geometry')
    gdf.to_file(output_gpkg, layer='rectangles', driver='GPKG')
    
    # Save upper_lines layer
    gdf_lines = gpd.GeoDataFrame(lines, crs=epsg_code, geometry='geometry')
    gdf_lines.to_file(output_gpkg, layer='upper_lines', driver='GPKG') 

    # Save orthoboxes layer
    gdf_orthobox = read_rcorthobox([{'orthoboxfile': Path(file_path).with_suffix('.rcortho')} for file_path in raster_files])
    gdf_orthobox.to_file(output_gpkg, layer='orthoboxes', driver='GPKG')
     

def process_sideview_files(file_paths, progress, status, should_continue):
    # Global list to store metadata about processed images
    processed_images = []
    num_files = len(file_paths)
    progress["maximum"] = num_files

    for i, file_path in enumerate(file_paths):
        print(f'Processing file {i+1} of {num_files}')
        if not should_continue.get():
            status.set("Aborted")
            break
        # Update the status
        status.set(f"Processing file {i+1} of {num_files}")

        output_path = Path(file_path).with_suffix('.tif')
        output_gpkg_path = Path(file_path).with_suffix('.gpkg')

        # Read the worldfile
        worldfile_path = Path(file_path)
        worldfile = worldfile_path.with_suffix('.tfw')
        with open(worldfile, 'r') as f:
            lines = f.readlines()

        # Get the pixel resolution and coordinates from the worldfile
        x_pixel = float(lines[0].strip())
        y_pixel = float(lines[3].strip())
        x_origin = float(lines[4].strip())
        y_origin = float(lines[5].strip())
        print(f'x_pixel: {x_pixel}, y_pixel: {y_pixel}, x_origin: {x_origin}, y_origin: {y_origin}')

        # Calculate the new origin for the image
        if processed_images:
            # If there are already processed images, add the width of the previous image and a buffer of 10
            x_origin_pseudo += processed_images[-1]['x_origin'] + processed_images[-1]['width'] * x_pixel + 10
            print(f'x_origin: {x_origin_pseudo}')
        else:
            # If this is the first image, set the x_origin to 0
            x_origin_pseudo = 0
            print(f'x_origin: {x_origin_pseudo}')

        # Apply the side view transformation
        output_gpkg_path = Path(file_path).with_suffix('.gpkg')
        #output_warp_path = output_gpkg_path.stem + "_warp" + output_gpkg_path.suffix
        print('Applying sideview transformation:')
        # Open the image ignoring georeference information
        with Env(GDAL_PAM_ENABLED='NO'):
            with rasterio.open(file_path) as src:
                # Get the width and height of the image
                width = src.width
                height = src.height
                #x_pixel, y_pixel = src.transform.a, src.transform.e

                # Generate new transform
                transform = from_origin(x_origin_pseudo, y_origin, x_pixel, -y_pixel)
                print(transform)

                # Prepare meta for writing new file
                meta = src.meta.copy()
                meta.update({'transform': transform})
                output_path = Path(file_path).with_stem(Path(file_path).stem + '_warp').with_suffix('.tif')
                # Write to the output file
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(src.read())
                    #close file
                    dst.close()
                # close file
                src.close()

        with Env(GDAL_PAM_ENABLED='YES'):
            # Translate to GeoPackage
            gdal.Translate(str(output_gpkg_path), str(output_path), format='GPKG', creationOptions=['TILE_FORMAT=PNG', 'NODATA_VALUE=0'])

            # Open the dataset
            ds = gdal.Open(str(output_gpkg_path), gdal.GA_Update)

            # Create a new transparency band where we set the black and near black pixels as transparent
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                band_array = band.ReadAsArray()
                band_array[band_array < 10] = 0
                band.SetNoDataValue(0)  # Set NoData value to 0
                band.WriteArray(band_array)

            # Generate overviews
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'JPEG')
            ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64, 128])

            # Close the dataset
            ds = None
            #ds.close()
        # Add the image to the processed_images list
        processed_images.append({'x_origin': x_origin_pseudo, 'width': width, 'outpath': output_gpkg_path})

        # Update the progress bar
        progress["value"] = i + 1
        progress.update()
    profileframes_gpkg_path = Path(file_paths[0]).parent / 'profilemapping.gpkg'
    profilemappping_to_gpkg([image['outpath'] for image in processed_images], profileframes_gpkg_path)

    

    
    if should_continue.get():
        status.set("Operation completed successfully!")
        messagebox.showinfo("Info", "Operation completed successfully!")



def abort_processing(should_continue, status):
    should_continue.set(False)
    status.set("Aborting...")




def open_file_browser_Orthobox2GIS():
    file_paths = filedialog.askopenfilenames(title="Select orthobox files", filetypes=(("Orthobox files", "*.rcortho"), ("All files", "*.*")))

    # Read and process the orthobox files
    if file_paths:
        diclist = [{'orthoboxfile': file_path} for file_path in file_paths]
        orthobox_gpdf = read_rcorthobox(diclist)

        # Save the output to the same folder as the input files
        output_folder = Path(file_paths[0]).parent
        orthobox_gpdf.to_file(output_folder / f"{output_folder.name}_output.gpkg", driver='GPKG')

        messagebox.showinfo("Info", "Orthobox files converted successfully!")


def read_rcorthobox(diclist):
    boxlist = []
    #print(series[rcorthofield])
    for item in diclist:
        if Path(item['orthoboxfile']).is_file:
            #print(item)s
            with open(Path(item['orthoboxfile']), 'r') as f:
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
            box1['name'] = Path(item['orthoboxfile']).stem
            box1['orthoprojection'] = ortho_xml
            boxlist.append(box1.copy())
    # Create a GeoDataFrame with the box geometry
    orthobox_gpdf = gpd.GeoDataFrame(boxlist, geometry='geometry')

    # Add CRS information if available
    if len(boxlist) > 0 and 'globalCoordinateSystem' in reconstruction_region.attrib:
        crs = reconstruction_region.attrib['globalCoordinateSystem']
        orthobox_gpdf.crs = crs

    return orthobox_gpdf

processed_images = []
def open_file_browser_hugeTiffs2GIS(root):
    file_paths = filedialog.askopenfilenames(title="Select GeoTIFF files", filetypes=(("GeoTIFF files", "*.tif"), ("All files", "*.*")))

    # Perform the operations on the GeoTIFF files
    if file_paths:
        # Create a progress bar
        progress = ttk.Progressbar(root, length=300, mode='determinate')
        progress.pack()

        # Create a status label
        status = tk.StringVar()
        status_label = tk.Label(root, textvariable=status)
        status_label.pack()

        # Create a continue variable and an abort button
        should_continue = tk.BooleanVar(value=True)
        abort_button = tk.Button(root, text="Abort", command=lambda: abort_processing(should_continue, status))
        abort_button.pack()

        # Start a new thread for the processing

        threading.Thread(target=process_files, args=(file_paths, progress, status, should_continue, root)).start()


def open_file_browser_sideviewTiffs2GIS(root):
    file_paths = filedialog.askopenfilenames(title="Select GeoTIFF files", filetypes=(("GeoTIFF files", ["*.tif","*.tiff"]), ("GeoTIFF files", "*.tiff"), ("All files", "*.*")))

    # Perform the operations on the GeoTIFF files
    if file_paths:
        # Create a progress bar
        progress = ttk.Progressbar(root, length=300, mode='determinate')
        progress.pack()

        # Create a status label
        status = tk.StringVar()
        status_label = tk.Label(root, textvariable=status)
        status_label.pack()

        # Create a continue variable and an abort button
        should_continue = tk.BooleanVar(value=True)
        abort_button = tk.Button(root, text="Abort", command=lambda: abort_processing(should_continue, status))
        abort_button.pack()

        # Start a new thread for the processing
        threading.Thread(target=process_sideview_files, args=(file_paths, progress, status, should_continue, root)).start()
