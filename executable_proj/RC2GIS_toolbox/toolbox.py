
# stdlib
from pathlib import Path
import threading
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# scientific/geo
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, LineString
from shapely.affinity import rotate

# rasterio (public APIs only)
import rasterio
from rasterio.env import Env
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

from affine import Affine
from pyproj import CRS
from osgeo import gdal

def ui_set(root, tkvar, value):
    root.after(0, lambda: tkvar.set(value))

def fmt_affine(A):
    # pretty-print a rasterio Affine as a 3x3
    return (f"| {A.a: .8f}, {A.b: .8f}, {A.c: .8f}|\n"
            f"| {A.d: .8f}, {A.e: .8f}, {A.f: .8f}|\n"
            f"|  0.00000000,  0.00000000,  1.00000000|")

def log_transform_diff(src_A, dst_A):
    diffs = []
    if not np.isclose(src_A.a, dst_A.a):    diffs.append(f"a (pixel width): {src_A.a} -> {dst_A.a}")
    if not np.isclose(src_A.e, dst_A.e):    diffs.append(f"e (pixel height): {src_A.e} -> {dst_A.e}")
    if not np.isclose(src_A.b, dst_A.b):    diffs.append(f"b (x-rotation/skew): {src_A.b} -> {dst_A.b}")
    if not np.isclose(src_A.d, dst_A.d):    diffs.append(f"d (y-rotation/skew): {src_A.d} -> {dst_A.d}")
    if not np.isclose(src_A.c, dst_A.c):    diffs.append(f"c (x origin): {src_A.c} -> {dst_A.c}")
    if not np.isclose(src_A.f, dst_A.f):    diffs.append(f"f (y origin): {src_A.f} -> {dst_A.f}")
    return diffs

def process_files(file_paths, progress, status, should_continue, root):
    epsg_code = ask_for_epsg(root)

    try:
        dest_crs = CRS.from_epsg(int(epsg_code))
    except ValueError:
        ui_set(root, status, f"Invalid EPSG code: {epsg_code}")
        return

    num_files = len(file_paths)
    progress["maximum"] = num_files

    for i, file_path in enumerate(file_paths):
        if not should_continue.get():
            ui_set(root, status, "Aborted")
            break

        p = Path(file_path)
        ui_set(root, status, f"Processing {i+1}/{num_files}: {p.name}")
        print(f"\n[{i+1}/{num_files}] {p}")

        output_path = p.with_stem(p.stem + "_warp").with_suffix(".tif")

        with rasterio.open(file_path) as src:
            if src.crs is None:
                ui_set(root, status, f"CRS not found: {p.name}")
                print("✖ CRS missing; skipping.")
                continue

            try:
                source_crs = CRS.from_string(src.crs.to_wkt())
                print(f"Source CRS EPSG: {source_crs.to_epsg()} | Dest CRS EPSG: {dest_crs.to_epsg()}")
            except Exception:
                print("Note: could not derive EPSG from source; will use full CRS.")
            
            # Proposed north-up grid
            transform_new, width_new, height_new = calculate_default_transform(
                src.crs, dest_crs, src.width, src.height, *src.bounds
            )

            print("-> Src transform:\n" + fmt_affine(src.transform))
            print("-> Proposed (north-up) transform:\n" + fmt_affine(transform_new))
            print(f"-> Proposed size: {width_new} x {height_new}")

            # Decide whether to warp: warp if grid differs OR CRS differs
            same_epsg = False
            try:
                same_epsg = (src.crs is not None) and (src.crs.to_epsg() == dest_crs.to_epsg())
            except Exception:
                same_epsg = False

            # grid equality test (tolerant)
            grids_equal = np.allclose(
                np.array(src.transform.to_gdal()),
                np.array(transform_new.to_gdal()),
                atol=1e-9
            )

            need_warp = (not same_epsg) or (not grids_equal)
            if not same_epsg:
                print("-> CRS differs -> will reproject.")
            elif not grids_equal:
                print("-> CRS equal but grid differs (rotation/scale/origin) -> will warp to north-up.")
                for d in log_transform_diff(src.transform, transform_new):
                    print("   Δ " + d)
            else:
                print("-> CRS and grid identical -> copy only (no warp).")

            # Build output metadata
            kwargs = src.meta.copy()
            kwargs.update({
                "driver": "GTiff",
                "compress": "deflate",
                "tiled": True,
                "predictor": 3 if src.dtypes[0].startswith("float") else 2,
                "bigtiff": "IF_SAFER",
            })
            if src.nodata is not None:
                kwargs["nodata"] = src.nodata

            if need_warp:
                kwargs.update({
                    "crs": dest_crs,
                    "transform": transform_new,
                    "width": width_new,
                    "height": height_new,
                })
            else:
                kwargs.update({
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                })

            # Choose resampling automatically (nearest for integer types)
            first_dtype = np.dtype(src.dtypes[0])
            is_integer = np.issubdtype(first_dtype, np.integer)
            resamp = Resampling.nearest if is_integer else Resampling.bilinear
            print(f"-> Resampling: {'NEAREST' if is_integer else 'BILINEAR'} (dtype={first_dtype})")

            with rasterio.open(output_path, "w", **kwargs) as dst:
                if need_warp:
                    print("-> Reprojecting bands…")
                    for b in range(1, src.count + 1):
                        print(f"   - Band {b}/{src.count}")
                        reproject(
                            source=rasterio.band(src, b),
                            destination=rasterio.band(dst, b),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform_new,
                            dst_crs=dest_crs,
                            resampling=resamp,
                        )
                else:
                    print("-> Copying pixels (no resampling).")
                    dst.write(src.read())

            print(f"✔ Wrote {output_path}")
            ui_set(root, status, f"Wrote: {output_path.name}")
            progress["value"] = i + 1
            progress.update()



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
    gdf_orthobox = read_rcorthobox([{'orthoboxfile': Path(file_path).with_suffix('.rsortho')} for file_path in raster_files])
    gdf_orthobox.to_file(output_gpkg, layer='orthoboxes', driver='GPKG')
     

def process_sideview_files(file_paths, progress, status, should_continue, root):
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
    file_paths = filedialog.askopenfilenames(title="Select orthobox files", filetypes=(("Orthobox files", "*.rsortho"), ("All files", "*.*")))

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
def open_file_browser_hugeTiffs2GIS_dem(root):
    file_paths = filedialog.askopenfilenames(
        title="Select DEM/continuous GeoTIFFs (top-down)",
        filetypes=(("GeoTIFF files", "*.tif;*.tiff"), ("All files", "*.*"))
    )
    if file_paths:
        progress = ttk.Progressbar(root, length=300, mode='determinate'); progress.pack()
        status = tk.StringVar(); tk.Label(root, textvariable=status).pack()
        should_continue = tk.BooleanVar(value=True)
        tk.Button(root, text="Abort", command=lambda: abort_processing(should_continue, status)).pack()
        threading.Thread(target=process_files_dem,
                         args=(file_paths, progress, status, should_continue, root),
                         daemon=True).start()

def open_file_browser_hugeTiffs2GIS_truecolor(root):
    file_paths = filedialog.askopenfilenames(
        title="Select truecolor RGB GeoTIFFs (top-down)",
        filetypes=(("GeoTIFF files", "*.tif;*.tiff"), ("All files", "*.*"))
    )
    if file_paths:
        progress = ttk.Progressbar(root, length=300, mode='determinate'); progress.pack()
        status = tk.StringVar(); tk.Label(root, textvariable=status).pack()
        should_continue = tk.BooleanVar(value=True)
        tk.Button(root, text="Abort", command=lambda: abort_processing(should_continue, status)).pack()
        threading.Thread(target=process_files_truecolor,
                         args=(file_paths, progress, status, should_continue, root),
                         daemon=True).start()


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
