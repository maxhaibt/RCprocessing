import shutil
import subprocess
from pathlib import Path
import numpy as np
import os
import json
import itertools
import pandas as pd
import pathconfig
from datetime import datetime 
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.affinity import rotate
import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
#from geopandas.tools import sjoin
#from shapely.geometry import Polygon
#import pickle




def loadconfigs(configpath):
    with open(configpath) as configfile:
        config = json.load(configfile)
    return config
config = loadconfigs('.\config_scanner.json')

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
            scan['scannerlogfile'] = [file for file in scan['scan_dir'].rglob("00-*")]
            imagelist = []
            for file in scan['scan_dir'].rglob(imageformat):
                if not file.stem.endswith('.mask'):
                    image_dict = {}
                    image_dict['rawimg_path']= Path(file)
                    mask = image_dict['rawimg_path'].with_name(image_dict['rawimg_path'].name + '.mask.png')

                    if mask.is_file():
                        image_dict['maskimg_path'] = mask
                    imagelist.append(image_dict.copy())
            scan['imagedf'] = pd.DataFrame(imagelist)
            scandf.append(scan.copy())
    return pd.DataFrame(scandf)


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


def importRegisteredParameters(scan):

    #find the registered parameters csv-file use glob
    camparam_path = [file for file in scan['scan_dir'].rglob("*camparam.csv")]
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

def calculateParametersStats(scan):
    print('calculateParametersStats: ', scan['id'])
    #calculate the mean and the standard deviation of the parameters for each camera
    #group the scan['image_df'] by the camera id, calculate the mean standard deviation for each of the parameters: f,px,py,k1,k2,k3,k4,t1,t2
    for parameter in ['f','px','py','k1','k2','k3','k4','t1','t2']:
        if parameter in scan['imagedf'].columns:
            #print(scan['imagedf'][parameter])
            scan['imagedf'][parameter+'_mean'] = scan['imagedf'].groupby('cam_id')[parameter].transform('mean')
            scan['imagedf'][parameter+'_std'] = scan['imagedf'].groupby('cam_id')[parameter].transform('std')
            #print (parameter, scan['imagedf'][parameter+'_mean'].unique(), scan['imagedf'][parameter+'_std'].unique())
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
            print('i: ', i)
            shotdf = pd.DataFrame(columns=(df.columns))
            # exclude from group the rows which are already in newdf
            groupmod = group[~group['rawimg_path'].isin(newdf['rawimg_path'].to_list())]
            # sort shotdf['cam_id'].to_list() and config['expected_cam_ids'] and compare
            t = 0
            for index, row in groupmod.iterrows():
                if not sorted(shotdf['cam_id'].to_list()) == sorted(config['expected_cam_ids']) :
                    
                    if row['cam_id'] not in shotdf['cam_id'].to_list() :
                        if int(row['imgnumber']) > shotdf['imgnumber'].max() + 6:
                            print('skip due to high imgnumber')
                            break

                        #print('Cam_id: ', row['cam_id'], ' is added')
                        #print(shotdf['cam_id'].to_list() )
                        row['shotnumber'] = i
                        shotdf = pd.concat([shotdf, row.to_frame().transpose()])
                        #print(row['rawimg_path'].stem, ' is added')
                    else :
                        print( 'skip cam_id: ', row['cam_id'])
                        t = t + 1
                        
                        if t>5:
                            print('t>5')
                            #shotdf = pd.concat([shotdf, row.to_frame().transpose()])
                            break
                else:
                    break
            #if sorted(shotdf['cam_id'].to_list()) == sorted(config['expected_cam_ids']) :
            print('shotdf complete ', len(shotdf),'. Range of imgnumbers: ', shotdf['imgnumber'].min(), ' - ', shotdf['imgnumber'].max())
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