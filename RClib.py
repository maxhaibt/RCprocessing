import shutil
import subprocess
from pathlib import Path
import numpy as np
import os
import json
import itertools
import pandas as pd
import pathconfig
#from geopandas.tools import sjoin
#from shapely.geometry import Polygon
#import pickle




def loadconfigs(configpath):
    with open(configpath) as configfile:
        config = json.load(configfile)
    return config
config = loadconfigs('.\config.json')

def provide_scandf(inputdirectory: str, imageformat = '*.dng') ->pd.DataFrame:
    scandf = []
    for scan_id in os.listdir(inputdirectory):
        scan = {}
        scan['id']= scan_id
        scan['scan_dir'] = Path(os.path.join(inputdirectory, scan_id))
        scan['pp3file'] = [file for file in scan['scan_dir'].rglob("*.pp3")]
        imagelist = []
        for file in scan['scan_dir'].rglob(imageformat):
            image_dict = {}
            image_dict['rawimg_path']= Path(file)
            imagelist.append(image_dict.copy())
        scan['imagedf'] = pd.DataFrame(imagelist)
        scandf.append(scan.copy())
    return pd.DataFrame(scandf)

def defineRawTherapeeOutput(series, foldername=''):
    series['RToutputfolder'] = Path(config['workspace']) / series['id'] / foldername
    series['RToutputfolder'].mkdir(exist_ok=True)
    return series

def defineRealityCaptureOutput(series, foldername=''):
    series['RCoutputfolder'] = Path(config['workspace']) / series['id'] / foldername
    series['RCoutputfolder'].mkdir(exist_ok=True)
    return series

def developwithRawTherapee(imageseries, pp3filepath , outputfolderpath, inputrawimagepathfield = 'rawimg_path', outputdevimagepathfield='dev-img_path'):
    outfile = outputfolderpath / Path(str(imageseries[inputrawimagepathfield].stem + '.png') )
    if not outfile.is_file() or outfile.is_file() and config['overwrite_dev-img'] :
        if imageseries['rawimg_path'].is_file():
            print('Inputfile: ', imageseries['rawimg_path'])
            print('Expect file: ', outfile)
        subprocess.check_output( '"' + str(config['RTpath']) + '"' \
             + ' -o ' + '"' + str(outfile.as_posix()) + '"'   + ' -n' + ' -q ' +  ' -Y ' \
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
    #imagelistpath.unlink()
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
    subprocess.check_output( '"' + str(Path(config['RCpath']).as_posix()) + '"' \
    + ' -headless' + ' -newScene'\
    + ' -save ' '"' + str(rcproj_path.as_posix()) + '"' \
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


def makeRCCMDfromListfield(scan, commandlistfield, rccmdpathfield='rccmdpath'):
    rccmdname = commandlistfield + '.rccmd'
    rccmdpath = scan['RCoutputfolder'] / rccmdname
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
def executeRCCMDuseRCproject(scan, rccmdpathfield='rccmdpath', instanceName = 'default'):
    subprocess.check_output('"' + str(Path(config['RCpath']).as_posix()) + '"' \
         + ' -setInstanceName ' + instanceName + ' -load ' \
        + str(scan['rcproj_path']) + ' -execRCCMD ' + '"' + str(scan[rccmdpathfield]) + '"' )

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