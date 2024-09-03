# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:37:59 2020

@author: mhaibt
"""
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
import time
import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import hsv_to_rgb
import pandas as pd
from subprocess import check_output
from sklearn.cluster import KMeans
import sys
sys.path.append("..")
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda) 
device = torch.device("cuda")
torch.Tensor(1).to(device)

#import skimage.data as data
#import skimage.segmentation as seg
#import skimage.filters as filters
#from skimage import ioSamPre

#import skimage.draw as draw
#import skimage.color as color

def loadconfigs(configpath):
    with open(configpath) as configfile:
        config = json.load(configfile)
    return config
config = loadconfigs('.\config_scanner.json')

def loadSAMpredictor():
    sam_checkpoint = "C:/Users/mhaibt/Downloads/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def loadSAM():
    sam_checkpoint = "C:/Users/mhaibt/Downloads/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return sam
  
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    
    
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
def show3Dcolourspace(img):
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    return plt.show()
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def showOpencvimg(img):
    plt.imshow(img) 
    return plt.show()
def showColourspace(img):
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print(len(flags))
    print(flags[40])
def resize(img, per):
    scale_percent = per
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized




def treshBySatuartion(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sv= (s/120) * (v/255)
    svnormal = (255/np.amax(sv)) * sv
    svnormal = svnormal.astype(np.uint8)
    print(np.amax(svnormal))

    ret, sat = cv2.threshold(v,0,180,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,8,2)
    #blur = cv2.GaussianBlur(h,(5,5),0)
    #kernel = np.ones((64,64),np.uint8)
    #gloveobject = cv2.morphologyEx(gloveobject, cv2.MORPH_CLOSE, kernel)

    return sat

def getHistogram(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def testgroupapp(series):
    print (type(series))

def write_rcimagelist(series):
    series['rcimagelistoutpath'] = os.path.join(workingdirectory, str(series.iloc[0]['scan_id']) + '_' + str(series.iloc[0]['imgnumber']) +'.imagelist')
    series['rcimagemaskslistoutpath'] = os.path.join(workingdirectory, str(series.iloc[0]['scan_id']) + '_' + str(series.iloc[0]['imgnumber']) +'_masks.imagelist')
    series['rcalignmentoutpath'] = os.path.join(workingdirectory, str(series.iloc[0]['scan_id']) + '_' + str(series.iloc[0]['imgnumber']) +'.rcalign')
    with open(series.iloc[0]['rcimagelistoutpath'], 'w') as f:
        for camera in series['img_path']:
            geometryimage = Path(camera)
            f.write("%s\n" % geometryimage)
    with open(series.iloc[0]['rcimagemaskslistoutpath'], 'w') as f:
        for camera in series['img_path']:
            mask = Path(camera + '.mask.png')
            f.write("%s\n" % mask)
    return series



def cumulativeAlignment(df):
    commandlist = []
    gcpconfig ='E:/testalign/MarkerCoords/gcpconfig.xml'
    detectconfig = 'E:/testalign/MarkerCoords/detectconfig.xml'  
    gcps = 'E:/testalign/MarkerCoords/markers_coordinates.csv'
    imagelistdf = df[['rcimagelistoutpath', 'rcimagemaskslistoutpath', 'rcalignmentoutpath']].drop_duplicates()
      
    for index, row in imagelistdf.iterrows():
        aligncommand = '-add ' + row['rcimagelistoutpath'] +  ' -add ' + row['rcimagemaskslistoutpath'] + ' -importImageSelection ' + row['rcimagelistoutpath'] + ' -detectMarkers ' +  str(detectconfig)  +  ' -align' + ' -setMinComponentSize 10' + ' -exportRegistration ' + row['rcalignmentoutpath'] + ' -deleteSelectedComponent ' +' -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableAlignment false' +' -importImageSelection ' + row['rcimagelistoutpath'] + ' -enableAlignment false'
        
        commandlist.append(aligncommand)    
    return commandlist


def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    img2 = img2.astype(np.uint8)

    #cv2.imshow("Biggest component", img2)
    #cv2.waitKey()
    return img2

def provide_imagelist(inputdirectory: str) ->pd.DataFrame:
    """
    @brief reads metadata of page images from inputdirectory and stores it into 
            a pandas DataFrame
    @param inputdirectory directory with page images stored in .png or .jpg format
    @return DataFrame with cols: pub_key, pub_value, pub_imagename, page_path 
    """

    imagelist = []
    for scan_id in os.listdir(inputdirectory):
        if not scan_id == 'masktemplates':
            scan_dir = os.path.join(inputdirectory, scan_id)
            
            for image in os.listdir(scan_dir):
                ##Hier Weitermachen du wolltest die images pro object_id speichern alle mit ihrem unique-name den du nochmal rekapitulieren solltest dann mit ffmpeg ein videoformat

                if image.endswith((".png", ".jpg")) and 'Thumbs' not in image and 'mask' not in image and 'frameless' not in image:
                    image_dict = {}
                    image_dict['scan_id']=scan_id
                    image_dict['img_path']= os.path.join(scan_dir, image)
                    cam, cam2 , objectidfile, roundnumber, imgnumber = image.split('_')
                    image_dict['cam_id'] = cam + '_'+ cam2
                    image_dict['objectid'] = scan_id.split('-')[2]
                    image_dict['roundnumber'] = roundnumber
                    imgnumber = imgnumber.replace('.jpg','')
                    image_dict['imgnumber'] = int(imgnumber.replace('test',''))

                    imagelist.append(image_dict.copy())

    return pd.DataFrame(imagelist)

def getBoundingBox(img):

    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
        return x,y,w,h

def cropToBoundingBox(img, x,y,w,h ):
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img


def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = hsv[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        #hsv = cv2.cvtColor(colors, cv2.COLOR_BGR2LAB)
        pickedcolorlist.append(colors)
        return colors

# Read an image, a window and bind the function to window




def getCammask(row, maskdir):
    maskpath = os.path.join(maskdir, row['cam_id'] + '_test15.jpg.mask.png' )
    print(maskpath)
    mask = cv2.imread(maskpath,0)
    invmask = cv2.bitwise_not(mask)
    return invmask

def applyMask(mask, original):
    truematrix = np.zeros(original.shape,dtype=np.uint8)
    truematrix.fill(True)
    invmask3ch = cv2.bitwise_and(truematrix, truematrix, mask=mask)
    maskedoriginal = original * invmask3ch
    return maskedoriginal

def colourkeyMask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    #blueglove34.4681, 29.3750, 62.7451
    #darkbackground 16.0000, 62.5000, 9.4118
    #Object 20.8264, 67.5978, 70.1961
    ##whiteglove###
    #lower_color = (0,0,190)
    #upper_color = (255,255,255)
    #blueglove
    #lower_color = (0,10,0)
    #upper_color = (50,180,255)
    #th3 = cv2.adaptiveThreshold(h,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #blur = cv2.GaussianBlur(h,(5,5),0)
    ret, gloveobject = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #lower_color = np.array([int(180/360*255),40,40])
    #upper_color = np.array([int(215/360*255),255,255])
    #glovemask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((64,64),np.uint8)
    gloveobject = cv2.morphologyEx(gloveobject, cv2.MORPH_CLOSE, kernel)
    #kernel = np.ones((32,32),np.uint8)
    #glovemask = cv2.dilate(glovemask, kernel)

    #showColourspace(glovemask)
    #glovemask = glovemask.astype(np.uint8)
    #showOpencvimg(gloveobject)
    
    #rgb = cv2.cvtColor(glovemask, cv2.COLOR_HSV2BGR)
    #print(bgr.shape[1])
    return gloveobject

def pickedoutMask(slic_av, pickedcolorlist):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(slic_av, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    #if hsv-vector is similar to picked hsv-vectors:
        #glovemask = cv2.inRange(hsv, lower_color, upper_color)


    #blueglove34.4681, 29.3750, 62.7451
    #darkbackground 16.0000, 62.5000, 9.4118
    #Object 20.8264, 67.5978, 70.1961
    ##whiteglove###
    #lower_color = (0,0,190)
    #upper_color = (255,255,255)
    #blueglove
    #lower_color = (0,10,0)
    #upper_color = (50,180,255)
    #th3 = cv2.adaptiveThreshold(h,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #blur = cv2.GaussianBlur(h,(5,5),0)
    ret, gloveobject = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #lower_color = np.array([int(180/360*255),40,40])
    #upper_color = np.array([int(215/360*255),255,255])
    #glovemask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((64,64),np.uint8)
    gloveobject = cv2.morphologyEx(gloveobject, cv2.MORPH_CLOSE, kernel)
    #kernel = np.ones((32,32),np.uint8)
    #glovemask = cv2.dilate(glovemask, kernel)

    #showColourspace(glovemask)
    #glovemask = glovemask.astype(np.uint8)
    #showOpencvimg(gloveobject)
    
    #rgb = cv2.cvtColor(glovemask, cv2.COLOR_HSV2BGR)
    #print(bgr.shape[1])
    return gloveobject


def kmeasClusterMask(img, masker, resizeper, n_segments):
    #img = cv2.imread(series['img_path'])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert BGR to HSV
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masker = resize(masker, resizeper)
    masker = cv2.cvtColor(masker, cv2.COLOR_BGR2GRAY)
    masker = cv2.bitwise_not(masker)
    print(masker.shape)
    #masker = 255 - masker
    #masker = cv2.threshold(masker,0,255,cv2.THRESH_BINARY)
    #print(masker)
    
    img = resize(rgb, resizeper)
    image_slic = seg.slic(img,n_segments=6, spacing=[0.1,1,1], multichannel=True, compactness=8.0, sigma=0.4, convert2lab=True, enforce_connectivity=False, mask=masker)
    #image, n_segments=100,  max_num_iter=10, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=1, mask=None, *, channel_axis=- 1
    #showOpencvimg(image_slic)
    image_slic_av = color.label2rgb(image_slic, img, kind='avg')
    image_slic_av = image_slic_av.astype(np.uint8)
    #image_slic_av = cv2.cvtColor(image_slic_av, cv2.COLOR_LAB2RGB)
    #text = image_slic_av.page()
    #image_show(text)
    return image_slic_av, img

def randomWalker(img, resizeper):
    img = resize(img, resizeper)
    #seg.random_walker(data, labels, beta=130, mode='cg_j', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None, *, prob_tol=0.001, channel_axis=None)

def rowAsJSONToFile(row, filepath):
    with open(filepath, 'a') as outfile:
        outfile.write(row.to_json())
        outfile.write('\n')


def grabcut(img, masker, resizeper):
    masker = masker[:,:,:1]
    #print(masker)
    masker = masker.astype(np.uint8)
    masker = resize(masker, resizeper)
    kernelwidth = 100
    kernelheight = 100
    kernel = np.ones((kernelheight,kernelwidth),np.uint8)
    masker = cv2.dilate(masker,kernel,iterations = 1)
    #cv2.imwrite(inputdirectory + 'testoutput/test.png', masker)
    img = resize(img, resizeper)
    #white = masker[masker == 255]
    #black = masker[masker == 0]
    masker[masker > 0] = cv2.GC_PR_BGD
    masker[masker == 0] = cv2.GC_PR_FGD
    print(masker[masker > 0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    (mask, bgModel, fgModel) = cv2.grabCut(img, masker, None, bgModel,fgModel, 10, mode=cv2.GC_INIT_WITH_MASK)
    values = (
        ("Definite Background", cv2.GC_BGD),
        ("Probable Background", cv2.GC_PR_BGD),
        ("Definite Foreground", cv2.GC_FGD),
        ("Probable Foreground", cv2.GC_PR_FGD),
    )
    
    valuemask = (mask == cv2.GC_PR_FGD).astype("uint8") * 255
    #detectedmask = detectedmask.astype("uint8") * 255

    return valuemask, img


def show_anns_class(anns, outpath):

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        
        # Set color based on class label
        if ann['class'] == 'object':
            color_mask = [1, 0, 0]
        elif ann['class'] == 'glove':
            color_mask = [0, 1, 0]
        else:
            color_mask = [0, 0, 1]
        
        for i in range(3):
            img[:,:,i] = color_mask[i]
        
        # Set alpha value to 0.35
        ax.imshow(np.dstack((img, m*0.35)))
        
        # Add label with area value
        if not 'area' in ann.keys():
            ann['area']= 'unknown'
        area = ann['area']
        if not 'predicted_iou' in ann.keys():
            ann['predicted_iou']= 0
        iou = ann['predicted_iou']
        classs = ann['class']
        
        # Calculate x and y from bbox
        if 'bbox' in ann.keys():
            bbox = ann['bbox']
        x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        
        ax.text(x, y, f'{classs}', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.savefig(outpath)
def show_anns(anns, outpath):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        print(m.shape)
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
        # Add label with area value
        area = ann['area']
        
        # Calculate x and y from bbox
        bbox = ann['bbox']
        x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        
        ax.text(x, y, f'{area:.2f}', fontsize=12, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.savefig(outpath)
def get_center_coordinates(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return [int(center_x), int(center_y)]

def get_masks_with_n_highest_scores(masks, scores, n):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masks = masks[sorted_indices]
    return sorted_masks[:n], np.argsort(scores)[:n]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)#






def convmasktoimg(mask):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def mouseCoords(event, x, y, flags, param, **kwargs):

    if event == cv2.EVENT_RBUTTONDOWN: #checks mouse right button down condition
        if row is not None:
            print('This is the row inside')
        ##Use last mask as input
        print('Right click')
        
        lastrow = kwargs.get('mask_input')
        #print(lastrow['samresult']['masks'])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
        point_labels=input_label,
        mask_input=lastrow['samresult']['masks'][0][None, :, :],
        multimask_output=False,
         )
        row['samresult'] = {}
        row['samresult']['masks'] = masks
        row['samresult']['scores'] = scores
        row['samresult']['logits'] = logits
        for i, (mask, score) in enumerate(zip(masks, scores)):
            #plt.figure(figsize=(10,10))
            #plt.imshow(image)
            mask_image = show_mask(mask, plt.gca())
            cv2.imshow('mouseRGB',cv2.resize(mask_image,(0,0),fx=0.5,fy=0.5))
            #show_points(input_po#int, input_label, plt.gca())
            #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            #plt.axis('off')
            #plt.show() 

    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        if row is not None:
            print('This is the row inside')
        print("Coordinates of pixel: X: ",x,"Y: ",y)
        input_point = np.array([[x*2, y*2]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            )
        row['samresult'] = {}
        row['samresult']['masks'] = masks
        row['samresult']['scores'] = scores
        row['samresult']['logits'] = logits
        for i, (mask, score) in enumerate(zip(masks, scores)):
            #plt.figure(figsize=(10,10))
            #plt.imshow(image)
            mask_image = show_mask(mask, plt.gca())
            cv2.imshow('mouseRGB',cv2.resize(mask_image,(0,0),fx=0.5,fy=0.5))
            #show_points(input_po#int, input_label, plt.gca())
            #plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            #plt.axis('off')
            #plt.show() 
        print (row['samresult']['masks'])

def findWhitestPixel(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Downscale image
    scale_percent = 5 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray_downscaled = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    # Threshold the image
    thresh = cv2.threshold(gray_downscaled , 230, 255, cv2.THRESH_BINARY)[1]

    # Count total number of white pixels
    total_white_pixels = cv2.countNonZero(thresh)

    # Find maximum value in image
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thresh)
    #max_loc_array = [max_loc[0],max_loc[1]]
    max_loc_array = [int(max_loc[0] * (100/scale_percent)), int(max_loc[1] * (100/scale_percent))]

    return max_loc_array

def findWhitestPixel_contours(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Downscale image
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    gray_downscaled = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    # Threshold the image
    thresh = cv2.threshold(gray_downscaled , 240, 255, cv2.THRESH_BINARY)[1]
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Iterate over contours and find centroids
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(cnt)
            centroids.append((cx, cy, area))

    # Sort centroids by area
    centroids = sorted(centroids, key=lambda x: x[2], reverse=True)

    # Extract coordinates of three centroids with highest area
    coords = centroids[:3]

    scaledcoord = np.array([[int(i[0] * (100/scale_percent)), int(i[1] * (100/scale_percent))] for i in coords])


    #print(f"Total number of white pixels: {total_white_pixels}")
    return scaledcoord
def binary_mask_to_rgb(mask):
    rgbmask = np.stack([mask* 255] , axis=-1).astype(np.uint8)
    return  rgbmask


def extract_histogram(series, image, maskfield='segmentation'): # applied to a dataframe of masks with the original image as input
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #print(series[maskfield])
    # Create a mask with only the pixels in the masked area
    #masked_hsv = np.where(series[maskfield][..., None], hsv, 0)
    #print(masked_hsv)
    #masked_hsv = applyMask(series[maskfield], hsv)
    
    mask = binary_mask_to_rgb(series[maskfield])

    # Calculate histogram of masked area
    #series['hist'] = cv2.calcHist([masked_hsv], [0], None, [256], [0, 256])
    #cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    # Calculate the histogram for each channel
    series['h_hist'] = cv2.calcHist([hsv], [0], mask, [256], [0, 256])
    series['s_hist'] = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    series['v_hist'] = cv2.calcHist([hsv], [2], mask, [256], [0, 256])

    # Plot the histograms
    #fig, axs = plt.subplots(3)
    #axs[0].plot(series['h_hist'] )
    #axs[0].set_title('Hue')
    #axs[1].plot(series['s_hist'])
    #axs[1].set_title('Saturation')
    #axs[2].plot(series['v_hist'])
    #axs[2].set_title('Value')
    #plt.show()
    return series

import numpy as np

def plothistograms(series, outpath):
    # Plot the histograms
    fig, axs = plt.subplots(1, 3, figsize=(34, 5))
    h_median = getMedianFromHist(series['h_hist'])
    axs[0].plot(series['h_hist'])
    axs[0].axvline(h_median, color='black', linestyle='dashed')
    axs[0].set_title('Hue', fontsize=34)
    axs[0].set_xlim([0, 179])  # Set the x-axis limit for Hue
    axs[0].tick_params(axis='both', labelsize=32)

    s_median = getMedianFromHist(series['s_hist'])
    axs[1].plot(series['s_hist'])
    axs[1].axvline(s_median, color='black', linestyle='dashed')
    axs[1].set_title('Saturation', fontsize=34)
    axs[1].set_xlim([0, 255])  # Set the x-axis limit for Saturation
    axs[1].tick_params(axis='both', labelsize=32)

    v_median = getMedianFromHist(series['v_hist'])
    axs[2].plot(series['v_hist'])
    axs[2].axvline(v_median, color='black', linestyle='dashed')
    axs[2].set_title('Value', fontsize=34)
    axs[2].set_xlim([0, 255])  # Set the x-axis limit for Value
    axs[2].tick_params(axis='both', labelsize=32)

    # Convert median HSV values to RGB color
    median_color = hsv_to_rgb(np.array([h_median, s_median, v_median]) / 255)

    # Create a small legend-like icon
    icon = Rectangle((0, 0), 0.05, 0.1, facecolor=median_color, transform=fig.transFigure, figure=fig)

    # Add the icon to the top right corner of the figure
    fig.patches.extend([icon])
    icon.set_clip_on(False)
    icon.set_xy((0.88, 0.98))

    fig.suptitle('cluster ' + series['class'], fontsize=40, y=1.2)
    plt.savefig(outpath + '_' + series['class'] + '.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def scaleimage(image, scale_percent):
    # Downscale image
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_downscaled = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image_downscaled

def get_overlapping(series, overlapagainst):
    # Check if masks overlap
    mask = binary_mask_to_rgb(series['segmentation'])

    overlap = cv2.bitwise_and(mask, overlapagainst)


    # Calculate percentage of overlap
    overlap_percentage = np.sum(overlap) / np.sum(mask)

    # If overlap percentage is greater than 10%, then masks overlap
    #print(overlap_percentage)
    if overlap_percentage > 0.5:
        #print('Masks overlap')
        series['overlap'] = True
    else:
        #print('Masks do not overlap')
        series['overlap'] = False

    return series
def getMedianFromHist(hist, part=0.5):
    # Get the median value from the histogram  
    #part should be between 0 and 1, 0,25 for the first quartile, 
    #0,5 for the median, 0,75 for the third quartile or in between
    median = 0
    for i in range(256):
        median += hist[i]
        if median > sum(hist) * part:
            break
    return i


def cluster_segments(masks_df):
    # Create empty similarity matrices for each histogram type
    h_sim = np.zeros((len(masks_df), len(masks_df)))
    s_sim = np.zeros((len(masks_df), len(masks_df)))
    v_sim = np.zeros((len(masks_df), len(masks_df)))

    # Iterate over each segment
    for i, row_i in masks_df.iterrows():
        # Extract histograms for this segment
        h_i = row_i['h_hist']
        s_i = row_i['s_hist']
        v_i = row_i['v_hist']

        # Iterate over all other segments
        for j, row_j in masks_df.iterrows():
            # Extract histograms for other segment
            h_j = row_j['h_hist']
            s_j = row_j['s_hist']
            v_j = row_j['v_hist']

            # Compute similarity between histograms using correlation metric
            h_sim[i, j] = cv2.compareHist(h_i, h_j, cv2.HISTCMP_CORREL)
            s_sim[i, j] = cv2.compareHist(s_i, s_j, cv2.HISTCMP_CORREL)
            v_sim[i, j] = cv2.compareHist(v_i, v_j, cv2.HISTCMP_CORREL)


    # Concatenate the two matrices into a single matrix
    data = s_sim
    #data = np.concatenate((s_sim.flatten(), v_sim.flatten()))

    # Reshape the data into a matrix of shape (18, 1)
    #data = data.reshape(-1, 1)


    # Perform k-means clustering with two clusters on similarity matrix
    if len(masks_df) > 1:
        kmeans = KMeans(n_clusters=2).fit(data)

        # Predict which cluster each data point belongs to
        labels = kmeans.predict(data)
        print('Inside cluster labels: ',labels)
        masks_df['class'] = labels.astype(str)
    else:
        masks_df['class'] = 'unknown'

    return masks_df

def merge_masks(df):
    # Group the DataFrame by class value
    groups = df.groupby('class')

    # Initialize an empty dictionary to store the merged masks and areas
    merged_masks = {}
    areas = {}
    boxes = {}

    # Loop over each group
    for name, group in groups:
        # Get the binary masks for this group
        masks = group['segmentation'].values

        # Merge the masks using element-wise OR
        merged_mask = np.zeros_like(masks[0])
        for mask in masks:
            merged_mask = np.logical_or(merged_mask, mask)

        # Calculate the area of the merged mask in pixels
        area = np.sum(merged_mask)

        # Calculate the bounding box of the merged mask in xyxy format
        rows = np.any(merged_mask, axis=1)
        cols = np.any(merged_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        box = np.array([xmin, ymin, xmax, ymax])

        # Store the merged mask, area, and bounding box in the dictionaries
        merged_masks[name] = merged_mask
        areas[name] = area
        boxes[name] = box


    # Create a new DataFrame with the merged masks, areas, and bounding boxes
    new_df = pd.DataFrame({'class': list(merged_masks.keys()), 'segmentation': list(merged_masks.values()), 'area': list(areas.values()), 'bbox': list(boxes.values())})

    return new_df

def defineGloveObject(df):
    if len(df) == 2:
   
        # Compare the histograms using cv2.compareHist()
        s_hist_corr = cv2.compareHist(df.iloc[0]['s_hist'], df.iloc[1]['s_hist'], cv2.HISTCMP_CORREL)
        v_hist_corr = cv2.compareHist(df.iloc[0]['v_hist'], df.iloc[1]['v_hist'], cv2.HISTCMP_CORREL)
        # If the correlation between the 's_hist' histograms is greater than 0.95,
        # classify both segments as object
        #print('Median of first hist S: ', getMedianFromHist(df.iloc[0]['s_hist']))
        #print('Median of second hist S: ', getMedianFromHist(df.iloc[1]['s_hist']))
        #print('Max of first hist V: ', getMedianFromHist(df.iloc[0]['v_hist'],part=0.75))
        #print('Max of second hist V: ', getMedianFromHist(df.iloc[1]['v_hist'],part=0.75))
        #print('Correlation between the two histograms: ', s_hist_corr)

        if getMedianFromHist(df.iloc[0]['s_hist']) < getMedianFromHist(df.iloc[1]['s_hist']) and getMedianFromHist(df.iloc[0]['v_hist'], part=0.75) > 190 and getMedianFromHist(df.iloc[0]['v_hist'], part=0.75) > getMedianFromHist(df.iloc[1]['v_hist'], part=0.75) :
            df['class'] = ['glove', 'object']
        elif getMedianFromHist(df.iloc[0]['s_hist']) > getMedianFromHist(df.iloc[1]['s_hist']) and getMedianFromHist(df.iloc[1]['v_hist'], part=0.75) > 190 and getMedianFromHist(df.iloc[1]['v_hist'], part=0.75) > getMedianFromHist(df.iloc[0]['v_hist'], part=0.75) :
            df['class'] = ['object', 'glove']
        elif s_hist_corr > 0.80:
            df['class'] = ['object','object']
        else:
            df['class'] = ['unknown','unknown']

    if len(df) == 1:
        df['class'] = 'object'


    return df   


def is_image_binary(image):
    #img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
    unique_pixel_values = np.unique(image[0]) 
    if np.array_equal(unique_pixel_values, np.array([False])) or np.array_equal(unique_pixel_values, np.array([False, True])) :  # Check if the image is binary
        return True
    else:
        return False


def show_mask_asoutline(maskdf, original, outpath, labelfield, dpi=96, reference_width=300):
    # Find the contours of the mask
    height, width = original.shape[:2]
    figsize = (width / dpi, height / dpi)

    # Calculate the scaling factor

    scaling_factor = width / reference_width

    plt.figure(figsize=figsize, dpi=dpi)
    #print('Here are the labels: ', maskdf[labelfield])
    plt.imshow(original, cmap='gray')
    unique_labels = maskdf[labelfield].unique()
    colormap = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))
    label_color_map = dict(zip(unique_labels, colormap.colors))

    for index, row in maskdf.iterrows():
        if is_image_binary(row['segmentation']):
            row['segmentation'] = binary_mask_to_rgb(row['segmentation'])
        contours, hierarchy = cv2.findContours(row['segmentation'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Find the centroid of the contour
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # Draw the contour
                label = row[labelfield] if labelfield in row.keys() and row[labelfield] else 'unknown'
                contour_color = label_color_map[label]
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=2 * scaling_factor, color=contour_color)

                # Draw the label
                if label != 'unknown':
                    plt.text(cx, cy, label, fontsize=12 * scaling_factor, color=contour_color, ha='center', va='center',
                             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
            else:
                print("Warning: Division by zero in centroid calculation skipped.")
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
    plt.close()





if config['maskobject_QS3D']:
    predictor = loadSAMpredictor()
    sam = loadSAM()

def maskoutobject_QS3D(series, scale_percent = 5):
    start_time = time.time()
        
    if 'maskimg_path' not in series.keys() or str(series.get('maskimg_path')) == 'nan' or config['overwrite_maskimg']:
        print('Generating mask for this image: ', series['dev-img_path'])
        original = cv2.imread(str(series['dev-img_path']))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        if Path(Path(config['workspace']) / 'masktemplates/basemasks').is_dir():
            cammask = getCammask(series, Path(config['workspace']) / 'masktemplates/basemasks')
        else:
            print('Cammask not found')

        frameless = applyMask(cammask, original)
        

        
        gloveobject = colourkeyMask(frameless)
        gloveobjectclean = undesired_objects(gloveobject)
        invgloveobjectclean  = cv2.bitwise_not(gloveobjectclean )
        bk = np.full(original.shape, 0, dtype=np.uint8)  
        fg_masked = cv2.bitwise_and(original, original, mask=gloveobjectclean)
        bk_masked = cv2.bitwise_and(bk, bk, mask=invgloveobjectclean) 
        final = cv2.bitwise_or(fg_masked, bk_masked)
        print(f"conventional-processing time: {time.time() - start_time:.4f} seconds")
        img = scaleimage(original, scale_percent)
        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_batch=32,
            points_per_side=16,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.70,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=20,  
        )
        masks2 = mask_generator_2.generate(img)
        print(f"sam-processing time: {time.time() - start_time:.4f} seconds")
        masks_df = pd.DataFrame(masks2)
        #show_masks_df = masks_df.copy()
        #show_masks_df['segmentation'] = show_masks_df['segmentation'].apply(binary_mask_to_rgb)
        masks_df['class'] = 'unknown'
        if config['maskobject_QS3D_visualized']:
            show_mask_asoutline(masks_df, img, outpath = str(series['rawimg_path'].with_name(series['rawimg_path'].name + '.sam.png')), labelfield='class')
        ###check if masks in masks_df overlap with gloveobjectclean take only overlapping
        gloveobjectclean_down = scaleimage(gloveobjectclean, scale_percent)

        masks_df = masks_df.apply(get_overlapping, overlapagainst = gloveobjectclean_down, axis=1)
        masks_df = masks_df[masks_df['overlap'] == True]
        #show_masks_df = masks_df.copy()
        #show_masks_df['segmentation'] = show_masks_df['segmentation'].apply(binary_mask_to_rgb)
        if config['maskobject_QS3D_visualized']:
            show_masks_df = masks_df.append(pd.DataFrame([{'segmentation':gloveobjectclean_down, 'class':'glove-object'}]))
        
            show_mask_asoutline(show_masks_df, img, outpath = str(series['rawimg_path'].with_name(series['rawimg_path'].name + '.samoverlap.png')), labelfield='class')
        if not masks_df.empty:
            #masks_df = masks_df[masks_df['area'] < 9000]
            masks_df = masks_df.nlargest(14, 'predicted_iou').reset_index(drop=True)
            #masks_df_area_iou = masks_df.nlargest(9, 'area').reset_index(drop=True)
            masks_df_area_iou  = masks_df.apply(extract_histogram, image = img, axis=1)
            masks_df_area_iou = cluster_segments(masks_df_area_iou)
            masks_df_area_iou = merge_masks(masks_df_area_iou)
            show_mask_asoutline(masks_df_area_iou, img, outpath = str(series['rawimg_path'].with_name(series['rawimg_path'].name + '.2clusters.png')), labelfield='class')
            masks_df_area_iou  = masks_df_area_iou.apply(extract_histogram, image = img, axis=1)
            masks_df_area_iou.apply(plothistograms, outpath = str(series['rawimg_path'].with_name(series['rawimg_path'].name + '.2clusters_hist.png')) , axis=1)
            masks_df_area_iou = defineGloveObject(masks_df_area_iou)
            masks_df_area_iou = merge_masks(masks_df_area_iou)



            if masks_df_area_iou['class'].str.contains('unknown').any(): 
                print('Second round of segmentation')

                whitestpixelcoords = findWhitestPixel_contours(final)
                #print('This should be the base for bbox', masks_df_area_iou.loc[0]['segmentation'])
                #rgbmask = ODlib.binary_mask_to_rgb(masks_df_area_iou.loc[0]['segmentation'], channels=3)
                
                #x,y,w,h = ODlib.getBoundingBox(rgbmask)         
                #input_box = np.array([x,y,x+w,y+h])
                input_box = masks_df_area_iou.loc[0]['bbox']
                #print('This should be the bbox: ', input_box)
                input_label = np.array([int(0) for i in range(len(whitestpixelcoords))])
                center = get_center_coordinates(input_box)
                input_points = whitestpixelcoords

                if input_points.size == 0:
                    print('No points found')
                    input_points = np.array([center])
                    input_label = np.array([int(1)])  
                            
                print(len(input_points), 'input points', len(input_label), 'input labels')
                
                predictor.set_image(img)
                masks, scores, logits = predictor.predict(
                    point_coords=input_points ,
                    point_labels=input_label,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                h, w = masks.shape[-2:]
                mask_image = masks.reshape(h, w)
                #print('mask image shape', mask_image.shape)
                masks_df_area_iou['segmentation'] = [mask_image]
                masks_df_area_iou['class'] = 'object'
    
 
        
        
        rgbmask = binary_mask_to_rgb(masks_df_area_iou[masks_df_area_iou['class']=='object'].iloc[0]['segmentation'])
        object_mask = scaleimage(rgbmask, 2000)
        kernel = np.ones((64,64),np.uint8)
        mask= cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        series['maskimg_path'] = series['dev-img_path'].with_name(series['dev-img_path'].name + '.mask.png')
        print(f"rest-processing time : {time.time() - start_time:.4f} seconds")
        cv2.imwrite(str(series['maskimg_path']),mask)
        print(f"writing time: {time.time() - start_time :.4f} seconds")
        #original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        #show_anns(masks2, str(series['rawimg_path'].with_name(series['rawimg_path'].name + '.sam.png')))
        finalmaskdf = pd.DataFrame([{'segmentation': mask, 'class': 'object'}])
        if config['maskobject_QS3D_visualized']:
            show_mask_asoutline(finalmaskdf, original, outpath = str(series['dev-img_path'].with_name(series['dev-img_path'].name + '.final.png')), labelfield='class')
        #plt.figure( figsize=(10,10))
        #plt.imshow(original)
        #show_mask(mask, plt.gca())
        #show_box(input_box, plt.gca())
        #show_points(input_points, input_label, plt.gca())
        #plt.show()
        #plt.savefig(series['rawimg_path'].with_name(series['rawimg_path'].name + '.segments.png'), format='jpg')

    return series


def colorcorrection(imageseries, colorreference, outputfolderpath, inputrawimagepathfield='rawimg_path', outputdevimagepathfield='dev-img_path'):
    outfile = outputfolderpath / Path(str(imageseries[inputrawimagepathfield].stem) + config['devimage_format'])

    for camref in colorreference['whitebalance']:
        if camref['cam_id'] == imageseries['cam_id']:
            whitepoint = camref['pos_color_019']
            blackpoint = camref['pos_color_024']

    if not outfile.is_file() or (outfile.is_file() and config['overwrite_dev-img']):
        if imageseries[inputrawimagepathfield].is_file():
            img = cv2.imread(str(imageseries[inputrawimagepathfield]))
            # Convert to LAB color space
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Extracting the average LAB color from a patch around the white point
            x, y = whitepoint
            lab_patch = lab_img[y-10:y+10, x-10:x+10]
            lab_avg_white = np.mean(lab_patch, axis=(0, 1))
            # Extracting the LAB color from the black point
            x, y = blackpoint
            lab_patch = lab_img[y-10:y+10, x-10:x+10]
            lab_avg_black = np.mean(lab_patch, axis=(0, 1))


            # Target LAB values for a neutral white patch
            white_color = np.uint8([[[254, 254, 254]]])
            white_color = cv2.cvtColor(white_color, cv2.COLOR_RGB2LAB)
            print('LAB average: ', lab_avg_white)
            print('LAB white: ', white_color)

            # Target LAB values for a neutral black patch
            black_color = np.uint8([[[30, 30, 30]]])
            black_color = cv2.cvtColor(black_color, cv2.COLOR_RGB2LAB)
            print('LAB average black: ', lab_avg_black)

            # Calculate correction factors
            # get the first channel of the white color and divide it by the average of the white color

            L_scale = (white_color[0][0][0] / lab_avg_white[0])
            A_shift = white_color[0][0][1] / lab_avg_white[1]
            B_shift = white_color[0][0][2] / lab_avg_white[2]
            print('Correction factors: ', L_scale, A_shift, B_shift)

            # Calculate correction factors black point
            L_scale_black = (black_color[0][0][0] / lab_avg_black[0])
            A_shift_black = black_color[0][0][1] / lab_avg_black[1]
            B_shift_black = black_color[0][0][2] / lab_avg_black[2]

            # Print and compare the correction factors
            print('Correction factors black: ', L_scale_black, A_shift_black, B_shift_black)
            print('Correction factors: ', L_scale, A_shift, B_shift)

            #average # white and black correction factors
            #L_scale = (L_scale + L_scale_black) / 2
            #A_shift = (A_shift + A_shift_black) / 2
            #B_shift = (B_shift + B_shift_black) / 2

            #blend the correction factors
            blendfactor = 0.7

            # Apply correction factors in LAB space
            lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] * L_scale * blendfactor, 0, 255)
            lab_img[:, :, 1] = np.clip(lab_img[:, :, 1] * A_shift, 0, 255)
            lab_img[:, :, 2] = np.clip(lab_img[:, :, 2] * B_shift, 0, 255)

            # Convert back to RGB
            corrected_img = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            # Write the corrected image
            cv2.imwrite(str(outfile), corrected_img)
            if outfile.is_file():
                print(f"Color corrected image saved to {outfile}")
                imageseries[outputdevimagepathfield] = outfile

    return imageseries

