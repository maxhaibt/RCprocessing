# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:37:59 2020

@author: mhaibt
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import os
from pathlib import Path
import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from subprocess import check_output
from sklearn.cluster import KMeans
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

#import skimage.data as data
#import skimage.segmentation as seg
#import skimage.filters as filters
#from skimage import io
#import skimage.draw as draw
#import skimage.color as color

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

# display the image frameless and get pixel coordinates from mouse click, use this mouse click to get the mask via the predictor, return the mask and display it
def show_anns_class(anns):

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

def show_anns(anns):
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
    ax.imshow(mask_image)

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
    print(f"Coordinates of most white pixel: {max_loc_array}")
    print(f"Total number of white pixels: {total_white_pixels}")
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
    print(coords)
    scaledcoord = np.array([[int(i[0] * (100/scale_percent)), int(i[1] * (100/scale_percent))] for i in coords])

    print(f"Coordinates of most white pixel: {scaledcoord}")
    #print(f"Total number of white pixels: {total_white_pixels}")
    return scaledcoord
def binary_mask_to_rgb(mask):
    rgbmask = np.stack([mask* 255] , axis=-1).astype(np.uint8)
    return  rgbmask


def extract_histogram(series, image, maskfield='segmentation'): # applied to a dataframe of masks with the original image as input
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(type(hsv))
    print('This is the mask type:', type(series[maskfield]))
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
def plothistograms(series):
    # Plot the histograms
    fig, axs = plt.subplots(3)
    axs[0].plot(series['h_hist'] )
    axs[0].set_title('Hue')
    axs[1].plot(series['s_hist'])
    axs[1].set_title('Saturation')
    axs[2].plot(series['v_hist'])
    axs[2].set_title('Value')
    plt.show()
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
    if overlap_percentage > 0.8:
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
    print(data.shape)
    print(type(data))
    print(data)

    # Perform k-means clustering with two clusters on similarity matrix
    if len(masks_df) > 1:
        kmeans = KMeans(n_clusters=2).fit(data)

        # Predict which cluster each data point belongs to
        labels = kmeans.predict(data)
        print(labels)
        masks_df['class'] = labels
    else:
        masks_df['class'] = 0

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
        print('Median of first hist S: ', getMedianFromHist(df.iloc[0]['s_hist']))
        print('Median of second hist S: ', getMedianFromHist(df.iloc[1]['s_hist']))
        print('Max of first hist V: ', getMedianFromHist(df.iloc[0]['v_hist'],part=0.75))
        print('Max of second hist V: ', getMedianFromHist(df.iloc[1]['v_hist'],part=0.75))
        print('Correlation between the two histograms: ', s_hist_corr)

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

