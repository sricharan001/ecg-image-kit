import imageio
from PIL import Image
import argparse
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import time
import random
from helper_functions import read_bounding_box_txt, write_bounding_box_txt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_directory', type=str, required=True)
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-r','--rotate',type=int,default=25)
    parser.add_argument('-n','--noise',type=int,default=25)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=6500)
    return parser

# Main function for running augmentations
def get_augment(image,rotate=25,noise=25,crop=0.01,temperature=6500,bbox=False, store_text_bounding_box=False):
      
    lead_bbs = []
    leadNames_bbs = []
    
    if bbox:
        for lead,cords in bbox.items():
            xmin,ymin,xmax,ymax = [float(i) for i in cords]
            box = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=lead)
            lead_bbs.append(box)        
        lead_bbs = BoundingBoxesOnImage(lead_bbs, shape=image.shape)

    if store_text_bounding_box:
        for lead,cords in store_text_bounding_box.items():
            xmin,ymin,xmax,ymax = [float(i) for i in cords]
            box = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=lead)
            leadNames_bbs.append(box)         
        leadNames_bbs = BoundingBoxesOnImage(leadNames_bbs, shape=image.shape)
       
    
    images = [image[:, :, :3]]
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    #Augment in a sequential manner. Create an augmentation object
    seq = iaa.Sequential([
          iaa.Affine(rotate=rot),
          iaa.AdditiveGaussianNoise(scale=(noise, noise)),
          iaa.Crop(percent=crop_sample),
          iaa.ChangeColorTemperature(temperature)
          ])
    
    seq_bbox = iaa.Sequential([
          iaa.Affine(rotate=-rot),
          iaa.Crop(percent=crop_sample)
          ])
   
    images_aug = seq(images=images)

    if bbox:
        temp, augmented_lead_bbs = seq_bbox(images=images, bounding_boxes=lead_bbs)
    
    if store_text_bounding_box:
        temp, augmented_leadName_bbs = seq_bbox(images=images, bounding_boxes=leadNames_bbs)

    
    return_dict = {'image' : images_aug[0]}    
    
    if bbox:
        out_bbox = {}
        for box in augmented_lead_bbs:
            xmin,ymin,xmax,ymax = box.x1,box.y1,box.x2,box.y2
            out_bbox[box.label] = [xmin,ymin,xmax,ymax]
        return_dict['lead_bbox'] = (out_bbox)

    if store_text_bounding_box:
        out_text_bbox = {}
        for box in augmented_lead_bbs:
            xmin,ymin,xmax,ymax = box.x1,box.y1,box.x2,box.y2
            out_text_bbox[box.label] = [xmin,ymin,xmax,ymax]
        return_dict['text_bbox'] = (out_text_bbox)

    return return_dict

