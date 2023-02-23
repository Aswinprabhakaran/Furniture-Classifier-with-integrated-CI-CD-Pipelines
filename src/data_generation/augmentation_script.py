#!/usr/bin/env python

import cv2
import numpy as np
import random

# # Shearing
def shearing(image):
    
    shear_factor = random.sample([0.1,0.2,0.3,0.4,0.5],1)[0]
    
    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                
    nW =  image.shape[1] + abs(shear_factor * image.shape[0])
        
    image = cv2.warpAffine(image, M, (int(nW), image.shape[0]))

    return image


# # Rotate
def rotate_im(image, angle):
    
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def rotate_box(corners,angle,  cx, cy, h, w):
    
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def Rotate(image):
    
    angle = random.sample(list(np.arange(-25,26)),1)[0]
    
    w,h = image.shape[1], image.shape[0]
    
    cx, cy = w//2, h//2
        
    image = rotate_im(image, angle)
        
    return image


# # Translate
def translate(image):
    
    translate = random.sample(list((np.arange(-25,26)/100)),2)
    
    translate_factor_x = translate[0]
    translate_factor_y = translate[1]
    
    img_shape = image.shape
        
    #get the top-left corner co-ordinates of the shifted box 
    corner_x = int(translate_factor_x * img_shape[1])
    corner_y = int(translate_factor_y * img_shape[0])
    
    image = image[ max(-corner_y, 0) : min(img_shape[0], -corner_y + img_shape[0]),
                  max(-corner_x, 0) : min(img_shape[1], -corner_x + img_shape[1]),
                  :]
    
    image_shape = [max(-corner_y, 0) , min(img_shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0) , min(img_shape[1], -corner_x + img_shape[1])]
    
    return image


# # Resize
def Resize(image):
    
    scale_percent = random.sample(list(np.arange(75,126)/100),1)[0] # percent of original size
    
    width = int(image.shape[1] * scale_percent )
    height = int(image.shape[0] * scale_percent )
    
    dim = (width, height)
    
    # resize image
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    return resized_image