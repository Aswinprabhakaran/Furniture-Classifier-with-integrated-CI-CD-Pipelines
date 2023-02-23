#!/usr/bin/env python

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import cv2

def convert_opencv_to_PIL (image):
    """Function to convert opencv image to PIL image"""

    # First need to convert the color as PIL reads image in RGB unlike opencv which is BGR.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil
        
def preprocess_image(image, inference=False, input_shape=None, openvino=False, local_scaling=False, expand_dims=False, global_scaling=True):
    
    """
    StandAlone preprocessing function to be used for both Trainng and Inference
    
    :param image: PIL Image
    :param inference: Flat which denotes it its training or inference
                      default to False which means its Training stage
                      
                      So during Training stage the PIL image which we get is already resised by keras 'flow_from_directory' API
                      and thus during inference we resise the image first and then begin the other preprocessing operations.
                      
    :param input_shape: Provide the input shape to be resized as per the model.
                        For Xception - (299,299)
                        For Mobilenet_V2 - (224,224)
                        
                        
    :param openvino: Default to False.
                     Set to true during inference if need to change data layout from HWC to CHW
    
           
    """
    
    if inference:
        # This resizing is done automatically by keras API during training
        # Thus for inference we do here
        image = image.resize(size=input_shape, resample=Image.NEAREST)
        
    # Keras loads in Image in PIL format. So convering it to img array
    image = img_to_array(image)
    
    if openvino:
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    
    if global_scaling:
        # Global Rescaling
        image /= 255.0
    
    if local_scaling:
        # Get the mean and standard deviation of the image
        mean = image.mean()
        std = image.std()
            
        # Subtract the mean from the image and then divide it by the standard deviation
        image -= mean
        image /= std
        
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    
    return image