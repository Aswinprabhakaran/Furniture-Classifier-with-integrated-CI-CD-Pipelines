#!/usr/bin/env python

import cv2
import numpy as np
# np.random.bit_generator = np.random._bit_generator
import random
import imgaug.augmenters as iaa
import os
import ast
import augmentation_script as img_aug
import argparse


# Functions to perform 3 types of augmentation on single image
def pipeline_RTShear(image):

    rotated_image= img_aug.Rotate(image.copy())
    translated_image = img_aug.translate(rotated_image)
    sheared_image = img_aug.shearing(translated_image)
    
    return sheared_image


def pipeline_RTResise(image):

    rotated_image = img_aug.Rotate(image.copy())
    translated_image = img_aug.translate(rotated_image)
    Resize_image = img_aug.Resize(translated_image)

    return Resize_image

def pipeline_RTBlur(image):

    rotated_image = img_aug.Rotate(image.copy())
    translated_image = img_aug.translate(rotated_image)
    
    blur_aug = iaa.MotionBlur(k = 10)
    blured_image = blur_aug(image = translated_image)
    
    return blured_image

def pipeline_RTSharp(image):

    rotated_image = img_aug.Rotate(image.copy())
    translated_image = img_aug.translate(rotated_image)
    
    sharp_aug = iaa.Sharpen(alpha = random.sample(list(np.arange(11)/10),2))
    sharped_image = sharp_aug( image = translated_image)
    
    return sharped_image


def pipeline_RTNoise(image):

    rotated_image = img_aug.Rotate(image.copy())
    translated_image = img_aug.translate(rotated_image)
    
    noise_aug = iaa.AdditiveGaussianNoise(loc = (0.0, 0.1*255),
                                      scale=(0.0, 0.1*255),
                                      per_channel = True)
    
    noised_image = noise_aug( image = translated_image)
    
    return noised_image



if __name__ == "__main__":

    # # Augmentations
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_path', '--images_path',required = True , type = str , help = 'path to images to augment')
    ap.add_argument('-o', '--output_path',required = True , type = str , help = 'path to save the Augmented annotations and images')
    args = vars(ap.parse_args())
    print(args)

    if not os.path.exists(args['output_path']):
        # os.makedirs(os.path.join(args['output_path'], 'RTShear'))
        # os.makedirs(os.path.join(args['output_path'], 'RTResize'))
        # os.makedirs(os.path.join(args['output_path'], 'RTBlur'))
        # os.makedirs(os.path.join(args['output_path'], 'RTNoise')) 
        # os.makedirs(os.path.join(args['output_path'], 'RTSharp'))
        os.makedirs(args['output_path'])

    # Reading the images form the images path
    images_to_aug = os.listdir(args['images_path'])
    print("\nNo of Images to Augment :", len(images_to_aug))

    for index, img in enumerate(sorted(images_to_aug)):
        
        image = cv2.imread(os.path.join(args['images_path'], img))
        
        # 1st Augmentation Pipeline
        p1_img= pipeline_RTShear(image =  image.copy())
        # cv2.imwrite('{}/RTShear/RTShear_{}'.format(args['output_path'], img), p1_img)
        cv2.imwrite('{}/RTShear_{}'.format(args['output_path'], img), p1_img)

        # 2nd Augmentation Pipeline
        p2_img = pipeline_RTResise(image.copy())
        # cv2.imwrite('{}/RTResize/RTResize_{}'.format(args['output_path'], img), p2_img)
        cv2.imwrite('{}/RTResize_{}'.format(args['output_path'], img), p2_img)

        #3rd Augmentation Pipeline        
        p3_img = pipeline_RTBlur(image =  image.copy())
        # cv2.imwrite('{}/RTBlur/RTBlur_{}'.format(args['output_path'], img), p3_img)
        cv2.imwrite('{}/RTBlur_{}'.format(args['output_path'], img), p3_img)

        #4th Augmentation Pipeline        
        p4_img = pipeline_RTNoise(image =  image.copy())
        # cv2.imwrite('{}/RTNoise/RTNoise_{}'.format(args['output_path'], img), p4_img)
        cv2.imwrite('{}/RTNoise_{}'.format(args['output_path'], img), p4_img)

        # 5th Augmentation Pipeline        
        p5_img = pipeline_RTSharp(image =  image.copy())
        # cv2.imwrite('{}/RTSharp/RTSharp_{}'.format(args['output_path'], img), p5_img)
        cv2.imwrite('{}/RTSharp_{}'.format(args['output_path'], img), p5_img)
        
        if index % 10 == 0:
            print("Processed = ", index)