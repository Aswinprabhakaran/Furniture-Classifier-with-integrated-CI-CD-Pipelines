#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("/mnt/f/github_repos/Project_Fulhaüs/")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import numpy as np
import datetime
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import time

from src.utils.preprocess import preprocess_image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-model_arch", "--model_arch", required=False, type=str, help="Model architecture to train base model from scratch (xception,mobilenet_v2)")
ap.add_argument("-basemodel_for_TL", "--basemodel_for_TL", required=False, type=str, help="Pretrained model to do tranfer learning")
ap.add_argument("-train_last_n_layers", "--train_last_n_layers", required=False, default = 1, type=int, help="Last N layers to be trained in Pretrained model to do transfer learning (Default : 1)")
ap.add_argument("-training_data_path", "--training_data_path", required=True, type=str, help="path to training images dir")
ap.add_argument("-validation_data_path", "--validation_data_path", required=False, type=str, help="path to validation images dir")
ap.add_argument("-o", "--outputloc", required=True, type=str, help="path to output training logs & models & artifacts")
ap.add_argument("-N_CLASSES", "--N_CLASSES", required=False, type=int, default=3, help="No of classes to be trained by the classifier. (Default:7)")
ap.add_argument("-N_EPOCHS", "--N_EPOCHS", required=False, type=int, default=50, help="No of epochs to run training for. (Default:10)")
ap.add_argument("-BATCH_SIZE", "--BATCH_SIZE", required=False, type=int, default=32, help="Batch size to process during training.Should be multiples of 8. (Default:32)")
ap.add_argument("-INIT_LR", "--INIT_LR", required=False, type=float, default=1e-4, help="Optimizer regularisation parameter. (Default: 1e-4")
ap.add_argument("--replace_last_layer_in_model", default = False, action="store_true")
ap.add_argument("--train_all_layers", default = False, action="store_true")
args = vars(ap.parse_args())
print("\n[INFO] Args supplied for Training : ", args)

print("\nTensorflow Version :", tf.__version__)
print("GPU NAME : ",tf.test.gpu_device_name())

SEED = 42

# Assume that you have 12GB of GPU memory and want to allocate ~6GB:
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

start = time.time()

# More models can be extended over the coarse of time
MODELS = { "mobilenet_v2": {'model': MobileNetV2, 'input_shape': (224, 224), 'channel_order':'RGB'}}

# Config Parameters
N_CLASSES = args['N_CLASSES']
N_EPOCHS = args['N_EPOCHS']
BATCH_SIZE = args['BATCH_SIZE']
INIT_LR = args['INIT_LR']
train_last_n_layers = args['train_last_n_layers']


if args['basemodel_for_TL']:
    
    if args['replace_last_layer_in_model']:
        
        # Loading the pretrained model to do transfer learning and this is the model that we will train
        source_model = load_model(args['basemodel_for_TL'])
        source_model._layers.pop()
        headModel = source_model.layers[-1].output
        headModel = Dense(N_CLASSES, activation="sigmoid", name="last_dense_layer")(headModel) # Using Sigmoid makes the output probabilities independent of each other (not summing to 1)

        # Create a new model and this is the model that we will train
        model = Model(inputs=source_model.input, outputs=headModel)
    else:
        # Loading the pretrained model to do transfer learning and this is the model that we will train
        model = load_model(args['basemodel_for_TL'])

    print("Final Model Summary")
    print(model.summary())
        
    if args['train_all_layers']:
        # Unfreeze all years in the model to be trained
        for layer in model.layers:
            layer.trainable = True
    else:
        # Freeze all years in the model before hand
        for layer in model.layers:
            layer.trainable = False

        # Unfreeze the last N layers set in args to be trained
        for layer in model.layers[-train_last_n_layers:]:
            layer.trainable = True

    # Display
    for layer in model.layers:
        print("Layer Name : {}, Layer Trainable? : {} ".format(layer.name, layer.trainable))
    
    # Get the model input size for the pretrained model from model itself
    img_height, img_width = model.input_shape[1:3]
    
elif args['model_arch']:
    
    model_arch = args['model_arch'].lower()
    
    # esnure a valid model name was supplied via command line argument
    if model_arch not in MODELS.keys():
        raise ValueError("Unknown/Unsupported model architecture - {}".format(model_arch))
        
    img_height, img_width = MODELS[model_arch]['input_shape']
    
    # Define Base Model
    input_tensor = Input(shape=(img_height, img_width, 3))
    base_model = MODELS[model_arch]['model'](weights='imagenet',
                                             include_top=False,
                                             input_shape=None,
                                             input_tensor=input_tensor)
    
    # Freeze all layers from base model
    for layer in base_model.layers:
        layer.trainable = False #TODO : we should also allow training of base model layers after the top layers are trained. 
                                # Training should be multi-stage. First - top layers are trained and then whole network

    # Attach our model on top of base model
    headModel = base_model.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel) #TODO : This should be one of the hyperparameters to tune up with large data
    headModel = Flatten(name="flatten")(headModel) #TODO : This should be one of the hyperparameters to tune up with large data
    headModel = Dense(256, activation="relu")(headModel) #TODO : This should be one of the hyperparameters to tune up with large data
    headModel = Dropout(0.5)(headModel) #TODO : This should be one of the hyperparameters to tune up with large data
    headModel = Dense(N_CLASSES, activation="softmax")(headModel) # Using Sigmoid makes the output probabilities independent of each other (not summing to 1)

    # Create a new model and this is the model that we will train
    model = Model(inputs=base_model.input, outputs=headModel)

    # Display
    for layer in model.layers:
        print("Layer Name : {}, Layer Trainable? : {} ".format(layer.name, layer.trainable))

else:
    raise AssertionError("Either provide base model for Transfer learning or choose model architecture to train from scratch")
    

## Setting up images path for model training
training_images_dir = args['training_data_path']

if args['validation_data_path']:
    
    validation_images_dir = args['validation_data_path']
    
    # Data Loader API , Load images from data folder and create generator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    
    # Training Images generator
    train_generator = datagen.flow_from_directory(training_images_dir, target_size=(img_height, img_width),
                                                  batch_size=32, class_mode='categorical', seed=SEED)

    # Validation Images generator
    validation_generator = datagen.flow_from_directory(validation_images_dir, target_size=(img_height, img_width),
                                                       batch_size=32, class_mode='categorical', seed=SEED)
    
else:
    
    # Data Loader API , Load images from data folder and create generator
    datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_image)

    # Training Images generator
    train_generator = datagen.flow_from_directory(training_images_dir, target_size=(img_height, img_width),
                                                  batch_size=32, class_mode='categorical', subset='training',
                                                  seed=SEED)

    # Validation Images generator
    validation_generator = datagen.flow_from_directory(training_images_dir, target_size=(img_height, img_width),
                                                       batch_size=32, class_mode='categorical', subset='validation',
                                                       seed=SEED)

# Setting the out directory to dump the files to
logdir = args['outputloc']
if not os.path.exists(logdir):
    print("Creating: {}".format(logdir))
    os.makedirs(logdir)

# Setting Tensorboard Callbacks
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

## Dumping Class name and its associate labels
class_label_map = train_generator.class_indices

with open(os.path.join(logdir, "class_label_map.pickle"), 'wb') as f:
    pickle.dump(class_label_map, f)

## Callback for early stopping the training
early_stopping = EarlyStopping(monitor='val_accuracy', 
                               restore_best_weights = True,
                               min_delta=0,
                               patience=15,
                               verbose=0, 
                               mode='max')

## Checkpoint for saving the best weights during training
checkpoint_filepath = os.path.join(logdir, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='val_accuracy', 
                             verbose = 1, 
                             save_best_only=True, 
                             mode='max')

# Compiling the model with loss functions & optimizers
model.compile(loss=CategoricalCrossentropy(), 
              optimizer=optimizers.Adam(learning_rate = INIT_LR),
              metrics=['accuracy'])

# Train model
history = model.fit(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, 
                    epochs=N_EPOCHS, validation_data=validation_generator, 
                    validation_steps = validation_generator.samples // BATCH_SIZE,
                    callbacks = [tensorboard_callback, checkpoint, early_stopping])


# Plotting Model Training Status
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.savefig('{}/trs_and_val_acc_and_loss_plot.jpg'.format(logdir))

model.save("{}/final_trained_model.h5".format(logdir))

end = time.time()

str_to_display = "Training Time Taken : {} Mins".format(round((end - start)/60))

print("\n", str_to_display)
print("\nTraining Completed !!")