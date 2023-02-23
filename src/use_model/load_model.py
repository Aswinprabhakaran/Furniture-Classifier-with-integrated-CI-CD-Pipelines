#!/usr/bin/env python
#description     :This script will will run inference/predictions on TF2 frozen graph(.pb) model & Keras model
#author          :Aswin Prabhakaran
#date            :22-Feb-2023
#version         :1.7
#usage           :python script.py <args> <args> <args>
#notes           :
#python_version  :
##############################################################################

# Necessary Imports

import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.utils.preprocess import preprocess_image, convert_opencv_to_PIL


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    
    """Load Forzen Graph in TF2"""
    
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_tensorflow_model(pb_file):

    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def = graph_def,
                                    inputs = ["x:0"],
                                    outputs = ["Identity:0"],
                                    print_graph = False)

    return frozen_func


class LoadModel:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.frozen_graph = False
        self.input_size = (224,224) # Default

    def load_model(self):

        """Function to load the model wither in keras format or in tensorflow frozen graph"""

        if not os.path.exists(self.model_path):
            raise ValueError("{} files doesn't exists".format(self.model_path))

        if self.model_path.endswith(".pb"):
            self.frozen_graph = True
            self.model = load_tensorflow_model(pb_file = self.model_path)
        elif self.model_path.endswith(".h5") or self.model_path.endswith(".hdf5"):
            self.model = load_model(self.model_path)
            self.input_size = self.model.input_shape[1:3]
        else:
            raise ValueError("Provided format not supported for loading")

    def preprocess(self, image):

        res_dict = dict()

        preproc_start = time.time()
        image = convert_opencv_to_PIL(image = image)
        altered_image = preprocess_image(image = image, inference = True, input_shape = self.input_size, 
                                        global_scaling = True, local_scaling = False, expand_dims = True)
        preproc_end = time.time()

        res_dict['preproc_time'] = preproc_end - preproc_start
        res_dict['altered_image'] = altered_image
        return res_dict

    def inference(self, altered_image):

        resdict = dict()

        det_start = time.time()
        if self.frozen_graph:
            predictions = self.model(x = tf.constant(altered_image))[0][0].numpy()
        else:
            predictions = self.model.predict(altered_image)[0]
        det_end = time.time()

        resdict['det_time'] = det_end - det_start
        resdict['predictions'] = predictions

        return resdict
