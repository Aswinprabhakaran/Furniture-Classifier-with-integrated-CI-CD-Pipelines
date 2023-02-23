# Deep Learning Based Furniture classifier and Integrated CI/CD Workflow with Docker

The AIM of this repo is to showcase the Deep learning model building capability with Automatic Docker Image Builds on Every commit to Main branch.

## 1) Model Training :

 **a) Training Data generation**

The given dataset consisit of 3 classes and 100 images in each classes. Since the number of images are not enough, we have done data augmentation to increase and generate new dataset of 1500 images.
A set of image augmentations like **Rotation, Translation, Shearing, Noise , Blur and Resizing** are applied on each image. Thus each image goes through 5 augmentations and produces 5 augmented images.

The 300 image test data provided is used only for testing and benchmarking purpose.

The [run_augmentation](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/src/data_generation/run_augmentation.py) script is used to produce the augmentated image for trainig data as follows : 

> python src/data_generation/run_augmentation.py -img_path ./data/test_data/bed/ -o ./data/augmented_training_data/bed/

**b) Model Training infused with Fine Tuning and Transfer Learning Technique:**

Now that we have accumulated 1500 images ( 500 images in each class), we start with training phase where I have introduced a custome model with the base layers are from MobilenetV2 loaded with Imagenet weigts and then the Top 5 layers are custom built for this task. This way I dont start the model training from scratch and hence achieve higher faster.

During the training phase, all the MobilenetV2 layers are frozen from training and only the head 5 layers which are custom built are trained as follows:

```
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
 ```
The [training script](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/src/build_model/train.py) also haas capabilities to load the trained model and **do stage-wise training** where layers are unfreezed depending on need allowing more flexibiloty during training.

**The model after 37 epochs achieved 97% training accuracy and 99% validation accuracy**

## 2) Building Flask API's : 

a) Now that the model is built, I have created a flask app file which encapsules the inference code of the custom built deep learning model as follows : 

```
def inference(abs_image_path):

    image = cv2.imread(abs_image_path)
    preprocess_res_dict = Model_obj.preprocess(image = image)
    inference_res_dict = Model_obj.inference(altered_image = preprocess_res_dict['altered_image'])

    print("Preprocessing TIME Taken : {} sec".format(preprocess_res_dict['preproc_time']))
    print("Inference TIME Taken : {} sec".format(inference_res_dict['det_time']))

    predictions = inference_res_dict['predictions']

    return predictions
```

b) For demo purpose, I have Also slightly toughed up on a Front-End in Flask which enables us to upload an image for the prediction/inference purposes.

![Landing Page...](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/display_images/view_1.png)

![Upload Image to be classified...](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/display_images/view_2.png)


c) The uploaded images is handled by flask forms and inference is done on the image and the results are routed back to the front end with Confidence percentage as well.

![Result Page](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/display_images/view_3.png)


## Libraries and Frameworks Used :

* Flask
* Numpy
* Opencv
* imgaug
* keras
* Tensorflow
* tqdm
* PILLOW


## CONTAINERIZATION using DOCKER

Docker Image is buit using the docker file with Python Image supported by Docker as follows

```
# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "src.use_model.app"]
```

This config seleccted the Python image and then copies the python virtual environment which we have set-up and the copies all the necessary files to run the flask application.
Once the files are copied, the python execution command is triggered.

![Docker Image](https://github.com/Aswinprabhakaran/project_fulhaus/blob/main/display_images/docker_image_built.png)