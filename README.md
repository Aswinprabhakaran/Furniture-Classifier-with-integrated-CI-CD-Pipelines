# project_fulhaus

The AIM of this repo is to showcase the Deep learning model building capability with Automatic Docker Image Builds on Every commit to Main branch.

1) Model Training :

a) Training Data generation

The given dataset consisit of 3 classes and 100 images in each classes. Since the number of images are not enough, we have done data augmentation to increase and generate new dataset of 1500 images.
A set of image augmentations like Rotation, Translation, Shearing, Noise , Blur and Resizing are applied on each image. Thus each image goes through 5 augmentations and produces 5 augmented images.

The 300 image test data provided is used only for testing and benchmarking purpose.

The scripts in src/data_generation/ folder is used to generate the augmented data.


b) Model Training : 

Now that we have accumulated 1500 images ( 500 images in each class), we start with training phase where I have introduced a customer model wit the base layers are from MobilenetV2 with Imagenet weigts and then the Top 5 layers are custom built for this task. This way I dont start the model training from scratch and hence achieve higher faster.

During the training phase, all the MobilenetV2 layers are frozen from training and only the head 5 layers which are custom built are trained.

The model after 37 epochs achieved 97% training accuracy and 99% validation accuracy.


2) Building Flask API's : 

a) Now that the model is built, I have created a flask app file which encapsules the inference code of the custom built deep learning model.

b) For demo purpose, I have also introduced a Front-End in Flask which enables us to upload an image for the prediction/inference purposes.

c) The uploaded images is handled by flask forms and inference is done on the image and the results are routed back to the front end with Confidence percentage as well.
