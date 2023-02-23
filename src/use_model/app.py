
import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from src.use_model.load_model import LoadModel 

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.secret_key = 'super secret key'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def inference(abs_image_path):

    image = cv2.imread(abs_image_path)
    preprocess_res_dict = Model_obj.preprocess(image = image)
    inference_res_dict = Model_obj.inference(altered_image = preprocess_res_dict['altered_image'])

    print("Preprocessing TIME Taken : {} sec".format(preprocess_res_dict['preproc_time']))
    print("Inference TIME Taken : {} sec".format(inference_res_dict['det_time']))

    predictions = inference_res_dict['predictions']

    return predictions


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello():
    return '''<h1>Welcome to the Site. The API's are alive. </h1> 
    <a href="upload_file"> Url for Uploading file for Classification </a>
    '''

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        # if user does not select file or browser also submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            result = inference(abs_image_path=save_path)
            pred_class = class_label_map[np.argmax(result)]
            pred_score = round(np.max(result)* 100, 2)
            return '''
            <h1>Inference is done on Deep Learning Model.</h1>
            <p> The Tested Image is: {} with conf: {}. </p>
            <a href="upload_file"> Url for Uploading file for Classification </a>
            '''.format(pred_class, pred_score)
        else:
            return '''<h1> Unknown File Format </h1>
            <p> Supply file of following extensions : {} </p>
            <a href="upload_file"> Url for Uploading file for Classification </a>
            '''.format(ALLOWED_EXTENSIONS)

    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value=Upload>
    </form>
    '''

if __name__ == '__main__':

    # Loading the keras h5 model trained
    Model_obj = LoadModel(model_path = "./model/weights-improvement-11-0.97.hdf5") 
    Model_obj.load_model()

    with open("./model/class_label_map.pickle", 'rb') as fin:
        class_label_map = pickle.load(fin)

    class_label_map = {value:key for key, value in class_label_map.items()}

    app.run(host="0.0.0.0", port = 3000)
