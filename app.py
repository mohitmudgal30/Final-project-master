from flask.helpers import send_file
from jinja2 import Template
import numpy as np 

import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

# Feature Scaling


from flask import Flask, request, jsonify, render_template



# coding=utf-8
import sys
import os
import glob
import re
import cv2
from  PIL import Image, ImageOps
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import io

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new/"  
 
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------------Data Preprocessing-------------------------------------------
# for data preprocessing
def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/downloadNewDataset')
def download_file():
    path1 = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path1,as_attachment=True)

#------------------------------Download Model-------------------------------------------
@app.route('/downloadmodel')
def download_model():
    path1 = "static/data-preprocess/model/model.pkl"
    return send_file(path1,as_attachment=True)

#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Artificial Neural network-------------------------------------------


@app.route('/ann')
def ann():
    return render_template('/ann/ann.html')

#----------------------Image Classification cat/ Dog------------------------------
model_cat = load_model("static/data-preprocess/model/FDPCNN1.h5")

def import_and_predict_cat(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(64, 64)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  result = model_cat .predict(img_reshape)
  print(result)
  #training_set.class_indices
  if result[0][0] == 1:
    prediction = "Dog" 
    
  else:
    prediction = 'Cat'
    #x = np.expand_dims(x, axis=1)
  
  
  return prediction


@app.route('/ann/cat/cat')
def cat():
    return render_template('/ann/cat/cat.html')


@app.route('/ann/cat/cat',  methods=['GET', 'POST'])
def cat1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict_cat(image)

        

        return render_template('/ann/cat/catoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )
#-----------------------Digit Recognition---------------------------------------------
model_digit = load_model("static/data-preprocess/model/MNISTANN.h5")

def import_and_predict(image_data):
  
  image_resized = cv2.resize(image_data, (28, 28)) 
   
  prediction = model_digit.predict(image_resized.reshape(1,784))
  print('Prediction Score:\n',prediction[0])
  thresholded = (prediction>0.5)*1
  print('\nThresholded Score:\n',thresholded[0])
  print('\nPredicted Digit:',np.where(thresholded == 1)[1][0])
  digit = np.where(thresholded == 1)[1][0]
  #st.image(image_data, use_column_width=True)
  return digit



@app.route('/ann/digit/digit')
def digit():
    return render_template('/ann/digit/digit.html')


@app.route('/ann/digit/digit',  methods=['GET', 'POST'])
def digit1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict(image)

        

        return render_template('/ann/digit/digitoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )


#------------------------------Convolution  Neural network-------------------------------------------


@app.route('/cnn')
def cnn():
    return render_template('/cnn/cnn.html')

#------------------------------Face Recognition-------------------------------------------
model_face = load_model("static/data-preprocess/model/Facemodel.h5")
FACE_CLASSES = ['ben_afflek', 'elton_john','jerry_seinfeld','madonna','mindy_kaling']
def predict_face(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  features = model_face.predict(img_reshape)
  
  label_index=features.argmax()
  print(label_index)
  
  
  
  return FACE_CLASSES[label_index]

@app.route('/cnn/face/face')
def face():
    return render_template('/cnn/Face/face.html')


@app.route('/cnn/face/face',  methods=['GET', 'POST'])
def face1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = predict_face(image)

        

        return render_template('/cnn/Face/faceoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )



#-------------------Flask Application--------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"

