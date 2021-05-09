from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
#from flask import make_response,session
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
from keras.models import model_from_json


# Define a flask app
app = Flask(__name__)
#model Json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        class_dict={0:'glioma tumour',1:'meningioma tumour',2:'no tumor',3:'pituitary tumor'}
        test_img1=cv2.imread(file_path)
        test_img1=cv2.resize(test_img1,(224,224))
        test_img1=np.expand_dims(test_img1,axis=0)
        pred=loaded_model.predict(test_img1)
        pred=np.argmax(pred)
        pred_class=class_dict[pred]
        result="The Brain MRI Scan shows there is " +pred_class
        os.remove(file_path)
        return result               
    return None

if __name__ == '__main__':
   app.run(debug=True)