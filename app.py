import urllib.request
from flask import Flask, request, jsonify
from keras.models import load_model
import keras.backend as K
import gc
import numpy as np
from PIL import Image
import io
import requests

app = Flask(__name__)

@app.route('/', methods=['get'])
def index():
	return '<h3>Documentation</h3> <p>URL request to : /predict</p> <p>Method : POST</p> <p>Format : JSON</p> <p>Example request: {"image":"https://d1d8o7q9jg8pjk.cloudfront.net/p/lg_5e7def84b7ad9.jpg"}</p><br/> <h3>Info</h3> <p>1 : "Aglonema", 2 : "Bonsai", 3 : "Matahari",4 : "Kalamansi"</p>'

@app.route('/predict', methods=['post'])
def upload_file():
	url = request.get_json()
	imgfromurl = requests.get(url['image'])
	img = Image.open(io.BytesIO(imgfromurl.content))
	if img.mode != 'RGB':
		img = img.convert('RGB')
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)
	img = img.astype('float32')
	img /= 255
	model = load_model('model.h5', compile = True)
	prediction = model(img)
	gc.collect()
	K.clear_session()
	classes = np.argmax(prediction)
	resp = jsonify({'prediction' : int(classes)+1})
	resp.status_code = 201
	
	return resp