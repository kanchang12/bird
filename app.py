from flask import Flask, request, jsonify, render_template
from roboflow import Roboflow
import base64
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 


roboflow_api_key = "282K9KJQbOG4dpF69t6D"
ROBOFLOW_WORKSPACE = "bird-v2"
ROBOFLOW_VERSION = 2

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project(ROBOFLOW_WORKSPACE)
model = project.version(ROBOFLOW_VERSION).model


@app.before_request
def before_request():
    request.headers['Permissions-Policy'] = 'geolocation=(self), camera=(), microphone=(), fullscreen=*, payment=()'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify_bird', methods=['POST'])
def identify_bird():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']

    try:
        # Read the image file
        image_data = image_file.read()
        
        # Perform bird identification
        result = model.predict(image_data, confidence=40, overlap=30).json()

        if result['predictions']:
            bird_name = result['predictions'][0]['class']
            print(bird_name)
            return jsonify({"bird_name": bird_name})
        else:
            return jsonify({"bird_name": "No bird detected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
