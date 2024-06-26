from flask import Flask, request, jsonify, render_template, make_response, send_file
from roboflow import Roboflow
import cv2
import numpy as np
import base64
from flask_cors import CORS
import os
from PIL import Image
import io
import requests

app = Flask(__name__)
CORS(app)

roboflow_api_key = "282K9KJQbOG4dpF69t6D"
ROBOFLOW_WORKSPACE = "bird-v2"
ROBOFLOW_VERSION = 2

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project(ROBOFLOW_WORKSPACE)
model = project.version(ROBOFLOW_VERSION).model

def add_permissions_policy_headers(response):
    response.headers['Permissions-Policy'] = 'geolocation=(self), camera=(), microphone=(), fullscreen=*, payment=()'
    return response

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    return add_permissions_policy_headers(response)

@app.route('/identify_bird', methods=['POST'])
def identify_bird():
    print("Received request to /identify_bird")
    try:
        data = request.json
        if not data or 'image_url' not in data:
            print("No image data received")
            return jsonify({"error": "No image data received"}), 400
        image_data = data.get('image_url').split(",")[1]
        
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        print("Sending image to model for prediction")
        result = model.predict(opencv_image, confidence=20, overlap=30).json()
        # Delete the image data after processing
        del image_data
        del image
        del opencv_image
        
        print("Raw prediction result:", result)  # Print raw result for debugging
        
        if result['predictions']:
            bird_name = result['predictions'][0]['class']
            confidence = result['predictions'][0]['confidence']
            print(f"Bird identified: {bird_name}, Confidence: {confidence}")
            return jsonify({"bird_name": f"{bird_name} (Confidence: {confidence:.2f})"})
        else:
            print("No bird detected")
            return jsonify({"bird_name": "No bird detected"})
    except Exception as e:
        print(f"Error in bird identification: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/proxy_thumbnail', methods=['GET'])
def proxy_thumbnail():
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return send_file(
            io.BytesIO(response.content),
            mimetype='image/jpeg',
            download_name='thumbnail.jpg'
        )
    else:
        return jsonify({"error": "Failed to fetch image"}), response.status_code

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint reached successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
