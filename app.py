from flask import Flask, request, jsonify, render_template, make_response
from roboflow import Roboflow
import cv2
import numpy as np
import base64
from flask_cors import CORS
import os
from PIL import Image
import io

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

@app.route('/identify_bird', methods=['GET', 'POST'])
def identify_bird():
    print("Received request to /identify_bird")
    try:
        data = request.json
        if not data or 'image_url' not in data:
            print("No image data received")
            return jsonify({"error": "No image data received"}), 400

        image_data = data.get('image_url').split(",")[1]
        print(f"Received image data of length: {len(image_data)}")

        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        print(f"Image opened. Size: {image.size}")

        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(f"Converted to OpenCV image. Shape: {opencv_image.shape}")

        print("Sending image to model for prediction")
        result = model.predict(opencv_image, confidence=40, overlap=30).json()
        print("Received result from model:", result)

        if result['predictions']:
            bird_name = result['predictions'][0]['class']
            print("Bird identified:", bird_name)
            return jsonify({"bird_name": bird_name})
        else:
            print("No bird detected")
            return jsonify({"bird_name": "No bird detected"})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint reached successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
