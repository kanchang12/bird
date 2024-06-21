from flask import Flask, request, jsonify, render_template, make_response
from roboflow import Roboflow
import cv2
import numpy as np
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
        print("Image data length:", len(image_data))

        image = base64.b64decode(image_data)
        image_array = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = model.predict(image, confidence=40, overlap=30).json()
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
