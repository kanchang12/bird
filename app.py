from flask import Flask, request, jsonify, render_template
from roboflow import Roboflow
import cv2
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = "282K9KJQbOG4dpF69t6D"
ROBOFLOW_WORKSPACE = "bird-v2"
ROBOFLOW_VERSION = 2

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_WORKSPACE)
model = project.version(ROBOFLOW_VERSION).model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify_bird', methods=['POST'])
def identify_bird():
    data = request.json
    image_data = data.get('image_url').split(",")[1]

    try:
        image = base64.b64decode(image_data)
        image_array = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = model.predict(image, confidence=40, overlap=30).json()

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
