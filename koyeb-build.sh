#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx

# Install Python dependencies
pip install -r requirements.txt

# Start your Flask application
gunicorn --bind 0.0.0.0:$PORT app:app
