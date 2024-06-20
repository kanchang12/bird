#!/bin/bash

# Install system dependencies


# Install Python dependencies
pip install -r requirements.txt

# Start your Flask application
gunicorn --bind 0.0.0.0:$PORT app:app
