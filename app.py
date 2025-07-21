import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import faiss
import json
from werkzeug.utils import secure_filename
from meal_sample import load_and_train_model, predict_pcos_type
import pickle 
from datetime import datetime, timedelta
import io

def create_model(num_outputs):
    resnet = models.resnet18(weights=None)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    model = nn.Sequential(
        feature_extractor,
        nn.Flatten(),
        nn.Linear(512, num_outputs)
    )
    return model

model = None
scaler = None
feature_names = []
column_types = {}
excel_df = None
result_data = None
# Image preprocessing (same as test.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['IMAGE_FOLDER'] = 'uploaded_images'
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# Load the model when the application starts
# print("Loading model...")
# model, scaler, feature_names, column_types = load_and_train_model("PCOS profilling.xlsx")
# Initialize global variables


# Define PCOS types mapping based on actual data
PCOS_TYPES = {
    0: 'PCOS Adrenal',
    1: 'PCOS Keradangan/Inflammation',
    2: 'PCOS Keradangan/Infllammation',
    3: 'PCOS Pil Perancang/Post Birth Control',
    4: 'PCOS Pos Pil Perancang/Post Birth Control',
    5: 'PCOS Rintangan Insulin/Insulin Resistance'
}
MEAL_TYPES = {
    0: 'Build muscle',
    1: 'Improve blood sugar regulation',
    2: 'Improve fertility',
    3: 'Lose weight'
}
PCOS_TYPES_FERTILITY = {
    0: 'Negative',
    1: 'Positive',
    
}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@app.route('/', methods=['POST'])
upload_file()


# @app.route('/female_predict', methods=['POST'])



# @app.route('/male_predict', methods=['POST'])



# @app.route('/image', methods=['POST'])



# @app.route('/image_predict', methods=['POST'])



# @app.route('/send', methods=['GET', 'POST'])



# @app.route('/fertility_predict', methods=['POST'])



# @app.route('/receive_data', methods=['POST'])
# receive_data()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True,use_reloader=False) 