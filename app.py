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
from utils import load_and_train_model, predict_pcos_type, allowed_file, load_pickle, allowed_image_file, calculate_ovulation_day, calculate_fertile_window, analyze_cycle_regularity, calculate_conception_probability
import pickle 
from datetime import datetime, timedelta
import io
from Meal_plan.male_meal_plans import get_meal_plan
from Meal_plan.female_meal_plans import get_female_meal_plan

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

def preprocess_female_data(input_data):
    """
    Convert text values from JSON input to numeric values expected by the model
    """
    processed_data = {}
    
    # Copy basic numeric fields as-is
    processed_data['name'] = input_data.get('name', '')
    processed_data['age'] = input_data.get('age', '')
    processed_data['height'] = input_data.get('height', '')
    processed_data['weight'] = input_data.get('weight', '')
    processed_data['How many hour you sleep per day?'] = input_data.get('How many hour you sleep per day?', '')
    
    # Convert 'Have you take the Quiz of PCOS Type?'
    pcos_quiz = input_data.get('Have you take the Quiz of PCOS Type?', '')
    if pcos_quiz.lower() == 'no':
        processed_data['Have you take the Quiz of PCOS Type?'] = 0
    elif pcos_quiz.lower() == 'yes':
        processed_data['Have you take the Quiz of PCOS Type?'] = 1
    else:
        processed_data['Have you take the Quiz of PCOS Type?'] = pcos_quiz
    
    # Convert 'Status'
    status = input_data.get('Status', '')
    if status.lower() == 'married':
        processed_data['Status'] = 0
    elif status.lower() == 'single':
        processed_data['Status'] = 1
    else:
        processed_data['Status'] = status
    
    # Convert 'Are you currently planning to get pregnant?'
    pregnant_plan = input_data.get('Are you currently planning to get pregnant?', '')
    if pregnant_plan.lower() == 'no':
        processed_data['Are you currently planning to get pregnant?'] = 0
    elif pregnant_plan.lower() == 'yes':
        processed_data['Are you currently planning to get pregnant?'] = 2
    else:
        processed_data['Are you currently planning to get pregnant?'] = pregnant_plan
    
    # Convert 'Do you use birth control pills?'
    birth_control = input_data.get('Do you use birth control pills?', '')
    if 'currently' in birth_control.lower():
        processed_data['Do you use birth control pills?'] = 0
    elif '3-6 months before' in birth_control.lower():
        processed_data['Do you use birth control pills?'] = 1
    elif 'never consume' in birth_control.lower():
        processed_data['Do you use birth control pills?'] = 2
    else:
        processed_data['Do you use birth control pills?'] = birth_control
    
    # Convert 'Do you take breakfast daily?'
    breakfast = input_data.get('Do you take breakfast daily?', '')
    if breakfast.lower() == 'no':
        processed_data['Do you take breakfast daily?'] = 0
    elif breakfast.lower() == 'sometimes':
        processed_data['Do you take breakfast daily?'] = 1
    elif breakfast.lower() == 'yes':
        processed_data['Do you take breakfast daily?'] = 2
    else:
        processed_data['Do you take breakfast daily?'] = breakfast
    
    # Convert 'What time do you take you breakfast?'
    breakfast_time = input_data.get('What time do you take you breakfast?', '')
    # Extract hour from time format (e.g., "08:00" -> "08")
    if ':' in breakfast_time:
        breakfast_time = breakfast_time.split(':')[0]
    time_mapping = {
        '00': 0, '06': 1, '07': 2, '08': 3, '09': 4, '10': 5, '11': 6, '12': 7
    }
    processed_data['What time do you take you breakfast?'] = time_mapping.get(breakfast_time, 8)  # 8 for nan
    
    # Convert 'Do you take lunch daily?'
    lunch = input_data.get('Do you take lunch daily?', '')
    if lunch.lower() == 'no':
        processed_data['Do you take lunch daily?'] = 0
    elif lunch.lower() == 'sometimes':
        processed_data['Do you take lunch daily?'] = 1
    elif lunch.lower() == 'yes':
        processed_data['Do you take lunch daily?'] = 2
    else:
        processed_data['Do you take lunch daily?'] = lunch
    
    # Convert 'What time do you take your lunch?'
    lunch_time = input_data.get('What time do you take your lunch?', '')
    # Extract hour from time format (e.g., "13:00" -> "13")
    if ':' in lunch_time:
        lunch_time = lunch_time.split(':')[0]
    lunch_time_mapping = {
        '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '09': 5, '10': 6, '11': 7,
        '12': 8, '13': 9, '14': 10, '15': 11, '16': 12, '22': 13
    }
    processed_data['What time do you take your lunch?'] = lunch_time_mapping.get(lunch_time, 14)  # 14 for nan
    
    # Convert 'Do you take dinner daily?'
    dinner = input_data.get('Do you take dinner daily?', '')
    if dinner.lower() == 'no':
        processed_data['Do you take dinner daily?'] = 0
    elif dinner.lower() == 'sometimes':
        processed_data['Do you take dinner daily?'] = 1
    elif dinner.lower() == 'yes':
        processed_data['Do you take dinner daily?'] = 2
    else:
        processed_data['Do you take dinner daily?'] = dinner
    
    # Convert 'What time do you take your dinner?'
    dinner_time = input_data.get('What time do you take your dinner?', '')
    # Extract hour from time format (e.g., "18:00" -> "18")
    if ':' in dinner_time:
        dinner_time = dinner_time.split(':')[0]
    dinner_time_mapping = {
        '00': 0, '05': 1, '06': 2, '07': 3, '08': 4, '09': 5, '10': 6, '12': 7,
        '14': 8, '17': 9, '18': 10, '19': 11, '20': 12, '21': 13, '22': 14, '23': 15
    }
    processed_data['What time do you take your dinner?'] = dinner_time_mapping.get(dinner_time, 16)  # 16 for nan
    
    # Convert 'How many  of water you drink per day?'
    water = input_data.get('How many  of water you drink per day?', '')
    water_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '6': 4}
    processed_data['How many  of water you drink per day?'] = water_mapping.get(str(water), 5)  # 5 for nan
    
    # Convert 'Are you exercise?'
    exercise = input_data.get('Are you exercise?', '')
    if exercise.lower() == 'no':
        processed_data['Are you exercise?'] = 0
    elif exercise.lower() == 'yes':
        processed_data['Are you exercise?'] = 1
    else:
        processed_data['Are you exercise?'] = 2  # nan
    
    # Convert 'How do you travel to work?'
    travel = input_data.get('How do you travel to work?', '')
    if travel.lower() == 'driving':
        processed_data['How do you travel to work?'] = 0
    elif travel.lower() == 'public transport':
        processed_data['How do you travel to work?'] = 1
    else:
        processed_data['How do you travel to work?'] = 2  # nan
    
    # Convert 'What time you are sleeping?'
    sleep_time = input_data.get('What time you are sleeping?', '')
    # Extract hour from time format (e.g., "22:00" -> "22")
    if ':' in sleep_time:
        sleep_time = sleep_time.split(':')[0]
    sleep_time_mapping = {
        '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '09': 6, '10': 7,
        '11': 8, '12': 9, '20': 10, '21': 11, '22': 12, '23': 13
    }
    processed_data['What time you are sleeping?'] = sleep_time_mapping.get(sleep_time, 14)  # 14 for nan
    
    # Convert 'Do you have an issue to fall asleep?'
    sleep_issue = input_data.get('Do you have an issue to fall asleep?', '')
    if sleep_issue.lower() == 'no':
        processed_data['Do you have an issue to fall asleep?'] = 0
    elif sleep_issue.lower() == 'sometimes':
        processed_data['Do you have an issue to fall asleep?'] = 1
    elif sleep_issue.lower() == 'yes':
        processed_data['Do you have an issue to fall asleep?'] = 2
    else:
        processed_data['Do you have an issue to fall asleep?'] = sleep_issue
    
    # Convert 'Can you make time to exercise 3 times a week?'
    exercise_3_times = input_data.get('Can you make time to exercise 3 times a week?', '')
    if exercise_3_times.lower() == 'no':
        processed_data['Can you make time to exercise 3 times a week?'] = 0
    elif exercise_3_times.lower() == 'sometimes':
        processed_data['Can you make time to exercise 3 times a week?'] = 1
    elif exercise_3_times.lower() == 'yes':
        processed_data['Can you make time to exercise 3 times a week?'] = 2
    else:
        processed_data['Can you make time to exercise 3 times a week?'] = exercise_3_times
    
    # Convert 'Do you have problems with irregular menstruation?'
    irregular_menstruation = input_data.get('Do you have problems with irregular menstruation?', '')
    if 'every month' in irregular_menstruation.lower() and 'too much' in irregular_menstruation.lower():
        processed_data['Do you have problems with irregular menstruation?'] = 0
    elif 'irregular' in irregular_menstruation.lower():
        processed_data['Do you have problems with irregular menstruation?'] = 1
    elif 'no' in irregular_menstruation.lower():
        processed_data['Do you have problems with irregular menstruation?'] = 2
    else:
        processed_data['Do you have problems with irregular menstruation?'] = irregular_menstruation
    
    # Convert 'Do you take any supplement or doctor medication at this moment?'
    supplement = input_data.get('Do you take any supplement or doctor medication at this moment?', '')
    if supplement.lower() == 'no':
        processed_data['Do you take any supplement or doctor medication at this moment?'] = 0
    elif supplement.lower() == 'sometimes':
        processed_data['Do you take any supplement or doctor medication at this moment?'] = 1
    elif supplement.lower() == 'yes':
        processed_data['Do you take any supplement or doctor medication at this moment?'] = 2
    else:
        processed_data['Do you take any supplement or doctor medication at this moment?'] = supplement
    
    return processed_data


def preprocess_male_data(input_data):
    """
    Convert text values from JSON input to numeric values expected by the male model
    """
    processed_data = {}
    
    # Copy basic numeric fields as-is
    processed_data['name'] = input_data.get('name', '')
    processed_data['age'] = input_data.get('age', '')
    processed_data['height'] = input_data.get('height', '')
    processed_data['weight'] = input_data.get('weight', '')
    
    # Convert 'How many Liters of water your drink per day?'
    water = input_data.get('How many Liters of water your drink per day?/  Berapa banyak liter air yang anda minum setiap hari?', '')
    water_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
    processed_data['How many Liters of water your drink per day?/  Berapa banyak liter air yang anda minum setiap hari?'] = water_mapping.get(str(water), 0)
    
    # Convert 'Lifestyle'
    lifestyle = input_data.get('Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):', '')
    if 'sedentary' in lifestyle.lower():
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = 0
    elif 'lightly active' in lifestyle.lower():
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = 1
    elif 'moderately active' in lifestyle.lower():
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = 2
    elif 'very active' in lifestyle.lower():
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = 3
    elif 'extremely active' in lifestyle.lower():
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = 4
    else:
        processed_data['Lifestyle (choose all that apply)/Gaya hidup (pilih semua yang berkenaan):'] = lifestyle
    
    # Convert 'Have you completed a sperm concentration Test Kit?'
    sperm_test = input_data.get('Have you completed a sperm concentration Test Kit?/Adakah anda telah menyelesaikan Kit Ujian Konsentrasi Sperma?', '')
    if sperm_test.lower() == 'no':
        processed_data['Have you completed a sperm concentration Test Kit?/Adakah anda telah menyelesaikan Kit Ujian Konsentrasi Sperma?'] = 0
    elif sperm_test.lower() == 'yes':
        processed_data['Have you completed a sperm concentration Test Kit?/Adakah anda telah menyelesaikan Kit Ujian Konsentrasi Sperma?'] = 1
    else:
        processed_data['Have you completed a sperm concentration Test Kit?/Adakah anda telah menyelesaikan Kit Ujian Konsentrasi Sperma?'] = sperm_test
    
    # Convert 'Do you have any specific dietary preferences or restrictions?'
    diet = input_data.get('Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?', '')
    if diet.lower() == 'no restrictions':
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = 0
    elif diet.lower() == 'vegetarian':
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = 1
    elif diet.lower() == 'vegan':
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = 2
    elif diet.lower() == 'keto':
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = 3
    elif diet.lower() == 'paleo':
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = 4
    else:
        processed_data['Do you have any specific dietary preferences or restrictions?/Adakah anda mempunyai sebarang pilihan atau sekatan diet tertentu?'] = diet
    
    # Convert 'What time do you typically have breakfast?'
    breakfast_time = input_data.get('What time do you typically have breakfast?/  Pukul berapa anda biasanya sarapan?', '')
    # Extract hour from time format (e.g., "12:00" -> "12")
    if ':' in breakfast_time:
        breakfast_time = breakfast_time.split(':')[0]
    breakfast_time_mapping = {
        '06': 0, '07': 1, '08': 2, '09': 3, '10': 4, '11': 5, '12': 6, '13': 7, '14': 8
    }
    processed_data['What time do you typically have breakfast?/  Pukul berapa anda biasanya sarapan?'] = breakfast_time_mapping.get(breakfast_time, 9)  # 9 for nan
    
    # Convert 'What time do you typically have dinner?'
    dinner_time = input_data.get('What time do you typically have dinner?/  Pukul berapa anda biasanya makan malam?', '')
    # Extract hour from time format (e.g., "19:00" -> "19")
    if ':' in dinner_time:
        dinner_time = dinner_time.split(':')[0]
    dinner_time_mapping = {
        '17': 0, '18': 1, '19': 2, '20': 3, '21': 4, '22': 5, '23': 6
    }
    processed_data['What time do you typically have dinner?/  Pukul berapa anda biasanya makan malam?'] = dinner_time_mapping.get(dinner_time, 7)  # 7 for nan
    
    # Convert 'How many meals do you typically eat per day?'
    meals = input_data.get('How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?', '')
    if '1 meal' in meals.lower():
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = 0
    elif '2 meals' in meals.lower():
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = 1
    elif '3 meals' in meals.lower():
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = 2
    elif '4 meals' in meals.lower():
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = 3
    elif '5 meals' in meals.lower():
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = 4
    else:
        processed_data['How many meals do you typically eat per day?/  Berapa banyak hidangan yang biasanya anda makan dalam sehari?'] = meals
    
    # Convert 'How often do you consume alcohol?'
    alcohol = input_data.get('How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?', '')
    if alcohol.lower() == 'never':
        processed_data['How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?'] = 0
    elif alcohol.lower() == 'monthly':
        processed_data['How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?'] = 1
    elif alcohol.lower() == 'weekly':
        processed_data['How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?'] = 2
    elif alcohol.lower() == 'daily':
        processed_data['How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?'] = 3
    else:
        processed_data['How often do you consume alcohol?/  Seberapa kerap anda meminum alkohol?'] = alcohol
    
    # Convert 'How many hours of sleep do you typically get per night?'
    sleep_hours = input_data.get('How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?', '')
    if '5 - 6 hours' in sleep_hours.lower():
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = 0
    elif '6 - 7 hours' in sleep_hours.lower():
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = 1
    elif '7 - 8 hours' in sleep_hours.lower():
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = 2
    elif '8 - 9 hours' in sleep_hours.lower():
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = 3
    elif '9+ hours' in sleep_hours.lower():
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = 4
    else:
        processed_data['How many hours of sleep do you typically get per night?/  Berapakah jam tidur yang biasanya anda dapatkan setiap malam?'] = sleep_hours
    
    # Convert 'Do you have any difficulty falling asleep or staying asleep?'
    sleep_difficulty = input_data.get('Do you have any difficulty falling asleep or staying asleep?/  Adakah anda mengalami kesukaran untuk tidur atau terus tidur?', '')
    if sleep_difficulty.lower() == 'no':
        processed_data['Do you have any difficulty falling asleep or staying asleep?/  Adakah anda mengalami kesukaran untuk tidur atau terus tidur?'] = 0
    elif sleep_difficulty.lower() == 'sometimes':
        processed_data['Do you have any difficulty falling asleep or staying asleep?/  Adakah anda mengalami kesukaran untuk tidur atau terus tidur?'] = 1
    elif sleep_difficulty.lower() == 'yes':
        processed_data['Do you have any difficulty falling asleep or staying asleep?/  Adakah anda mengalami kesukaran untuk tidur atau terus tidur?'] = 2
    else:
        processed_data['Do you have any difficulty falling asleep or staying asleep?/  Adakah anda mengalami kesukaran untuk tidur atau terus tidur?'] = sleep_difficulty
    
    return processed_data


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    print("Upload request received")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    gender = request.form.get('gender')
    print(f"Received gender: {gender}")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Attempting to save file to: {filepath}")
        try:
            file.save(filepath)
            print("File saved successfully")
            
            # Reload the model with the new data
            #global model, scaler, feature_names, column_types
            # Reset globals to None (optional, for clarity)
            model = scaler = feature_names = column_types = None
            print("Loading and training model...")
            model, scaler, feature_names, column_types = load_and_train_model(filepath)
            
            # Save model and related objects
            if gender == 'male':
                with open('uploads/male_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('uploads/male_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                with open('uploads/male_feature_names.pkl', 'wb') as f:
                    pickle.dump(feature_names, f)
                with open('uploads/male_column_types.pkl', 'wb') as f:
                    pickle.dump(column_types, f)
            elif gender == 'female':
                with open('uploads/female_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('uploads/female_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                with open('uploads/female_feature_names.pkl', 'wb') as f:
                    pickle.dump(feature_names, f)
                with open('uploads/female_column_types.pkl', 'wb') as f:
                    pickle.dump(column_types, f)
            
            # Format/validate globals
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            if model is None or scaler is None or not feature_names:
                raise ValueError("Model, scaler, or feature_names not properly loaded.")
            
            print("Model updated successfully")
            if gender == 'female':
                return jsonify({'message': 'File uploaded and model updated successfully', 'file': " uploads/female_model.pkl"})
            else:
                return jsonify({'message': 'File uploaded and model updated successfully', 'file': " uploads/male_model.pkl"})
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    print("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/female_predict', methods=['POST'])
def predict():
    try:
        # Load model and related objects from file
        female_model = load_pickle('uploads/female_model.pkl')
        female_features = load_pickle('uploads/female_feature_names.pkl')
        female_scaler = load_pickle('uploads/female_scaler.pkl')
        
        # Get JSON data from request
        input_data = request.get_json() 
        
        # Preprocess the data to convert text values to numeric values
        processed_data = preprocess_female_data(input_data)
        
        #print(f"Original input_data: {input_data}")
        #print(f"Processed data: {processed_data}")
        
        # Convert to DataFrame and select only the features needed by the model
        input_df = pd.DataFrame([processed_data])
        input_df = input_df[female_features] 

        #For Model
        # input_data = request.get_json() 
        # print(f"input_data: {input_data}")
        # input_df = pd.DataFrame([input_data])
        # input_df = input_df[features] 
        # Make prediction
        prediction, type_probabilities = predict_pcos_type(input_df, female_model, female_scaler)
                
        # Get the PCOS type name
        pcos_type = PCOS_TYPES.get(prediction, 'Unknown Type')
        print(f"prediction: {pcos_type}, type_probabilities:{type_probabilities }")
        # Include meal plan data based on the prediction type
        response_data = {
            'name': processed_data['name'],
            'height': processed_data['height'],
            'age': processed_data['age'],
            'weight': processed_data['weight'],
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        }
        if pcos_type in ["PCOS Rintangan Insulin", "PCOS Adrenal", "PCOS Keradangan/ Inflamantion", "PCOS Pil Perancang/ Post Birth Control", "PCOS Keradangan/ Infllamantion", "PCOS Rintangan Insulin/Insulin Resistance"]:
            # Get meal plan data for the predicted goal
            meal_plan = get_female_meal_plan(pcos_type)
            response_data['meal_plan'] = meal_plan
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/male_predict', methods=['POST'])
def meal_predict():
    try:
        # Load model and related objects from file
        male_model = load_pickle('uploads/male_model.pkl')
        male_features = load_pickle('uploads/male_feature_names.pkl')
        male_scaler = load_pickle('uploads/male_scaler.pkl')
        
        # Get JSON data from request
        input_data = request.get_json() 
        
        # Preprocess the data to convert text values to numeric values
        processed_data = preprocess_male_data(input_data)
        
        #print(f"Original input_data: {input_data}")
        #print(f"Processed data: {processed_data}")
        
        # Convert to DataFrame and select only the features needed by the model
        input_df = pd.DataFrame([processed_data])
        input_df = input_df[male_features] 
        #For Model
        # input_data = request.get_json() 
        # print(f"input_data: {input_data}")
        # input_df = pd.DataFrame([input_data])
        # input_df = input_df[features] 
        # Make prediction
        prediction, type_probabilities = predict_pcos_type(input_df, male_model, male_scaler)
        
        # Get the PCOS type name
        pcos_type = MEAL_TYPES.get(prediction, 'Unknown Type')
        print(f"prediction: {pcos_type}, type_probabilities:{type_probabilities }")
        
        # Prepare response data
        response_data = {
            'name': processed_data['name'],
            'height': processed_data['height'],
            'age': processed_data['age'],
            'weight': processed_data['weight'],
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        }
        
        # Include meal plan data based on the prediction type
        if pcos_type in ["Improve fertility", "Improve blood sugar regulation", "Lose weight", "Build muscle"]:
            # Get meal plan data for the predicted goal
            meal_plan = get_meal_plan(pcos_type)
            response_data['meal_plan'] = meal_plan
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/image', methods=['POST'])
def image_upload_file1():
    print("Upload request received")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Attempting to save file to: {filepath}")
        try:
            file.save(filepath)
            print("File saved successfully")
            
            # Reload the model with the new data
            # global excel_df
            excel_df = pd.read_excel(filepath)
            with open('uploads/image_excel_df.pkl', 'wb') as f:
                pickle.dump(excel_df, f)
            # Reset globals to None (optional, for clarity)
            # model = scaler = feature_names = column_types = None
            model = scaler = feature_names = column_types = None
            print("Loading and training model...")
            model, scaler, feature_names, column_types = load_and_train_model(filepath)
            with open('uploads/image_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('uploads/image_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open('uploads/image_feature_names.pkl', 'wb') as f:
                pickle.dump(feature_names, f)
            with open('uploads/image_column_types.pkl', 'wb') as f:
                pickle.dump(column_types, f)
            # Format/validate globals
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
                # print(feature_names)
            if model is None or scaler is None or not feature_names:
                raise ValueError("Model, scaler, or feature_names not properly loaded.")
            
            print("Model updated successfully")
            return jsonify({'message': 'File uploaded and model updated successfully', 'files': 'uploads/image_model.pkl'})
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    print("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/image_predict', methods=['POST'])
def image_predict():
    try:
        result_data = None
        # Check if model and scaler are loaded
        excel_df = load_pickle('uploads/image_excel_df.pkl')
        image_model = load_pickle('uploads/image_model.pkl')
        image_features = load_pickle('uploads/image_feature_names.pkl')
        image_scaler = load_pickle('uploads/image_scaler.pkl')
        if image_model is None or image_scaler is None:
            return jsonify({'error': 'Model not loaded. Please upload an Excel file first.'}), 400
        
        # print("=== PREDICT ROUTE DEBUG ===")
        # print(f"Available form fields: {list(request.form.keys())}")
        print(f"Feature names: {image_features}")
        
        # Get input data from the form
        input_data = {}
        
        data = request.get_json()
        truths = data.get('left_truths')
        right_truths = data.get('right_truths')
        MCD = data.get('Menstrual_Cycle_Data')
        Cli_sign = data.get('Clinical_signs')
        Bio_signs = data.get('Biochemical_signs')
        pcos = data.get('PCOS')
        # print(f"Truths from form: {truths}")
        # print(f"Right truths from form: {right_truths}")

        if MCD == '29 days':
            input_data['Menstrual Cycle Data'] = 0
        elif MCD == '32 days':
            input_data['Menstrual Cycle Data'] = 1
        elif MCD == '50 days':
            input_data['Menstrual Cycle Data'] = 2
        elif MCD == '60 days':
            input_data['Menstrual Cycle Data'] = 3
        
        elif MCD == '35 days':
            input_data['Menstrual Cycle Data'] = 5
        else:
            input_data['Menstrual Cycle Data'] = 4


        if Cli_sign == 'Acne':
            input_data['Clinical signs'] = 0
        elif Cli_sign == 'Acne & Hirsutism':
            input_data['Clinical signs'] = 1
        elif Cli_sign == 'Hair Loss':
            input_data['Clinical signs'] = 2
        elif Cli_sign == 'No':
            input_data['Clinical signs'] = 3
        else:
            input_data['Clinical signs'] = 4


        if Bio_signs == 'High':
            input_data['Biochemical signs'] = 0
        elif Bio_signs == 'Low':
            input_data['Biochemical signs'] = 1
        else:
            input_data['Biochemical signs'] = 2

        if pcos == 'Non PCOS':
            input_data['PCOS Assessment Result'] = 0
        elif pcos == 'PCOS Inflammation':
            input_data['PCOS Assessment Result'] = 1
        elif pcos == 'PCOS Rintangan Insulin':
            input_data['PCOS Assessment Result'] = 2
        else:
            input_data['PCOS Assessment Result'] = 3

        if truths:
            # truths is already a dict, no need to parse
            for key, value in truths.items():
                if key == 'Follicle Appearance(L)':
                    if value == 'Few small follicles seen':
                        value = 0
                    else:
                        value = 1
                elif key == 'Peripheral Follicle Pattern(L)':
                    if value == 'Absent':
                        value = 0
                    elif value == "Partial/Not prominent":
                        value = 1
                    else:
                        value = 2
                input_data[key] = str(value)
        if right_truths:
            # right_truths is already a dict, no need to parse
            for key, value in right_truths.items():
                if key == 'Follicle Appearance(R)':
                    if value == 'Few small follicles seen':
                        value = 0
                    else:
                        value = 1
                elif key == 'Peripheral Follicle Pattern(R)':
                    if value == 'Absent':
                        value = 0
                    elif value == "Partial":
                        value = 1
                    else:
                        value = 2
                input_data[key] = str(value)

        

        # Process time values in input_data (fix the dictionary access)
        for key, value in input_data.items():
            if ':' in str(value):
                input_data[key] = str(value).split(':')[0]
            if '.' in str(value):
                input_data[key] = str(value).split('.')[0]
            # Ensure all values are strings
            input_data[key] = str(input_data[key])
        
        # Create DataFrame with ONLY the features the model was trained on
        model_input_data = {}
        # print("input_data:", input_data)
        for feature in image_features:
            if feature in input_data:
                model_input_data[feature] = input_data[feature]
            else:
                print(f"Warning: Feature '{feature}' not found in input data, using default value 0")
                model_input_data[feature] = "0"  # Default as string
        
        # Convert input to DataFrame with only training features
        # print("=== DEBUG INFO ===")
        # print(f"Model was trained on features: {feature_names}")
        # print(f"Features being sent for prediction: {list(model_input_data.keys())}")
        
        input_df = pd.DataFrame([model_input_data])
        
        # # Verify the DataFrame has the correct features in the correct order
        # if list(input_df.columns) != image_features:
        #     print("ERROR: Feature mismatch!")
        #     print(f"Expected: {image_features}")
        #     print(f"Got: {list(input_df.columns)}")
        #     return jsonify({'error': 'Feature names or order mismatch with training data'}), 400
        
        # # Make prediction
        prediction, type_probabilities = predict_pcos_type(input_df, image_model, image_scaler)
        print(f"Prediction: {prediction}")
        # Get the PCOS type name
        pcos_type = PCOS_TYPES_FERTILITY.get(prediction, 'Unknown Type')
        print(f"prediction: {pcos_type}, type_probabilities:{type_probabilities }")
        return jsonify({
            'left': truths,
            'right': right_truths,
            'Morphology Assessment': str(pcos_type),
            'type_probabilities': type_probabilities,
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/send', methods=['GET', 'POST'])
def send_images():
    try:
        # Check if excel_df is loaded
        if excel_df is None:
            return jsonify({'error': 'No training data loaded. Please upload an Excel file first.'}), 400
        # Load model (change num_outputs if needed)
        image_model = create_model(num_outputs=7)
        # If you have trained weights, uncomment the next line:
        # model.load_state_dict(torch.load('ovary_model.pth', map_location='cpu'))
        image_model.eval()
        if 'Left Image' in excel_df.columns:
            excel_df['Left Image'] = excel_df['Left Image'].astype(str).str.strip().str.lower()
        # Target columns
        if 'Right Image' in excel_df.columns:
            excel_df['Right Image'] = excel_df['Right Image'].astype(str).str.strip().str.lower()
        TARGET_LEFT_COLS = [
            'Ovary Length (L)',
            'Ovary Width (L)',
            'Ovary Height (L)',
            'Ovarian Volume(L)',
            'Follicle Count(L)',
            'Follicle Appearance(L)',
            'Peripheral Follicle Pattern(L)'
        ]
        TARGET_RIGHT_COLS = [
            'Ovary Length (R)',
            'Ovary Width (R)',
            'Ovary Height (R)',
            'Ovarian Volume(R)',
            'Follicle Count(R)',
            'Follicle Appearance(R)',
            'Peripheral Follicle Pattern(R)'
        ]
        IMAGE_DIR = 'train_data/image'
        def get_image_embedding(image_path):
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                emb = image_model[0](img_tensor).cpu().numpy().flatten()  # Use image_model
            return emb
        
        # Build image list and embeddings
        image_paths = []
        right_image_paths = []
        embeddings = []
        right_embeddings = []
        for idx, row in excel_df.iterrows():
            rel_path = row['Left Image']
            right_path = row['Right Image']
            img_path = os.path.join(IMAGE_DIR, rel_path)
            right_img_path = os.path.join(IMAGE_DIR, right_path)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                emb = get_image_embedding(img_path)
                embeddings.append(emb)
            if os.path.exists(right_img_path):
                right_image_paths.append(right_img_path)
                right_emb = get_image_embedding(right_img_path)
                right_embeddings.append(right_emb)
        embeddings = np.vstack(embeddings) if embeddings else np.zeros((0, 512))
        right_embeddings = np.vstack(right_embeddings) if right_embeddings else np.zeros((0, 512))

        # Build FAISS indices
        faiss_index_left = None
        faiss_index_right = None
        if len(embeddings) > 0:
            dim = embeddings.shape[1]
            faiss_index_left = faiss.IndexFlatL2(dim)
            faiss_index_left.add(embeddings)
        if len(right_embeddings) > 0:
            dim = right_embeddings.shape[1]
            faiss_index_right = faiss.IndexFlatL2(dim)
            faiss_index_right.add(right_embeddings)
            
        most_similar = None
        truths = None
        right_truths = None
        uploaded_img = None
        if request.method == 'POST':
            
            file = request.files['left_image']
            right_file = request.files['right_image']
            img_bytes = file.read()
            right_img_bytes = right_file.read()
            uploaded_img = img_bytes
            right_uploaded_img = right_img_bytes
            # Get embedding for uploaded image
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            right_image = Image.open(io.BytesIO(right_img_bytes)).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)
            right_img_tensor = transform(right_image).unsqueeze(0)
            with torch.no_grad():
                query_emb = image_model[0](img_tensor).cpu().numpy().flatten().reshape(1, -1)
                right_query_emb = image_model[0](right_img_tensor).cpu().numpy().flatten().reshape(1, -1)
            # Search for most similar image
            if faiss_index_left is not None and len(image_paths) > 0:
                D, I = faiss_index_left.search(query_emb, k=1)
                idx = I[0][0]
                most_similar = image_paths[idx]
                # Find the corresponding row in df
                rel_path = os.path.relpath(most_similar, IMAGE_DIR).replace('\\', '/').lower()
                match = excel_df[excel_df['Left Image'] == rel_path]
                if not match.empty:
                    truths = match.iloc[0][TARGET_LEFT_COLS].to_dict()
            else:
                most_similar = None
                truths = None
                
            if faiss_index_right is not None and len(right_image_paths) > 0:
                right_D, right_I = faiss_index_right.search(right_query_emb, k=1)
                right_idx = right_I[0][0]
                right_most_similar = right_image_paths[right_idx]
                # Find the corresponding row in df
                right_rel_path = os.path.relpath(right_most_similar, IMAGE_DIR).replace('\\', '/').lower()
                right_match = excel_df[excel_df['Right Image'] == right_rel_path]
                if not right_match.empty:
                    right_truths = right_match.iloc[0][TARGET_RIGHT_COLS].to_dict()
            else:
                right_most_similar = None
                right_truths = None
        # print(truths)
        # print(right_truths)
        return jsonify({
            'success': True,
            'message': 'Images uploaded and processed successfully',
            'left_truths': truths,
            'right_truths': right_truths
        })
            
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


@app.route('/fertility_predict', methods=['POST'])
def fertility_predict():
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        # If not JSON, try to get form data
        data = request.form.to_dict()
        # Convert string values to appropriate types
        for key in data:
            if key.startswith('Cycle_Length_') or key.startswith('Period_Duration_'):
                data[key] = int(data[key])
            elif key == 'is_pcos':
                data[key] = data[key].lower() == 'true'
    
    if not data:
        return jsonify({'error': 'No data received'}), 400
    
    print(f"Received data: {data}")
    print(f"Content-Type: {request.content_type}")
    
    # --- Data Extraction and Processing ---
    name = data.get('name')
    cycle_lengths = [int(data['Cycle_Length_1']), int(data['Cycle_Length_2']), int(data['Cycle_Length_3'])]
    period_durations = [int(data['Period_Duration_1']), int(data['Period_Duration_2']), int(data['Period_Duration_3'])]
    
    lmp_date_str = data['LMP_Cycle_3']  # Use the most recent LMP date
    lmp_date = datetime.strptime(lmp_date_str, '%Y-%m-%d')
    
    # The value from JSON will be a string 'True' or 'False', convert to boolean
    has_pcos = data.get('is_pcos', 'False').lower() == 'true'
    
    # Note: For JSON requests, file uploads are typically handled separately
    # If you need to handle BBT data, consider sending it as base64 encoded data in JSON
    # or handle file uploads in a separate endpoint
    if has_pcos:
        # You can add BBT data processing here if needed
        # For now, we'll just log that PCOS is detected
        print(f"PCOS detected for user '{name}'")

    # --- Calculations ---
    avg_cycle_length = round(sum(cycle_lengths) / len(cycle_lengths))
    avg_period_duration = round(sum(period_durations) / len(period_durations))
    
    # Ensure avg_cycle_length is an integer for timedelta
    avg_cycle_length = int(avg_cycle_length)
    
    # Predict the start of the next cycle
    next_lmp_date = lmp_date + timedelta(days=avg_cycle_length)
    
    ovulation_day = calculate_ovulation_day(next_lmp_date, avg_cycle_length, has_pcos)
    fertile_window = calculate_fertile_window(ovulation_day)
    cycle_regularity = analyze_cycle_regularity(cycle_lengths, has_pcos)
    conception_probability = calculate_conception_probability(fertile_window)
    
    # --- Format Response ---
    response = {
        'fertileWindow': {
            'start': fertile_window['start'].strftime('%A, %B %d, %Y'),
            'end': fertile_window['end'].strftime('%A, %B %d, %Y')
        },
        'ovulationDay': ovulation_day.strftime('%A, %B %d, %Y'),
        'cycleRegularity': cycle_regularity,
        'conceptionProbability': conception_probability,
        'insights': {
            'averageCycleLength': avg_cycle_length,
            'averagePeriodDuration': avg_period_duration,
        }
    }
    
    return jsonify(response)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        data = request.form
        name = data.get('name')
        print(f"Received name: {name}")
        bbt_file = request.files.get('bbtFile_data')
        result = {'success': False, 'message': ''}
        result['name'] = name
        columns_to_load = [
            'LMP_Cycle_1', 'LMP_Cycle_2', 'LMP_Cycle_3',
            'Cycle_Length_1', 'Cycle_Length_2', 'Cycle_Length_3',
            'Period_Duration_1', 'Period_Duration_2', 'Period_Duration_3',
            'is_pcos'
        ]

        if bbt_file and bbt_file.filename != '':
            try:
                # Read the file into a DataFrame
                if bbt_file.filename.lower().endswith('.csv'):
                    df = pd.read_csv(bbt_file)
                else:
                    df = pd.read_excel(bbt_file)

                # Filter rows where user_id matches name
                if 'user_id' in df.columns:
                    user_rows = df[df['user_id'] == name]
                    if not user_rows.empty:
                        user_row = user_rows.iloc[0]  # Take the first match
                        for col in columns_to_load:
                            value = user_row.get(col, None)
                            if value is not None:
                                result[col] = str(value)
                            else:
                                result[col] = None
                        result['success'] = True
                        result['message'] = f"Data loaded for user '{name}'."
                    else:
                        result['message'] = f"No rows found for user_id '{name}'."
                else:
                    result['message'] = "No 'user_id' column found in the file."
            except Exception as e:
                result['message'] = f"Error reading file: {e}"
        else:
            result['message'] = "No file uploaded."
        print(f"Result: {result}")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True,use_reloader=False) 