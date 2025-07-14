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
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
# Define PCOS types mapping based on actual data

def calculate_ovulation_day(lmp_date, cycle_length, has_pcos):
    days_to_ovulation = cycle_length - 14
    if has_pcos:
        days_to_ovulation = int(cycle_length * 0.6)
    
    ovulation_date = lmp_date + timedelta(days=days_to_ovulation)
    return ovulation_date

def calculate_fertile_window(ovulation_day):
    fertile_start = ovulation_day - timedelta(days=5)
    fertile_end = ovulation_day + timedelta(days=1)
    return {'start': fertile_start, 'end': fertile_end}

def analyze_cycle_regularity(cycle_lengths, has_pcos):
    if has_pcos:
        return "Irregular - PCOS may cause irregular cycles"
    if max(cycle_lengths) - min(cycle_lengths) > 8:
        return "Irregular"
    return "Regular"

def calculate_conception_probability(fertile_window):
    probabilities = [
        {'day_offset': -5, 'prob': 'Low (~4%)'},
        {'day_offset': -4, 'prob': 'Low (~10%)'},
        {'day_offset': -3, 'prob': 'Medium (~15%)'},
        {'day_offset': -2, 'prob': 'High (~27%)'},
        {'day_offset': -1, 'prob': 'High (~30%)'},
        {'day_offset': 0, 'prob': 'Peak (~33%)'},
        {'day_offset': 1, 'prob': 'Very Low'}
    ]
    
    probability_list = []
    for i in range(7):
        day = fertile_window['start'] + timedelta(days=i)
        probability_list.append({
            'date': day.strftime('%B %d'),
            'probability': probabilities[i]['prob']
        })
    return probability_list


@app.route('/home')
def home():
    pcos_features = load_pickle('uploads/female_feature_names.pkl')
    return render_template('./pcos/index.html', features=pcos_features)

@app.route('/')
def upload_page():
    return render_template('./pcos/upload.html')

@app.route('/meal')
def meal_upload_page():
    return render_template('./meal/upload.html')

@app.route('/meal_home')
def meal_home():
    meal_features = load_pickle('uploads/male_feature_names.pkl')
    return render_template('./meal/index.html', features=meal_features)

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
            return jsonify({'message': 'File uploaded and model updated successfully'})
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
        
        input_data = {}
       # For admin dashboard
        for feature in female_features:
            value = request.form.get(feature)
            if value is None:
                continue
            input_data[feature] = value
        # Convert input to DataFrame
        if input_data:
            input_df = pd.DataFrame([input_data])
        else:
            input_data = request.get_json() 
            #print(f"input_data: {input_data}")
            input_df = pd.DataFrame([input_data])
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
        return jsonify({
            # 'name':input_data['name'],
            # 'height':input_data['height'],
            # 'age':input_data['age'],
            # 'weight':input_data['weight'],
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    name = request.args.get('name')
    height = request.args.get('height')
    weight = request.args.get('weight')
    
    # Parse the probabilities from URL
    import json
    probabilities_str = request.args.get('probabilities')
    type_probabilities = {}
    if probabilities_str:
        try:
            type_probabilities = json.loads(probabilities_str)
        except:
            type_probabilities = {}
    
    return render_template('./pcos/result.html', 
                         prediction=prediction, 
                         error=error,
                         name=name,
                         height=height,
                         weight=weight,
                         type_probabilities=type_probabilities)


@app.route('/male_predict', methods=['POST'])
def meal_predict():
    try:
        # Load model and related objects from file
        male_model = load_pickle('uploads/male_model.pkl')
        male_features = load_pickle('uploads/male_feature_names.pkl')
        male_scaler = load_pickle('uploads/male_scaler.pkl')
        input_data = {}
        # For admin dashboard
        for feature in male_features:
            print(f"feature: {feature}:{request.form.get(feature)}")
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_data[feature] = value
        # Convert input to DataFrame
        if input_data:
            input_df = pd.DataFrame([input_data])
        else:
            input_data = request.get_json() 
            #print(f"input_data: {input_data}")
            input_df = pd.DataFrame([input_data])
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
        return jsonify({
            # 'name':input_data['name'],
            # 'height':input_data['height'],
            # 'age':input_data['age'],
            # 'weight':input_data['weight'],
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/meal_result')
def meal_result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    name = request.args.get('name')
    height = request.args.get('height')
    weight = request.args.get('weight')
    age = request.args.get('age')
    # Parse the probabilities from URL
    import json
    probabilities_str = request.args.get('probabilities')
    type_probabilities = {}
    if probabilities_str:
        try:
            type_probabilities = json.loads(probabilities_str)
        except:
            type_probabilities = {}
    
    return render_template('./meal/result.html', 
                         prediction=prediction, 
                         error=error,
                         name=name,
                         height=height,
                         age=age,
                         weight=weight,
                         type_probabilities=type_probabilities)

@app.route('/image')
def image_upload_file():
    return render_template('./image/upload.html', features=feature_names)


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
            global excel_df
            excel_df = pd.read_excel(filepath)
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
            return jsonify({'message': 'File uploaded and model updated successfully'})
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    print("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/image_home')
def image_home():
    image_features = load_pickle('uploads/image_feature_names.pkl')
    return render_template('./image/home.html', features=image_features)

@app.route('/image_predict', methods=['POST'])
def image_predict():
    try:
        # Check if model and scaler are loaded
        image_model = load_pickle('uploads/image_model.pkl')
        image_features = load_pickle('uploads/image_feature_names.pkl')
        image_scaler = load_pickle('uploads/image_scaler.pkl')
        if image_model is None or image_scaler is None:
            return jsonify({'error': 'Model not loaded. Please upload an Excel file first.'}), 400
        
        # print("=== PREDICT ROUTE DEBUG ===")
        # print(f"Available form fields: {list(request.form.keys())}")
        # print(f"Feature names: {feature_names}")
        
        # Get input data from the form
        input_data = {}
        truths = request.form.get('truths')
        right_truths = request.form.get('right_truths')
        
        # print(f"Truths from form: {truths}")
        # print(f"Right truths from form: {right_truths}")
        
        if truths:
            try:
                truths = json.loads(truths)
                # print(f"Parsed truths: {truths}")
                # Add truths data to input_data
                for key, value in truths.items():
                    # print(f"Adding truth key: {key}, value: {value}")
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
                    
                    input_data[key] = str(value)  # Convert to string
            except json.JSONDecodeError as e:
                print(f"Error parsing truths JSON: {e}")
                return jsonify({'error': f'Invalid truths data format: {e}'}), 400
                
        if right_truths:
            try:
                right_truths = json.loads(right_truths)
                # print(f"Parsed right_truths: {right_truths}")
                # Add right_truths data to input_data
                for key, value in right_truths.items():
                    # print(f"Adding right truth key: {key}, value: {value}")
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
                    input_data[key] = str(value)  # Convert to string
            except json.JSONDecodeError as e:
                print(f"Error parsing right_truths JSON: {e}")
                return jsonify({'error': f'Invalid right_truths data format: {e}'}), 400
        
        missing_features = []
        for feature in image_features:
            # print(f"Checking feature: {feature}")
            value = request.form.get(feature)
            # print(f"Value for {feature}: {value}")
            if value is None or value == '':
                missing_features.append(feature)
            else:
                input_data[feature] = str(value)  # Convert to string immediately
        result_data = input_data

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
            'pcos_type': str(pcos_type),
            'type_probabilities': type_probabilities,
            'result_data': result_data
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
            'truths': truths,
            'right_truths': right_truths
        })
            
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500

@app.route('/image_result')
def image_result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    name = request.args.get('name')
    result_data = request.args.get('result_data')
    # Parse the probabilities from URL
    import json
    probabilities_str = request.args.get('probabilities')
    type_probabilities = {}
    if probabilities_str:
        try:
            type_probabilities = json.loads(probabilities_str)
        except:
            type_probabilities = {}
    
    return render_template('./image/result.html', 
                         prediction=prediction, 
                         error=error,
                         name=name,
                         type_probabilities=type_probabilities,
                         result_data=result_data)

@app.route('/fertility')
def fertility():
    return render_template('./fertility/index.html')

@app.route('/fertility_predict', methods=['POST'])
def fertility_predict():
    # Data is now in request.form because we are using FormData
    data = request.form
    
    # --- Data Extraction and Processing ---
    name =  data.get('name')
    cycle_lengths = [int(data['cycleLength1']), int(data['cycleLength2']), int(data['cycleLength3'])]
    period_durations = [int(data['periodDuration1']), int(data['periodDuration2']), int(data['periodDuration3'])]
    
    lmp_date_str = data['lmpDate3']
    lmp_date = datetime.strptime(lmp_date_str, '%Y-%m-%d')
    
    # The value from the form will be a string 'true' or 'false'
    has_pcos = data.get('pcos') == 'true'
    
    # Check for an uploaded file if PCOS is true
    if has_pcos:
        bbt_file = request.files.get('bbtFile')
        if bbt_file and bbt_file.filename != '':
            print(f"Received file: {bbt_file.filename}")
            try:
                if bbt_file.filename.lower().endswith('.csv'):
                    df = pd.read_csv(bbt_file)
                else:
                    df = pd.read_excel(bbt_file)
                
                # --- Calculate average BBT for this user ---
                if 'user_id' in df.columns:
                    user_rows = df[df['user_id'] == name]
                    if not user_rows.empty:
                        for col in ['bbt', 'BBT', 'temperature', 'Temperature']:
                            if col in user_rows.columns:
                                avg_bbt = user_rows[col].mean()
                                print(f"Average BBT for user '{name}': {avg_bbt:.2f}")
                                break
                        else:
                            print("No BBT column found in the file.")
                    else:
                        print(f"No rows found for user_id '{name}'.")
                else:
                    print("No 'user_id' column found in the file.")
            except Exception as e:
                print(f"Error reading file: {e}")

    # --- Calculations ---
    avg_cycle_length = round(sum(cycle_lengths) / len(cycle_lengths))
    avg_period_duration = round(sum(period_durations) / len(period_durations))
    
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