import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from datetime import datetime, timedelta

# Constants
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

# Columns to exclude from model training
EXCLUDED_COLUMNS = [
    'timestamp', 'name', 'age','occupation', 'height', 'weight', 
    'please','jika','email','LINK', 'phone',
    'What kind of exercise do you do','If yes, what was the result of your sperm concentration test?',
    'How many time you exercise per week and how many hour','What do you usually eat for breakfast?','What time do you typically have lunch?',
    'What do you usually eat for lunch?', 'How often do you consume sugary drinks?','What do you usually eat for dinner?'
]

def preprocess_data(df):
    """
    Preprocess the data to handle mixed data types
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Store column types and skip columns
    column_types = {}
    skip_columns = []

    # First, identify and remove excluded columns
    for col in df_processed.columns:
        # Convert column name to string for safe comparison
        col_str = str(col).lower()
        if any(excluded.lower() in col_str for excluded in EXCLUDED_COLUMNS):
            skip_columns.append(col)
            df_processed = df_processed.drop(col, axis=1)
    
    # Convert date columns to numeric (days since epoch)
    for col in df_processed.columns:
        # print(df_processed[col])
        # Convert time values to hours
        for index, value in df_processed[col].items():
            
            if ':' in str(value):
                # print(value)
                df_processed.at[index, col] = (str(value).split(':')[0])
                # print(df_processed[col][index])
            if '.' in str(value):
                # print(value)
                df_processed.at[index, col] = (str(value).split('.')[0])
                # print(df_processed[col][index])
        
        if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
            # Skip timestamp columns
            skip_columns.append(col)
            df_processed = df_processed.drop(col, axis=1)
        elif pd.api.types.is_object_dtype(df_processed[col]):
            # Convert categorical/text columns to numeric using LabelEncoder
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            column_types[col] = 'categorical'
            print(f"\nMeaning of numbers in column '{col}':")
            for value, encoded in zip(le.classes_, le.transform(le.classes_)):
                print(f"{encoded} = {value}")
        else:
            column_types[col] = 'numeric'

    # print(f"Excluded columns: {skip_columns}")
    # print(f"Remaining columns for training: {df_processed.columns.tolist()}")
    
    # print("\nProcessed Data:")
    # print(df_processed)
    # print("\n" + "="*10 + "\n")
    
    return df_processed, column_types, skip_columns

def load_and_train_model(excel_file):
    """
    Load data from Excel file and train the model
    """
    # Load the data
    print("Loading data from Excel file...")
    df = pd.read_excel(excel_file)
    
    # Preprocess the data
    print("Preprocessing data...")
    df_processed, column_types, skip_columns = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.iloc[:, :-1]  # All columns except the last one
    y = df_processed.iloc[:, -1]   # Last column (PCOS type)
    print(f"adsf:{y}")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_accuracy:.2f}")
    print(f"Testing accuracy: {test_accuracy:.2f}")
    
    return model, scaler, X.columns, column_types

def predict_pcos_type(input_data, model, scaler):
    """
    Predict PCOS type for new data and return both prediction and probability distribution
    """
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Get prediction probabilities for all classes
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Get the predicted type
    predicted_type = model.predict(input_scaled)[0]
    
    # Create a dictionary of type probabilities
    type_probabilities = {}
    for type_value, probability in enumerate(probabilities):
        type_probabilities[type_value] = float(probability * 100)  # Convert to percentage
    
    # Print the probability distribution
    print("\nProbability distribution for prediction:")
    for type_value, probability in type_probabilities.items():
        prediction_marker = "✓" if type_value == predicted_type else " "
        print(f"Type {type_value}: {probability:.2f}% {prediction_marker}")
    
    return predicted_type, type_probabilities

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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

