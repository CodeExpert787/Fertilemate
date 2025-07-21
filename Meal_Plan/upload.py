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
