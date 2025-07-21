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
