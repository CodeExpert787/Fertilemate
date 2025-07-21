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
