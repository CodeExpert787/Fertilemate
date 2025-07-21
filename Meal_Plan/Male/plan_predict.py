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
