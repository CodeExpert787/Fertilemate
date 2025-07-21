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