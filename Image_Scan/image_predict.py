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
