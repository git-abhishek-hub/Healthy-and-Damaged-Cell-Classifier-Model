
    print("Loading processed dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dat()
    print("Building model...")
    
    model = build_model()
    print("Training model...")
    
    history = model.fit(