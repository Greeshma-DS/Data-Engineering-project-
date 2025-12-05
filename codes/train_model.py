
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("🎯 Training model on ACTUAL student data...")

try:
    # Load the actual student data
    df = pd.read_parquet('student_actual_data.parquet')
    print(f"✅ Loaded actual student data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check data types and basic info
    print("\n📊 Data Overview:")
    print(f"Data types:\n{df.dtypes}")
    print(f"G3 value counts:\n{df['g3'].value_counts().sort_index()}")
    
    # Use only numeric columns (same as original approach)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n🔢 Numeric columns: {numeric_columns}")
    
    # Make sure g3 is numeric
    if 'g3' not in numeric_columns:
        df['g3'] = pd.to_numeric(df['g3'], errors='coerce')
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove g3 from features
    feature_columns = [col for col in numeric_columns if col != 'g3']
    print(f"🎯 Using features: {feature_columns}")
    
    # Prepare data
    X = df[feature_columns]
    y = df['g3']
    
    print(f"\n📈 Dataset Info:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Convert G3 to categories (0: Low, 1: Medium, 2: High)
    # Since your original training did this categorization
    y_categorized = pd.cut(y, bins=[-1, 9, 14, 20], labels=[0, 1, 2])
    print(f"Target after categorization: {y_categorized.value_counts().to_dict()}")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorized, test_size=0.2, random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n🎯 Model Performance:")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📊 Top 5 Important Features:")
    print(feature_importance.head(5))
    
    # Save the REAL model
    model_dict = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_info': 'RandomForest trained on ACTUAL student data from S3 Parquet',
        'training_accuracy': train_score,
        'test_accuracy': test_score,
        'data_source': 'student_actual_data.parquet',
        'dataset_size': df.shape[0]
    }
    
    joblib.dump(model_dict, 'real_student_model.joblib')
    print(f"\n✅ REAL model saved: 'real_student_model.joblib'")
    print(f"📁 Trained on {df.shape[0]} actual student records")
    print(f"🎯 Using {len(feature_columns)} features from the real dataset")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
