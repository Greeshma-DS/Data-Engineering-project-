
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student Performance Prediction - REAL DATA", layout="wide")
st.title("🎓 Student Performance Prediction - REAL DATA")
st.write("**Trained on Actual Student Dataset from S3**")

@st.cache_resource
def load_real_model():
    """Load the model trained on actual student data"""
    try:
        model_path = 'real_student_model.joblib'
        if os.path.exists(model_path):
            model_dict = joblib.load(model_path)
            st.success("✅ REAL Student Data Model Loaded!")
            return model_dict
        else:
            st.error("❌ Real model not found. Please train it first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Load model
    if 'model_dict' not in st.session_state:
        with st.spinner("Loading AI model trained on real student data..."):
            st.session_state.model_dict = load_real_model()

    # Show model info if loaded
    if st.session_state.model_dict:
        model_dict = st.session_state.model_dict
        st.info(f"**Model Info**: {model_dict['model_info']}")
        st.info(f"**Dataset**: {model_dict['dataset_size']} student records | **Accuracy**: {model_dict['test_accuracy']:.1%}")

    # Input form
    with st.sidebar:
        st.header("📋 Student Details")

        st.subheader("🎓 Academic Info")
        g1 = st.slider("First Period Grade (0-20)", 0, 20, 10)
        g2 = st.slider("Second Period Grade (0-20)", 0, 20, 10)
        studytime = st.slider("Study Time (1-4)", 1, 4, 2)
        failures = st.slider("Past Failures", 0, 4, 0)
        absences = st.slider("Absences", 0, 93, 6)

        st.subheader("👨‍👩‍👧‍👦 Personal & Family Info")
        age = st.slider("Age", 15, 22, 18)
        medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
        fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
        famrel = st.slider("Family Relations (1-5)", 1, 5, 4)
        freetime = st.slider("Free Time (1-5)", 1, 5, 3)
        goout = st.slider("Going Out (1-5)", 1, 5, 3)
        health = st.slider("Health (1-5)", 1, 5, 3)

        predict_btn = st.button("🎯 Predict Performance", type="primary", use_container_width=True)

    # Prediction logic
    if predict_btn:
        if st.session_state.model_dict is None:
            st.error("Real model not loaded. Please train it first.")
            return

        try:
            model_dict = st.session_state.model_dict
            model = model_dict['model']
            scaler = model_dict['scaler']
            feature_columns = model_dict['feature_columns']

            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age], 'medu': [medu], 'fedu': [fedu],
                'traveltime': [2], 'studytime': [studytime], 'failures': [failures],
                'famrel': [famrel], 'freetime': [freetime], 'goout': [goout],
                'dalc': [1], 'walc': [1], 'health': [health],
                'absences': [absences], 'g1': [g1], 'g2': [g2]
            })

            input_data = input_data[feature_columns]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)

            # Display results
            st.header("📊 Prediction Results")

            categories = {0: "📉 Low (0-9)", 1: "📊 Medium (10-14)", 2: "📈 High (15-20)"}
            predicted_category = categories.get(prediction[0], "Unknown")

            st.success(f"### Predicted Performance: {predicted_category}")

            # Show confidence
            st.subheader("🎯 Prediction Confidence")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Low", f"{probabilities[0][0]:.1%}")
            with col2:
                st.metric("Medium", f"{probabilities[0][1]:.1%}")
            with col3:
                st.metric("High", f"{probabilities[0][2]:.1%}")

            # Show feature importance
            st.subheader("🔍 Top Influencing Factors (from Real Data)")
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                
                for _, row in feature_importance.iterrows():
                    importance_bar = "█" * int(row['Importance'] * 50)
                    st.write(f"- **{row['Feature']}**: `{importance_bar}` ({row['Importance']:.3f})")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
