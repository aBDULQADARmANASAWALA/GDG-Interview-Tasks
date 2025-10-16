# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import F1Predictor

class F1PredictionApp:
    def __init__(self):
        self.predictor = F1Predictor()
        self.df = None
        self.load_data()
        self.load_model()

    def load_data(self) -> None:
        try:
            self.df = pd.read_csv('f1_dnf.csv')
        except Exception as e:
            st.warning(f"Could not load dataset for dropdown options: {str(e)}")
            self.df = None

    def load_model(self) -> None:
        try:
            FILE = 'f1_xgboost_model_with_cv.pkl'
            
            model_loaded = False
            try:
                self.predictor.load_model(FILE)
                model_loaded = True
            except FileNotFoundError:
                st.warning(f'File not found: {FILE}')
            except Exception as e:
                st.warning(f"Error loading {FILE}: {str(e)}")
            
            if not model_loaded:
                st.error("⚠️ Model file not found. Please train the model first by running: python model.py")
        except Exception as e:
            st.error(f"Unexpected error loading model: {str(e)}")

    def predict_race(self, input_data) -> tuple: # Returns (int: prediction, list: probabilities)
        try:
            input_df = pd.DataFrame([input_data])
            
            if self.predictor.feature_columns:
                input_df = input_df[self.predictor.feature_columns]

            # Apply target encoding
            if hasattr(self.predictor, 'target_encoder') and self.predictor.target_encoder is not None:
                input_encoded = self.predictor.target_encoder.transform(input_df)
            else:
                st.error("Target encoder is not loaded. Please ensure the model and encoder are available.")
                return None, None
            # ---

            if self.predictor.model is not None:
                prediction = self.predictor.model.predict(input_encoded)
                probabilities = self.predictor.model.predict_proba(input_encoded)
                return prediction[0], probabilities[0]
            else:
                st.error("Model is not loaded. Please ensure the model file is available and loaded correctly.")
                return None, None
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

    def run(self) -> None:
        st.set_page_config(
            page_title="F1 Race Predictor",
            page_icon="🏎️",
            layout="wide",
        )

        st.title("🏎️ F1 Race Outcome Predictor")
        st.markdown("""
        This app predicts F1 race outcomes using XGBoost with Target Encoding.
        Enter the race details below to get predictions.
        """)

        # Check if model is properly loaded
        if self.predictor.model is None:
            st.error("⚠️ **Model Not Loaded!**")
            st.warning("The model file was not found or could not be loaded.")
            return

        # Create input form
        with st.form("prediction_form"):
            st.header("Race Details")

            col1, col2 = st.columns(2)

            with col1:
                # Get unique values from dataset 
                drivers = sorted(self.df['driverRef'].dropna().unique().tolist())
                constructors = sorted(self.df['constructorRef'].dropna().unique().tolist())
                circuits = sorted(self.df['circuitRef'].dropna().unique().tolist())
                locations = sorted(self.df['location'].dropna().unique().tolist())

                driver = st.selectbox(
                    "Driver",
                    drivers
                )

                constructor = st.selectbox(
                    "Constructor",
                    constructors
                )

                circuit = st.selectbox(
                    "Circuit",
                    circuits
                )

            with col2:
                grid = st.slider(
                    'Grid Position',
                    min_value=0,
                    max_value=34,
                    value=1
                )

                year = st.selectbox(
                    "Year",
                    [2025, 2026],
                )

                loc = st.selectbox(
                    "Location",
                    locations
                )

            submitted = st.form_submit_button("Predict Race Outcome")

            if submitted:
                input_data = {
                    'year': year,
                    'location': loc,
                    'circuitRef': circuit,
                    'constructorRef': constructor,
                    'driverRef': driver,
                    'grid': grid
                }

                st.divider()
                prediction, probabilities = self.predict_race(input_data)

                if prediction is not None:
                    st.header("Prediction Results: ")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        if int(prediction):
                            st.markdown(f"## 🏆 Will Finish the Race" )
                        else:
                            st.markdown(f"## ❌ Will Not Finish the Race")

                    with result_col2:
                        max_prob = np.max(probabilities) * 100
                        st.markdown(f"## Confidence: {max_prob:.1f}%")

                    st.divider()
                    st.subheader("Prediction Probabilities")

                    prediction_labels = ['Will Not Finish', 'Will Finish']
                    prob_df = pd.DataFrame({
                        'Prediction': prediction_labels,
                        'Probability': probabilities * 100
                    })

                    with st.container(width='stretch', horizontal_alignment='center'):
                        st.bar_chart(prob_df.set_index('Prediction'), horizontal=True, height=300, color=[(0,255,0)])

                    st.divider()
                    st.subheader("Detailed Probabilities")
                    
                    st.write(f'Will Not Finish: {probabilities[0]*100:.2f}%')
                    st.write(f'Will Finish: {probabilities[1]*100:.2f}%')

        # Model information section
        with st.expander("Model Information"):
            st.write("""
            **Model Details:**
            - Algorithm: XGBoost Classifier
            - Encoding: Target Encoding for categorical features
            - Features: 6 important features identified during EDA

            **Feature Importance:**
            The model uses the following features:
            1. driverRef
            2. constructorRef
            3. circuitRef
            4. grid
            5. year
            6. location
            """)

def main():
    app = F1PredictionApp()
    app.run()

if __name__ == "__main__":
    main()