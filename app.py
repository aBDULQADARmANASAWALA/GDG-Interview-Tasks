# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model import F1Predictor

class F1PredictionApp:
    def __init__(self):
        self.predictor = F1Predictor()
        self.df = None
        self.load_data()
        self.load_model()

    def load_data(self):
        """Load the F1 dataset to get unique values for dropdowns"""
        try:
            self.df = pd.read_csv('f1_dnf.csv')
            # st.info("Dataset loaded successfully for input options!")
        except Exception as e:
            st.warning(f"Could not load dataset for dropdown options: {str(e)}")
            self.df = None

    def load_model(self):
        """Load the trained model"""
        try:
            # Try the model with CV first, then fall back to other names
            model_file = 'f1_xgboost_model_with_cv.pkl'
            
            model_loaded = False
            try:
                self.predictor.load_model(model_file)
                # st.success(f"Model loaded successfully from {model_file}!")
                model_loaded = True
            except FileNotFoundError:
                st.warning(f'File not found: {model_file}')
            except Exception as e:
                st.warning(f"Error loading {model_file}: {str(e)}")
            
            if not model_loaded:
                st.error("⚠️ Model file not found. Please train the model first by running: python model.py")
                # st.info("This will create the model file 'f1_xgboost_model_with_cv.pkl'")
        except Exception as e:
            st.error(f"Unexpected error loading model: {str(e)}")

    def predict_race(self, input_data):
        """Make prediction for input data"""
        try:
            # Convert input to DataFrame with correct column order
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns match the feature_columns used during training
            if self.predictor.feature_columns:
                input_df = input_df[self.predictor.feature_columns]

            # Apply target encoding
            if hasattr(self.predictor, 'target_encoder') and self.predictor.target_encoder is not None:
                input_encoded = self.predictor.target_encoder.transform(input_df)
                
                # Explicitly cast to float to match training data format
                input_encoded = input_encoded.astype(float)
            else:
                st.error("Target encoder is not loaded. Please ensure the model and encoder are available.")
                return None, None

            # Make prediction
            if self.predictor.model is not None:
                prediction = self.predictor.model.predict(input_encoded)
                probability = self.predictor.model.predict_proba(input_encoded)
                return prediction[0], probability[0]
            else:
                st.error("Model is not loaded. Please ensure the model file is available and loaded correctly.")
                return None, None
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None, None

    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="F1 Race Predictor",
            page_icon="🏎️",
            layout="wide"
        )

        st.title("🏎️ F1 Race Outcome Predictor")
        st.markdown("""
        This app predicts F1 race outcomes using XGBoost with Target Encoding.
        Enter the race details below to get predictions.
        """)

        # Check if model is properly loaded
        if self.predictor.model is None:
            st.error("⚠️ **Model Not Loaded!**")
            st.warning("""
            The model file was not found or could not be loaded. 
            
            **To train the model, run:**
            ```bash
            python model.py
            ```
            This will train the model and save it as 'f1_xgboost_model_with_cv.pkl'
            """)
            return

        # Create input form
        with st.form("prediction_form"):
            st.header("Race Details")

            col1, col2 = st.columns(2)

            with col1:
                # Get unique values from dataset or use defaults
                # if self.df is not None:
                drivers = sorted(self.df['driverRef'].dropna().unique().tolist())
                constructors = sorted(self.df['constructorRef'].dropna().unique().tolist())
                circuits = sorted(self.df['circuitRef'].dropna().unique().tolist())
                locations = sorted(self.df['location'].dropna().unique().tolist())
                # else:
                #     # Fallback defaults if dataset not loaded
                #     drivers = ["hamilton", "max_verstappen", "leclerc", "sainz", "norris"]
                #     constructors = ["mercedes", "red_bull", "ferrari", "mclaren"]
                #     circuits = ["monaco", "silverstone", "spa", "monza"]
                #     locations = ["Europe", "Asia", "America"]

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
                # Get year range from dataset
                # if self.df is not None:
                max_grid = int(self.df['grid'].max())
                # else:
                #     min_year = 1950
                #     max_year = 2024
                #     max_grid = 20

                grid = st.slider(
                    'Grid Position',
                    min_value=0,
                    max_value=max_grid,
                    value=1
                )

                year = st.selectbox(
                    "Year",
                    [2025, 2026],
                    # value=2026
                )

                loc = st.selectbox(
                    "Location",
                    locations
                )

                

            # Submit button
            submitted = st.form_submit_button("Predict Race Outcome")

            if submitted:
                # Prepare input data - match the feature columns from model.py
                input_data = {
                    'year': year,
                    'location': loc,
                    'circuitRef': circuit,
                    'constructorRef': constructor,
                    'driverRef': driver,
                    'grid': grid
                }

                # Make prediction
                prediction, probabilities = self.predict_race(input_data)

                if prediction is not None:
                    # Display results
                    st.header("Prediction Results: ")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        # st.subheader("Predicted Final Position")
                        if int(prediction):
                            st.markdown(f"## 🏆 Will Finish the Race" )
                        else:
                            st.markdown(f"## ❌ Will Not Finish the Race")

                    with result_col2:
                        if probabilities is not None:
                            max_prob = np.max(probabilities) * 100
                        else:
                            st.warning("Probability data is not available.")
                        st.markdown(f"## Confidence: {max_prob:.1f}%")
                        # st.markdown(f"# {max_prob:.1f}%")

                    # Show probability distribution
                    st.subheader("Position Probabilities")

                    # Create probability chart
                    positions = range(1, len(probabilities) + 1)
                    prob_df = pd.DataFrame({
                        'Position': positions,
                        'Probability': probabilities * 100
                    })

                    st.bar_chart(prob_df.set_index('Position'))

                    # Show detailed probabilities
                    st.subheader("Detailed Probabilities")
                    for pos, prob in zip(positions, probabilities):
                        st.write(f"Position {pos}: {prob*100:.2f}%")

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

        # # Sample predictions section
        # with st.expander("Sample Predictions"):
        #     st.write("Try these pre-defined scenarios:")

        #     sample_scenarios = [
        #         {
        #             "name": "Top Team - Front Row",
        #             "year": 2025,
        #             "driverRef": "max_verstappen",
        #             "constructorRef": "red_bull",
        #             "circuitRef": "monaco",
        #             "grid": 1,
        #             "location": "Budapest"
        #         },
        #         {
        #             "name": "Midfield - Middle Grid",
        #             "year": 2025,
        #             "driverRef": "lando_norris",
        #             "constructorRef": "mclaren",
        #             "circuitRef": "silverstone",
        #             "grid": 8,
        #             "location": "Budapest"
        #         }
        #     ]

        #     for scenario in sample_scenarios:
        #         if st.button(f"Test: {scenario['name']}"):
        #             prediction, probabilities = self.predict_race(scenario)
        #             if int(prediction):
        #                     st.markdown(f"🏆 Will Finish the Race" )
        #             else:
        #                 st.markdown(f"❌ Will Not Finish the Race")

def main():
    app = F1PredictionApp()
    app.run()

if __name__ == "__main__":
    main()