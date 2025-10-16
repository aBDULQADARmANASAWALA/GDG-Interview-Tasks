# model_training_with_cv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from category_encoders import TargetEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class F1Predictor:
    def __init__(self):
        self.model = None
        self.target_encoder = None
        self.label_encoders = {}
        self.feature_columns = None
        self.cv_scores = None

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess F1 data"""
        df = pd.read_csv(file_path)

        # Assuming these are your 5 important features (adjust based on your EDA)
        self.feature_columns = [
            'year',
            'location',
            'circuitRef',
            'constructorRef',
            'driverRef',
            'grid',
        ]

        # Target variable (e.g., final position, points, etc.)
        target_column = 'target_finish'  # Adjust based on your target

        # Handle missing values
        df = df.dropna(subset=self.feature_columns + [target_column])

        return df[self.feature_columns], df[target_column]

    def create_target_encoding(self, X, y):
        """Apply target encoding to categorical features"""
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Initialize target encoder
        self.target_encoder = TargetEncoder(cols=categorical_cols)

        # Fit and transform the data
        X_encoded = self.target_encoder.fit_transform(X, y)

        return X_encoded

    def perform_cross_validation(self, X, y, cv_folds=5):
        """
        Perform k-fold cross-validation on the entire pipeline
        """
        print("Performing Cross-Validation...")

        # Initialize CV strategy
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Identify categorical columns
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            # Create a new target encoder for each fold to avoid data leakage
            fold_target_encoder = TargetEncoder(cols=categorical_cols)
            
            # Fit and transform training data
            X_train_encoded = fold_target_encoder.fit_transform(X_train, y_train)
            
            # Transform validation data using the fitted encoder
            X_val_encoded = fold_target_encoder.transform(X_val)

            # Explicitly cast to float to avoid dtype issues with XGBoost
            X_train_encoded = X_train_encoded.astype(float)
            X_val_encoded = X_val_encoded.astype(float)


            # Initialize and train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

            model.fit(X_train_encoded, y_train)

            # Validate
            y_pred = model.predict(X_val_encoded)
            fold_accuracy = accuracy_score(y_val, y_pred)
            fold_scores.append(fold_accuracy)

            print(f"Fold {fold}: Accuracy = {fold_accuracy:.4f}")

        # Store CV results
        self.cv_scores = fold_scores

        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores) * 2:.4f})")
        print(f"All Scores: {[f'{score:.4f}' for score in fold_scores]}")

        return fold_scores

    def train_model(self, X, y, test_size=0.2, random_state=42, perform_cv=True):
        """Train XGBoost model with optional cross-validation"""

        # Perform cross-validation first if requested
        if perform_cv:
            cv_scores = self.perform_cross_validation(X, y, cv_folds=5)

        # Now train the final model on entire training set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Apply target encoding - fit on training data
        X_train_encoded = self.create_target_encoding(X_train, y_train)
        
        # Transform test data using the fitted encoder (self.target_encoder is set by create_target_encoding)
        if self.target_encoder is not None:
            X_test_encoded = self.target_encoder.transform(X_test)
        else:
            raise ValueError("Target encoder not initialized")

        # Explicitly cast to float to avoid dtype issues with XGBoost
        X_train_encoded = X_train_encoded.astype(float)
        X_test_encoded = X_test_encoded.astype(float)

        # Initialize and train final XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )

        self.model.fit(X_train_encoded, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_encoded)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nFinal Model Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return X_test_encoded, y_test, y_pred

    def save_model(self, filepath='f1_model.pkl'):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'cv_scores': self.cv_scores
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='f1_model.pkl'):
        """Load the trained model and encoders"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.target_encoder = model_data['target_encoder']
        self.feature_columns = model_data['feature_columns']
        self.cv_scores = model_data.get('cv_scores', None)
        print(f"Model loaded from {filepath}")

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = F1Predictor()

    # Load your F1 dataset
    X, y = predictor.load_and_preprocess_data('f1_dnf.csv')

    # Train the model with cross-validation
    X_test, y_test, y_pred = predictor.train_model(X, y, perform_cv=True)

    # Save the model
    predictor.save_model('f1_xgboost_model_with_cv.pkl')