# Maternal Health Risk Prediction - SDG 3: Good Health and Well-being
# AI for Good Assignment
# Author: AI Specialist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class MaternalHealthPredictor:
    """
    A machine learning system to predict maternal health risks and prevent mortality
    Addresses SDG 3: Good Health and Well-being
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = []

    def generate_synthetic_data(self, n_samples=1000):
        print("Generating synthetic maternal health dataset...")
        np.random.seed(42)
        data = {
            'age': np.random.normal(26, 6, n_samples).astype(int),
            'systolic_bp': np.random.normal(120, 20, n_samples),
            'diastolic_bp': np.random.normal(80, 15, n_samples),
            'blood_sugar': np.random.normal(100, 30, n_samples),
            'body_temp': np.random.normal(98.6, 1.5, n_samples),
            'heart_rate': np.random.normal(75, 15, n_samples),
            'prenatal_visits': np.random.poisson(6, n_samples),
            'previous_pregnancies': np.random.poisson(1.5, n_samples),
            'education_years': np.random.normal(8, 4, n_samples),
            'income_level': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'urban_rural': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'health_facility_distance': np.random.exponential(5, n_samples)
        }
        df = pd.DataFrame(data)
        df['age'] = np.clip(df['age'], 15, 45)
        df['systolic_bp'] = np.clip(df['systolic_bp'], 80, 200)
        df['diastolic_bp'] = np.clip(df['diastolic_bp'], 50, 120)
        df['blood_sugar'] = np.clip(df['blood_sugar'], 60, 300)
        df['body_temp'] = np.clip(df['body_temp'], 95, 105)
        df['heart_rate'] = np.clip(df['heart_rate'], 50, 120)
        df['prenatal_visits'] = np.clip(df['prenatal_visits'], 0, 15)
        df['education_years'] = np.clip(df['education_years'], 0, 16)
        df['health_facility_distance'] = np.clip(df['health_facility_distance'], 0.5, 50)
        df['risk_score'] = (
            (df['age'] < 18).astype(int) * 2 +
            (df['age'] > 35).astype(int) * 2 +
            (df['systolic_bp'] > 140).astype(int) * 3 +
            (df['diastolic_bp'] > 90).astype(int) * 2 +
            (df['blood_sugar'] > 125).astype(int) * 2 +
            (df['prenatal_visits'] < 4).astype(int) * 3 +
            (df['education_years'] < 6).astype(int) * 2 +
            (df['income_level'] == 1).astype(int) * 2 +
            (df['urban_rural'] == 0).astype(int) * 1 +
            (df['health_facility_distance'] > 10).astype(int) * 2
        )
        df['risk_level'] = pd.cut(df['risk_score'], bins=[-1, 3, 7, 12, 20],
                                  labels=['Low', 'Medium', 'High', 'Critical'])
        return df

    def explore_data(self, df):
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {df.shape}")
        print(df.info())
        print(f"\nRisk level distribution:")
        print(df['risk_level'].value_counts())
        return df

    def preprocess_data(self, df):
        print("\n=== DATA PREPROCESSING ===")
        feature_cols = ['age', 'systolic_bp', 'diastolic_bp', 'blood_sugar', 
                        'body_temp', 'heart_rate', 'prenatal_visits', 
                        'previous_pregnancies', 'education_years', 'income_level',
                        'urban_rural', 'health_facility_distance']
        X = df[feature_cols].copy()
        y = df['risk_level'].copy()
        self.feature_names = feature_cols
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def train_models(self, X_train, y_train):
        print("\n=== MODEL TRAINING ===")
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        model_scores = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")
        return model_scores

    def evaluate_model(self, X_test, y_test):
        print("\n=== MODEL EVALUATION ===")
        y_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        return accuracy, y_pred

    def feature_importance_analysis(self):
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            print("Top Important Features:")
            print(feature_importance.head())
            return feature_importance
        else:
            print("Feature importance not available.")
            return None

    def predict_risk(self, patient_data):
        patient_scaled = self.scaler.transform([patient_data])
        risk_pred = self.best_model.predict(patient_scaled)[0]
        risk_proba = self.best_model.predict_proba(patient_scaled)[0]
        risk_level = self.label_encoder.inverse_transform([risk_pred])[0]
        return risk_level, risk_proba

    def ethical_analysis(self):
        print("\n=== ETHICAL CONSIDERATIONS ===")
        ethical_points = [
            "BIAS ANALYSIS:",
            "- Data may underrepresent certain ethnic/geographic groups",
            "- Rural vs urban healthcare access bias exists",
            "- Socioeconomic factors may introduce systematic bias",
            "",
            "FAIRNESS MEASURES:",
            "- Model should be tested across different demographic groups",
            "- False negative rates should be minimized (missing high-risk cases)",
            "- Healthcare resource allocation should consider equity",
            "",
            "PRIVACY & CONSENT:",
            "- Patient data must be anonymized and secured",
            "- Informed consent required for data collection",
            "- Transparent communication about AI decision-making",
            "",
            "SUSTAINABILITY IMPACT:",
            "- Reduces maternal mortality (SDG 3)",
            "- Improves healthcare efficiency and resource allocation",
            "- Enables preventive care and early intervention",
            "- Supports healthcare worker decision-making in resource-limited settings"
        ]
        for point in ethical_points:
            print(point)

def main():
    import os
    os.makedirs("demo_screenshots", exist_ok=True)
    print("=" * 60)
    print("MATERNAL HEALTH RISK PREDICTION SYSTEM")
    print("SDG 3: Good Health and Well-being")
    print("AI for Good Assignment")
    print("=" * 60)
    predictor = MaternalHealthPredictor()
    df = predictor.generate_synthetic_data(1000)
    df = predictor.explore_data(df)
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.preprocess_data(df)
    model_scores = predictor.train_models(X_train, y_train)
    accuracy, y_pred = predictor.evaluate_model(X_test, y_test)
    feature_importance = predictor.feature_importance_analysis()
    print("\n=== EXAMPLE PREDICTION ===")
    sample_patient = [28, 140, 95, 110, 99.2, 85, 3, 1, 10, 2, 0, 8.5]
    risk_level, risk_proba = predictor.predict_risk(sample_patient)
    print(f"Sample Patient Risk Level: {risk_level}")
    print("Risk Probabilities:")
    for i, prob in enumerate(risk_proba):
        print(f"  {predictor.label_encoder.classes_[i]}: {prob:.3f}")
    predictor.ethical_analysis()
    print("\n=== PROJECT SUMMARY ===")
    print(f"✓ Successfully trained ML model with {accuracy:.1%} accuracy")
    print("✓ Addresses SDG 3: Good Health and Well-being")
    print("✓ Enables early intervention to prevent maternal mortality")
    print("✓ Considers ethical implications and bias mitigation")
    print("✓ Supports sustainable healthcare in developing regions")

if __name__ == "__main__":
    main()
