import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class DiabetesPredictor:
    def __init__(self):
        self.rf_model = RandomForestClassifier(random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, file_path):
        # Load data
        data = pd.read_csv('diabetes_dataset.csv')
        
        # Handle missing values (replace 0 with median)
        columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_fix:
            # Calculate median of non-zero values for each column
            median_value = data[data[column] != 0][column].median()
            # Replace 0 with median
            data.loc[data[column] == 0, column] = median_value
        
        # Separate features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        # Train both models
        self.rf_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        # Make predictions for both models
        # Random Forest
        rf_train_pred = self.rf_model.predict(X_train)
        rf_test_pred = self.rf_model.predict(X_test)
        
        # Logistic Regression
        lr_train_pred = self.lr_model.predict(X_train)
        lr_test_pred = self.lr_model.predict(X_test)
        
        # Calculate accuracies
        rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
        rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
        lr_train_accuracy = accuracy_score(y_train, lr_train_pred)
        lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
        
        # Print results
        print("\nRandom Forest Performance:")
        print(f"Training Accuracy: {rf_train_accuracy:.2f}")
        print(f"Testing Accuracy: {rf_test_accuracy:.2f}")
        print("\nRandom Forest Classification Report (Test Set):")
        print(classification_report(y_test, rf_test_pred))
        
        print("\nLogistic Regression Performance:")
        print(f"Training Accuracy: {lr_train_accuracy:.2f}")
        print(f"Testing Accuracy: {lr_test_accuracy:.2f}")
        print("\nLogistic Regression Classification Report (Test Set):")
        print(classification_report(y_test, lr_test_pred))
        
        # Plot accuracies
        self.plot_accuracies(rf_train_accuracy, rf_test_accuracy, 
                           lr_train_accuracy, lr_test_accuracy)
        
    def plot_accuracies(self, rf_train_acc, rf_test_acc, lr_train_acc, lr_test_acc):
        plt.figure(figsize=(10, 6))
        
        # Set bar positions
        x = np.arange(2)
        width = 0.35
        
        # Create bars
        bars1 = plt.bar(x - width/2, [rf_train_acc, rf_test_acc], width, 
                       label='Random Forest', color=['blue', 'lightblue'])
        bars2 = plt.bar(x + width/2, [lr_train_acc, lr_test_acc], width,
                       label='Logistic Regression', color=['green', 'lightgreen'])
        
        # Add value labels on top of each bar
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.ylabel('Accuracy Score')
        plt.title('Model Accuracy Comparison')
        plt.legend()
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
    
    def predict(self, patient_data):
        # Scale the input data
        scaled_data = self.scaler.transform([patient_data])
        
        # Make predictions with both models
        rf_prediction = self.rf_model.predict(scaled_data)
        rf_probability = self.rf_model.predict_proba(scaled_data)
        
        lr_prediction = self.lr_model.predict(scaled_data)
        lr_probability = self.lr_model.predict_proba(scaled_data)
        
        return {
            'random_forest': (rf_prediction[0], rf_probability[0][1]),
            'logistic_regression': (lr_prediction[0], lr_probability[0][1])
        }

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Specify the path to your dataset
    dataset_path = "diabetes_dataset.csv"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess(dataset_path)
    
    # Train models
    predictor.train(X_train, y_train)
    
    # Evaluate models and plot accuracies
    predictor.evaluate(X_train, X_test, y_train, y_test)
    
    # Example of making a prediction for a new patient
    sample_patient = [2, 130, 80, 25, 150, 28, 0.6, 45]
    
    predictions = predictor.predict(sample_patient)
    
    print("\nSample Patient Predictions:")
    print("\nRandom Forest:")
    print(f"Diagnosis: {'Diabetic' if predictions['random_forest'][0] == 1 else 'Non-diabetic'}")
    print(f"Probability of diabetes: {predictions['random_forest'][1]:.2f}")
    
    print("\nLogistic Regression:")
    print(f"Diagnosis: {'Diabetic' if predictions['logistic_regression'][0] == 1 else 'Non-diabetic'}")
    print(f"Probability of diabetes: {predictions['logistic_regression'][1]:.2f}")
