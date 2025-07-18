import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class DiabetesPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, file_path):
        # Load data
        data = pd.read_csv(file_path)
        
        # Handle missing values (replace 0 with median)
        columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_fix:
            # Calculate median of non-zero values for each column
            # Replace 0 with median
            median_value = data[data[column] != 0][column].median()

        
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
        # Train model
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        # Make predictions for both train and test sets
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Print results
        print("\nModel Performance:")
        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
        
        # Plot accuracies
        self.plot_accuracies(train_accuracy, test_accuracy)
        
    def plot_accuracies(self, train_accuracy, test_accuracy):
        plt.figure(figsize=(8, 6))
        bars = plt.bar (['Training', 'Testing'], 
                      [train_accuracy, test_accuracy],
                      color=['blue', 'green'])
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy Score')
        plt.ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 to show full bars
        plt.tight_layout()
        plt.show()
    
    def predict(self, patient_data):
        # Scale the input data
        scaled_data = self.scaler.transform([patient_data])
        
        # Make prediction
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        return prediction[0], probability[0][1]

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Specify the path to your dataset
    dataset_path = "diabetes_dataset.csv"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess(dataset_path)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate model and plot accuracies
    predictor.evaluate(X_train, X_test, y_train, y_test)
    
    # Example of making a prediction for a new patient
    sample_patient = [4, 10, 160, 25, 150, 28, 0.6, 45]
    
    prediction, probability = predictor.predict(sample_patient)
    print(f"\nSample Patient Prediction:")
    print(f"Diagnosis: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability of diabetes: {probability:.2f}")
