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
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, file_path):
        # Load data
        data = pd.read_csv(file_path)

        # Fix: Convert columns to float to avoid dtype warning
        columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        data[columns_to_fix] = data[columns_to_fix].astype(float)
        
        for column in columns_to_fix:
            median_value = data[data[column] != 0][column].median()
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
        self.rf_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        rf_train_pred = self.rf_model.predict(X_train)
        rf_test_pred = self.rf_model.predict(X_test)
        
        lr_train_pred = self.lr_model.predict(X_train)
        lr_test_pred = self.lr_model.predict(X_test)
        
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        lr_train_acc = accuracy_score(y_train, lr_train_pred)
        lr_test_acc = accuracy_score(y_test, lr_test_pred)
        
        print("\nRandom Forest Performance:")
        print(f"Training Accuracy: {rf_train_acc:.2f}")
        print(f"Testing Accuracy: {rf_test_acc:.2f}")
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_test_pred))
        
        print("\nLogistic Regression Performance:")
        print(f"Training Accuracy: {lr_train_acc:.2f}")
        print(f"Testing Accuracy: {lr_test_acc:.2f}")
        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, lr_test_pred))
        
        self.plot_accuracies(rf_train_acc, rf_test_acc, lr_train_acc, lr_test_acc)
    
    def plot_accuracies(self, rf_train, rf_test, lr_train, lr_test):
        plt.figure(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        
        bars1 = plt.bar(x - width/2, [rf_train, rf_test], width, label='Random Forest', color=['#2980b9', '#3498db'])
        bars2 = plt.bar(x + width/2, [lr_train, lr_test], width, label='Logistic Regression', color=['#f39c12', '#e67e22'])
        
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x, ['Training Accuracy', 'Testing Accuracy'])
        plt.ylim(0, 1.1)
        plt.legend()

        # Add values on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
    
    def predict(self, patient_data):
        scaled_data = self.scaler.transform([patient_data])
        rf_prediction = self.rf_model.predict(scaled_data)
        rf_probability = self.rf_model.predict_proba(scaled_data)
        
        lr_prediction = self.lr_model.predict(scaled_data)
        lr_probability = self.lr_model.predict_proba(scaled_data)
        
        return {
            'random_forest': (rf_prediction[0], rf_probability[0][1]),
            'logistic_regression': (lr_prediction[0], lr_probability[0][1])
        }

# Direct testing (optional)
if __name__ == "__main__":
    predictor = DiabetesPredictor()
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess("diabetes_dataset.csv")
    predictor.train(X_train, y_train)
    predictor.evaluate(X_train, X_test, y_train, y_test)
    
    sample_patient = [2, 130, 80, 25, 150, 28, 0.6, 45]
    predictions = predictor.predict(sample_patient)

    print("\nRandom Forest:")
    print(f"Diagnosis: {'Diabetic' if predictions['random_forest'][0] == 1 else 'Non-Diabetic'}")
    print(f"Probability: {predictions['random_forest'][1]:.2f}")

    print("\nLogistic Regression:")
    print(f"Diagnosis: {'Diabetic' if predictions['logistic_regression'][0] == 1 else 'Non-Diabetic'}")
    print(f"Probability: {predictions['logistic_regression'][1]:.2f}")
