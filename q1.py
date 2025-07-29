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
        data = pd.read_csv(file_path)
        columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        data[columns_to_fix] = data[columns_to_fix].astype(float)

        for column in columns_to_fix:
            median = data[data[column] != 0][column].median()
            data.loc[data[column] == 0, column] = median
        
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, patient_data):
        scaled_data = self.scaler.transform([patient_data])
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        return prediction[0], probability[0][1]
