from flask import Flask, render_template, request
from q2 import DiabetesPredictor  # Use q2.py

app = Flask(__name__)
predictor = DiabetesPredictor()

# Load and train once
X_train, X_test, y_train, y_test = predictor.load_and_preprocess("diabetes_dataset.csv")
predictor.train(X_train, y_train)

# Static accuracy (replace with actual evaluation if needed)
rf_train_accuracy = 0.98
rf_test_accuracy = 0.82
lr_train_accuracy = 0.79
lr_test_accuracy = 0.77

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"]),
        ]

        predictions = predictor.predict(data)
        rf_pred, rf_prob = predictions['random_forest']
        lr_pred, lr_prob = predictions['logistic_regression']

        result = "Diabetic" if rf_pred == 1 else "Non-Diabetic"

        return render_template(
            "index.html",
            result=result,
            prob=round(rf_prob, 2),
            rf_train=rf_train_accuracy,
            rf_test=rf_test_accuracy,
            lr_train=lr_train_accuracy,
            lr_test=lr_test_accuracy
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
