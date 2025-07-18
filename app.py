from flask import Flask, render_template, request
from q1 import DiabetesPredictor  # or from q2 import DiabetesPredictor

app = Flask(__name__)
predictor = DiabetesPredictor()

# Load and train model once
X_train, X_test, y_train, y_test = predictor.load_and_preprocess("diabetes_dataset.csv")
predictor.train(X_train, y_train)

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
        prediction, probability = predictor.predict(data)
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return render_template("index.html", result=result, prob=round(probability, 2))
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
