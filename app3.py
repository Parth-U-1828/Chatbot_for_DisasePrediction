from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_pymongo import PyMongo
import bcrypt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mode
import os
from difflib import get_close_matches

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot_members"
mongo = PyMongo(app)

# Load dataset
try:
    data = pd.read_csv("Original_Dataset.csv")
except FileNotFoundError:
    print("Error: Dataset file not found. Make sure 'Original_Dataset.csv' is present.")
    exit()

# Convert symptoms into standardized sets
data.fillna("", inplace=True)
data["Symptoms"] = data.iloc[:, 1:].apply(lambda row: set(str(val).strip().lower().replace(" ", "_") for val in row.values if val), axis=1)

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data["Symptoms"])
y = data["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
final_svm_model = SVC().fit(X_train, y_train)
final_nb_model = GaussianNB().fit(X_train, y_train)
final_rf_model = RandomForestClassifier().fit(X_train, y_train)

# Calculate accuracies
svm_accuracy = accuracy_score(y_test, final_svm_model.predict(X_test))
nb_accuracy = accuracy_score(y_test, final_nb_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, final_rf_model.predict(X_test))

# Store model details in app configuration
app.config["MODEL_ACCURACIES"] = {
    "SVM": svm_accuracy,
    "Naive Bayes": nb_accuracy,
    "Random Forest": rf_accuracy
}
app.config["BEST_MODEL"] = max(app.config["MODEL_ACCURACIES"], key=app.config["MODEL_ACCURACIES"].get)

# Get available symptoms from dataset
known_symptoms = set(mlb.classes_)

# Function to process and validate symptoms
def process_symptoms(user_symptoms):
    user_symptoms = [symptom.strip().lower().replace(" ", "_") for symptom in user_symptoms]

    valid_symptoms = list(set(user_symptoms) & known_symptoms)
    unknown_symptoms = list(set(user_symptoms) - known_symptoms)

    # Suggest close matches for unknown symptoms
    symptom_suggestions = {
        symptom: get_close_matches(symptom, known_symptoms, n=3) for symptom in unknown_symptoms
    }

    if not valid_symptoms:
        return None, unknown_symptoms, symptom_suggestions

    return mlb.transform([valid_symptoms]), unknown_symptoms, symptom_suggestions

# Function to predict disease
def predict_disease(user_symptoms):
    input_vector, unknown_symptoms, suggestions = process_symptoms(user_symptoms)

    if input_vector is None:
        return {
            "error": "No valid symptoms recognized.",
            "unknown_symptoms": unknown_symptoms,
            "suggestions": suggestions
        }

    predictions = {
        "SVM": final_svm_model.predict(input_vector)[0],
        "Naive Bayes": final_nb_model.predict(input_vector)[0],
        "Random Forest": final_rf_model.predict(input_vector)[0]
    }

    # Get doctor recommendations
    try:
        doctors_df = pd.read_csv("Doctor_Versus_Disease.csv")
        matching_doctors = doctors_df[doctors_df['Disease'].str.lower() == predictions["Random Forest"].lower()]
        recommendations = matching_doctors.to_dict('records')
    except Exception as e:
        recommendations = []
        print(f"Error loading doctor recommendations: {str(e)}")

    return {
        "individual_predictions": predictions,
        "best_model": "Random Forest",
        "best_prediction": predictions["Random Forest"],
        "ensemble_prediction": predictions["Random Forest"],
        "unknown_symptoms": unknown_symptoms,
        "suggestions": suggestions,
        "recommendations": recommendations
    }

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        user = mongo.db.users.find_one({'username': username})

        if user and bcrypt.checkpw(password, user['password']):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('chatbot'))
        else:
            flash("Invalid credentials. Please try again.", "error")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())

        if mongo.db.users.find_one({'username': username}):
            flash("Username already exists!", "error")
        else:
            mongo.db.users.insert_one({'username': username, 'password': password})
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    return render_template('chatbot.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized", "redirect": url_for('login')}), 401

    try:
        data = request.get_json()
        if not data or "symptoms" not in data:
            return jsonify({"error": "No symptoms provided"}), 400

        user_symptoms = data["symptoms"]

        if isinstance(user_symptoms, str):
            user_symptoms = user_symptoms.split(",")
        elif not isinstance(user_symptoms, list):
            return jsonify({"error": "'symptoms' must be a list or a comma-separated string"}), 400

        prediction_results = predict_disease(user_symptoms)

        if "error" in prediction_results:
            return jsonify(prediction_results), 400

        return jsonify({
            "predictions": prediction_results["individual_predictions"],
            "accuracies": {model: round(acc * 100, 2) for model, acc in app.config["MODEL_ACCURACIES"].items()},
            "best_model": prediction_results["best_model"],
            "best_prediction": prediction_results["best_prediction"],
            "ensemble_prediction": prediction_results["ensemble_prediction"],
            "unknown_symptoms": prediction_results["unknown_symptoms"],
            "suggestions": prediction_results["suggestions"],
            "recommendations": prediction_results.get("recommendations", [])
        })

    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500

@app.route('/doc')
def doctors():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    disease = request.args.get('disease')
    if not disease:
        flash("No disease specified", "error")
        return redirect(url_for('chatbot'))

    return render_template('doc.html', disease=disease)

@app.route('/api/doc')
def api_doctors():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    disease = request.args.get('disease')
    if not disease:
        return jsonify({"error": "No disease specified"}), 400

    try:
        doctors_df = pd.read_csv("Doctor_Versus_Disease.csv")
        matching_doctors = doctors_df[doctors_df['Disease'].str.lower() == disease.lower()]
        
        if matching_doctors.empty:
            return jsonify({"doctors": []})
        
        return jsonify({
            "doctors": matching_doctors.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": f"Error fetching doctors: {str(e)}"}), 500

@app.route('/evaluate')
def evaluate():
    if 'username' not in session:
        flash("You need to log in first!", "error")
        return redirect(url_for('login'))

    accuracies = app.config["MODEL_ACCURACIES"]

    return render_template('evaluate.html',
                          rf_accuracy=round(accuracies["Random Forest"]*100, 2),
                          nb_accuracy=round(accuracies["Naive Bayes"]*100, 2),
                          svm_accuracy=round(accuracies["SVM"]*100, 2),
                          best_model=app.config["BEST_MODEL"])

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/get_username')
def get_username():
    if 'username' in session:
        return jsonify({"username": session['username']})
    return jsonify({"username": None})

if __name__ == '__main__':
    app.run(debug=True)