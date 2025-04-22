import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from difflib import get_close_matches  # To suggest similar symptoms

# Load dataset
df = pd.read_csv("Original_Dataset.csv")

# Fill NaN values with empty strings
df.fillna("", inplace=True)

# Convert symptoms into sets of symptoms
df["Symptoms"] = df.iloc[:, 1:].apply(lambda row: set(str(val).strip().lower().replace(" ", "_") for val in row.values if val), axis=1)

# Prepare one-hot encoding
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
svm_model = SVC().fit(X_train, y_train)
nb_model = MultinomialNB().fit(X_train, y_train)
rf_model = RandomForestClassifier().fit(X_train, y_train)

# Model accuracies
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# Store models and accuracies
models = {"SVM": svm_model, "Naive Bayes": nb_model, "Random Forest": rf_model}
accuracies = {"SVM": svm_acc, "Naive Bayes": nb_acc, "Random Forest": rf_acc}

# Get known symptoms from dataset
known_symptoms = set(mlb.classes_)

# Function to predict disease
def predict_disease(symptoms):
    symptoms_set = set(symptoms)
    
    # Find intersection with known symptoms
    valid_symptoms = list(symptoms_set & known_symptoms)
    unknown_symptoms = list(symptoms_set - known_symptoms)
    
    # Suggest closest symptoms for unknown ones
    symptom_suggestions = {symptom: get_close_matches(symptom, known_symptoms, n=3) for symptom in unknown_symptoms}

    if not valid_symptoms:
        return None, None, None, unknown_symptoms, symptom_suggestions  # No valid symptoms found

    input_vector = mlb.transform([valid_symptoms])
    predictions = {model_name: model.predict(input_vector)[0] for model_name, model in models.items()}
    best_model = max(accuracies, key=accuracies.get)
    final_prediction = models[best_model].predict(input_vector)[0]

    return predictions, final_prediction, best_model, unknown_symptoms, symptom_suggestions

# User input
user_symptoms = input("Enter symptoms separated by commas: ").split(",")
user_symptoms = [symptom.strip().lower().replace(" ", "_") for symptom in user_symptoms]

predictions, final_prediction, best_model, unknown_symptoms, symptom_suggestions = predict_disease(user_symptoms)

# Output results
if not predictions:
    print("Error: No valid symptoms were recognized from your input. Please try again with different symptoms.")
    if unknown_symptoms:
        print(f"Unknown symptoms: {', '.join(unknown_symptoms)}")
        for symptom, suggestions in symptom_suggestions.items():
            if suggestions:
                print(f"Did you mean: {', '.join(suggestions)}?")
else:
    if unknown_symptoms:
        print(f"Warning: The following symptoms were not found in the dataset and were ignored: {', '.join(unknown_symptoms)}")
    
    print("\nPredictions by all models:")
    for model, disease in predictions.items():
        print(f"{model}: {disease}")
    print(f"\nFinal prediction ({best_model} - highest accuracy): {final_prediction}")
