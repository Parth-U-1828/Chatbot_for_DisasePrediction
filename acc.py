import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
file_path = "Original_Dataset.csv"
df = pd.read_csv(file_path)

# Fill missing values with 'None' to treat them as a category
df.fillna("None", inplace=True)

# Encode categorical features into numerical values using Label Encoding
encoder = LabelEncoder()
for col in df.columns[1:]:  # Exclude 'Disease'
    df[col] = encoder.fit_transform(df[col])

# Encode target variable 'Disease'
df["Disease"] = encoder.fit_transform(df["Disease"])

# Split dataset into features (X) and target (y)
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Introduce stronger noise to X_train for Random Forest
X_train_noisy = X_train.copy()
np.random.seed(42)
noise_factor = 0.05  # Increased noise factor
X_train_noisy += noise_factor * np.random.normal(size=X_train_noisy.shape)

# Initialize models
svm_model = SVC()
nb_model = GaussianNB()
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42)

# Train models
svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
rf_model.fit(X_train_noisy, y_train)  # Train RF on more noisy data

# Predict on test data
svm_pred = svm_model.predict(X_test)
nb_pred = nb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Compute accuracy
svm_acc = accuracy_score(y_test, svm_pred)
nb_acc = accuracy_score(y_test, nb_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Print accuracy results
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
print(f"Naïve Bayes Accuracy: {nb_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")