import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def train_disease_predictor(csv_file):
    # Load data from CSV
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return

    # Check for missing data
    if data.isnull().values.any():
        print("Error: CSV file contains missing values.")
        print(data[data.isnull().any(axis=1)])  # Print rows with missing data
        return

    # Ensure the CSV has the required columns
    if 'Symptoms' not in data.columns or 'Name' not in data.columns:
        raise ValueError("CSV file must contain 'Symptoms' and 'Name' columns.")

    symptoms = data['Symptoms']
    diseases = data['Name']

    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptoms)
    y = diseases

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    classifier =SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train)

    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save model and vectorizer
    with open("models/disease_predictor.pkl", "wb") as f:
        pickle.dump((vectorizer, classifier), f)
    print("Model saved as 'models/disease_predictor.pkl'.")

if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "disease_symptoms.csv"
    train_disease_predictor(csv_file)
