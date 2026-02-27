import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import logging
import sqlite3
import bcrypt
from functools import wraps
from sklearn.neighbors import KNeighborsClassifier

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database setup
def init_db():
    with sqlite3.connect("users.db") as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (username TEXT UNIQUE, password TEXT)''')
        conn.commit()

# Mock entity extraction (updated in entity_extraction.py)
from entity_extraction import extract_entities

# Train the disease predictor model (unchanged)
def train_disease_predictor(csv_file):
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"Error: File '{csv_file}' not found.")
        return
    if data.isnull().values.any():
        logging.error("Error: CSV file contains missing values.")
        return
    if 'Symptoms' not in data.columns or 'Name' not in data.columns:
        raise ValueError("CSV file must contain 'Symptoms' and 'Name' columns.")
    symptoms = data['Symptoms']
    diseases = data['Name']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptoms)
    y = diseases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier =SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")
    with open("models/disease_predictor.pkl", "wb") as f:
        pickle.dump((vectorizer, classifier), f)
    logging.info("Model saved.")

# Load disease precautions
def load_disease_precautions(csv_file):
    try:
        data = pd.read_csv(csv_file)
        return dict(zip(data['Name'], data['Treatments']))
    except FileNotFoundError:
        logging.error("Error loading CSV file.")
        return {}

# Load model
csv_file = "disease_symptoms.csv"
disease_precaution_map = load_disease_precautions(csv_file)
try:
    with open("models/disease_predictor.pkl", "rb") as f:
        vectorizer, classifier = pickle.load(f)
except FileNotFoundError:
    logging.error("Model file not found.")
    vectorizer = None
    classifier = None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            return render_template("signup.html", error="Username and password are required.")
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            with sqlite3.connect("users.db") as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Username already exists.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        with sqlite3.connect("users.db") as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
                session['username'] = username
                return redirect(url_for('home'))
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/")
@login_required
def home():
    return render_template("index.html", greeting="Hello, welcome to the Medical Chatbot! How can I assist you today?")

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    if not vectorizer or not classifier:
        return jsonify({"error": "Model not loaded."})
    
    user_input = request.json.get("message", "").strip().lower()
    logging.info(f"User input: {user_input}")
    
    # Extract entities
    entities = extract_entities(user_input)
    if "error" in entities:
        return jsonify({"error": entities["error"]})
    
    symptoms = entities.get("SYMPTOM", [])
    response = {}
    
    if symptoms:
        symptom_str = ", ".join(symptoms)
        try:
            vectorized_input = vectorizer.transform([symptom_str])
            predicted_disease = classifier.predict(vectorized_input)[0]
            logging.info(f"Predicted disease: {predicted_disease}")
            precaution = disease_precaution_map.get(predicted_disease, "Consult a doctor.")
            response = {
                "symptoms": symptoms,
                "predicted_disease": predicted_disease,
                "precaution": precaution,
            }
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return jsonify({"error": "Prediction failed."})
    else:
        response["error"] = "No medical symptoms identified. Please provide symptoms or diseases."
    
    return jsonify(response)

if __name__ == "__main__":
    init_db()  # Initialize database
    app.run(debug=True)