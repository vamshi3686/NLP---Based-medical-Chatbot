import re

# Expanded medical term dictionary (extend as needed)
MEDICAL_PATTERNS = {
    "SYMPTOM": r"\b(fever|headache|cough|sore throat|sugar levels|fatigue|nausea|pain|dizziness|vomiting|diarrhea)\b",
    "DISEASE": r"\b(malaria|flu|diabetes|covid|hypertension|asthma|tuberculosis|dengue|arthritis|cancer)\b",
    "PRECAUTION": r"\b(consult a doctor|exercise regularly|take rest|hydrate|medication)\b",
}

# Common non-medical terms to filter out (e.g., names, greetings)
NON_MEDICAL_PATTERNS = r"\b(john|jane|smith|hi|hello|hey|howdy|good morning|good evening)\b"

def extract_entities(text):
    # Check for non-medical terms
    if re.search(NON_MEDICAL_PATTERNS, text, re.IGNORECASE):
        return {"error": "Input contains non-medical terms (e.g., names or greetings). Please use medical terms or diseases only."}
    
    entities = {}
    for label, pattern in MEDICAL_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities[label] = matches
    return entities

if __name__ == "__main__":
    # Test the function
    text = "I have fever and headache, not John or hi"
    print(extract_entities(text))