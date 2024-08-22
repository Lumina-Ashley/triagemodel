from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv", encoding="ISO-8859-1")

# Extract unique diseases for the dropdown
diseases = sorted(data['Disease'].unique())

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features and target variable for diagnostic classifier
X_diagnostic = data.drop(['Outcome Variable', 'Disease'], axis=1)
y_diagnostic = data['Outcome Variable']
X_train_diagnostic, X_test_diagnostic, y_train_diagnostic, y_test_diagnostic = train_test_split(X_diagnostic, y_diagnostic, test_size=0.2, random_state=42)

# Split data into features and target variable for predictive classifier
X_predictive = data[['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']]
y_predictive = data['Outcome Variable']
X_train_predictive, X_test_predictive, y_train_predictive, y_test_predictive = train_test_split(X_predictive, y_predictive, test_size=0.2, random_state=42)

# Initialize the Random Forest classifiers
clf_diagnostic = RandomForestClassifier(n_estimators=100, random_state=42)
clf_diagnostic.fit(X_train_diagnostic, y_train_diagnostic)

clf_predictive = RandomForestClassifier(n_estimators=100, random_state=42)
clf_predictive.fit(X_train_predictive, y_train_predictive)

def predict_new_case(disease, symptoms, age, gender, blood_pressure, cholesterol_level):
    # Transform input data
    gender = label_encoders['Gender'].transform([gender])[0]
    blood_pressure = label_encoders['Blood Pressure'].transform([blood_pressure])[0]
    cholesterol_level = label_encoders['Cholesterol Level'].transform([cholesterol_level])[0]

    # Transform symptoms into binary form
    for key in symptoms:
        symptoms[key] = 1 if symptoms[key] == 'Yes' else 0

    new_patient = pd.DataFrame({
        'Disease': [disease],
        'Fever': [symptoms['Fever']],
        'Cough': [symptoms['Cough']],
        'Fatigue': [symptoms['Fatigue']],
        'Difficulty Breathing': [symptoms['Difficulty Breathing']],
        'Age': [age],
        'Gender': [gender],
        'Blood Pressure': [blood_pressure],
        'Cholesterol Level': [cholesterol_level]
    })

    # Predict the outcome using both classifiers
    outcome_diagnostic = clf_diagnostic.predict(new_patient.drop(['Disease'], axis=1))
    outcome_predictive = clf_predictive.predict(new_patient[['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']])

    # Update threshold for diagnostic classifier to prioritize sensitivity
    threshold_diagnostic = 0.5  # Example threshold, can be adjusted based on sensitivity requirements
    outcome_prob_diagnostic = clf_diagnostic.predict_proba(new_patient.drop(['Disease'], axis=1))[:, 1]
    outcome_adjusted_diagnostic = (outcome_prob_diagnostic >= threshold_diagnostic).astype(int)

    if outcome_adjusted_diagnostic[0] == 0:
        diagnostic_prediction = "Non-Emergency, No Serious Disease Detected."
    else:
        diagnostic_prediction = "Non-Emergency, Serious Disease Detected."

    # Perform proactive disease management based on diagnostic prediction
    if outcome_diagnostic[0] == 1:  # If diagnostic classifier predicts a positive outcome
        if outcome_predictive[0] == 1:  # If predictive classifier also predicts a positive outcome
            proactive_action = "Proactive intervention recommended."
        else:
            proactive_action = "No proactive intervention recommended."
    else:
        proactive_action = "No proactive intervention recommended."

    return diagnostic_prediction, proactive_action

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        form_data = request.form

        # Convert form data to appropriate types
        try:
            disease = form_data.get('disease')
            symptoms = {
                'Fever': form_data.get('fever'),
                'Cough': form_data.get('cough'),
                'Fatigue': form_data.get('fatigue'),
                'Difficulty Breathing': form_data.get('difficulty_breathing')
            }
            age = int(form_data.get('age'))
            gender = form_data.get('gender')
            blood_pressure = form_data.get('blood_pressure')
            cholesterol_level = form_data.get('cholesterol')
        except Exception as e:
            return render_template('index.html', error="Invalid input. Please check your entries.", diseases=diseases)

        # Predict using the model
        diagnostic_prediction, proactive_action = predict_new_case(disease, symptoms, age, gender, blood_pressure, cholesterol_level)

        # Calculate scores for the diagnostic model
        y_pred_diagnostic = clf_diagnostic.predict(X_test_diagnostic)
        recall_diagnostic = round(recall_score(y_test_diagnostic, y_pred_diagnostic) * 100, 2)
        accuracy_diagnostic = round(accuracy_score(y_test_diagnostic, y_pred_diagnostic) * 100, 2)

        # Calculate scores for the predictive model
        y_pred_predictive = clf_predictive.predict(X_test_predictive)
        precision_predictive = round(precision_score(y_test_predictive, y_pred_predictive) * 100, 2)
        f1_predictive = round(f1_score(y_test_predictive, y_pred_predictive) * 100, 2)

        # Prepare message text with scores
        return render_template('result.html',
                               diagnostic_prediction=diagnostic_prediction,
                               proactive_action=proactive_action,
                               recall_diagnostic=recall_diagnostic,
                               accuracy_diagnostic=accuracy_diagnostic,
                               precision_predictive=precision_predictive,
                               f1_predictive=f1_predictive)

    return render_template('index.html', diseases=diseases)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)