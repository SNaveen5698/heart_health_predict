from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='template3')

# Load the dataset (replace 'your_dataset.csv' with the path to your dataset)
df = pd.read_csv("C:\\Users\\yello\\Downloads\\heart.csv")

# Preprocessing (Add preprocessing steps specific to your dataset)
# For simplicity, let's assume that preprocessing has already been done

# Define features (X) and target variable (y)
X = df.drop(['DEATH_EVENT', 'creatinine_phosphokinase', 'ejection_fraction', 'serum_sodium', 'serum_creatinine', 'time'], axis=1)
y = df['DEATH_EVENT']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate and fit the Logistic Regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Render the index.html template for the root URL
@app.route('/')
def index():
    return render_template('death.html')

# Handle form submission and prediction
@app.route('/predict_death_event', methods=['POST'])
def predict_death_event():
    # Extract the form data
    age = float(request.form['age'])
    anaemia = int(request.form['anaemia'])
    diabetes = int(request.form['diabetes'])
    high_blood_pressure = int(request.form['high_blood_pressure'])
    platelets = float(request.form['platelets'])
    sex = int(request.form['sex'])
    smoking = int(request.form['smoking'])

    # Preprocess the input data (Feature scaling)
    input_data = scaler.transform([[age, anaemia, diabetes, high_blood_pressure, platelets, sex, smoking]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Render the result back to the UI
    return render_template('death.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
