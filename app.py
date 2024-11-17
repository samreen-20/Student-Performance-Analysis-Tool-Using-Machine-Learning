from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' part is in the request
    if 'file' not in request.files:
        return jsonify(success=False, message="No file part in request")

    file = request.files['file']
    
    # Check if algorithm is in the form data
    if 'algorithm' not in request.form:
        return jsonify(success=False, message="No algorithm selected")

    algorithm = request.form['algorithm']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify(success=False, message="No selected file")

    # Check if file is allowed (e.g., CSV)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Sanitize filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(filepath)
        
        # Process the file
        try:
            prediction_result = process_file(filepath, algorithm)
            return jsonify(success=True, result=prediction_result)
        except Exception as e:
            return jsonify(success=False, message=str(e))

    return jsonify(success=False, message="Invalid file format. Only CSV files are allowed.")

def process_file(filepath, algorithm):
    # Load dataset
    df = pd.read_csv(filepath)

    # Assuming the dataset has 'math score', 'reading score', 'writing score' as features
    X = df[['math score', 'reading score', 'writing score']]  # Update as per your dataset

    # Create a target column based on whether the student passes (average score > 50)
    df['target'] = (df[['math score', 'reading score', 'writing score']].mean(axis=1) > 50).astype(int)
    y = df['target']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Choose model based on selected algorithm
    if algorithm == 'decision_tree':
        model = DecisionTreeClassifier()
    elif algorithm == 'random_forest':
        model = RandomForestClassifier()
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    else:
        raise ValueError("Unknown algorithm")

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)

    # Optionally save the model
    model_path = os.path.join(UPLOAD_FOLDER, f'{algorithm}_model.pkl')
    joblib.dump(model, model_path)

    return f"{algorithm.replace('_', ' ').title()} Model Accuracy: {score * 100:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)

