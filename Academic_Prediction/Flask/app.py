from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
loaded_model = pickle.load(open('final_model2.pkl', 'rb'))
loaded_preprocessor = pickle.load(open('preprocessor1.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
     # Collect input features from the form
    course = request.form['father_occupatio']
    prev_qualification_grade = float(request.form['prev_qualification_grade'])
    father_occupation = request.form['father_occupation']
    admission_grade = float(request.form['admission_grade'])
    age_at_enrollment = float(request.form['age_at_enrollment'])
    cu1s_approved = float(request.form['cu1s_approved'])
    cu1s_grade = float(request.form['cu1s_grade'])
    cu2s_evaluations = float(request.form['cu2s_evaluations'])
    cu2s_approved = float(request.form['cu2s_approved'])
    cu2s_grade = float(request.form['cu2s_grade'])
   
    # Create a dictionary with collected data
    data = {
        'course': [course],
        'prev_qualification_grade': [prev_qualification_grade],
        'father_occupation': [father_occupation],
        'admission_grade': [admission_grade],
        'age_at_enrollment': [age_at_enrollment],
        'cu1s_approved': [cu1s_approved],
        'cu1s_grade': [cu1s_grade],
        'cu2s_evaluations': [cu2s_evaluations],
        'cu2s_approved': [cu2s_approved],
        'cu2s_grade': [cu2s_grade],
       
    }

    # Create a DataFrame from the collected data
    data_df = pd.DataFrame(data)

    # Transform the input data using the loaded preprocessor
    processed_data = loaded_preprocessor.transform(data_df)
    print(processed_data)
    # Make predictions using the loaded model
    pred = loaded_model.predict(processed_data)
    print(pred)
    def check_enrollment_status(prediction):
        if prediction == 0:
            return "DROPOUT"
        elif prediction == 2:
            return "GRADUATED"
        else:
            return "ENROLLED"
    pred_=check_enrollment_status(pred)
    print(pred_)
    return render_template("blog.html", predict=pred_)

if __name__ == '__main__':
    app.run(debug=True)
