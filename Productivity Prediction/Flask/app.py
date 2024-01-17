from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)
loaded_model = pickle.load(open('final_model1.pkl', 'rb'))
loaded_preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
     # Collect input features from the form
    quarter = request.form['Quarter']
    department = request.form['Department']
    day = request.form['Day']
    team = request.form['Team'] 
    targeted_productivity = request.form['TargetedProductivity']
    smv = request.form['Standard_Minute_Value']
    over_time = request.form['over_time']
    incentive = request.form['Incentive']
    idle_time = request.form['idle_time']
    idle_men = request.form['idle_men']
    no_of_style_change = request.form['no_of_style_change']
    no_of_workers = request.form['no_of_workers']
    data = {
    'quarter': [quarter],
    'department': [department],
    'day': [day],
    'team': [float(team)],
    'targeted_productivity': [float(targeted_productivity)],
    'smv': [float(smv)],
    'over_time': [float(over_time)],
    'incentive': [float(incentive)],
    'idle_time': [float(idle_time)],
    'idle_men': [float(idle_men)],
    'no_of_style_change': [float(no_of_style_change)],
    'no_of_workers': [float(no_of_workers)]}

    data=pd.DataFrame(data)
    print(data)
    # Transform the input data using the loaded preprocessor
    processed_data = loaded_preprocessor.transform(data)
    print(processed_data)
    # Make predictions using the loaded model
    pred = loaded_model.predict(processed_data)
    print(pred)
    #loaded_model.predict(data)
    def check(x):
        if x[0]==1:
            return("Productive")
        else:
            return( "Unproductive")
    pred_=check(pred)
   
    return render_template("inner-page.html", predict=pred_)

if __name__ == '__main__':
    app.run(debug=True)
