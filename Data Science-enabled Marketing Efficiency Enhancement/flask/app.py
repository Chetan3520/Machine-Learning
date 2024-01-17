from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
loaded_model = pickle.load(open('Regressor2.pkl', 'rb'))
Industry_l1=pickle.load(open('l1.pkl', 'rb'))
Fund_category_l2=pickle.load(open('l2.pkl', 'rb'))
Geography_l3=pickle.load(open('l3.pkl', 'rb'))  
Designation_l4=pickle.load(open('l4.pkl', 'rb'))
Last_lead_update_l5=pickle.load(open('l5.pkl', 'rb'))
Resource_l6=pickle.load(open('l6.pkl', 'rb'))
'''# Load the pickled model
with open('Regressor2.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the pickled label encoders
with open('label_encoders1.pkl', 'rb') as file:
    loaded_label_encoders = pickle.load(file)'''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect input features from the form
    industryInput = request.form['industryInput']
    Deal_value = request.form['Deal_value']
    Weighted_amount = request.form['Weighted_amount']
    fundCategoryInput = request.form['fundCategoryInput']
    geographyInput = request.form['geographyInput']
    designationInput = request.form['designationInput']
    resourceInput = request.form['resourceInput']
    lastLeadUpdateInput = request.form['lastLeadUpdateInput']
    internalRating = request.form['internalRating']

    input_features =[[Industry_l1.transform([industryInput])[0],float(Deal_value),float(Weighted_amount),Fund_category_l2.transform([fundCategoryInput])[0],Geography_l3.transform([geographyInput])[0],Designation_l4.transform([designationInput])[0],Resource_l6.transform([resourceInput])[0],Last_lead_update_l5.transform([lastLeadUpdateInput])[0],float(internalRating)]]

    print(input_features)
    '''name = ['Industry','Deal_value','Weighted_amount','Fund_category','Geography', 'Designation', 'Resource','Last_lead_update','Internal_rating']
    x=[np.array(input_features)]
    df =pd.DataFrame(x,columns=name)'''

    pred =loaded_model.predict(input_features)
    pred=np.round(pred,2)
    #loaded_model.predict(data)
    return render_template("inner-page.html", predict=pred[0])

if __name__ == '__main__':
    app.run(debug=True)
