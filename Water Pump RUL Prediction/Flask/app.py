from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
loaded_model = pickle.load(open('final_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    x = [np.array(input_feature)]
    print(input_feature)
    names = ['sensor_22', 'sensor_30', 'sensor_29', 'sensor_25', 'sensor_23', 'sensor_32', 'sensor_26', 'sensor_28',
             'sensor_35', 'sensor_00', 'sensor_36', 'sensor_13']
    data = pd.DataFrame(x, columns=names)
    print(data)

   
    pred = (loaded_model.predict(data))
    pred_int=int(np.round(pred))
    print(pred_int)
    def months_to_years_months(n):
        years = n // 12
        months = n % 12
        return '{} years, {} months'.format(years, months)
    pred=months_to_years_months(pred_int)
    print(pred)
    return render_template("inner-page.html", predict=pred)

if __name__ == '__main__':
    app.run(debug=True)
