from flask import Flask, request, jsonify
import json
import csv
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# define the Flask app
app = Flask(__name__)

# define the endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # read the input JSON data
    input_data = request.get_json()

    # convert the JSON to a CSV file
    with open('data/data.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(input_data[0].keys())
        for item in input_data:
            writer.writerow(item.values())

    # load the CSV file
    df = pd.read_csv('data/data.csv', index_col=0)

    # preprocess the input data
    ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
    X = ct.fit_transform(df.iloc[:, :-1].values)
    y = df.iloc[:, -1].values

    # train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # save the model to disk
    joblib.dump(model, 'sales_model.joblib')

    # make a prediction using the loaded model
    y_pred = model.predict(X)

    # calculate the accuracy of the model
    accuracy = r2_score(y, y_pred) * 100

    # return the predicted value and accuracy as JSON response
    output = {'prediction': y_pred[0], 'accuracy': "{:.2f}%".format(accuracy)}

    return jsonify(output)

# run the app
if __name__ == '__main__':
    app.run(debug=True)
