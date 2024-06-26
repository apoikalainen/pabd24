"""House price prediction service"""
import os

from dotenv import dotenv_values
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
from joblib import load
import pandas as pd

MODEL_SAVE_PATH = 'models/catboost_v1.joblib'

app = Flask(__name__)
CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config['APP_TOKEN']: "pabd24",
}

model = load(MODEL_SAVE_PATH)


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]


def predict(in_data: dict) -> int:
    """ Predict house price from input data parameters.
    :param in_data: house parameters.
    :raise Error: If something goes wrong.
    :return: House price, RUB.
    :rtype: int
    """
    col = ['author_type', 'floor', 'floors_count', 'rooms_count', 'total_meters', 'underground']
    price = model.predict(pd.DataFrame(in_data, index=[0])[col])
    return int(price.squeeze())



@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/")
def home():
    return """
    <html>
    <head>
    <link rel="shortcut icon" href="/favicon.ico">
    </head>
    <body>
    <h1>Housing price service.</h1> Use /predict endpoint
    </body>
    </html>
    """


@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    """Dummy service"""
    # in_data = request.get_json()['area']
    # price = predict_cpu_multithread(in_data)
    in_data = request.get_json()
    price = predict(in_data)
    return {'price': price}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)