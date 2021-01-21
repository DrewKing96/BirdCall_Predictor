import os
import json
import requests
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from datetime import date, datetime

import makePredictions

#Create Instance of Flask App
app = Flask(__name__)

headers = {
        'Content-Type': 'application/json'
        }

@app.route("/ml_sensordata", methods=['POST'])
def receive():
    if(request.is_json):
        content = request.get_json()
    prediction = makePredictions.predict(content['url'], content['filename'])
    print(prediction)
    print(str(prediction))
    #send prediction to webapp server
    today = date.today()
    d1 = today.strftime("%B %d %Y")
    time = datetime.now()
    time_string = time.strftime("%H:%M:%S")
    payload = {}
    payload["classification"] = prediction
    payload["date"] = d1
    payload["time"] = time_string

    response = requests.post('http://ip_address_webapplication:/prediction', headers = headers, data = json.dumps(payload))
    print(response.text)
    return str(prediction)

if __name__ == "__main__":
    app.run(host='ip_address', debug=False)
