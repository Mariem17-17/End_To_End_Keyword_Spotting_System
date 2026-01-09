

"""
server

client -> POST request -> server -> prediction back to client

"""
from keyword_spotting_service import Keyword_Spotting_Service
from flask import Flask ,   request, jsonify
import os
import random


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke the keyword spotting service
    kss = Keyword_Spotting_Service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)


    # remove the audio
    os.remove(file_name)

    # send back the predicted keyword in json format
    data={"keyword":predicted_keyword}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)