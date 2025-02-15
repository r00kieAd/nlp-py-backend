from flask import Flask, jsonify
from models import static_spacy
import requests

app = Flask(__name__)

class App:
    def __init__(self):
        self.reply = "empty"
        self.static_model = static_spacy.Model()

    @app.route('/')
    def home():
        return jsonify({"msg": "Flask app initialized!"})

    @app.route('/reply')
    def getReply(self):
        _input = requests.get("input")
        obj = self.static_model.static_response()

if __name__ == '__main__':
    app.run(debug=True)