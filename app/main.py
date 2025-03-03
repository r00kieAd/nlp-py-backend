from flask import Flask, jsonify, request
from models import static_spacy, trained_spacy
from trainers import training_spacy
import json, os

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.static_model = static_spacy.Static_Model()
        self.dynamic_model = trained_spacy.Dynamic_Model()
        self.train_spacy = training_spacy.Training_Spacy()
        self.setup_routes()
        self.intents_path = os.path.join(os.path.dirname(__file__), '.', 'data', 'intent_mapping.json')

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return jsonify({"msg": "App initialized!"})

        @self.app.route('/get_reply', methods=['POST'])
        def get_reply():
            req_data = request.get_json()
            if not req_data or "input" not in req_data:
                return jsonify({"error": "Missing 'input' parameter"}), 400
            msg = req_data["input"]
            mode = req_data["mode"]
            if mode == 1:
                response = self.dynamic_model.dynamic_response(msg)
            else:
                mode = 0
                response = self.static_model.static_response(msg)
            return jsonify({"reply": response, "mode": mode})

        @self.app.route('/collect_feedback', methods=['POST'])
        def collect_feedback():
            req_data = request.get_json()
            example = req_data["example"]
            intent = req_data["intent"]
            with open(self.intents_path, 'r') as file:
                data = json.load(file)
            data['intents'].append({example: intent})
            with open(self.intents_path, 'w') as file:
                json.dump(data, file, indent=4)
            return jsonify({"status": "success"})
        
        @self.app.route('/train_spacy', methods=['GET'])
        def train_spacy_function():
            try:
                return jsonify(self.train_spacy.train_model())
            except Exception as e:
                return jsonify({"status": "failed", "error": str(repr(e))})


    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    app_instance = App()
    app_instance.run()