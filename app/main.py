from flask import Flask, jsonify, request
from models import static_spacy, trained_spacy, trained_tensorflow
from trainers import training_spacy, training_tensorflow
from feedback import collect_feedback
import json, os

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.static_model = static_spacy.Static_Model()
        self.dynamic_model = trained_spacy.Dynamic_Model()
        self.tensor_model = trained_tensorflow.Tensor_Model()
        self.train_spacy = training_spacy.Training_Spacy()
        self.train_tensor = training_tensorflow.Train_Tensor()
        self.collect_feedback = collect_feedback.Collect_Feedback()
        self.setup_routes()
        self.intents_path = os.path.join(os.path.dirname(__file__), '.', 'data', 'intent_mapping.json')

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return jsonify({"msg": "App initialized!"})

        @self.app.route('/get_reply', methods=['POST'])
        def get_reply():
            try:
                req_data = request.get_json()
                if not req_data or "input" not in req_data:
                    return jsonify({"error": "Missing 'input' parameter"}), 400
                msg = req_data["input"]
                model = req_data["model"]
                if not msg:
                    return jsonify({"error": "Missing 'input' param"}), 400
                if model == 2:
                    response = self.tensor_model.predictIntent(msg)
                elif model == 1:
                    response = self.dynamic_model.dynamic_response(msg)
                else:
                    response = self.static_model.static_response(msg)
                if response['status'] == 'failed':
                    return jsonify(response), 500
                return jsonify(response)
            except Exception as e:
                return jsonify({"error": f"error while getting reply {str(e)}"}), 500

        @self.app.route('/collect_feedback', methods=['POST'])
        def collect_feedback():
            req_data = request.get_json()
            example = req_data["example"]
            intent = req_data["intent"]
            responses = req_data["responses"]
            model = req_data["model"]
            if model == 0:
                return jsonify({"status": "failed", "error": "can't collect feedback for static model", "suggestion": "try 1 for spacy, 2 for tensor"}), 400
            else:
                resp = self.collect_feedback.saveFeedback(intent, example, responses, model)
            if resp['status'] == "failed":
                return jsonify(resp), 500
            return jsonify(resp)
        
        @self.app.route('/train_spacy', methods=['GET'])
        def train_spacy_function():
            epochs = request.args.get('n', default=50, type=int)
            try:
                result = self.train_spacy.train_model(epochs)
                if result['status'] == "failed":
                    return jsonify(result), 500
                return jsonify(result)
            except Exception as e:
                return jsonify({"status": "failed", "error": str(e)}), 500

        @self.app.route('/train_tensor', methods=['GET'])
        def train_tensor_function():
            epochs = request.args.get('n', default=50, type=int)
            try:
                result = self.train_tensor.createModel(epochs)
                if result['status'] == "failed":
                    return jsonify(result), 500
                return jsonify(result)
            except Exception as e:
                return jsonify({"status": "failed", "error": str(e)}), 500


    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    app_instance = App()
    app_instance.run()