from flask import Flask, jsonify, request
from app.models import static_spacy, trained_spacy, trained_tensorflow
from app.trainers import training_spacy, training_tensorflow
from app.feedback import collect_feedback
from app.history import get_history
import json, os, logging

logging.basicConfig(level=logging.ERROR)

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.static_model = static_spacy.Static_Model()
        self.dynamic_model = trained_spacy.Dynamic_Model()
        self.tensor_model = trained_tensorflow.Tensor_Model()
        self.train_spacy = training_spacy.Training_Spacy()
        self.train_tensor = training_tensorflow.Train_Tensor()
        self.collect_feedback = collect_feedback.Collect_Feedback()
        self.get_history = get_history.Get_History()
        self.setup_routes()
        self.intents_path = os.path.join(os.path.dirname(__file__), '.', 'data', 'intent_mapping.json')

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return jsonify({"msg": "App initialized!"})

        @self.app.route('/get_reply', methods=['POST'])
        def getReply():
            try:
                req_data = request.get_json()
                if not req_data or "input" not in req_data:
                    return jsonify({"error": "Missing 'input' parameter"}), 400
                msg = req_data["input"]
                model = req_data["model"]
                if not msg:
                    return jsonify({"error": "Missing 'input' param"}), 400
                if model == 2:
                    top_p = req_data["top_p"]
                    top_k = req_data["top_k"]
                    temp = req_data["temperature"]
                    conf_thresh = req_data["confidence_threshold"]
                    response = self.tensor_model.predictIntent(msg, top_p, top_k, temp, conf_thresh)
                elif model == 1:
                    response = self.dynamic_model.dynamic_response(msg)
                else:
                    response = self.static_model.static_response(msg)
                if response['status'] == 'failed':
                    return jsonify(response), 500
                return jsonify(response)
            except Exception as e:
                logging.error(f"Error while getting reply...", exc_info=True)
                return jsonify({"error": f"error while getting reply {str(e)}"}), 500

        @self.app.route('/collect_feedback', methods=['PUT'])
        def collectFeedback():
            try:
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
            except Exception as e:
                logging.error("Error while collecting feedback...", exc_info=True)
                return jsonify({"status": "failed", "error": str(e)}), 500
        
        @self.app.route('/train_spacy', methods=['GET'])
        def trainSpacyFunction():
            try:
                epochs = request.args.get('n', default=50, type=int)
                try:
                    result = self.train_spacy.train_model(epochs)
                    if result['status'] == "failed":
                        return jsonify(result), 500
                    return jsonify(result)
                except Exception as e:
                    return jsonify({"status": "failed", "error": str(e)}), 500
            except Exception as e:
                logging.error("Error while training spacy...", exc_info=True)
                return jsonify({"status": "failed", "error": str(e)}), 500

        @self.app.route('/train_tensor', methods=['GET'])
        def trainTensorFunction():
            epochs = request.args.get('n', default=50, type=int)
            try:
                result = self.train_tensor.createModel(epochs)
                if result['status'] == "failed":
                    return jsonify(result), 500
                return jsonify(result)
            except Exception as e:
                logging.error("Error while training tensor...", exc_info=True)
                return jsonify({"status": "failed", "error": str(e)}), 500

        @self.app.route('/tensor_training_history', methods=['GET'])
        def getTensorHistory():
            try:
                data = self.get_history.getData()
                if data["status"] == "failed":
                    return jsonify(data), 500
                return jsonify(data)
            except Exception as e:
                logging.error('Exception in get tensor history', exc_info=True)
                return jsonify({"status": "failed", "error": "unable to get history"}), 500

    # def run(self):
    #     env = os.getenv("FLASK_ENV", "dev")
    #     if env == "prod":
    #         logging.info("Running in production...")
    #         self.app.run(host="0.0.0.0", port=8000)
    #     else:
    #         logging.info("Running in development...")
    #         self.app.run(debug=True)
    #     logging.warning("App stopped.")

if __name__ == '__main__':
    app_instance = App()
    # app = app_instance.app
    app_instance.run()

app = App().app