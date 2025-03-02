from flask import Flask, jsonify, request
from models import static_spacy, trained_spacy

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.static_model = static_spacy.Static_Model()
        self.dynamic_model = trained_spacy.Dynamic_Model()
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return jsonify({"msg": "App initialized!"})

        @self.app.route('/get_reply', methods=['POST'])
        def get_static_reply():
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

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    app_instance = App()
    app_instance.run()