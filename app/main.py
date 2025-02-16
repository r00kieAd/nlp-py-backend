from flask import Flask, jsonify, request
from models import static_spacy

class App:
    def __init__(self):
        self.app = Flask(__name__)
        self.static_model = static_spacy.Model()
        self.setup_routes()  # This helps the routes to work inside the class

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return jsonify({"msg": "Flask app initialized!"})

        @self.app.route('/reply', methods=['POST'])
        def get_reply():
            req_data = request.get_json()
            if not req_data or "input" not in req_data:
                return jsonify({"error": "Missing 'input' parameter"}), 400
            msg = req_data["input"]
            response = self.static_model.static_response(msg)
            return jsonify({"reply": response})

    def run(self):
        self.app.run(debug=True)

# Instantiating app for all the routes to work when it's running
if __name__ == '__main__':
    app_instance = App()
    app_instance.run()