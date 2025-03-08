import spacy, random, os, json

class Dynamic_Model:
    def __init__(self):
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_trained_model')
        self.json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
        self.nlp = self.load_model()
        self.responses = self.load_responses()

    def load_model(self):
        try:
            return spacy.load(self.trained_model_path)
        except:
            return None

    def load_responses(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as file:
                return json.load(file)
        else:
            default_responses = {
                "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
                "goodbye": ["Goodbye!", "See you later!", "Take care!"],
                "thanks": ["You're welcome!", "No problem!", "Anytime!"],
                "unknown": ["Sorry, I didn't understand that.", "Can you rephrase?"]
            }
            self.save_responses(default_responses)
            return default_responses

    def save_responses(self, data):
        with open(self.json_path, 'w') as file:
            json.dump(data, file, indent=4)

    def predict_intent(self, text):
        doc = self.nlp(text)
        scores = doc.cats
        best_intent = max(scores, key=scores.get)
        return best_intent if scores[best_intent] > 0.6 else "unknown"

    def dynamic_response(self, user_input):
        intent = self.predict_intent(user_input)
        return random.choice(self.responses.get(intent, self.responses["unknown"]))

