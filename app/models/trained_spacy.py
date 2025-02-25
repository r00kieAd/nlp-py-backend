import spacy, random, os

class Dynamic_Model:
    def __init__(self):
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'trainers', 'custom_trained_model')
        self.nlp = spacy.load(self.trained_model_path)
        self.responses = {
            "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
            "goodbye": ["Goodbye!", "See you later!", "Take care!"],
            "thanks": ["You're welcome!", "No problem!", "Anytime!"],
            "unknown": ["Sorry, I didn't understand that.", "Can you rephrase?"]
        }

    def predict_intent(self, text):
        doc = self.nlp(text)
        scores = doc.cats  # Intent scores
        best_intent = max(scores, key=scores.get)  # Highest probability intent
        return best_intent if scores[best_intent] > 0.6 else "unknown"

    def dynamic_response(self, user_input):
        intent = self.predict_intent(user_input)
        return random.choice(self.responses.get(intent, self.responses["unknown"]))

    def run_model(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            print("Chatbot:", self.dynamic_response(user_input))

obj = Dynamic_Model()
obj.run_model()