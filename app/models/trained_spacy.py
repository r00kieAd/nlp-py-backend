'''
How This Works
1.	Train a classifier using TextCategorizer in spaCy.
2.	Assign intent labels (greeting, goodbye, thanks) to phrases.
3.	Train the model using the given dataset.
4.	Predict intent based on user input.
5.	Generate dynamic responses based on intent.
'''

import spacy
import random

class Dynamic_Model:
    def __init__(self):
        self.nlp = spacy.load("trainers/custom_trained_model")
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