import spacy, random, os, json
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

class Dynamic_Model:
    def __init__(self):
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'trainers', 'custom_trained_model')
        self.json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
        self.nlp = spacy.load(self.trained_model_path)
        self.responses = self.load_responses()

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

    def expand_responses(self, intent):
        if intent not in self.responses:
            return

        expanded_responses = set(self.responses[intent])

        for response in self.responses[intent]:
            words = response.split()
            new_sentence = []
            for word in words:
                synonyms = [lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()]
                new_word = random.choice(synonyms) if synonyms else word
                new_sentence.append(new_word)
            expanded_responses.add(" ".join(new_sentence))

        self.responses[intent] = list(expanded_responses)
        self.save_responses(self.responses)

    def predict_intent(self, text):
        doc = self.nlp(text)
        scores = doc.cats
        best_intent = max(scores, key=scores.get)
        return best_intent if scores[best_intent] > 0.6 else "unknown"

    def dynamic_response(self, user_input):
        """Return a response and dynamically expand responses if needed."""
        intent = self.predict_intent(user_input)
        self.expand_responses(intent)
        return random.choice(self.responses.get(intent, self.responses["unknown"]))

