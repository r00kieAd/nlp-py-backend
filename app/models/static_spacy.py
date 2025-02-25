import spacy
from spacy.matcher import Matcher
import random

class Static_Model:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.patterns = {
            "greeting": [[{"LOWER": "hello"}], [{"LOWER": "hi"}], [{"LOWER": "hey"}], [{"LOWER": "good"}, {"LOWER": "morning"}]],
            "goodbye": [[{"LOWER": "bye"}], [{"LOWER": "goodbye"}], [{"LOWER": "see"}, {"LOWER": "you"}]],
            "thanks": [[{"LOWER": "thanks"}], [{"LOWER": "thank"}, {"LOWER": "you"}, {"LOWER": "good"}, {"LOWER": "hear"}]],
            "chitchat": [[{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}]]
        }
        self.responses = {
            "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
            "goodbye": ["Goodbye!", "See you later!", "Take care!"],
            "thanks": ["You're welcome!", "No problem!", "Anytime!"],
            "chitchat": ["I am as good as the developer can make me", "I am fine, how are you?"],
            "unknown": ["Sorry, I didn't understand that.", "Can you rephrase?"]
        }
        for intent, pattern_list in self.patterns.items():
            self.matcher.add(intent, pattern_list)


    def static_response(self, user_input):
        doc = self.nlp(user_input)
        matches = self.matcher(doc)
        
        if matches:
            match_id, start, end = matches[0]  # Get first match
            print(match_id, start, end)
            intent = self.nlp.vocab.strings[match_id]  # Convert match ID to string
            print(intent)
            return random.choice(self.responses[intent])  # Random response for matched intent
        return random.choice(self.responses["unknown"])  # Default response
