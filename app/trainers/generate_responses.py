from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

class Response_Generation():
    def __init__(self):
        self.resp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
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
    
    