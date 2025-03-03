import nltk, os, json, logging, random
from nltk.corpus import wordnet

nltk.download('wordnet')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class Response_Generation():
    def __init__(self):
        self.intent_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.resp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
        self.responses = self.load_responses()
        print(self.responses)

    def load_responses(self):
        if os.path.exists(self.resp_path):
            logging.info('loading responses...')
            with open(self.resp_path, 'r') as file:
                return json.load(file)
        else:
            logging.warning('!! loading default responses !!')
            default_responses = {
                "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
                "goodbye": ["Goodbye!", "See you later!", "Take care!"],
                "thanks": ["You're welcome!", "No problem!", "Anytime!"],
                "unknown": ["Sorry, I didn't understand that.", "Can you rephrase?"]
            }
            self.save_responses(default_responses)
            return default_responses
    
    def save_responses(self, data):
        logging.info('saving responses...')
        with open(self.resp_path, 'w') as file:
            json.dump(data, file, indent=4)

    def expand_responses(self):
        logging.info('starting process...')
        intents = []
        with open(self.intent_path, 'r') as file:
            intents = json.load(file)
        logging.info('intents loaded...')
        for intent in intents['intents']:
            curr = intent['intent']
            if self.responses.get(curr, -1) == -1:
                logging.info(f'{curr} not in responses')
                continue
            expanded_responses = set(self.responses[curr])
            for response in self.responses.keys():
                words = response.split()
                new_sentence = []
                for word in words:
                    synonyms = [lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()]
                    new_word = random.choice(synonyms) if synonyms else word
                    new_sentence.append(new_word)
                logging.info(f'new sentence generated for {curr}: {new_sentence}')
                expanded_responses.add(" ".join(new_sentence))

            self.responses[curr] = list(expanded_responses)
            self.save_responses(self.responses)
    
# process = Response_Generation()
# process.expand_responses()