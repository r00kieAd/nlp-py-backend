import nltk, random, json, os, logging
from nltk.corpus import wordnet

nltk.download('wordnet')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Sentence_Generation:
    def __init__(self):
        self.sentences_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.json')
        self.intent_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.expanded_data = {"intents": []}

    def get_valid_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                    synonyms.append(synonym)

        return list(set(synonyms))[:3]

    def generate(self):
        logging.info('Loading sentences...')
        with open(self.sentences_path, 'r') as file:
            data = json.load(file)

        logging.info('Loading intents...')
        with open(self.intent_path, 'r') as file:
            intent_data = json.load(file)

        intent_mapping = {ex.lower(): entry["intent"] for entry in intent_data["intents"] for ex in entry["examples"]}

        for sentence in data["sentences"]:
            intent = intent_mapping.get(sentence.lower(), "unknown")
            variations = []

            words = sentence.split()
            for _ in range(3):
                new_sentence = []
                for word in words:
                    synonyms = self.get_valid_synonyms(word)
                    new_word = random.choice(synonyms) if synonyms else word
                    new_sentence.append(new_word)
                variations.append(" ".join(new_sentence))

            self.expanded_data["intents"].append({
                "intent": intent,
                "examples": [sentence] + variations 
            })

        with open(self.intent_path, 'w') as file:
            json.dump(self.expanded_data, file, indent=4)

        logging.info("Updated intents.json saved successfully!")
