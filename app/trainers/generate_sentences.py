import nltk, random, json, os, logging
from nltk.corpus import wordnet

nltk.download('wordnet')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Sentence_Generation:

    def __init__(self):
        self.sentences_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.json')
        self.intent_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.intent_map = os.path.join(os.path.dirname(__file__), '..', 'data', 'intent_mapping.json')
        self.expanded_data = {"intents": []}

    def synonyms(self, sentence, num_variations=5):
        words = sentence.split()
        sentence_variations = set()

        for _ in range(num_variations):
            new_sentence = []
            for word in words:
                synonyms_lst = [lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()]
                if synonyms_lst:
                    new_word = random.choice(synonyms_lst)
                    new_sentence.append(new_word)
                else:
                    new_sentence.append(word)
            sentence_variations.add(" ".join(new_sentence))

        return list(sentence_variations)

    def generate(self):
        logging.info('loading sentences...')
        with open(self.sentences_path, 'r') as file:
            data = json.load(file)
        logging.info('loaded sentences, loading mapped intents...')
        with open(self.intent_map, 'r') as file:
            mapped_data = json.load(file)
        logging.info(f'loaded mapped intents, starting synonyms generation process for new intent examples...')
        for intent_mapping in mapped_data['intents']:

            for sentence in data["sentences"]:
                intent = intent_mapping.get(sentence, "unknown")
                variations = self.synonyms(sentence, 5)
                self.expanded_data["intents"].append({
                    "intent": intent,
                    "examples": [sentence] + variations
                })

        with open(self.intent_path, 'w') as file:
            json.dump(self.expanded_data, file, indent=4)

        logging.info("Updated JSON saved successfully!")

# process = Sentence_Generation()
# process.generate()