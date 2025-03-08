import random, json, os, logging
from nltk.corpus import wordnet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Response_Generation:
    def __init__(self):
        self.intent_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.resp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
        self.responses = self.load_responses()

    def load_responses(self):
        """Load existing responses or create an empty dictionary."""
        if os.path.exists(self.resp_path):
            with open(self.resp_path, 'r') as file:
                return json.load(file)
        return {}

    def get_valid_synonyms(self, word):
        """Fetch valid synonyms ensuring meaning is preserved."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                    synonyms.append(synonym)

        return list(set(synonyms))[:2]

    def expand_responses(self):
        logging.info('expanding intents...')
        with open(self.intent_path, 'r') as file:
            intent_data = json.load(file)

        valid_intents = {entry["intent"] for entry in intent_data["intents"]}

        updated_responses = {key: set(value) for key, value in self.responses.items()}

        for intent_name, response_list in self.responses.items():
            if intent_name not in valid_intents:
                logging.warning(f"Skipping {intent_name} because it's not in intents.json")
                continue

            for response in response_list:
                words = response.split()
                new_sentence = []
                for word in words:
                    synonyms = self.get_valid_synonyms(word)
                    new_word = random.choice(synonyms) if synonyms else word
                    new_sentence.append(new_word)

                new_response = " ".join(new_sentence)

                if intent_name != "unknown":
                    updated_responses[intent_name].add(new_response)

        self.responses = {key: list(value) for key, value in updated_responses.items()}
        self.save_responses()

    def save_responses(self):
        logging.info('updating intents...')
        with open(self.resp_path, 'w') as file:
            json.dump(self.responses, file, indent=4)

# Run process
# process = Response_Generation()
# process.expand_responses()