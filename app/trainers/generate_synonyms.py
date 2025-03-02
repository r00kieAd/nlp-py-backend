import nltk, random, json, os
from nltk.corpus import wordnet

class SynonymsGeneration:

    def __init__(self):
        self.json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.json')
        self.json_path_2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.expanded_data = {"intents": []}

    def synonyms(self, sentence, num_variations=5):
        """Generate multiple variations of a sentence using synonyms."""
        nltk.download('wordnet')
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
        with open(self.json_path, 'r') as file:
            data = json.load(file)

        intent_mapping = {
            "hello": "greeting",
            "hi": "greeting",
            "bye": "goodbye",
            "thank you": "gratitude",
            "how are you": "small_talk",
            "what's up": "small_talk"
        }

        for sentence in data["sentences"]:
            intent = intent_mapping.get(sentence, "unknown")
            variations = self.synonyms(sentence, 5)

            self.expanded_data["intents"].append({
                "intent": intent,
                "examples": [sentence] + variations
            })

        with open(self.json_path_2, 'w') as file:
            json.dump(self.expanded_data, file, indent=4)

        print("Updated JSON saved successfully!")

process = SynonymsGeneration()
process.generate()