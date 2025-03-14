import spacy, os, json, logging, random
from spacy.pipeline.textcat import Config, single_label_cnn_config
from app.trainers import generate_sentences, generate_responses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Training_Spacy:

    def __init__(self):
        self.intents_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_trained_model')
        self.sentence_generation = generate_sentences.Sentence_Generation()
        self.response_generation = generate_responses.Response_Generation()
        self.INTENT_KEY = "cats"

        self.nlp = spacy.blank("en")
        self.config = Config().from_str(single_label_cnn_config)
        self.textcat = self.nlp.add_pipe("textcat", config=self.config, last=True)

    def gen_sentences(self):
        self.sentence_generation.generate()

    def gen_responses(self):
        self.response_generation.expand_responses()

    def load_data(self):
        logging.info('Loading intent data...')
        with open(self.intents_path, 'r') as file:
            data = json.load(file)

        train_data = []
        intent_distribution = {}

        for entry in data["intents"]:
            intent = entry["intent"]
            examples = list(set(entry["examples"]))

            if len(examples) > 100:
                examples = random.sample(examples, 100) 

            intent_distribution[intent] = len(examples)
            self.textcat.add_label(intent) 
            
            for example in examples:
                train_data.append((example, {self.INTENT_KEY: {intent: 1.0}}))

        logging.info(f"Intent Distribution: {intent_distribution}")
        return train_data

    def train_model(self, epochs):
        train_data = self.load_data()
        random.shuffle(train_data)
        logging.info("Training started...")
        try:
            self.gen_sentences()
            self.gen_responses()
            optimizer = self.nlp.begin_training()

            for epoch in range(epochs):
                losses = {}
                for text, annotations in train_data:
                    doc = self.nlp.make_doc(text)
                    example = spacy.training.Example.from_dict(doc, annotations)
                    self.nlp.update([example], losses=losses)

                logging.info(f"Epoch {epoch+1}, Loss: {losses.get('textcat', 0)}")

            self.nlp.to_disk(self.trained_model_path)
            logging.info(f"Model trained and saved to {self.trained_model_path}")
            return {"status": "success", "epochs": epochs}

        except Exception as e:
            logging.error(str(repr(e)))
            return {"status": "failed", "error": "Error while training the custom model, see logs..."}