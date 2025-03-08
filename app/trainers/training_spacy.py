import spacy, os, json, logging
from spacy.pipeline.textcat import Config, single_label_cnn_config
from trainers import generate_sentences, generate_responses
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Training_Spacy():

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
        logging.info('loading intent data...')
        with open(self.intents_path, 'r') as file:
            data = json.load(file)

        train_data = []
        for intents in data["intents"]:
            self.textcat.add_label(intents["intent"])
            for example in intents["examples"]:
                item = (example, {self.INTENT_KEY: {intents["intent"]: 1.0}})
                train_data.append(item)
        logging.info('data loaded...')
        return train_data

    def train_model(self, ephocs):
        n_iter = ephocs
        train_data = self.load_data()
        logging.info("training started...")
        try:
            self.gen_sentences()
            self.gen_responses()
            optimizer = self.nlp.begin_training()
            for _ in range(n_iter):
                losses = {}
                for text, annotations in train_data:
                    print('annotations', annotations)
                    doc = self.nlp.make_doc(text)
                    example = spacy.training.Example.from_dict(doc, annotations)
                    self.nlp.update([example], losses=losses, sgd=optimizer)
                logging.info(f"Loss: {losses['textcat']}")
            self.nlp.to_disk(self.trained_model_path)
            return {"status": "success", "ephocs": ephocs}
        except Exception as e:
            logging.info(str(repr(e)))
            return {"status": "failed", "error": "Error while training the custom model, see logs..."}

