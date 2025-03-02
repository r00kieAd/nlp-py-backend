import spacy, os, json
from spacy.pipeline.textcat import Config, single_label_cnn_config
print('')
json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'trainers', 'custom_trained_model')
print('path', json_path)
INTENT_KEY = "cats"

nlp = spacy.blank("en")
config = Config().from_str(single_label_cnn_config)
textcat = nlp.add_pipe("textcat", config=config, last=True)

with open(json_path, 'r') as file:
    data = json.load(file)

train_data = []
for intents in data["intents"]:
    textcat.add_label(intents["intent"])
    for example in intents["examples"]:
        item = (example, {INTENT_KEY: {intents["intent"]: 1.0}})
        train_data.append(item)

def train_model(nlp, train_data, n_iter=10):
    optimizer = nlp.begin_training()
    for _ in range(n_iter):
        losses = {}
        for text, annotations in train_data:
            print('annotations', annotations)
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses, sgd=optimizer)
        print(f"Loss: {losses['textcat']}")

print("training start")
train_model(nlp, train_data)
nlp.to_disk(trained_model_path)
print("Model saved to 'custom_trained_model'")