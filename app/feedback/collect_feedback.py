import json, os, logging
import traceback

class Collect_Feedback():

    def __init__(self):
        self.intent_path_1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.intent_path_2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents2.json')
        self.response_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'responses.json')
        self.sentences = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentences.json')
        self.intent = ""
        self.examples = ""
        self.responses = ""
    
    def forSpacy(self):
        try:
            intent_data = None
            with open(self.intent_path_1, 'r') as file:
                intent_data = json.load(file)
            response_data = None
            with open(self.response_path, 'r') as file:
                response_data = json.load(file)
            with open(self.response_path, 'w') as file:
                if response_data.get(self.intent, None) == None:
                    response_data[self.intent] = [self.responses]
                else:
                    response_data[self.intent].append(self.responses)
                json.dump(response_data, file, indent=4)
            sentence_data = None
            with open(self.sentences, 'r') as file:
                sentence_data = json.load(file)
            with open(self.sentences, 'w') as file:
                sentence_data["sentences"].append(self.examples)
                json.dump(sentence_data, file, indent=4)
            with open(self.intent_path_1, 'w') as file:
                all_intents = set([item["intent"] for item in intent_data["intents"]])
                curr_intents = [item for item in intent_data["intents"]]
                if self.intent in all_intents:
                    for intent in curr_intents:
                        if self.intent == intent["intent"]:
                            intent['examples'].append(self.examples)
                else:
                    new_intent = {'intent': self.intent, 'examples': [self.examples]}
                    curr_intents.append(new_intent)
                intent_data["intents"] = curr_intents
                json.dump(intent_data, file, indent=4)
            return {"status": "success", "message": "Feedback saved for training spacy"}
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}
    
    def forTensor(self):
        try:
            intent_data = {}
            with open(self.intent_path_2, 'r') as file:
                intent_data = json.load(file)
            with open(self.intent_path_2, 'w') as file:
                all_intents = [item for item in intent_data["intents"]]
                added = False
                for intents in all_intents:
                    if intents["intent"] == self.intent:
                        intents["examples"].append(self.examples)
                        intents["responses"].append(self.responses)
                        added = True
                if not added:
                    examples = [self.examples]
                    print(examples)
                    new_intent = {"intent": self.intent, "examples": examples, "responses": [self.responses]}
                    all_intents.append(new_intent)
                intent_data["intents"] = all_intents
                json.dump(intent_data, file, indent=4)
            return {"status": "success", "message": "Feedback saved for training tensor"}
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}

    def saveFeedback(self, intent, example, responses, model):
        try:
            self.intent = intent
            self.examples = example
            self.responses = responses
            if model == 1:
                return self.forSpacy()
            else:
                return self.forTensor()
        except:
            return {"status": "failed", "error": "unexpected error while saving feedback, see logs"}

# process = Collect_Feedback()
# process.saveFeedback("news", "How bla bla", "bla bla bla", 2)
