import logging
import os
import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Tensor_Model:
    def __init__(self):
        self.trained_model_path = os.path.join(os.path.dirname(__file__), 'custom_tensor.keras')
        print('model path: ', self.trained_model_path)
        self.model_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tokenizer.json')
        self.responses_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents2.json')
        self.label_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'label_encoder.pkl')
        self.max_length_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'max_length.json')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 20
        self.responses = {}
        self.loadModel()
        self.loadTokenizer()
        self.loadLabelEncoder()
        self.loadMaxLength()
        self.loadResponses()

    def loadModel(self):
        try:
            logging.info('Loading trained model...')
            self.model = tf.keras.models.load_model(self.trained_model_path)
            logging.info(f'Model loaded successfully..{type(self.model)}')
        except Exception as e:
            logging.error(f'Error loading model: {e}')

    def loadTokenizer(self):
        try:
            logging.info('Loading tokenizer...')
            with open(self.model_tokenizer_path, "r") as file:
                tokenizer_data = json.load(file)
                self.tokenizer = tokenizer_from_json(tokenizer_data)
            logging.info('Tokenizer loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading tokenizer: {e}')

    def loadLabelEncoder(self):
        try:
            logging.info("Loading Label Encoder...")
            with open(self.label_encoder_path, 'rb') as file:
                self.label_encoder = pickle.load(file)
            logging.info("Label Encoder loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Label Encoder: {e}")
            self.label_encoder = LabelEncoder()

    def loadMaxLength(self):
        try:
            logging.info("Loading max sequence length...")
            with open(self.max_length_path, "r") as file:
                self.max_length = json.load(file)["max_length"]
            logging.info(f"Max sequence length: {self.max_length}")
        except Exception as e:
            logging.error(f"Error loading max sequence length: {e}")
            self.max_length = 20 

    def loadResponses(self):
        try:
            logging.info('Loading responses...')
            with open(self.responses_path, "r") as file:
                data = json.load(file)
                for item in data["intents"]:
                    self.responses[item["intent"]] = item["responses"]
            logging.info('Responses loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading responses: {e}')
            self.responses = {}

    def predictIntent(self, text):
        try:
            logging.info('Processing input...')
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_length, padding="post")

            logging.info('Predicting intent...')
            prediction = self.model.predict(padded)
            probabilities = tf.nn.softmax(prediction)[0].numpy()
            prediction = self.model.predict(padded)
            intent_index = np.argmax(tf.nn.softmax(prediction)[0])
            intent_name = self.label_encoder.inverse_transform([intent_index])[0]
            confidence = float(f'{probabilities[intent_index]:.2f}')
            logging.info(f'Predicted intent: {intent_name} (Confidence: {confidence})')
            if confidence < 0.12:
                return {"response": "I'm not sure. Can you rephrase?", "intent": "unknown", "confidence": confidence, "model": "TensorFlow", "status": "success"}
            
            response = np.random.choice(self.responses.get(intent_name, ["Sorry, I didn't understand that."]))
            logging.info(f'Response: {response}')
            return {"status": "success", "reply": response, "model": "TensorFlow", "predicted_intent": intent_name, "confidence_score": confidence}

        except Exception as e:
            logging.critical(f'Error during prediction: {e}')
            return {"status": "failed", "error": "Sorry, something went wrong during prediction", "model": "TensorFlow"}

# chatbot = Tensor_Model()
# print(f'\npredition 1: {chatbot.predictIntent("hi")}\n')
# print(f'\npredition 2: {chatbot.predictIntent("bye")}\n')