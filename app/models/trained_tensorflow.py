import logging, traceback
import os
import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from app.history import get_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Tensor_Model:
    def __init__(self):
        self.trained_model_path = os.path.join(os.path.dirname(__file__), 'custom_tensor.keras')
        self.trained_trasnformer_path = os.path.join(os.path.dirname(__file__), 'resp_transformer.keras')
        self.model_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tokenizer.json')
        self.transformer_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transformer_tokenizer.json')
        self.responses_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tensor_intents.json')
        self.label_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'label_encoder.pkl')
        self.max_length_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'max_length.json')
        self.model = None
        self.transformer_model = None
        self.tokenizer = None
        self.transformer_tokenizer = None
        self.label_encoder = None
        self.max_length = 20
        self.responses = {}
        self.get_history = get_history.Get_History()
        self.history = None
        self.training_date = 'undefined'
        self.top_p = 0
        self.top_k = 0
        self.temperature = 0
        self.loadModel()
        self.loadTokenizer()
        self.loadLabelEncoder()
        self.loadMaxLength()
        self.loadResponses()

    def loadModel(self):
        try:
            logging.info('Loading trained models...')
            self.model = tf.keras.models.load_model(self.trained_model_path)
            logging.info(f'Intent model loaded successfully..{type(self.model)}')
            try:
                self.history = self.get_history.getData()
                if self.history["status"] == "failed":
                    logging.error("Could not fetch intent training history.")
                    return
                self.training_date = self.history["last_training_date"]
            except:
                logging.error("Could not fetch history.")
            self.transformer_model = tf.keras.models.load_model(self.trained_trasnformer_path)
            logging.info(f'Transfomer model loaded successfully..{type(self.transformer_model)}')
        except Exception as e:
            logging.error(f'Error loading models: {e}')

    def loadTokenizer(self):
        try:
            logging.info('Loading tokenizers...')
            with open(self.model_tokenizer_path, "r") as file:
                tokenizer_data = json.load(file)
                self.tokenizer = tokenizer_from_json(tokenizer_data)
            logging.info('Intent tokenizer loaded successfully.')
            with open(self.transformer_tokenizer_path, "r") as file:
                tokenizer_data = json.load(file)
                self.transformer_tokenizer = tokenizer_from_json(tokenizer_data)
            logging.info('Transformer tokenizer loaded successfully')
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

    def top_p_top_k_sampling(self, logits, top_p, top_k, temperature):
        logits = logits / temperature
        probs = tf.nn.softmax(logits).numpy()
        if top_k > 0:
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[top_k_indices] = top_k_probs
            probs = filtered_probs / np.sum(filtered_probs) 
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            cutoff_index = np.where(cumulative_probs > top_p)[0][0] + 1
            top_p_indices = sorted_indices[:cutoff_index]

            filtered_probs = np.zeros_like(probs)
            filtered_probs[top_p_indices] = probs[top_p_indices]
            probs = filtered_probs / np.sum(filtered_probs)  # Normalize

        next_token = np.random.choice(len(probs), p=probs)
        return next_token


    def generate_response(self, user_input):
        transformer_model = self.transformer_model
        tokenizer = self.transformer_tokenizer
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=20, padding="post")
        generated_tokens = []

        for _ in range(20):
            logits = transformer_model.predict(padded_sequence)[0, -1]
            next_token = self.top_p_top_k_sampling(logits, self.top_p, self.top_k, self.temperature)
            generated_tokens.append(next_token)
            padded_sequence = np.concatenate([padded_sequence[:, 1:], np.array([[next_token]])], axis=1)
            if next_token == tokenizer.word_index.get("<eos>", -1):
                break

        generated_response = " ".join([tokenizer.index_word.get(token, "") for token in generated_tokens if token > 0])

        return generated_response if generated_response.strip() else "I'm not sure how to respond."
        
    def checkTrainingDate(self):
        logging.info("Checking training date...")
        try:
            data = self.get_history.getData()
            if data["status"] == "success":
                curr_date = data["last_training_date"]
                if curr_date == self.training_date:
                    logging.info("Check complete.")
                    return True
                logging.info("Check complete.")
                return False
        except Exception as e:
            logging.error(f'Error while getting history data: {str(e)}')
            return None

    def predictIntent(self, text, top_p, top_k, temperature, confidence_threshold):
        try:
            logging.info('Processing input...')
            self.top_p = top_p
            self.top_k = top_k
            self.temperature = temperature
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
            response = np.random.choice(self.responses.get(intent_name, []))
            if not response or confidence < confidence_threshold:
                response = self.generate_response(text)
            logging.info(f'Response: {response}')
            return {
                "status": "success", 
                "reply": response, 
                "model": "TensorFlow", 
                "predicted_intent": intent_name, 
                "confidence_score": confidence, 
                "model_upToDate": self.checkTrainingDate()
            }

        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            logging.critical(f'{e}, line number: {line_no}')
            logging.critical(traceback.extract_tb(e.__traceback__))
            return {"status": "failed", "error": f"Sorry, something went wrong during prediction, error came at line number {line_no}", "model": "TensorFlow/Transformer"}

# chatbot = Tensor_Model()
# print(f'\npredition 1: {chatbot.predictIntent("hi")}\n')
# print(f'\npredition 2: {chatbot.predictIntent("bye")}\n')