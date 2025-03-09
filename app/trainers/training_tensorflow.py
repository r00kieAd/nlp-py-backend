import json
import nltk
import logging
import os, pickle
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")
# logging.basicConfig(filename="training.log",filemode="a",level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger()
# for handler in logger.handlers:
    # handler.flush()


class Train_Tensor:
    def __init__(self):
        self.log_messages = []
        self.sentences = []
        self.labels = []
        self.responses = {}
        self.intent_labels = []
        self.padded_sequences = None
        self.encoded_labels = None
        self.vocab_size = 0
        self.num_classes = 0
        self.tokenizer = None
        self.max_length = 0
        self.label_encoder = None
        self.log_path = os.path.join(os.path.dirname(__file__), 'training.log')
        self.intents_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents2.json')
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_tensor.keras')
        self.model_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tokenizer.json')
        self.data = self.loadData()
        self.pickle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'label_encoder.pkl')
        self.max_len_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'max_length.json')

    def setupLogging(self):
        class ListLogger(logging.Handler):
            def emit(handler_self, record):
                formatted_record = handler_self.format(record)
                self.log_messages.append(formatted_record)

        list_handler = ListLogger()
        list_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(list_handler)
        logging.getLogger().setLevel(logging.INFO)

    def loadData(self):
        try:
            logging.info("Loading intent data...")
            with open(self.intents_path, "r") as file:
                data = json.load(file)
            logging.info("Loading complete.")
            return data
        except Exception as e:
            logging.error(f"Error loading intent data: {e}")
            return None

    def extractData(self):
        if not self.data:
            logging.error("No data loaded. Cannot extract.")
            return

        logging.info("Extracting data...")
        for intent in self.data["intents"]:
            intent_name = intent["intent"]
            self.intent_labels.append(intent_name)
            self.responses[intent_name] = intent["responses"]

            for example in intent["examples"]:
                self.sentences.append(example)
                self.labels.append(intent_name)

        logging.info("Data extraction complete.")

    def encodeLabels(self):
        logging.info("Encoding labels...")
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        label_map = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        logging.info(f"Label Mapping: {label_map}")
        logging.info("Encoding complete.")

    def tokenize(self):
        logging.info("Tokenizing data...")
        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.sentences)
        word_index = self.tokenizer.word_index
        sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.padded_sequences = pad_sequences(sequences, padding="post")
        self.vocab_size = len(word_index) + 1
        self.max_length = self.padded_sequences.shape[1]
        self.num_classes = len(set(self.labels))
        logging.info(
            f"Vocabulary Size: {self.vocab_size}, Max Length: {self.max_length}, Classes: {self.num_classes}")
        logging.info("Tokenization complete.")

    def createModel(self, n, lstm=128):
        try:
            try:
                self.setupLogging()
            except:
                self.log_messages = "error while setting up logs"
            self.extractData()
            self.encodeLabels()
            self.tokenize()
            logging.info('Training...')
            # model = tf.keras.Sequential([
            #     tf.keras.layers.Embedding(self.vocab_size, lstm, input_length=self.max_length),
            #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, return_sequences=True, dropout=drop)),
            #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, dropout=drop)),
            #     tf.keras.layers.Dense(lstm, activation="relu"),
            #     tf.keras.layers.Dense(self.num_classes, activation="softmax")
            # ])
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, lstm, input_length=self.max_length),  
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)),  
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, dropout=0.4, recurrent_dropout=0.2)),  
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.4),  # 🔥 Forces model to generalize
                tf.keras.layers.Dense(self.num_classes, activation="softmax")
            ])
            model.compile(loss="sparse_categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
            history = model.fit(self.padded_sequences, np.array(self.encoded_labels), epochs=n, batch_size=8)
            model.save(self.trained_model_path)
            logging.info(f"Final Training Loss: {history.history['loss'][-1] * 100:.4f}%")
            logging.info(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.4f}%")

            with open(self.model_tokenizer_path, "w") as file:
                json.dump(self.tokenizer.to_json(), file)

            with open(self.pickle_path, 'wb') as file:
                pickle.dump(self.label_encoder, file)

            with open(self.max_len_path, "w") as file:
                json.dump({"max_length": self.max_length}, file)
            logging.info('Training complete.')
            return {"status": "success", "epochs": n, "logs": self.log_messages}
        except Exception as e:
            logging.critical(f'{e}')
            return {"status": "failed", "error": "error occured while training the model", "logs": self.log_messages}   

