import json
import nltk
import logging, traceback
import os, pickle, datetime
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model

nltk.download("punkt")


class Train_Tensor:
    def __init__(self):
        self.log_messages = []
        self.sentences = []
        self.labels = []
        self.responses = {}
        self.intent_labels = []
        self.padded_sequences = None
        self.encoded_labels = None
        self.input_sequences = None
        self.output_sequences = None
        self.vocab_size = 0
        self.num_classes = 0
        self.tokenizer = None
        self.max_length = 0
        self.label_encoder = None
        self.total_epochs = 0
        self.input_texts = []
        self.output_texts = []
        self.class_weights = None
        self.cornell_corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cornell_corpus.json')
        self.history_path = os.path.join(os.path.dirname(__file__), '..', 'history', 'training_history.pkl')
        self.intents_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tensor_intents.json')
        self.trained_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_tensor.keras')
        self.model_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tokenizer.json')
        self.transformer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'resp_transformer.keras')
        self.t_tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transformer_tokenizer.json')
        self.data = self.loadData()
        self.pickle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'label_encoder.pkl')
        self.max_len_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'max_length.json')

    def setupLogging(self):
        self.log_messages.clear()
        logger = logging.getLogger()
        logger.handlers = []

        class ListLogger(logging.Handler):
            def emit(handler_self, record):
                formatted_record = handler_self.format(record)
                self.log_messages.append(formatted_record)

        list_handler = ListLogger()
        list_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(list_handler)
        logger.setLevel(logging.INFO)

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
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.encoded_labels), y=self.encoded_labels)
        self.class_weights = dict(enumerate(class_weights))
        # logging.info(f"Label Mapping: {label_map}")
        # logging.info(f"Class Weights: {self.class_weights}")
        logging.info("Encoding complete.")

    def tokenize(self, model_classification):
        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        if model_classification == 0:
            logging.info("Tokenizing intent data...")
            self.sentences = [" ".join(text_to_word_sequence(sentence)) for sentence in self.sentences]
            self.tokenizer.fit_on_texts(self.sentences)
            word_index = self.tokenizer.word_index
            sequences = self.tokenizer.texts_to_sequences(self.sentences)
            self.padded_sequences = pad_sequences(sequences, padding="post")
            self.vocab_size = len(word_index) + 1
            self.max_length = self.padded_sequences.shape[1]
            self.num_classes = len(set(self.labels))
            logging.info(f"Vocabulary Size: {self.vocab_size}, Max Length: {self.max_length}, Classes: {self.num_classes}")
            logging.info("Tokenization of intents complete.")
        else:
            logging.info("Tokenizing cornell corpus...")
            self.tokenizer.fit_on_texts(self.input_texts + self.output_texts)
            self.input_sequences = pad_sequences(self.tokenizer.texts_to_sequences(self.input_texts), padding="post", maxlen=20)
            self.output_sequences = pad_sequences(self.tokenizer.texts_to_sequences(self.output_texts), padding="post", maxlen=20)
            logging.info("Cornell corpus tokenized successfully.")

    def loadCornellData(self):
        try:
            logging.info("Loading Cornell movie dialogues dataset...")
            with open(self.cornell_corpus_path, "r") as file:
                dialogues = json.load(file)
            input_texts, output_texts = [], []
            convo_map = {}

            for entry in dialogues:
                convo_id = entry["conversation_id"]
                if convo_id not in convo_map:
                    convo_map[convo_id] = []
                convo_map[convo_id].append(entry["text"])

            for convo in convo_map.values():
                for i in range(len(convo) - 1):
                    input_texts.append(convo[i])
                    output_texts.append(convo[i + 1])

            self.input_texts = input_texts
            self.output_texts = output_texts
            logging.info(f"Loaded {len(input_texts)} conversation pairs.")

        except Exception as e:
            logging.error(f"Error loading Cornell data: {e}")

    def buildTransformerModel(self, vocab_size=5000, d_model=128, num_heads=4, ff_dim=256, max_len=20):
        logging.info("Building improved transformer model...")
    
        inputs = Input(shape=(max_len,))
        embedding = Embedding(vocab_size, d_model, mask_zero=True)(inputs)

        attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(embedding, embedding, embedding)
        attention = Dropout(0.1)(attention) 
        norm1 = LayerNormalization(epsilon=1e-6)(attention + embedding)  

        dense_ff = Dense(ff_dim, activation="relu")(norm1)
        dense_ff = Dropout(0.1)(dense_ff)
        dense_ff = Dense(d_model)(dense_ff)
        norm2 = LayerNormalization(epsilon=1e-6)(dense_ff + norm1)  

        outputs = Dense(vocab_size, activation="softmax")(norm2)

        transformer = Model(inputs, outputs)
        lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        transformer.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.transformer_model = transformer

        logging.info("Transformer model improved and built successfully.")


    def trainTransformer(self, epochs, batch_size=32):
        epochs = 1
        logging.info("Training transformer model...")
        if not hasattr(self, "input_sequences") or self.input_sequences is None:
            self.loadCornellData()
            self.tokenize(1)
        if self.input_sequences is None or self.output_sequences is None:
            raise ValueError("Tokenization failed. Check Cornell dataset preprocessing.")
        if not hasattr(self, "transformer_model") or self.transformer_model is None:
            self.buildTransformerModel()
        if self.transformer_model is None:
            raise ValueError("Transformer model build failed.")
        history = self.transformer_model.fit(
            self.input_sequences, 
            self.output_sequences, 
            epochs=epochs, 
            batch_size=batch_size
        )
        logging.info(f"Final transformer training loss: {history.history['loss'][-1] * 100:.4f}")
        logging.info(f"Final transformer training accuracy: {history.history['accuracy'][-1] * 100:.4f}")
        self.transformer_model.save(self.transformer_path)
        with open(self.t_tokenizer_path, "w") as file:
            json.dump(self.tokenizer.to_json(), file)
        logging.info("Transformer training complete and model saved.")
        return history

    def updateHistory(self, n, history, t_history):
        try:
            logging.info('Updating training history...')
            if os.path.exists(self.history_path) and os.path.getsize(self.history_path) > 0:
                with open(self.history_path, "rb") as file:
                    history_data = pickle.load(file)
                    self.total_epochs = history_data.get("total_epochs", 790)
                    self.total_transformer_epochs = history_data.get("total_transformer_epochs", 60)
            else:
                logging.info("History file is empty or doesn't exist. Creating a new history record...")
                self.total_epochs = 0
            training_info = {
                "date": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), 
                "last_epoch_count": 100,
                "total_epochs": self.total_epochs + n,
                "history": history.history,
                "last_transformer_epoch": n,
                "total_transformer_epochs": self.total_transformer_epochs+n,
                "transformer_history": t_history.history
            }
            with open(self.history_path, "wb") as file:
                pickle.dump(training_info, file)
            logging.info('History updated successfully.')
        except Exception as e:
            logging.error(f'Unexpected exception occurred: {str(e)}')

    def createModel(self, n, lstm=128):
        try:
            try:
                self.setupLogging()
                logging.info('Starting process...')
            except:
                self.log_messages = "error while setting up logs"
            self.extractData()
            self.encodeLabels()
            self.tokenize(0)
            logging.info('Training intent model...')
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, lstm, input_length=self.max_length),  
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),  
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm, dropout=0.3, recurrent_dropout=0.2)),  
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(self.num_classes, activation="softmax")
            ])
            model.compile(loss="sparse_categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
            history = model.fit(self.padded_sequences, np.array(self.encoded_labels), epochs=n, batch_size=8, class_weight=self.class_weights)
            model.save(self.trained_model_path)
            logging.info(f"Final Training Loss: {history.history['loss'][-1] * 100:.4f}%")
            logging.info(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.4f}%")

            logging.info('Updating tokens...')
            with open(self.model_tokenizer_path, "w") as file:
                json.dump(self.tokenizer.to_json(), file)

            logging.info('Updating labels...')
            with open(self.pickle_path, 'wb') as file:
                pickle.dump(self.label_encoder, file)

            logging.info('Updating max length...')
            with open(self.max_len_path, "w") as file:
                json.dump({"max_length": self.max_length}, file)
            logging.info('Intent training complete.')
            t_history = self.trainTransformer(n)
            self.updateHistory(n, history, t_history)
            logging.info('Process complete.')
            return {"status": "success", "epochs": n, "logs": self.log_messages}
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            logging.critical(f'{e}, line number: {line_no}')
            return {"status": "failed", "error": "error occured while training the model", "logs": self.log_messages}   

