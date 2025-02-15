'''
Steps to Build a Trainable NLP Chatbot
1.	Load training data (from PostgreSQL or a JSON file).
2.	Train spaCy's TextCategorizer model for intent classification.
3.	Use the trained model to classify new inputs.
4.	Fetch dynamic responses from a database or generate AI-based replies.
'''



import spacy
import random
from spacy.training.example import Example

# Load a small English model
nlp = spacy.load("en_core_web_sm")

# Add a text classifier to the pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")

# Define the intent categories
textcat.add_label("greeting")
textcat.add_label("goodbye")
textcat.add_label("thanks")

# Training data: (text, {"cats": {"intent": score}})
train_data = [
    ("hello", {"cats": {"greeting": 1.0, "goodbye": 0.0, "thanks": 0.0}}),
    ("hi", {"cats": {"greeting": 1.0, "goodbye": 0.0, "thanks": 0.0}}),
    ("hey", {"cats": {"greeting": 1.0, "goodbye": 0.0, "thanks": 0.0}}),
    ("good morning", {"cats": {"greeting": 1.0, "goodbye": 0.0, "thanks": 0.0}}),
    ("bye", {"cats": {"greeting": 0.0, "goodbye": 1.0, "thanks": 0.0}}),
    ("see you later", {"cats": {"greeting": 0.0, "goodbye": 1.0, "thanks": 0.0}}),
    ("thanks a lot", {"cats": {"greeting": 0.0, "goodbye": 0.0, "thanks": 1.0}}),
    ("thank you", {"cats": {"greeting": 0.0, "goodbye": 0.0, "thanks": 1.0}}),
]

# Training function
def train_chatbot(nlp, train_data, n_iter=10):
    optimizer = nlp.begin_training()
    
    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        
        print(f"Iteration {i+1}, Loss: {losses}")

    return nlp

# Train the model
nlp = train_chatbot(nlp, train_data)

# Save the trained model
nlp.to_disk("chatbot_model")
