'''
How This Works
1.	Train a classifier using TextCategorizer in spaCy.
2.	Assign intent labels (greeting, goodbye, thanks) to phrases.
3.	Train the model using the given dataset.
4.	Predict intent based on user input.
5.	Generate dynamic responses based on intent.
'''

import spacy
import random

# Load trained model
nlp = spacy.load("chatbot_model")

# Possible responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Anytime!"],
    "unknown": ["Sorry, I didn't understand that.", "Can you rephrase?"]
}

# Function to predict intent
def predict_intent(text):
    doc = nlp(text)
    scores = doc.cats  # Get intent probabilities
    best_intent = max(scores, key=scores.get)  # Pick highest probability intent
    
    if scores[best_intent] > 0.6:  # Confidence threshold
        return best_intent
    return "unknown"

# Chatbot response function
def chatbot_response(user_input):
    intent = predict_intent(user_input)
    return random.choice(responses.get(intent, responses["unknown"]))

# Chatbot interaction
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))