import os
import pandas as pd
from convokit import Corpus, download

# Download and load the corpus
corpus = Corpus(filename=download("movie-corpus"))

# Extract utterances (dialogues)
data = []
for convo_id, convo in corpus.conversations.items():  # ✅ Get conversation ID correctly
    for utt in convo.iter_utterances():
        data.append({
            "conversation_id": convo_id,  # ✅ Get conversation ID from conversation object
            "utterance_id": utt.id,
            "speaker": utt.speaker.id if utt.speaker else "unknown",
            "text": utt.text
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to JSON
path = os.path.join(os.path.dirname(__file__), 'data', 'cornell_corpus.json')
df.to_json(path, orient="records", indent=4)

print("Dialogue dataset saved as JSON!")