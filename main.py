import json
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("intents.json", "r") as file:
    data = json.load(file)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


patterns = []
tags = []
responses_map = {}

for intent in data["intents"]:
    responses_map[intent["tag"]] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(clean_text(pattern))
        tags.append(intent["tag"])


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)


def get_response(user_input):
    user_input = clean_text(user_input)

    user_vec = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vec, X)[0]

  
    top_indices = similarity.argsort()[-3:][::-1]

    best_score = similarity[top_indices[0]]

    if best_score < 0.25:
        return "Hmm... I'm not sure I understand that. Can you rephrase?"

    best_tag = tags[top_indices[0]]

    responses = responses_map.get(best_tag, [])

    base_response = random.choice(responses)

    
    if best_tag == "greeting":
        return base_response + " "

    elif best_tag == "goodbye":
        return base_response + "  Take care!"

    elif best_tag == "help":
        return base_response + " I'm here whenever you need me."

    elif best_tag == "order_status":
        return base_response + " You can track it anytime from your account."

    else:
        return base_response

print(" Mini ChatGPT Bot is running! Type 'exit' to stop.")

while True:
    user_input = input("You: ")

    if clean_text(user_input) == "exit":
        print("Bot: Bye!")
        break

    response = get_response(user_input)
    print("Bot:", response)