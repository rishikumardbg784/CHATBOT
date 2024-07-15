from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import json
from datetime import datetime
import string
import nltk
from nltk.corpus import stopwords
import ssl


# Create an unverified SSL context for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK data files
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Create a new instance of a ChatBot
chatbot = ChatBot('Example Bot')

# Load the data from the JSON file
with open('C:\\Users\\rishi\\ChatBot\\Sample Question Answers.json', 'r') as f:
    data = json.load(f)

# Transform the list of dictionaries into separate lists for questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Set the trainer
trainer = ListTrainer(chatbot)

# Train the chatbot
for question, answer in zip(questions, answers):
    trainer.train([question, answer])

conversation_history = []

# Function to preprocess text
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    print(f"User input received: {user_input}")  # Debugging statement

    preprocessed_input = preprocess(user_input)

    if any(preprocessed_input in preprocess(question) for question in questions):
        response = chatbot.get_response(user_input)
        response_text = str(response)
    else:
        response_text = "Please contact us directly for this inquiry."

    print(f"Response: {response_text}")  # Debugging statement

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conversation_history.append({"user": user_input, "bot": response_text, "timestamp": timestamp})

    return jsonify({"response": response_text, "history": conversation_history})

if __name__ == "__main__":
    app.run(debug=True)




