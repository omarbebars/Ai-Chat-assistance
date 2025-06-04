import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load resources
with open('intents.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[i], 'probability': str(prob)} for i, prob in results]


def extract_city_from_message(message):
    """Extract city name from user message"""
    # Convert to lowercase for easier matching
    message_lower = message.lower()

    # Common weather question patterns and extract city
    words = message_lower.split()

    # Look for common patterns like "weather in [city]", "temperature in [city]", etc.
    trigger_phrases = ["in ", "for ", "at ", "weather in", "temperature in"]

    for phrase in trigger_phrases:
        if phrase in message_lower:
            # Find the word after the trigger phrase
            phrase_index = message_lower.find(phrase)
            remaining_text = message[phrase_index + len(phrase):].strip()
            city = remaining_text.split()[0] if remaining_text.split() else None
            if city:
                return city.title()  # Capitalize first letter

    # If no pattern found, look for city names at the end of the message
    if len(words) > 1:
        potential_city = words[-1]
        if potential_city.isalpha():  # Only letters, likely a city name
            return potential_city.title()

    return None


def get_weather(city=None):
    API_KEY = os.getenv("WEATHER_API_KEY")


    # Use provided city or default to Berlin
    if not city:
        city = "Berlin"

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        return f"The current temperature in {city} is {temp}°C with {description}."
    elif response.status_code == 404:
        return f"Sorry, I couldn't find weather information for '{city}'. Please check the city name."
    else:
        return "Sorry, I couldn't fetch the weather right now."


def get_response(intents_list, intents_json, user_message=""):
    if len(intents_list) == 0:
        return "Sorry, I don't understand."

    tag = intents_list[0]['intent']

    if tag == "weather_query":
        city = extract_city_from_message(user_message)
        return get_weather(city)

    elif tag == "news":
        return get_latest_news()

    # Fallback to static response
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I don't understand."


def get_latest_news():
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=3&apiKey={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if not articles:
            return "Sorry, I couldn't find any news right now."

        headlines = [f"- {article['title']}" for article in articles[:3]]
        return "Here are the top headlines:\n" + "\n".join(headlines)
    else:
        return "Sorry, I couldn't fetch the news at this moment."



# ------------------------
# TKINTER GUI APPLICATION
# ------------------------
def send_message():
    user_input = entry_box.get()
    if user_input.strip() == "":
        return

    # Add user message with modern styling
    chat_window.config(state='normal')
    chat_window.insert(tk.END, "You: ", "user_label")
    chat_window.insert(tk.END, user_input + "\n", "user_text")

    # Get bot response
    ints = predict_class(user_input)
    res = get_response(ints, intents, user_input)

    # Add bot response with modern styling
    chat_window.insert(tk.END, "🤖 Chatty: ", "bot_label")
    chat_window.insert(tk.END, res + "\n\n", "bot_text")

    chat_window.config(state='disabled')
    chat_window.see(tk.END)  # Auto-scroll to bottom
    entry_box.delete(0, tk.END)


def on_enter_key(event):
    send_message()


def on_entry_focus_in(event):
    if entry_box.get() == "Type your message here...":
        entry_box.delete(0, tk.END)
        entry_box.config(fg='black')


def on_entry_focus_out(event):
    if entry_box.get() == "":
        entry_box.insert(0, "Type your message here...")
        entry_box.config(fg='gray')


# Set up modern GUI
window = tk.Tk()
window.title("Chatty - AI Assistant")
window.geometry("600x700")
window.configure(bg='#1a1a2e')
window.resizable(True, True)
window.minsize(500, 300)  # Set minimum size

# Create main container with padding
main_frame = tk.Frame(window, bg='#1a1a2e')
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Title label
title_label = tk.Label(main_frame, text="💬 Chatty AI",
                       font=("Segoe UI", 18, "bold"),
                       fg='#00d2ff', bg='#1a1a2e')
title_label.pack(pady=(0, 20))

# Chat window container
chat_container = tk.Frame(main_frame, bg='#1a1a2e')
chat_container.pack(fill='both', expand=True, pady=(0, 10))

# Chat window with modern colors
chat_frame = tk.Frame(chat_container, bg='#16213e', relief='flat', bd=2)
chat_frame.pack(fill='both', expand=True)

chat_window = scrolledtext.ScrolledText(
    chat_frame,
    wrap=tk.WORD,
    font=("Segoe UI", 11),
    bg='#16213e',
    fg='#ecf0f1',
    insertbackground='#00d2ff',
    selectbackground='#3498db',
    selectforeground='white',
    relief='flat',
    bd=0,
    padx=10,
    pady=10
)
chat_window.pack(fill='both', expand=True, padx=2, pady=2)

# Configure text tags for styling
chat_window.tag_configure("user_label", foreground="#00d2ff", font=("Segoe UI", 11, "bold"))
chat_window.tag_configure("user_text", foreground="#ecf0f1", font=("Segoe UI", 11))
chat_window.tag_configure("bot_label", foreground="#ff6b6b", font=("Segoe UI", 11, "bold"))
chat_window.tag_configure("bot_text", foreground="#bdc3c7", font=("Segoe UI", 11))

# Fixed input frame at bottom
input_frame = tk.Frame(main_frame, bg='#1a1a2e', height=96)
input_frame.pack(fill='x', side='bottom', pady=(5, 0))
input_frame.pack_propagate(False)  # Maintain fixed height

# Entry box with placeholder
entry_box = tk.Entry(
    input_frame,
    font=("Segoe UI", 12),
    bg='white',
    fg='black',
    insertbackground='#00d2ff',
    relief='solid',
    bd=2,
    highlightthickness=2,
    highlightcolor='#00d2ff',
    highlightbackground='#cccccc'
)
entry_box.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 10), ipady=8, pady=10)
entry_box.insert(0, "Type your message here...")
entry_box.bind("<FocusIn>", on_entry_focus_in)
entry_box.bind("<FocusOut>", on_entry_focus_out)
entry_box.bind("<Return>", on_enter_key)

# Modern send button
send_button = tk.Button(
    input_frame,
    text="Send ➤",
    command=send_message,
    font=("Segoe UI", 12, "bold"),
    bg='#00d2ff',
    fg='white',
    relief='flat',
    bd=0,
    padx=20,
    pady=8,
    cursor='hand2'
)
send_button.pack(side=tk.RIGHT, pady=10)
send_button.pack(side=tk.RIGHT)


# Button hover effects
def on_button_enter(event):
    send_button.config(bg='#0056b3')


def on_button_leave(event):
    send_button.config(bg='#00d2ff')


send_button.bind("<Enter>", on_button_enter)
send_button.bind("<Leave>", on_button_leave)

# Welcome message
chat_window.config(state='normal')
chat_window.insert(tk.END, "🤖 Chatty: ", "bot_label")
chat_window.insert(tk.END,
                   "Hello! I'm Chatty, your AI assistant. I can help you with weather information and answer your questions. Try asking 'What's the weather in London?' or just say hello!\n\n",
                   "bot_text")
chat_window.config(state='disabled')

# Focus on entry box
entry_box.focus()

window.mainloop()
