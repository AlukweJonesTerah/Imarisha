import json
import os
import logging
import random
import nltk
import pickle
import string
import numpy as np
import tensorflow as tf
from home import app, db
from flask import render_template, url_for, redirect, request, flash,  jsonify
from flask_login import current_user, login_required, login_user, logout_user
from tensorflow.keras.preprocessing.sequence import pad_sequences
from home.forms import LoginForm, RegistrationForm, EditProfileForm, ChatBotForm
from home.models import User, Conversation, SQLAlchemyError
from datetime import datetime
from nltk.corpus import stopwords
from config import tokenizer_file_path, label_encoder_file_path, model_file_path

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

try:
    # Load tokenizer, label encoder, and model
    with open(tokenizer_file_path, 'rb') as token_file:
    # with open('home/tokenizer(1).pkl', 'rb') as token_file:
        tokenizer = pickle.load(token_file)
    with open(label_encoder_file_path, 'rb') as label_file:
    # with open('home/label_encoder(1).pkl', 'rb') as label_file:
        le = pickle.load(label_file)

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as trained_model:
            model = tf.keras.models.load_model(model_file_path)
    else:
        print(f"Error: Model file '{model_file_path}' not found.")
        logging.error(f"Model file '{model_file_path}' not found")
        raise FileNotFoundError(f"Error: Model file '{model_file_path}' not found.")

except FileNotFoundError as e:
    print(e)
    logging.error(e)
except Exception as e:
    print(f"Error during initialization: {e}")
    logging.error(f"Error during initialization: {e}")

try:
    with open('home/mwalimu_sacco.json') as content:
        data1 = json.load(content)
    # data = open('mwalimu_sacco.json', 'r')

except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    logging.error(f"JSON decoding error: {e}")
    logging.error(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    raise ValueError(f"JSON decoding error: {e}") from None
except FileNotFoundError as e:
    print(e)
    logging.error(e)
except Exception as e:
    print(f"Error during initialization: {e}")
    logging.error(f"Error during initialization: {e}")

data = data1
responses = {}
for intent in data["intents"]:
    responses[intent['tag']] = intent['responses']

# Combine all responses into a single string
all_responses_combined = ' '.join([' '.join(response) for response in responses.values()])

# Split the combine responses into individual words
dataset_words = all_responses_combined.split()

tag_keywords = {
    "loan_inquiry": ["loan", "apply", "requirements", "interest rates", "options", "types"],
    "greeting": ["hello", "hi", "howdy", "hey"],
}


def preprocess_input(user_input):
    processed_text = ' '.join(
        [ltrs.lower() for ltrs in user_input.split() if ltrs not in (string.punctuation, stop_words)])
    return processed_text

@app.route('/', methods=['GET'])
# @login_required
def chatbot():
    """
    renders that
    :return:
    """
    form = ChatBotForm()
    return render_template('chatting.html', form=form)

@app.route('/', methods=['POST'])
def chatbot_response():
    form = ChatBotForm(request.form)

    if request.method == 'POST' or request.is_json:
        if request.is_json:
            data = request.get_json()
            user_input = data.get('user_input', '')
        else:
            user_input = form.user_input.data
        # Preprocess user input
        processed_input = preprocess_input(user_input)
        # Tokenize and pad the input
        input_sequence = tokenizer.texts_to_sequences([processed_input])
        padded_input = pad_sequences(input_sequence, maxlen=model.input_shape[1])
        # Make a prediction
        predictions = model.predict(padded_input)
        predicted_class = np.argmax(predictions)
        # Convert the predicted class back to the original tag using the label encoder
        predicted_tag = le.inverse_transform([predicted_class])[0]
        # Get a random response for the predicted tag
        response = random.choice(responses.get(predicted_tag, ['Sorry, I don\'t understand.']))
        try:
            # Create a new Conversation object
            # user = current_user if current_user.is_authenticated else None,
            conversation = Conversation(user_input=user_input,
                                        bot_response=response)
            # Add and commit the conversation to the database
            db.session.add(conversation)
            db.session.commit()
        except SQLAlchemyError as e:
            db.session.rollback()
            logging.error(f'Error saving user conversation: {str(e)}')
            flash(f'Error saving user conversation: {str(e)}', 'danger')
            return jsonify({'error': 'Database error'}), 500
        except Exception as e:
            db.session.rollback()
            logging.error(f'Unexpected error during conversation saving: {str(e)}')
            flash('An unexpected error occurred during conversation saving.', 'danger')
            return jsonify({'error': 'Unexpected error'}), 500
        # Check if the client expects JSON (Accept header contains 'application/json')
        if request.is_json:  # check if request is AJAX request
            return jsonify({'user_input': user_input, 'response': response})
        else:
            return render_template('chatting.html', form=form, user_input=user_input, response=response)
    if request.is_json:
        return jsonify({'error': 'Invalid form submission'}), 400  # HTTP 400 Bad Request
    else:
        flash('Invalid form submission', 'danger')
        return render_template('chatting.html', form=form)




@app.route('/home')
# @login_required
def home():
    """
    renders that
    :return:
    """
    return render_template("base1.html")

