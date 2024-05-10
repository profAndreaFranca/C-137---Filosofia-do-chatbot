#🗿🍷
import nltk
nltk.download('punkt')

import json 
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib🗿🍷
import tensorflow 
from data_preprocessing import get_stem_words
model = tensorflow.keras.models.load_model("./chat_bot_model.h5")

intents = json.loads(open("./intents.json").read())
words = pickle.load(open("./words.pkl",'rb'))
classes = pickle.load(open("./classes.pkl",'rb'))

def preprocess_user_input(user_input):
    input_word_token1 = nltk.word_tokenize(user_input)
    input_word_token2 = get_stem_words(input_word_token1, ignore_words)
    input_word_token2 = sorted(list(set(input_word_token2)))

    bag = []
    bag_of_words = []

    for word in words:
        if word in input_word_token2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

    bag.append(bag_of_words)
    return(np.array(bag))

def bot_class_prediction(user_input):
    imp = preprocess_user_input(user_input)
    prediction = model.predict(imp)
    predict_class_label = np.argmax(prediction[0])
    return(predict_class_label)

def bot_response(user_input):
    predict_class_label = bot_class_prediction(user_input)
    predict_class = classes[predict_class_label]
    for intent in intents["intents"]:
        if intent["tag"] == predict_class:
            bot_response = random.choice(intent["responses"])
            return(bot_response)
        #🗿🍷
print("oi eu sou o Jeremias, como eu posso ajudar?")
while True :
    user_input = input("escreva sua mensagem aqui")
    print("user_input :", user_input)
    response = bot_response(user_input)
    print("Jeremias :",response)