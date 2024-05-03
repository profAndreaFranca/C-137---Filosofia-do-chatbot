#Biblioteca de pré-processamento de dados de texto
#pip install nltk
import nltk
nltk.download('punkt')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words = []
classes = []
word_tags_list = []
ignore_words = ['?','!',',','.',"'s","'m"]
train_data_file = open("intents.json").read()
intents = json.loads(train_data_file)

#função para anexar palavras-tronco🗿🍷
def getStemWord(words,ignore_words) :
    stemWords = [] 
    for word in words :
        if word not in ignore_words :
            w = stemmer.stem(word.lower())
            stemWords.append(w) 
    return stemWords    


for intent in intents["intents"]:
        for pattern in intent["patterns"]:
                pattern_word = nltk.word_tokenize(pattern)
                words.extend(pattern_word)
                word_tags_list.append((pattern_word,intent["tag"]))
        if intent["tag"] not in classes :
                classes.append(intent["tag"])
                stem_words = getStemWord(words,ignore_words)   
print(stem_words)
print(word_tags_list[0])
print(classes)
         
        
        
def CreateBotCorpus(stemWords,classes) :
      stemWords = sorted(list(set(stemWords)))
      classes = sorted(list(set(classes)))
      pickle.dump(stemWords, open("words.pkl", "wb"))
      pickle.dump(classes, open("classes.pkl", "wb"))
      return(stemWords, classes)

stem_words, classes = CreateBotCorpus(stem_words, classes)
print("****************************")
print(stem_words)
print(classes)

    
        # Adicione todas as palavras dos padrões à lista
       
        # Adicione todas as tags à lista classes


#Crie o corpus de palavras para o chatbot
