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

#Aula 137 e 138

#função para anexar palavras-tronco🗿🍷

    
        
       
        

def getStemWord(words,ignore_words) :
    
    stemWords = [] 
    for word in words :
        if word not in ignore_words :
            w = stemmer.stem(word.lower())
            stemWords.append(w) 
    return stemWords    


for intent in intents["intents"]:
        # Adicione todas as palavras dos padrões à lista
        for pattern in intent["patterns"]:
                pattern_word = nltk.word_tokenize(pattern)
                words.extend(pattern_word)
                word_tags_list.append((pattern_word,intent["tag"]))
        # Adicione todas as tags à lista classes
        if intent["tag"] not in classes :
                classes.append(intent["tag"])
                stem_words = getStemWord(words,ignore_words)   
print(stem_words)
print(word_tags_list[0])
print(classes)
         
        
#Crie o corpus de palavras para o chatbot        
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


traning_data = []
number_of_tags = len(classes)
labels = [0] * number_of_tags #[0,0,0,0]🗿🍷

# Para criar um saco de palavras, execute um loop
# for em word_tags_list. Cada elemento é chamado
# de word_tags.
#       ● Para cada elemento, criaremos um array de
#       zeros e números um.
#       ● Defina um array vazio com o nome
#       bag_of_words.
#       ● Em seguida, defina pattern_words
#       (palavras-padrão) como o primeiro índice de
#       word_tags.

#       ● Execute um loop for em pattern_words
#       para encontrar suas palavras-tronco.

#       ● Se essas palavras estiverem presentes nas
#       palavras-tronco, anexaremos 1 ao
#       bag_of_words; caso contrário, anexaremos 0.

#       ● Reproduza o bag_of_words para verificar o
#       resultado.
for  word_tags in word_tags_list:
        bag_of_words = []
        pattern_words = word_tags[0]
        for word in pattern_words: 
              index = pattern_words.index(word)
              word = stemmer.stem(word.lower())
              pattern_words[index] = word

        for word in stem_words: 
                if word in pattern_words: 
                        bag_of_words.append(1)
                else: 
                        bag_of_words.append(0)
        print(bag_of_words)
# Criando a label_encoding (codificação de etiquetas):        
        labels_encoding = list(labels)
        tag = word_tags[1]
        index=classes.index(tag)
        labels_encoding[index] = 1
        traning_data.append([bag_of_words,labels_encoding])


print(traning_data[0])

def preprocess_traning_data(traning_data) :
       traning_data = np.array(traning_data,dtype = object)
       traning_X = list(traning_data[:,0])
       traning_Y = list(traning_data[:,1])
       print(traning_X[0],traning_Y[0])
       return(traning_X,traning_Y)

traning_X, traning_Y = preprocess_traning_data(traning_data)
