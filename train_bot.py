from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam
#🗿🍷
from data_preprocessing import preprocess_train_data

def train_bot_model(trainingX,trainingY):
    model = Sequential()
    model.add(Dense(128,input_shape = (len(trainingX[0]),),activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(trainingY[0]),activation = "softmax"))
    
    model.compile(optimizer= "adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(trainingX,trainingY,epochs = 200,batch_size = 5,verbose = True)
    model.save("chat_bot_model.h5",history)
    print("modelo criado com sucesso")

trainX, trainY = preprocess_train_data()
train_bot_model(trainX, trainY)