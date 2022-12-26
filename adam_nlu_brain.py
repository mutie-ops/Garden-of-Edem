import json
import string
import random
import keras
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Dropout

#nltk.download('all')
# nltk.download('wordnet')

# loading json file
data_file = open("C:\\Users\\benja\\Desktop\\pythonProject\\adam\\intents.json").read()
data = json.loads(data_file)

words = []  # vocabulary for patterns
classes = []  # vocabulary for tags
data_x = []  # storing each pattern
data_y = []  # storing tags corresponding to each pattern in data_x

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent['tag'])

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

lementizer = WordNetLemmatizer()

# lememntize the vocabulary and convert them to lower case
# if the words don't appear in punctuation

words = [lementizer.lemmatize(word.lower) for word in words if word not in string.punctuation]

# sorting the words and classes

# using sets to ensure no duplication
words = sorted(set(words))
classes = sorted(set(classes))

# converting the text to numbers
training = []
out_empty = [0] * len(classes)

# creating the bag of words_model

for index, doc in enumerate(data_x):
    bow = []

    text = lementizer.lemmatize(doc.lower())
    for word in words:
        if word in text:
            bow.append(1)
        else:
            bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(data_y[index])] = 1

    #  adding the one hot encoded bow and associated classes to training
    training.append([bow, output_row])

# shuffle the data and convert it to an array
random.shuffle(training)

training = np.array(training, dtype=object)

# split the features and target label
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

#  building the neural network

model = Sequential()
model.add(Dense(128, input_shape=len(train_x[0], ),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))


# Adam optimizer to correctly adjust the weights

adam = tf.optimizers.Adam(learning_rate=0.01, decay =1e-6)
model.compile(loss= 'categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())

model.fit(x= train_x,y=train_y, epochs= 150, verbose=1)
model.save('model.h5')




