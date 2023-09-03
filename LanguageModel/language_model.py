import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random


class ModelContext:
    def __init__(self):
        self.model = None
        self.classes = []
        self.words = []
        self.lemmatizer = WordNetLemmatizer()

    def train(self, data):
        dataX, dataY = self._intents_preporcessing(data)
        # lemmatize all the words in the vocab and convert them to lowercase
        # if the words don't appear in punctuation
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]
        # sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        training = self._trainingset_creating(dataX, dataY)
        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_X = np.array(list(training[:, 0]))
        train_Y = np.array(list(training[:, 1]))
        self.model = self._create_model(train_X, train_Y)
        adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=["accuracy"])
        self.model.fit(x=train_X, y=train_Y, epochs=250, verbose=1)
        print("Model is ready")

    def predict(self, msg):
        return self._predict_class(msg)

    def _clean_text(self,text, word):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def _bag_of_words(self,text):
        tokens = self._clean_text(text, self.words)
        bow = [0] * len(self.words)
        for w in tokens:
            for idx, word in enumerate(self.words):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)

    def _predict_class(self,text):
        bow = self._bag_of_words(text)
        result = self.model.predict(np.array([bow]))[0]  # Extracting probabilities
        thresh = 0.5
        y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)  # Sorting by values of probability in decreasing order
        return_list = []
        for r in y_pred:
            return_list.append(self.classes[r[0]])  # Contains labels(tags) for highest probability
        return return_list

    def _create_model(self, input_size_x, output_size_y):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(input_size_x[0]),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(output_size_y[0]), activation="softmax"))
        return model

    def _intents_preporcessing(self, data):
        dataX = []
        dataY = []
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)  # tokenize each pattern
                self.words.extend(tokens)  # and append tokens to words
                dataX.append(pattern)  # appending pattern to data_X
                dataY.append(intent["tag"]),  # appending the associated tag to each pattern
            # adding the tag to the classes if it's not there already
            if intent["tag"] not in self.classes:
                self.classes.append(intent["tag"])
        return dataX, dataY

    def _trainingset_creating(self,dataX, dataY):
        training = []
        out_empty = [0] * len(self.classes)
        # creating the bag of words model
        for idx, doc in enumerate(dataX):
            bow = []
            text = self.lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                bow.append(1) if word in text else bow.append(0)
            # mark the index of class that the current pattern is associated
            output_row = list(out_empty)
            output_row[self.classes.index(dataY[idx])] = 1
            # add the one hot encoded BoW and associated classes to training
            training.append([bow, output_row])
        return training

    def is_trained(self):
        return self.model is not None
