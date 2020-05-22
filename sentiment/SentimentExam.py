# -*- coding: utf-8 -*-
"""
Created on Mon April 6 14:43:11 2020
@author: Sila
"""

# Slightly changed version of
# Petr Baudis and Martin Holecek tutorial from MLPrague 2018
# for information extraction from documents.

# Vocabulary: All words used, starting by the most frequent
with open('aclImdb/imdb.vocab','r',encoding='utf-8')as f:
    vocab = [word.rstrip() for word in f]
    # Keep only most frequent 5000 words rather than all 90000
    # Just saving memory - the long tail occurs too few times
    # for the model to learn anything anyway
    vocab = vocab[:5000]
    print('%d words in vocabulary' % (len(vocab),))

import re

def text_tokens(text):
    text = text.lower()
    text = re.sub("\\s", " ", text)
    text = re.sub("[^a-zA-Z' ]", "", text)
    tokens = text.split(' ')
    return tokens

import os

def load_dataset(dirname):
    X, y = [], []
    # Review files: neg/0_3.txt neg/10000_4.txt neg/10001_4.txt ...
    for y_val, y_label in enumerate(['neg', 'pos']):
        y_dir = os.path.join(dirname, y_label)
        for fname in os.listdir(y_dir):
            fpath = os.path.join(y_dir, fname)
            # print('\r' + fpath + '   ', end='')
            with open(fpath, 'r',encoding='utf-8') as f:
                tokens = text_tokens(f.read())
            X.append(tokens)
            y.append(y_val)  # 0 for 'neg', 1 for 'pos'
    print()
    return X, y

print("Load training set:")
X_train, y_train = load_dataset('aclImdb/train/')

# We are cheating here - this is a test set, not a validation set.
# This is just to make results quickly comparable to outside results
# during the tutorial, but you should normally never use the test set
# during training, of course!
print("Load test set:")
X_val, y_val = load_dataset('aclImdb/test/')

print( len(X_train), len(X_val) )

#A one hot encoding is a representation of categorical variables as binary vectors.
# This first requires that the categorical values be mapped to integer values.
# Then, each integer value is represented as a binary vector that is
# all zero values except the index of the integer, which is marked with a 1.
def bow_onehot_vector(tokens):
    vector = [0] * len(vocab)
    for t in tokens:
        try:
            vector[vocab.index(t)] = 1
        except:
            pass  # ignore missing words
    return vector

import sklearn
from sklearn.model_selection import train_test_split

# Only using 20 percent of the data
X_train_using, X_train_notusing, y_train_using, y_train_notusing= train_test_split(X_train, y_train, test_size=0.001)
X_val_using, X_val_notusing, y_val_using, y_val_notusing= train_test_split(X_val, y_val, test_size=0.001)

#Create BOW vectors.
print("Creating BOW vectors")
X_bow_train = [bow_onehot_vector(x) for x in X_train_using]
X_bow_val = [bow_onehot_vector(x) for x in X_val_using]

print("Start fitting model")
def best_train_history(history):
    best_epoch = np.argmax(history.history['val_accuracy'])
    print('Accuracy (epoch %d): %.4f train, %.4f val' % \
          (best_epoch + 1, history.history['accuracy'][best_epoch], history.history['val_accuracy'][best_epoch]))


from keras.layers import Activation, Dense, Input
from keras.models import Model
import numpy as np


class BOWSentimentModel(object):
    def __init__(self):
        bow = Input(shape=(len(vocab),), name='bow_input')
        # weights of all inputs
        sentiment = Dense(1)(bow)
        # normalize to [0, 1] range
        sentiment = Activation('sigmoid')(sentiment)

        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=100, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


# sentiment = BOWSentimentModel()
# history = sentiment.train(X_bow_train, y_train_using, X_bow_val, y_val_using)
# best_train_history(history)

# test_text = 'The movie is overflowing with life, rich with all the grand emotions and vital juices of existence, up to and including blood. Without a doubt, one of the best films of all time. As far as crime drama/mobster genre goes, it starts and ends with this masterpiece. For those who have already witnessed the rise and fall of the Corleone family, which mirrors the trajectory of the American dream, now is as good a time as ever to revisit. For those who havent seen Coppolas mafia movies: brace yourselves. This is it.'
# test_tokens = text_tokens(test_text)
# print(test_text, sentiment.predict([bow_onehot_vector(test_tokens)])[0])
#
# test_text = 'On a journey to San Francisco, Richard, his father and a girl are shipwrecked. The two children escape while their father is on another lifeboat. Come to an island. And then nothing happens for 2 hours. I have never been so bored in my life. Repeat, nothing happes.  The children fish and build huts. But just isnt very exciting to watch. Some people actually fell asleep in their seats next to me. The only thing that kept me from walking out was the fact that it was raining outside.'
# test_tokens = text_tokens(test_text)
# print(test_text, sentiment.predict([bow_onehot_vector(test_tokens)])[0])
#
# print('Interstellar (2014) | 10')
# test_text = 'This movie was the best written, acted, visual effected, etc. movie. This movie was the best movie I have ever seen. I am a huge Christopher Nolan fan and this movie was his finest. Matthew McConaughey turned in his best performance of his lifetime. Anne Hathaway was an amazing supporting actress and compared to her performance in Les Miserables, I have no idea how she didnt get an Oscar for this. The visual effects were more than just Oscar worthy. They were pioneering. I have never seen anything like it. One thing I would recommend is having a little previous knowledge about space. Not like Einstein stuff though. I would recommend you see this movie as fast as you can if you are a Nolan fan or not. I give this movie a rating of 97 out of 100.'
# test_tokens = text_tokens(test_text)
# print(test_text, sentiment.predict([bow_onehot_vector(test_tokens)])[0])
#
# print('The Banker (2020) | 5')
# test_text = 'This is a simplified racial story told time and time again isnt it 2020? Samuel L plays his normal great portrayal of a role and the rest of the cast do adequately, but there really is no pizazz to this film. It drags along with no surprises. I definitely wouldnt watch this again and if we werent in a pandemic probably wouldnt have made it through once.'
# test_tokens = text_tokens(test_text)
# print(test_text, sentiment.predict([bow_onehot_vector(test_tokens)])[0])
#
# print('Batman & Robin (1997) | 2')
# test_text = 'Itd be easy to write off "Batman & Robin" as just a toy commercial. But honestly, there are merchandise-driven movies that are actually good. No, thats not the problem with this movie; its woes are much more serious. Clooney couldve made a decent Batman in another movie, but hes pretty much just Clooney here. None of these characters are paid any actual mind; Freeze and Ivy arent close to the mark at all (the less that can be said about Bane here, the better), while Batgirl is awkwardly shoehorned into the plot; the conflict between Batman and Robin is forced, as well as Alfreds illness. And none of this occurs organically; its clumsy in every respect. If you take all of that and throw in the hysterical score, the overacting and awful dialogue, its not hyperbole to call this the bottom of the barrel for superhero movies. Its loud, fetishistic and just plain uncomfortable. And theres serious money behind this. What the hell were they thinking? '
# test_tokens = text_tokens(test_text)
# print(test_text, sentiment.predict([bow_onehot_vector(test_tokens)])[0])

import matplotlib.pyplot as plt
def plot_train_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

# plot_train_history(history)

class BOWHiddenSentimentModel(object):
    def __init__(self, N=64):
        bow = Input(shape=(len(vocab),), name='bow_input')
        hidden = Dense(N, activation='tanh')(bow)
        sentiment = Dense(1, activation='sigmoid')(hidden)

        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=100, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))

# Try the whole thing again. Now with a hidden layer in the model.
# sentiment = BOWHiddenSentimentModel()
# history = sentiment.train(X_bow_train, y_train_using, X_bow_val, y_val_using)
# best_train_history(history)
#
# plot_train_history(history)
# exit()
from keras.layers import Dropout
from keras import regularizers

class BOWHiddenRegularizedSentimentModel(object):
    def __init__(self, N=128):
        bow = Input(shape=(len(vocab),), name='bow_input')
        # hidden = Dropout(0.5)(Dense(N, activation='tanh')(bow))
        hidden = Dropout(0.5)(Dense(N, kernel_regularizer=regularizers.l2(1e-3))(bow))
        sentiment = Dense(1, activation='sigmoid')(hidden)

        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=100, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


class BOWHiddenRegularizedSentimentModel2(object):
    def __init__(self, N=128):

        bow = Input(shape=(len(vocab),), name='bow_input')
        hidden = Dropout(0.2)(bow)
        hidden = Dense(32, activation='relu')(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(16, activation='relu')(hidden)
        hidden = Dropout(0.5, )(hidden)
        sentiment = Dense(1, activation='sigmoid')(hidden)

        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=100, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


class BOWHiddenRegularizedSentimentModel3(object):
    def __init__(self, N=128):

        bow = Input(shape=(len(vocab),), name='bow_input')
        hidden = Dropout(0.2)(bow)
        hidden = Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu')(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(16, activation='relu')(hidden)
        hidden = Dropout(0.5)(hidden)

        #hidden = Dense(128, kernel_regularizer=regularizers.l1(0.01))(bow)
        #hidden = Dropout(0.1)(bow)
        #hidden = Dense(32, activation='relu')(hidden)
        #hidden = Dropout(0.5)(hidden)
        #hidden = Dense(64, activation='relu')(bow)
        #hidden = Dense(N, kernel_regularizer=regularizers.l2(1e-3))(hidden)
        #hidden = Dropout(0.5, )(hidden)
        #hidden = Dense(16, activation='relu')(hidden)
        #hidden = Dropout(0.5, )(hidden)
        #hidden = Dropout(0.2, noise_shape=None, seed=None)(hidden)
        #hidden = Dropout(0.2)(hidden)
        #hidden = Dropout(0.5)(hidden)
        #hidden = Dense(N, kernel_regularizer=regularizers.l2(1e-3))(hidden)
        #hidden = Dense(16, activation='relu')(hidden)

        sentiment = Dense(1, activation='sigmoid')(hidden)
        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=30, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


class BOWHiddenRegularizedSentimentModel4(object):
    def __init__(self, N=32):
        bow = Input(shape=(len(vocab),), name='bow_input')
        #hidden = Dropout(0.5)(Dense(N, activation='tanh')(bow))
        hidden = Dropout(1.0)(Dense(N, kernel_regularizer=regularizers.l2(1e-3))(bow))
        sentiment = Dense(1, activation='sigmoid')(hidden)

        self.model = Model(inputs=[bow], outputs=[sentiment])
        self.model.summary()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, X_val, y_val):
        print('Fitting...')
        return self.model.fit(np.array(X), y, validation_data=(np.array(X_val), y_val), epochs=30, verbose=1)

    def predict(self, X):
        return self.model.predict(np.array(X))


sentiment = BOWHiddenRegularizedSentimentModel()
history = sentiment.train(X_bow_train, y_train_using, X_bow_val, y_val_using)
best_train_history(history)

plot_train_history(history)