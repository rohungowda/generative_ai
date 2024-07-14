import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (layers, models, callbacks, utils, metrics, losses, optimizers)
import string
import re

BATCH_SIZE = 64
VOCAB_SIZE = 20000
EMBEDDING_SIZE = 250
UNITS = 256
EPOCHS = 25
max_tokens = 300


def create_lstm_research():
    inputs = layers.Input(shape=(None,), dtype='int32')
    x = layers.Embedding(VOCAB_SIZE,EMBEDDING_SIZE)(inputs)


    x = layers.LSTM(512,return_sequences=True)(x)
    x= layers.Dropout(0.25)(x)
    attention = layers.Attention()([x,x])
    x = layers.concatenate([x,attention])



    x = layers.LSTM(256,return_sequences=True)(x)
    x= layers.Dropout(0.25)(x)
    attention = layers.Attention()([x,x])
    x = layers.concatenate([x,attention])


    x = layers.LSTM(128,return_sequences=True)(x)
    x= layers.Dropout(0.25)(x)
    attention = layers.Attention()([x,x])
    x = layers.concatenate([x,attention])

    outputs = layers.Dense(VOCAB_SIZE,activation='softmax')(x)
    lstm = models.Model(inputs,outputs)
    return lstm

def create_basic_lstm():
    inputs = layers.Input(shape=(None,), dtype='int32')
    x = layers.Embedding(VOCAB_SIZE,EMBEDDING_SIZE)(inputs)
    x = layers.LSTM(UNITS,return_sequences=True)(x)
    outputs = layers.Dense(VOCAB_SIZE,activation='softmax')(x)
    lstm = models.Model(inputs,outputs)
    return lstm


def create_stacked_lstm():
    inputs = layers.Input(shape=(None,), dtype='int32')
    x = layers.Embedding(VOCAB_SIZE,EMBEDDING_SIZE)(inputs)
    x = layers.LSTM(UNITS,return_sequences=True)(x)
    x= layers.Dropout(0.30)(x)
    x = layers.LSTM(UNITS // 2,return_sequences=True)(x)
    x= layers.Dropout(0.30)(x)
    x = layers.LSTM(UNITS // 2,return_sequences=True)(x)
    x= layers.Dropout(0.30)(x)
    outputs = layers.Dense(VOCAB_SIZE,activation='softmax')(x)
    lstm = models.Model(inputs,outputs)
    return lstm



text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize='lower',
    output_mode='int',
    output_sequence_length=max_tokens + 1,
)



def prepare_texts(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    text = re.sub(f"([{string.punctuation}])",r'',text)
    text.strip()
    return text

def convert_data(text):
    text = tf.expand_dims(text, -1)
    tokens = text_vectorizer(text)
    return tokens[:,:-1], tokens[:,1:] # 0- end-1, 1 - end


class SaveModel(callbacks.Callback):
    def __init__(self, name):
        self.name = name
    def on_epoch_end(self,epochs,logs=None):
        self.model.save(f"./autoregressive_models_lstms/{self.name}_{epochs}.keras")

class LyricGenerator(callbacks.Callback):
    def __init__(self,index_to_word,top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {word:index for index,word in enumerate(self.index_to_word)}
    def sample_from(self,probs,temperature):
        # greater temperature probabilites more adventures +   # lesser temperature probabilities are less adventures
        probs = probs ** (1/temperature)
        # normalizes the probabilites so between 0 - 1
        probs = probs / np.sum(probs)
        # choose a random index based on the probability distribution provided
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens,temperature):
        start_tokens = [self.word_to_index.get(x,1) for x in start_prompt]
        sample_token = None

        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose='0')
            sample_token,probs = self.sample_from(y[0][-1], temperature)
            start_tokens.append(sample_token)
            start_prompt += (' ' + self.index_to_word[sample_token])
        print(f"\n Generated Text: \n {start_prompt} \n")

    def on_epoch_end(self, epochs,logs=None):
        print()
        print("0.8 temperature")
        self.generate("I cannot stop loving you baby", max_tokens=max_tokens, temperature=0.8)
        print("0.7 temperature")
        self.generate("I cannot stop loving you baby", max_tokens=max_tokens, temperature=0.7)
        print("0.6 temperature")
        self.generate("I cannot stop loving you baby", max_tokens=max_tokens, temperature=0.6)


directory = "../../datasets/csv/"
raw_text_dataset = []

for file in os.listdir(directory):
    filename = os.path.join(directory, file)
    df = pd.read_csv(filename)
    df = df.dropna(subset=['Lyric'])
    df = df[df['Lyric'] != 'lyrics for this song have yet to be released please check back once the song has been released']
    lyrics = df['Lyric'].values
    for song in lyrics:
        song = prepare_texts(song)
        song = song.split(' ')
        i = 0
        while i < len(song):
            text = song[i:(i+max_tokens)]
            if len(text) >= 100:
                text = ' '.join(text)
                raw_text_dataset.append(text)
            i += max_tokens
print(len(raw_text_dataset))


#raw_text_dataset = [prepare_texts(text) for text in raw_text_dataset]


text_dataset = tf.data.Dataset.from_tensor_slices(raw_text_dataset).batch(BATCH_SIZE).shuffle(1500)

#text_batch = next(iter(text_dataset.take(1)))[0]
#print(text_batch)

text_vectorizer.adapt(text_dataset)

vocab = text_vectorizer.get_vocabulary()


processed_text_data = text_dataset.map(convert_data)



lstm = create_lstm_research()
lstm.summary()

loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate=0.0005)
lstm.compile(optimizer,loss_fn)
history = lstm.fit(processed_text_data,epochs=EPOCHS, callbacks=[SaveModel("attention_lstm"), LyricGenerator(vocab)])


fig1 = plt.figure()
plt.plot(history.history['loss'])

plt.show()