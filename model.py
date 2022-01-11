import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

maxlen=50
def get_sequences(tokenizer, sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, truncating = 'post', padding='post', maxlen=maxlen)
    return padded

# read data, divide to sentences and labels
data = pd.read_csv('data.csv')
sentences = data['1']
labels = data ['0']

# divide to train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2)

# init tokenizer and fit to our data
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_data)

padded_train_seq = get_sequences(tokenizer, train_data)

classes = set(train_labels)
class_to_index = dict((c,i) for i, c in enumerate(classes))
index_to_class = dict((v,k) for k, v in class_to_index.items())
names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])
train_labels = names_to_ids(train_labels)

print(train_labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
     loss='sparse_categorical_crossentropy',
     optimizer='adam',
     metrics=['accuracy']
)

test_seq = get_sequences(tokenizer, test_data)
test_labels = names_to_ids(test_labels)

h = model.fit(
     padded_train_seq, train_labels,
     validation_data=(test_seq, test_labels),
     epochs=20,
     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)]
)

model.save("model.h5")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved model and tokenizer to disk")

#sentence = 'olen vihainen sinulle'
#sequence = tokenizer.texts_to_sequences([sentence])
#paddedSequence = pad_sequences(sequence, truncating = 'post', padding='post', maxlen=maxlen)
#p = model.predict(np.expand_dims(paddedSequence[0], axis=0))[0]
#pred_class=index_to_class[np.argmax(p).astype('uint8')]
#print('Sentence:', sentence)
#print('Predicted Emotion: ', pred_class)