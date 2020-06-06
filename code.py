import os
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import keras_tpu as kt
from keras import layers
import pandas as pd
from tensorflow.contrib import seq2seq as sq
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

input_text = open('Input_text.txt', 'rb').read().decode(encoding='utf-8')
vocabulary = sorted(set(input_text))
idxcnv = {unique:idx for idx, unique in enumerate(vocabulary)}
chrcnv = np.array(vocabulary)
text_to_int = np.array([idxcnv[char] for char in input_text])
seq_len = 100
ex_per_epoch = len(input_text) // (seq_len + 1)
chrdata = tf.data.Dataset.from_tensor_slices(text_to_int)
sequences = chrdata.batch(seq_len+1, drop_remainder=True)

def slipt_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(slipt_input_target)

Batch_size = 64
Buffer_size = 20000

dataset = dataset.shuffle(Buffer_size).batch(Batch_size, drop_remainder=True)
VocabularySize = len(vocabulary)
embedding_dim = 256
rnn_units = 1024

def build_rnn(VocabularySize, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VocabularySize, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(VocabularySize)])
    return model

def loss(lables, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(lables, logits, from_logits=True)

checkpoint_dir = './training_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 25

model = build_rnn(VocabularySize, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  num_generate = 1000
  input_eval = [idxcnv[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 0.5
  model.reset_states()
  for i in range(num_generate):
     predictions = model(input_eval)
     predictions = tf.squeeze(predictions, 0)
     predictions = predictions / temperature
     predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
     input_eval = tf.expand_dims([predicted_id], 0)
     text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"The "))