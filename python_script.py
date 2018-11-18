# coding: utf-8

# # Detect Language from Text
#
# In this project we will focus on Text Classification. Text Classification has been used in the past for [Sentiment
# Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis). Here, we will use text classification to predict the
# language of the input text.
#
# In order to classify text, we will look at a class of Neural Network where connections between units form a
# directed cycle, called Recurrent Neural Network (RNNs). RNNs use an internal memory to process sequences of
# elements and is able to learn from the syntactic structure of text. Our model will be able to classify text based
# on the text we train it with.
#
# RNNs are [very effective](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) when understanding sequence of
# elements and have been used in the past to generate or classify text. I will use a Recurrent Neural Network to
# predict the language of the input text.
#
# The solution below, can be useful in the following scenarios:
# - A Global Customer Service team receives all their requests via a single email address. The model below can be
# used to identify the language of the text and forward it to the right person/group.
# - Tagging an email.
# - A Government agency of a country that has several regional languages, e.g. Spain, which is what we've used in
# this notebook to train our model.
#
# Alternative solutions:
#
# The Translation API by Google can be used to detect the language of a text.
# https://cloud.google.com/translate/docs/detecting-language
#

# ## Metrics
#
# The purpose of this project, is to create a model that will be able to identify the language the input text is in.
#
# The performance of our model will be measure by the accuracy of identifying the right language.

# ## Datasets and Input
#
# To train our model we will use the [recognised languages](https://en.wikipedia.org/wiki/Spain#cite_note-c-6) in [
# Spain](https://en.wikipedia.org/wiki/Spain) as input. Apart from the official language - Castilian (Spanish),
# there are different languages spoken in various autonomous communities, e.g. Catalan, Gailicia, Basque, Aragonese,
# Asturian and Occitan.
#
# The text used as input comes from Wikipedia. We are using the Leipzig Corpora Collection -
# http://wortschatz.uni-leipzig.de/en/download/ dataset available for each language. We've trained the model with ~
# 10,000 sentences per language.
#
# Each dataset, e.g. [Asturian](http://pcai056.informatik.uni-leipzig.de/downloads/corpora/ast_wikipedia_2016_10K.tar
# .gz), contains a file with sentences, minimal preprocessing (remove index and append the language name) was done to
#  this file to train our model.
#
# The file included in the Github repository (all_sentences.txt) doesn't require further processing.

# In[1]:


# Lets extract our sentences and have a look at the data we will be dealing with

import pandas as pd

data = pd.read_csv("all-sentences.txt", names=["sentence", "language"], header=None, delimiter="|")
data.describe()

# In[2]:


import re


def process_sentence(sentence):
    '''Removes all special characters from sentence. It will also strip out
    extra whitespace and makes the string lowercase.
    '''
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())


# In[3]:


from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# As our sentences in all_sentences.txt are in order, we need to shuffle it first.
sss = StratifiedShuffleSplit(test_size=0.2, random_state=0)

# Clean the sentences
X = data["sentence"].apply(process_sentence)
y = data["language"]

# Split all our sentences
elements = (' '.join([sentence for sentence in X])).split()

X_train, X_test, y_train, y_test = None, None, None, None

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# In[4]:


len(X_train), len(X_test)

# In[5]:


languages = set(y)
print("Languages in our dataset: {}".format(languages))

# In[6]:


print("Feature Shapes:")
print("\tTrain set: \t\t{}".format(X_train.shape),
      "\n\tTest set: \t\t{}".format(X_test.shape))
print("Totals:\n\tWords in our Dataset: {}\n\tLanguages: {}".format(len(elements), len(languages)))

# In[7]:


# Lets look at our training data
X_train[:10], y_train[:10]


# In[8]:


def create_lookup_tables(text):
    """Create lookup tables for vocabulary
    :param text: The text split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)

    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {v: k for k, v in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


# In[9]:


elements.append("<UNK>")

# Map our vocabulary to int
vocab_to_int, int_to_vocab = create_lookup_tables(elements)
languages_to_int, int_to_languages = create_lookup_tables(y)

print("Vocabulary of our dataset: {}".format(len(vocab_to_int)))


# In[10]:


def convert_to_int(data, data_int):
    """Converts all our text to integers
    :param data: The text to be converted
    :return: All sentences in ints
    """
    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])

    return all_items


# In[11]:


# Convert our inputs
X_test_encoded = convert_to_int(X_test, vocab_to_int)
X_train_encoded = convert_to_int(X_train, vocab_to_int)

y_data = convert_to_int(y_test, languages_to_int)

# In[12]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

enc.fit(y_data)

# One hot encoding our outputs
y_train_encoded = enc.fit_transform(convert_to_int(y_train, languages_to_int)).toarray()
y_test_encoded = enc.fit_transform(convert_to_int(y_test, languages_to_int)).toarray()

# In[13]:


# Sample of our encoding
print(y_train_encoded[:10], '\n', y_train[:10])

# ### Check the Version of TensorFlow and Access to GPU

# In[20]:


from distutils.version import LooseVersion

import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# In[22]:


# Import Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# In[23]:


# Hyperparameters
max_sentence_length = 200
embedding_vector_length = 300
dropout = 0.5

# In[24]:


import numpy

with tf.device('/gpu:0'):
    # Truncate and pad input sentences
    X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=max_sentence_length)
    X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=max_sentence_length)

    # Create the model
    model = Sequential()

    model.add(Embedding(len(vocab_to_int), embedding_vector_length, input_length=max_sentence_length))
    model.add(LSTM(256, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
    model.add(LSTM(256, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(len(languages), activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

# In[27]:


# Train the model
model.fit(X_train_pad, y_train_encoded, epochs=2, batch_size=256)

# Final evaluation of the model
scores = model.evaluate(X_test_pad, y_test_encoded, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# In[29]:


import paperspace

# In[30]:


paperspace.run()


# ## Results
#
# After 2 epochs, our model is able to achieve an accuracy of 99.57%. Which is quite good compared to the time it
# will take someone to classify the text and that the person will need to be familiar with all languages.

# In[19]:


def predict_sentence(sentence):
    """Converts the text and sends it to the model for classification
    :param sentence: The text to predict
    :return: string - The language of the sentence
    """

    # Clean the sentence
    sentence = process_sentence(sentence)

    # Transform and pad it before using the model to predict
    x = numpy.array(convert_to_int([sentence], vocab_to_int))
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)

    prediction = model.predict(x)

    # Get the highest prediction
    lang_index = numpy.argmax(prediction)

    return int_to_languages[lang_index]


# In[20]:


predict_sentence(
    "foi Vicepresidente de la institución y responsable del programa de formación de la mesma, dirixendo la UABRA. Al "
    "empar, foi l'entamador y direutor de la coleición académica")

# In[21]:


predict_sentence(
    "Els socis a l’Ajuntament de Barcelona han pactat la discrepància (de fet, divendres van votar en sentit contrari "
    "al ple municipal en una moció presentada per Ciutadans")

# In[22]:


predict_sentence(
    "Espardióse hacia'l sur cola Reconquista y cola conversión de Castiella nel reinu con más puxu de los cristianos, "
    "impúnxose como llingua de la población na mayor parte de lo que dempués sedría España.")

# In[23]:


predict_sentence(
    "Eklipse bat gorputz bat beste baten itzalean sartzen denean gertatzen den fenomeno astronomikoa da. Alegia, "
    "argizagi baten estaltzea, haren eta begiaren artean tartekatzen den beste argizagi batek eragina.")

# In[24]:


predict_sentence(
    "Como intelectual os seus amplos desenvolvementos teóricos e filosóficos do marxismo produciron o "
    "marxismo-leninismo, unha aplicación pragmática rusa do marxismo que fai fincapé no papel fundamental")

# In[25]:


predict_sentence(
    "L’èuro es la moneda comuna dels 28 estats de l’Union Europèa (UE) — e la moneda unica de 19 estats membres pel "
    "moment — que succedís a l’ECU (European Currency Unit, unitat de compte europèa) que n'èra la moneda comuna.")

# In[26]:


predict_sentence(
    "Así se tiene, por ejemplo, que tanto el trigo como los otros cereales se han empleado en Europa y parte de "
    "África; el maíz es frecuente en América; el arroz, en Asia.")

# In[27]:


predict_sentence(
    "America d'o Sur, tamién clamada Sudamerica, ye un subcontinent trescruzato por l'Equador, con a mayor parti d'a "
    "suya aria en l'hemisferio sud. Ye situato entre l'Ocián Pacifico y l'Ocián Atlantico.")

# In[28]:


predict_sentence(
    "O vin ye una bebida alcoholica obtenita d'a fermentación alcoholica, por l'acción d'os recientos, d'o suco u "
    "mosto d'os fruitos d'a planta Vitis vinifera (vinyers) que transforma os zucres d'o fruito en alcohol etilico y "
    "gas anhidrido carbonico.")

# In[29]:


predict_sentence(
    "As Xornadas de Xullo de 1917 foi o nome que recibiron as protestas armadas apoiadas polos anarcocomunistas e os "
    "bolxeviques, finalmente fracasadas, que trataron de derrocar ao Goberno Provisional Ruso e traspasar o poder aos "
    "soviets (consellos) en xullo dese ano.")

# In[30]:


predict_sentence(
    "El monestir té 2.000 metres quadrats d'extensió i és de planta irregular. Al llarg dels més de 500 anys de la "
    "seva història, ha passat per diverses modificacions, sobretot arran del terratrèmol de Lisboa de 1755.")

# In[31]:


predict_sentence(
    "Esta plusvalía é apropiada polo capitalista e dela procede a ganancia. Esta apropiación constitúe a base "
    "fundamental do modo de produción capitalista e a súa vez estas condicións materiais determinan a superestrutura")

# In[32]:


predict_sentence("Putin cambia a su embajador en Washington, figura clave de la trama rusa")

# In[33]:


predict_sentence(
    "Erigit durant els segles XIV-XV, hi destaquen pel seu interès artístic l'església d'estil gòticmudèjar, "
    "així com les estances decorades amb frescos de Daniel Vázquez Díaz, el claustre i el museu, on es conserven "
    "nombrosos objectes commemoratius del descobriment d'Amèrica, i una escultura de l'advocació mariana sota la qual es troba el convent,")

# In[ ]:


