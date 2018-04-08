"""
Script that train a ConvNet model for the Yelp dataset
This is inspired from Yoon Kim paper : 
https://arxiv.org/abs/1408.5882

# Note #1: 
To execute the script, you need to download the 
Google's pre-trained Word2Vec model available at : 
https://code.google.com/archive/p/word2vec/ 

# Note #2: 
You might need to install gensim package using either: 
- pip install gensim 
- conda install -c anaconda gensim (anaconda environment)

# Compile command : 
python ConvNet.py --dataset ../yelp_academic_dataset_review.pickle --word2vec ../GoogleNews-vectors-negative300.bin

"""

## Import libraries
# remove warnings
import warnings
warnings.filterwarnings('ignore')

import argparse
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.optimizers import Adam
from keras.models import model_from_json
from matplotlib import pyplot as plt

# Making output consistent
seed = 46

## Construct the argument parse and parse the arguments
"""
	--dataset: The path to the Yelp directory residing on disk
	--word2vec : The path to the word2vec pre-trained model
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, \
	help='path to Yelp dataset')
parser.add_argument("-w", "--word2vec", require=True, \
	help='path to word2vec pre-trained model')
args = parser.parse_args

# Retrieve the path to files
path_data = args.dataset
path_word2vec = args.word2vec


## Load data
data = pd.read_pickle(path_data)

# Removing all ('\n') characters using list comprehensions
data['text'] = [txt.replace('\n', '') for txt in data['text']]

# Taking only text and stars columns
data = data.loc[:, ['text', 'stars']]


## Tokenize sentences 
data["tokens"] = data.apply(lambda row: word_tokenize(row["text"]), axis=1)


## Bag of words counts
def count_vectorize(data):
    count_vectorizer = CountVectorizer()
    
    embedding = count_vectorizer.fit_transform(data)
    
    return embedding, count_vectorizer

X = toy_data["text"].tolist()
y = toy_data["stars"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train_counts, count_vectorizer = count_vectorize(X_train)
X_test_counts = count_vectorizer.transform(X_test)


## Load word2vec model 
word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_word2vec, binary=True)


## Embed the text
all_words = [word for tokens in data["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data["tokens"]]
vocabulary = sorted(list(set(all_words)))

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 35
VOCAB_SIZE = len(vocabulary)
VALIDATION_SPLIT = 0.2

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(data["text"].tolist())
sequences = tokenizer.texts_to_sequences(data["text"].tolist())

word_index = tokenizer.word_index
print("Found %s unique tokens." % (len(word_index)))

cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(data["stars"]))

indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, index in word_index.items():
    embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

print("embedding_weights shape: ", embedding_weights.shape)

## Construct model 
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, trainable=False):
    
    embedding_layer = Embedding(num_words, \
                                embedding_dim, \
                                weights=[embeddings], \
                                input_length=max_sequence_length, \
                                trainable=trainable)
    
    sequence_input = Input(shape=(max_sequence_length, ), dtype="int32", name="Input")
    embedded_sequences = embedding_layer(sequence_input)
    
    convs = []
    filter_sizes = [3, 4, 5]
    
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)
    
    l_merge = Merge(mode='concat', concat_axis=1, name="Merge")(convs)
    
    l_conv1 = Conv1D(filters=128, kernel_size=5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(pool_size=3)(l_conv1)
    
    l_drop  = Dropout(0.5)(l_pool1)
    
    l_flat  = Flatten(name="Flatten")(l_drop)
    
    l_dense = Dense(128, activation='relu', name="Dense")(l_flat)
    

    preds = Dense(6, activation='softmax', name="Output")(l_dense)
    
    model = Model(sequence_input, preds)
    
    return model


X_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
X_val  = cnn_data[-num_validation_samples:]
y_val  = labels[-num_validation_samples:]

model = ConvNet(embedding_weights, \
                MAX_SEQUENCE_LENGTH, \
                len(word_index)+1, \
                EMBEDDING_DIM, \
                trainable=False)


## Training the model
adam = Adam(lr=1e-4)

model.compile(optimizer=adam, \
              loss='categorical_crossentropy', \
              metrics=['acc'])

# Set a small number of epochs to validate the model, then set this to be like 50, 100
num_epochs = 5
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=128)


## Serialize model to JSON
model_json = model.to_json()
with open("model_convnet.json", "w") as json_file:
    json_file.write(model_json)
    
## Serialize weights to HDF5

model.save_weights("model_convnet.h5")
print("Saved model to disk")


## Plotting the training and validation loss 
history_dict = history.history
train_loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, num_epochs + 1)

plt.figure(num=1, figsize=(16, 12))

plt.plot(epochs, train_loss_values, 'r', label="Training loss")
plt.plot(epochs, val_loss_values, 'b', label="Validation loss")
plt.title("Training and validation loss", fontsize=12)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

## Plotting the training and validation accuracy 
history_dict = history.history
train_acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

num_epochs = 5
epochs = range(1, num_epochs + 1)

plt.figure(num=2, figsize=(16, 12))

plt.plot(epochs, train_acc_values, 'r', label="Training accuracy")
plt.plot(epochs, val_acc_values, 'b', label="Validation accuracy")
plt.title("Training and validation accuracy", fontsize=12)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

## Load model 
#
# # Load json and create model
# with open("model_convnet.json", "r") as json_file:
# 	loaded_model_json = json_file.read()
# loaded_model = model_from_json(loaded_model_json)
# # Load weights into new model
# loaded_model.load_weights("model_convnet.h5")
# print("Loaded model from disk")
# 
# 
# ## Evaluate loaded model on test data
# loaded_model.compile(optimizer=adam, \
#               		loss='categorical_crossentropy', \
#               		metrics=['acc'])
# scores = loaded_model.evaluate(X_test, y_test, verbose=1) # Verbose=1 for progress bar
# 
# print("%s on test data: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))


## Evaluate model directly without loading json and hdf5 files
scores = model.evaluate(X_test, y_test)
print("%s on test data: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))