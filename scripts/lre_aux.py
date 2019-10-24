import json
import cv2
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Embedding, Concatenate, Conv1D, MaxPooling1D, Multiply, Dense, Add, Input, Reshape, LSTM, Lambda, Permute
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle
import tensorflow as tf


def get_captions(list_captions_tokens, list_captions_synsets):
    MAX_SEQ_LEN = 1000
    
    with open('tutorial/datasamples/tokenizer_tokens.pickle', 'rb') as handle:
        tokenizer_tokens = pickle.load(handle)

    with open('tutorial/datasamples/tokenizer_synsets.pickle', 'rb') as handle:
        tokenizer_synsets = pickle.load(handle)

    sequences = tokenizer_tokens.texts_to_sequences(list_captions_tokens)
    captions_tokens = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    sequences = tokenizer_synsets.texts_to_sequences(list_captions_synsets)
    captions_synsets = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    
    return captions_tokens, captions_synsets


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, 0, cls]
    conv_output = input_model.layers[layer_name].output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    

    cam = cv2.resize(cam, (300, 1000))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    
    
    
    return cam

def get_model():
    with open('tutorial/datasamples/tokenizer_tokens.pickle', 'rb') as handle:
        tokenizer_tokens = pickle.load(handle)

    with open('tutorial/datasamples/tokenizer_synsets.pickle', 'rb') as handle:
        tokenizer_synsets = pickle.load(handle)

    tokenizers = [tokenizer_tokens, tokenizer_synsets]
    
    EMB_FILE = "./tutorial/datasamples/scigraph_wordnet.tsv"
    DIM = 100
    MAX_SEQ_LEN = 1000

    file = open(EMB_FILE, "r", encoding="utf-8", errors="surrogatepass")
    embeddings_index_tokens = {}
    embeddings_index_synsets = {}

    for line in file:
      values = line.split()
      comp_len = len(values)-DIM
      word = "+".join(values[0:comp_len])
      if (line.startswith("wn31")):
        vector = np.asarray(values[comp_len:], dtype='float32')
        embeddings_index_synsets[word] = vector
      else:
        if (line.startswith("grammar#")):
          continue
        else:
          vector = np.asarray(values[comp_len:], dtype='float32')
          embeddings_index_tokens[word] = vector
    file.close()

    embedding_indexes = [embeddings_index_tokens, embeddings_index_synsets]

    embedding_matrices = []
    for tok_i in range(len(tokenizers)):
      embedding_matrix = np.zeros((len(tokenizers[tok_i].word_index) + 1, DIM))
      for word, i in tokenizers[tok_i].word_index.items():
        embedding_vector = embedding_indexes[tok_i].get(word)
        if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
      embedding_matrices.append(embedding_matrix)
    embedding_matrix_tokens, embedding_matrix_synsets = embedding_matrices

    modelFigures = Sequential()
    modelFigures.add(InputLayer(input_shape=(224,224,3)))
    modelFigures.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(MaxPooling2D(2))
    modelFigures.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(MaxPooling2D(2))
    modelFigures.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(MaxPooling2D(2))
    modelFigures.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
    modelFigures.add(BatchNormalization())
    modelFigures.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
    modelFigures.add(BatchNormalization())
    modelFigures.add(MaxPooling2D((28,28),2))
    modelFigures.add(Flatten())

    modelCaptionsScratch = Sequential()
    modelCaptionsScratch.add(Embedding(len(tokenizer_tokens.word_index)+1, DIM, embeddings_initializer="uniform", input_length=MAX_SEQ_LEN, trainable=True))
    modelCaptionsVecsiTokens = Sequential()
    modelCaptionsVecsiTokens.add(Embedding(len(tokenizer_tokens.word_index) + 1, DIM, weights = [embedding_matrix_tokens], input_length = MAX_SEQ_LEN, trainable = False))
    modelCaptionsVecsiSynsets = Sequential()
    modelCaptionsVecsiSynsets.add(Embedding(len(tokenizer_synsets.word_index) + 1, DIM, weights = [embedding_matrix_synsets], input_length = MAX_SEQ_LEN, trainable = False))
    modelMergeEmbeddings = Concatenate()([modelCaptionsScratch.output,modelCaptionsVecsiTokens.output,modelCaptionsVecsiSynsets.output])
    modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
    modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
    modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
    modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
    modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
    modelMergeEmbeddings = MaxPooling1D(35)(modelMergeEmbeddings)
    modelMergeEmbeddings = Flatten()(modelMergeEmbeddings)
    modelCaptions = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelCaptionsVecsiSynsets.input], modelMergeEmbeddings)

    mergedOut = Multiply()([modelCaptions.output,modelFigures.output])  
    mergedOut = Dense(128, activation='relu')(mergedOut)
    mergedOut = Dense(2, activation='softmax')(mergedOut)
    model = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelCaptionsVecsiSynsets.input, modelFigures.input], mergedOut)

    model.load_weights("model_weights_BIG.h5")

    return modelFigures, modelCaptions, model

def get_figure_vis_model():
    modelF = Sequential()
    modelF.add(InputLayer(input_shape=(224,224,3)))
    modelF.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(MaxPooling2D(2))
    modelF.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(MaxPooling2D(2))
    modelF.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(MaxPooling2D(2))
    modelF.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
    modelF.add(BatchNormalization())
    modelF.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
    modelF.add(BatchNormalization())
    modelF.add(MaxPooling2D((28,28),2))
    modelF.add(Flatten())
    modelF.load_weights("./tutorial/datasamples/modelFigures-qualitative.h5")

    return modelF

def get_caption_vis_model():
    with open('tutorial/datasamples/tokenizer_tokens.pickle', 'rb') as handle:
        tokenizer_tokens = pickle.load(handle)

    with open('tutorial/datasamples/tokenizer_synsets.pickle', 'rb') as handle:
        tokenizer_synsets = pickle.load(handle)

    tokenizers = [tokenizer_tokens, tokenizer_synsets]
    
    EMB_FILE = "./tutorial/datasamples/scigraph_wordnet.tsv"
    DIM = 100
    MAX_SEQ_LEN = 1000

    file = open(EMB_FILE, "r", encoding="utf-8", errors="surrogatepass")
    embeddings_index_tokens = {}
    embeddings_index_synsets = {}

    for line in file:
      values = line.split()
      comp_len = len(values)-DIM
      word = "+".join(values[0:comp_len])
      if (line.startswith("wn31")):
        vector = np.asarray(values[comp_len:], dtype='float32')
        embeddings_index_synsets[word] = vector
      else:
        if (line.startswith("grammar#")):
          continue
        else:
          vector = np.asarray(values[comp_len:], dtype='float32')
          embeddings_index_tokens[word] = vector
    file.close()

    embedding_indexes = [embeddings_index_tokens, embeddings_index_synsets]

    embedding_matrices = []
    for tok_i in range(len(tokenizers)):
      embedding_matrix = np.zeros((len(tokenizers[tok_i].word_index) + 1, DIM))
      for word, i in tokenizers[tok_i].word_index.items():
        embedding_vector = embedding_indexes[tok_i].get(word)
        if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
      embedding_matrices.append(embedding_matrix)
    embedding_matrix_tokens, embedding_matrix_synsets = embedding_matrices
    
    modelC = get_model()[1]
    
    modelEmbScratch = Sequential()
    modelEmbScratch.add(Embedding(len(tokenizer_tokens.word_index)+1, DIM, embeddings_initializer="uniform", input_length=MAX_SEQ_LEN, trainable=True))
    modelEmbVecsiTokens = Sequential()
    modelEmbVecsiTokens.add(Embedding(len(tokenizer_tokens.word_index) + 1, DIM, weights = [embedding_matrix_tokens], input_length = MAX_SEQ_LEN, trainable = False))
    modelEmbVecsiSynsets = Sequential()
    modelEmbVecsiSynsets.add(Embedding(len(tokenizer_synsets.word_index) + 1, DIM, weights = [embedding_matrix_synsets], input_length = MAX_SEQ_LEN, trainable = False))
    modelEmbMerge = Concatenate()([modelEmbScratch.output,modelEmbVecsiTokens.output,modelEmbVecsiSynsets.output])
    modelEmbeddings = Model([modelEmbScratch.input,modelEmbVecsiTokens.input,modelEmbVecsiSynsets.input], modelEmbMerge)

    modelVisualize = Sequential()
    modelVisualize.add(InputLayer((MAX_SEQ_LEN,DIM*3,)))
    modelVisualize.add(Conv1D(512, 5, activation="relu"))
    modelVisualize.add(MaxPooling1D(5))
    modelVisualize.add(Conv1D(512, 5, activation="relu"))
    modelVisualize.add(MaxPooling1D(5))
    modelVisualize.add(Conv1D(512, 5, activation="relu"))
    modelVisualize.add(MaxPooling1D(35))

    for i in range(len(modelEmbeddings.layers)):
        modelEmbeddings.layers[i].set_weights(modelC.layers[i].get_weights())
    for j in range(len(modelVisualize.layers)):
        modelVisualize.layers[j].set_weights(modelC.layers[j+len(modelEmbeddings.layers)].get_weights())

    return modelEmbeddings, modelVisualize, modelC     

def similarity(x):
    return tf.matmul(x[0],x[1], transpose_a=True)

def output_similarity(input_shape):
    return (input_shape[0][0], input_shape[0][2], input_shape[1][2])

def reduce_max_layer(x):
    return tf.reduce_max(x, axis=2, keepdims=True)

def output_reduce_max_layer(input_shape):
    return (input_shape[0], input_shape[1], 1)

def answerer(x):
    return tf.multiply(x[1], x[0])

def output_answerer(input_shape):
    return (input_shape[1][0], input_shape[1][1], input_shape[1][2])

def reduce_sum_layer(x):
    return tf.reduce_sum(x, axis=2, keepdims=True)

def output_reduce_sum_layer(input_shape):
    return (input_shape[0], input_shape[1], 1)

def get_TQAmodel(dim,emb_tok,emb_syn, dout,rdout, tokenizers, max_lens):
    vocab_sizes = [len(x.word_index)+1 for x in tokenizers]

    M_scratch_input = Input(shape = (max_lens[0],), name="M_scratch_input")
    M_tokens_input = Input(shape = (max_lens[0],), name="M_tokens_input")
    M_synsets_input = Input(shape = (max_lens[0],), name="M_synsets_input")
    U_scratch_input = Input(shape = (max_lens[1],), name="U_scratch_input")
    U_tokens_input = Input(shape = (max_lens[1],), name="U_tokens_input")
    U_synsets_input = Input(shape = (max_lens[1],), name="U_synsets_input")
    C1_scratch_input = Input(shape = (max_lens[2],), name="C1_scratch_input")
    C1_tokens_input = Input(shape = (max_lens[2],), name="C1_tokens_input")
    C1_synsets_input = Input(shape = (max_lens[2],), name="C1_synsets_input")
    C2_scratch_input = Input(shape = (max_lens[3],), name="C2_scratch_input")
    C2_tokens_input = Input(shape = (max_lens[3],), name="C2_tokens_input")
    C2_synsets_input = Input(shape = (max_lens[3],), name="C2_synsets_input")
    C3_scratch_input = Input(shape = (max_lens[4],), name="C3_scratch_input")
    C3_tokens_input = Input(shape = (max_lens[4],), name="C3_tokens_input")
    C3_synsets_input = Input(shape = (max_lens[4],), name="C3_synsets_input")
    C4_scratch_input = Input(shape = (max_lens[5],), name="C4_scratch_input")
    C4_tokens_input = Input(shape = (max_lens[5],), name="C4_tokens_input")
    C4_synsets_input = Input(shape = (max_lens[5],), name="C4_synsets_input")

    modelMF = Sequential()
    modelMF.add(InputLayer(input_shape=(512,), name="input_MF"))
    modelMF.add(Dense(256, activation="tanh", name="perceptron_MF_1"))
    modelMF.add(Dense(dim, activation="tanh", name="perceptron_MF_2"))
    modelMF.add(Reshape((dim, 1,), name="reshape_MF"))
    modelUF = Sequential()
    modelUF.add(InputLayer(input_shape=(512,), name="input_UF"))
    modelUF.add(Dense(256, activation="tanh", name="perceptron_UF_1"))
    modelUF.add(Dense(dim, activation="tanh", name="perceptron_UF_2"))
    modelUF.add(Reshape((dim, 1,), name="reshape_UF"))

    embedding_scratch_layer = Embedding(vocab_sizes[0], dim, embeddings_initializer="uniform", trainable=True)
    embedding_tokens_layer = Embedding(vocab_sizes[0], dim, weights = [emb_tok], trainable = False)
    embedding_synsets_layer = Embedding(vocab_sizes[1], dim, weights = [emb_syn], trainable = False)

    M_scratch_input_embedded = embedding_scratch_layer(M_scratch_input)
    M_tokens_input_embedded = embedding_tokens_layer(M_tokens_input)
    M_synsets_input_embedded = embedding_synsets_layer(M_synsets_input)
    U_scratch_input_embedded = embedding_scratch_layer(U_scratch_input)
    U_tokens_input_embedded = embedding_tokens_layer(U_tokens_input)
    U_synsets_input_embedded = embedding_synsets_layer(U_synsets_input)
    C1_scratch_input_embedded = embedding_scratch_layer(C1_scratch_input)
    C1_tokens_input_embedded = embedding_tokens_layer(C1_tokens_input)
    C1_synsets_input_embedded = embedding_synsets_layer(C1_synsets_input)
    C2_scratch_input_embedded = embedding_scratch_layer(C2_scratch_input)
    C2_tokens_input_embedded = embedding_tokens_layer(C2_tokens_input)
    C2_synsets_input_embedded = embedding_synsets_layer(C2_synsets_input)
    C3_scratch_input_embedded = embedding_scratch_layer(C3_scratch_input)
    C3_tokens_input_embedded = embedding_tokens_layer(C3_tokens_input)
    C3_synsets_input_embedded = embedding_synsets_layer(C3_synsets_input)
    C4_scratch_input_embedded = embedding_scratch_layer(C4_scratch_input)
    C4_tokens_input_embedded = embedding_tokens_layer(C4_tokens_input)
    C4_synsets_input_embedded = embedding_synsets_layer(C4_synsets_input)
    M_input_embedded = Add()([M_scratch_input_embedded, M_tokens_input_embedded, M_synsets_input_embedded])
    U_input_embedded = Add()([U_scratch_input_embedded, U_tokens_input_embedded, U_synsets_input_embedded])
    C1_input_embedded = Add()([C1_scratch_input_embedded, C1_tokens_input_embedded, C1_synsets_input_embedded])
    C2_input_embedded = Add()([C2_scratch_input_embedded, C2_tokens_input_embedded, C2_synsets_input_embedded])
    C3_input_embedded = Add()([C3_scratch_input_embedded, C3_tokens_input_embedded, C3_synsets_input_embedded])
    C4_input_embedded = Add()([C4_scratch_input_embedded, C4_tokens_input_embedded, C4_synsets_input_embedded])

    LSTM_layer = LSTM(units=dim, return_sequences=True, name="lstm_layer", dropout=dout, recurrent_dropout=rdout)

    M_input_encoded = LSTM_layer(M_input_embedded)
    M_input_encoded = Permute((2, 1), input_shape=(max_lens[0], dim,), name="M_permute")(M_input_encoded)
    modelM = Model([M_scratch_input,M_tokens_input,M_synsets_input], M_input_encoded)
    modelInMMF = Concatenate(name="concatenateMMF")([modelM.output, modelMF.output])
    modelMMF = Model([M_scratch_input, M_tokens_input, M_synsets_input, modelMF.input], modelInMMF)

    U_input_encoded = LSTM_layer(U_input_embedded)
    U_input_encoded = Permute((2, 1), input_shape=(max_lens[1], dim,), name="U_permute")(U_input_encoded)
    modelU = Model([U_scratch_input,U_tokens_input,U_synsets_input], U_input_encoded)
    modelInUUF = Concatenate(name="concatenateUUF")([modelU.output, modelUF.output])
    modelUUF = Model([U_scratch_input, U_tokens_input, U_synsets_input, modelUF.input], modelInUUF)

    C1_input_encoded = LSTM_layer(C1_input_embedded)
    C1_input_encoded = Permute((2, 1), input_shape=(max_lens[2], dim,), name="C1_permute")(C1_input_encoded)
    C1_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C1_reduce_sum")(C1_input_encoded)
    modelC1 = Model([C1_scratch_input,C1_tokens_input,C1_synsets_input], C1_input_encoded)

    C2_input_encoded = LSTM_layer(C2_input_embedded)
    C2_input_encoded = Permute((2, 1), input_shape=(max_lens[3], dim,), name="C2_permute")(C2_input_encoded)
    C2_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C2_reduce_sum")(C2_input_encoded)
    modelC2 = Model([C2_scratch_input,C2_tokens_input,C2_synsets_input], C2_input_encoded)

    C3_input_encoded = LSTM_layer(C3_input_embedded)
    C3_input_encoded = Permute((2, 1), input_shape=(max_lens[4], dim,), name="C3_permute")(C3_input_encoded)
    C3_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C3_reduce_sum")(C3_input_encoded)
    modelC3 = Model([C3_scratch_input,C3_tokens_input,C3_synsets_input], C3_input_encoded)

    C4_input_encoded = LSTM_layer(C4_input_embedded)
    C4_input_encoded = Permute((2, 1), input_shape=(max_lens[5], dim,), name="C4_permute")(C4_input_encoded)
    C4_input_encoded = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="C4_reduce_sum")(C4_input_encoded)
    modelC4 = Model([C4_scratch_input,C4_tokens_input,C4_synsets_input], C4_input_encoded)

    modelIna = Lambda(similarity, output_shape=output_similarity,name="similarityMU")([modelMMF.output, modelUUF.output])
    modelIna = Lambda(reduce_max_layer, output_shape=output_reduce_max_layer, name="a_reduce_max")(modelIna)
    modelIna = Permute((2, 1), input_shape=(max_lens[0]+1, 1,), name="a_permute")(modelIna)
    modelIna = Dense(max_lens[0]+1,activation="softmax",name="softmax_a")(modelIna)
    modela = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input], modelIna)

    modelInm = Lambda(answerer, output_shape=output_answerer,name="answerer") ([modela.output, modelMMF.output])
    modelInm = Lambda(reduce_sum_layer, output_shape=output_reduce_sum_layer, name="m_reduce_sum") (modelInm)
    modelm = Model([M_scratch_input,M_tokens_input,M_synsets_input, modelMF.input, U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input], modelInm)

    modelInC1 = Lambda(similarity, output_shape=output_similarity,name="similaritymC1")([modelm.output,modelC1.output])
    modelIn1 = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,
                                      U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input,
                                      C1_scratch_input,C1_tokens_input,C1_synsets_input], modelInC1)
    modelInC2 = Lambda(similarity, output_shape=output_similarity,name="similaritymC2")([modelm.output,modelC2.output])
    modelIn2 = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,
                                      U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input,
                                      C2_scratch_input,C2_tokens_input,C2_synsets_input], modelInC2)
    modelInC3 = Lambda(similarity, output_shape=output_similarity,name="similaritymC3")([modelm.output,modelC3.output])
    modelIn3 = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,
                                      U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input,
                                      C3_scratch_input,C3_tokens_input,C3_synsets_input], modelInC3)
    modelInC4 = Lambda(similarity, output_shape=output_similarity,name="similaritymC4")([modelm.output,modelC4.output])
    modelIn4 = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,
                                      U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input,
                                      C4_scratch_input,C4_tokens_input,C4_synsets_input], modelInC4)

    modelIn = Concatenate(name = "concatenate")([modelIn1.output,modelIn2.output,modelIn3.output,modelIn4.output])
    modelIn = Flatten()(modelIn)
    modelIn = Dense(4, activation="softmax",name="softmax_y") (modelIn)
    model = Model([M_scratch_input,M_tokens_input,M_synsets_input,modelMF.input,
                                   U_scratch_input,U_tokens_input,U_synsets_input,modelUF.input,
                                   C1_scratch_input,C1_tokens_input,C1_synsets_input,
                                   C2_scratch_input,C2_tokens_input,C2_synsets_input,
                                   C3_scratch_input,C3_tokens_input,C3_synsets_input,
                                   C4_scratch_input,C4_tokens_input,C4_synsets_input], modelIn)
    return model

