def get_model():
    EMB_FILE = "./tutorial/datasamples/scigraph_wordnet.tsv"
    DIM = 100

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

