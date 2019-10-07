# classification.py
# defines methods for performing text classification using neural nets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import array
import io
import os
from tqdm import tqdm
import six
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, LSTM, Concatenate, Lambda
from keras.callbacks import EarlyStopping
import keras
import gc
import nltk
import tensorflow as tf
import urllib

# make sure we have the stopwords data in this environment
nltk.download('stopwords')


def read_classification_corpus(df, text_fields=['text'], tag_field='tag',
                               tag2id=None):
    '''Reads a corpus from a dataFrame into a python dictionary
    The resulting dictionay will have fields `texts`, `categories`,
    `tags` and `id2tag`.
    '''
    tagset = df[tag_field].unique()
    if tag2id is None:
        tag2id = {tag_name: tag_id for tag_id, tag_name in enumerate(tagset)}
    else:
        msg = 'Provided tag2id does not cover all tags in the dataset'
        assert set(tag2id.keys()) == set(tagset), msg
    id2tag = {tag_id: tag_name for tag_name, tag_id in tag2id.items()}
    categories = df[tag_field].values
    tags = [tag2id[tag_name] for tag_name in categories]
    text_dict = {}
    for tf in text_fields:
        text_dict[tf] = df[tf].values  # skip header
        msg = "number of texts {} in field {} must match number of tags {}"
        assert len(text_dict[tf]) == len(tags), msg.format(
            len(text_dict[tf]), tf, len(tags))
    return {
        'texts': text_dict,
        'categories': categories,
        'tags': tags,
        'id2tag': id2tag}


def sanity_check(raw_dataset, idx=0, max_chars=100):
    '''Returns an easy to inspect example from a raw dataset.
    The `raw_dataset` should be a python dict as output by method
    `read_classification_corpus`.
    '''
    _texts = {field: val[idx] for field, val in raw_dataset['texts'].items()}
    _tag = raw_dataset['tags'][idx]
    for field, text in _texts.items():
        assert type(text) == str
    assert type(raw_dataset['tags'][idx]) == int
    _capped_texts = {}
    for field, text in _texts.items():
        max_chars = min(max_chars, len(text))
        _capped_texts[field] = _texts[field][:max_chars]
    return _capped_texts, _tag, raw_dataset['id2tag'][_tag]


def determine_max_seq_len(texts, plot=True):
    '''Returns a suitable value for max_seq_len based on the texts to categorise
    It returns the mean + std of the lengths, i.e. a value that should cover
    most of the documents, especially, since texts tend to have a powerlog
    distribution.
    '''
    doc_len = pd.Series([len(doc.split(" ")) for doc in texts])
    max_seq_len = np.round(doc_len.mean() + doc_len.std()).astype(int)
    sns.distplot(doc_len, hist=True, kde=True, color='b', label='doc len')
    plt.axvline(x=max_seq_len, color='k', linestyle='--',
                label='max len=%d' % max_seq_len)
    plt.title('text length in words')
    plt.legend()
    plt.show()
    return max_seq_len


def load_tsv_embeddings(name, sep='\t', max_words=None,
                        word_filter_fn=lambda x: True,
                        word_map_fn=lambda x: x):
    '''Reads embeddings from tsv file `name`
    args
    ----
    name path to a tsv file, this should be a text file with an
         optional header line and subsequent lines
         have format <word>(<sep><float>)+
    sep separator string between elements in each line
    max_words the maximum number of words to read from the tsv file
    word_filter function that filters the words as read from the
         tsv file, only words that pass this filter
         are included in the returned embeddings
    word_map_fn function that maps the words as read from the tsv
         file to a different representation. E.g.
         this can be used to map 'lem_word' to 'word'
    '''
    path = os.path.join(name)

    if not os.path.isfile(path):
        raise RuntimeError('no vectors found at {}'.format(path))

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    itos, vectors, dim = [], array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(path, encoding="utf-8") as f:
            lines = [line for line in f]

        # If there are malformed lines, read in binary mode
        # and manually decode each word from utf-8
    except Exception as e:
        print("Could not read {} as UTF8 file {}, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(path, e))
        with open(path, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    print("Loading vectors from {} lines in {}".format(len(lines), path))
    for line in tqdm(lines):
        # Explicitly splitting on "\t" is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(sep)
        word, entries = entries[0], entries[1:]
        if not word_filter_fn(word):
            continue
        if dim is None and len(entries) > 1:
            dim = len(entries)
        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(
                        word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except Exception as e:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue
        vectors.extend(float(x) for x in entries)
        itos.append(word_map_fn(word))
        if max_words is not None and len(itos) > max_words:
            break

    return {'itos': itos,
            'stoi': {word: i for i, word in enumerate(itos)},
            'vecs': vectors,
            'source': name,
            'dim': dim}


def concat_embs(embA, embB):
    '''Produces a new emb dict by concatenating two inputs.
    The result concatenates the vectors in two input emb dicts and
    merges their vocabs.'''
    itos = []
    for w in embA['itos']:
        itos.append(w)
    A_ws = set(itos)
    for w in embB['itos']:
        if w in A_ws:
            continue
        else:
            itos.append(w)
    vectors = array.array(str('d'))
    for w in itos:
        i_A, dim_A = embA['stoi'].get(w), embA['dim']
        vecA = [0.0 for j in range(dim_A)] if i_A is None else embA['vecs'][i_A*dim_A:(i_A+1)*dim_A]
        msg = 'Expected dim {}, but was {} for {} at index {}'
        assert len(vecA) == dim_A, msg.format(dim_A, len(vecA), w, i_A)
        i_B, dim_B = embB['stoi'].get(w), embB['dim']
        vecB = [0.0 for j in range(dim_B)] if i_B is None else embB['vecs'][i_B*dim_B:(i_B+1)*dim_B]
        assert len(vecB) == dim_B, msg.format(dim_B, len(vecB), w, i_B)
        vectors.extend(vecA)
        vectors.extend(vecB)
    dim = embA['dim'] + embB['dim']
    msg = 'Size of Vectors {} does not match dim {} * number of words {}'
    assert len(vectors) == len(itos) * dim, msg.format(
        len(vectors), dim, len(itos))
    return {'itos': itos,
            'stoi': {word: i for i, word in enumerate(itos)},
            'vecs': vectors,
            'source': 'concat_{}_{}'.format(embA['source'], embB['source']),
            'dim': dim}


def as_keras_emb_weights(emb):
    '''Converts an embedding as returned by `load_tsv_embeddings`
    into a np array of arrays as required by the Keras Embedding
    `weight` parameter.
    '''
    if emb['vecs'] is None:
        return None
    voc_size = len(emb['itos']) + 2  # the words + the pad 'word'
    emb_weights = np.zeros((voc_size, emb['dim']))
    print('Creating emb weights matrix of shape', emb_weights.shape)
    for word, i in emb['stoi'].items():
        start_i = i * emb['dim']
        vec = emb['vecs'][start_i:start_i+emb['dim']] if len(emb['vecs']) > start_i else None
        if vec is not None:
            emb_weights[i] = np.asarray(vec)
    return emb_weights


def to_categorical_tags(raw_dataset):
    tags = raw_dataset['tags']
    num_tags = len(set(tags))
    return np.array([to_categorical(i, num_classes=num_tags) for i in tags])


def default_stop_words():
    from nltk.corpus import stopwords
    result = set(stopwords.words('english'))
    result.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    return result


def default_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    return RegexpTokenizer(r'\w+')


def clean_texts(texts, tokenizer=default_tokenizer(),
                stop_words=default_stop_words()):
    '''Pre-processes the input texts by filtering stopwords
    '''
    result = []
    for doc in tqdm(texts):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        result.append(" ".join(filtered))
    return result


def all_texts(raw_dataset):
    result = []
    for field, texts in raw_dataset['texts'].items():
        result.extend(texts)
    return result


def clean_ds_texts(raw_dataset, tokenizer=default_tokenizer(),
                   stop_words=default_stop_words()):
    return {
        'texts': {field: clean_texts(
            text_vals, tokenizer=tokenizer,
            stop_words=stop_words) for
                  field, text_vals in raw_dataset['texts'].items()},
        'categories': raw_dataset['categories'],
        'tags': raw_dataset['tags'],
        'id2tag': raw_dataset['id2tag']
    }


def extract_vocab_embedding(raw_dataset, max_words=None, emb_dim=150):
    texts = all_texts(raw_dataset)
    flatten_docs = [word for doc in texts for word in doc.split()]
    words = list(set(flatten_docs))  # deduplicate and define an order
    word2idx = {w: i for i, w in enumerate(words)}
    return {
          'stoi': word2idx,
          'vecs': None,
          'weights': None,
          'itos': words,
          'dim': emb_dim
        }


def as_stoi(word_index):
    max_i = max(word_index.values())
    result = ['<invalid>' for i in range(max_i + 1)]
    for w, i in word_index.items():
        result[i] = w
    return result


def index_texts_using_tokenizer(texts, max_words=None, lower=True):
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=False)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return {'w2i': word_index,
            'vecs': None,
            'weights': None,
            'i2w': as_stoi(word_index),
            'dim': 150}, tokenizer.texts_to_sequences(texts)


def index_texts_using_embeddings(texts, emb):
    w2i = emb['stoi']
    pad_word_id = len(emb['itos']) + 1
    X = [[w2i.get(word, pad_word_id) for word in doc.split()] for doc in texts]
    oov_cnt, iv_cnt = 0, 0
    for seq in X:
        for wid in seq:
            if wid == pad_word_id:
                oov_cnt = oov_cnt + 1
            else:
                iv_cnt = iv_cnt + 1
    msg = 'field has %d (%.3f) tokens in and %d (%.3f) tokens out of the embvocab'
    print(msg % (iv_cnt, iv_cnt / (iv_cnt + oov_cnt), oov_cnt,
                 oov_cnt / (iv_cnt + oov_cnt)))
    emb['w2i'] = emb['stoi']
    emb['i2w'] = emb['itos']
    return emb, X


def index_ds_using_vocab(raw_dataset, vocab_emb):
    inputs = {}
    for field, texts in raw_dataset['texts'].items():
        _, inputs[field] = index_texts_using_embeddings(texts, vocab_emb)
    return {'vocab_embedding': vocab_emb,
            'inputs': inputs,
            'outputs': to_categorical_tags(raw_dataset)}


def simple_index_ds(raw_dataset, max_words=None):
    '''Returns an indexed version of the texts.
    Returns a tuple, the first element is an `emb` dict,
    the second is an array of int arrays, where each value is
    a word index.
    '''
    voc_emb = extract_vocab_embedding(raw_dataset, max_words=max_words)
    return index_ds_using_vocab(raw_dataset, voc_emb)


def wnet_val_if_gt(val):
    '''Returns the value, but only if it's a grammar subtoken'''
    return val if wnet_is_grammar(val) else None


def wnet_val_if_concept(val):
    '''Returns the value, but only if it's a concept subtoken'''
    return val if wnet_is_concept(val) else None


def wnet_is_grammar(val):
    '''Returns true if the string value is a grammar subtoken'''
    return val.startswith('GT_')


def wnet_is_lemma(val):
    '''Returns true if the string value is a lemma subtoken'''
    return val.startswith('lem_')


def wnet_is_concept(val):
    '''Returns true if the string value is a synset subtoken'''
    return val.startswith('wn31_')


def wnet_tlgs_subtok(sub_tokens, subtok_type):
    '''Returns the subtoken of the requested subtok_type or None
    Processes the list of sub_tokens and returns the subtoken of
    the requested subtok_type, if available. Otherwise, returns None'''
    if sub_tokens is None:
        raise ValueError("No sub_tokens passed", sub_tokens)
    if len(sub_tokens) < 2:
        raise ValueError("Expecting at least 2 sub token values, but found",
                         sub_tokens)
    if subtok_type == 't':
        return sub_tokens[0].replace('+', ' ')
    elif subtok_type == 'l':
        lem = sub_tokens[1] if wnet_is_lemma(sub_tokens[1]) else None
        return None if lem is None else lem.replace('lem_', '').replace('+', ' ')
    elif subtok_type == 'g':
        end = min(3, len(sub_tokens))
        vals = [wnet_val_if_gt(val) for val in sub_tokens[1:end]]
        gt_vals = [v.replace('GT_', '') for v in vals if v is not None]
        return gt_vals[0] if len(gt_vals) > 0 else None
    elif subtok_type == 's':
        end = min(4, len(sub_tokens))
        vals = [wnet_val_if_concept(val) for val in sub_tokens[1:end]]
        s_vals = [v for v in vals if v is not None]
        return s_vals[0] if len(s_vals) > 0 else None
    else:
        raise ValueError('Subtok type %s not supported for format tlgs' %
                         subtok_type)


def wnet_sub_tok(sub_tokens, subtok_type, expected_format='tlgs'):
    '''Returns the subtoken of the requested type, for the expected format'''
    if expected_format == 'tlgs':
        return wnet_tlgs_subtok(sub_tokens, subtok_type)
    else:
        raise ValueError('unsupported expected_format %s' % expected_format)


def wnet_ls(tok, emb):
    dec_tok = urllib.parse.unquote(tok)
    subdec_toks = dec_tok.split(sep='|')
    lem = wnet_sub_tok(subdec_toks, 'l')
    syn = wnet_sub_tok(subdec_toks, 's')
    return [emb['stoi'].get(lem, 0), emb['stoi'].get(syn, 0)]


def add_w2i_i2w(emb):
    emb['w2i'] = emb['stoi']
    emb['i2w'] = emb['itos']
    return emb


def index_texts_wnet_ls(texts, emb):
    result = [[wnet_ls(tok, emb) for tok in doc.split()] for
              doc in texts]
    lems_oov, syns_oov = 0, 0
    lems_inv, syns_inv = 0, 0
    for seq in result:
        for tok in seq:
            assert len(tok) == 2
            if tok[0] == 0:
                lems_oov = lems_oov + 1
            else:
                lems_inv = lems_inv + 1
            if tok[1] == 0:
                syns_oov = syns_oov + 1
            else:
                syns_inv = syns_inv + 1
    tot_toks = lems_oov + lems_inv
    msg = 'Found %d (%.3f) %s in and %d (%.3f) out of vocab'
    print(msg % (lems_inv, lems_inv/tot_toks, 'lems',
                 lems_oov, lems_oov/tot_toks))
    print(msg % (syns_inv, syns_inv/tot_toks, 'syns',
                 syns_oov, syns_oov/tot_toks))
    return add_w2i_i2w(emb), result


def index_ds_wnet(raw_dataset, vocab_emb,
                  texts_indexing_fn=index_texts_wnet_ls):
    msg = 'You must pass a valid texts indexing function'
    assert texts_indexing_fn is not None, msg
    inputs = {}
    for field, texts in raw_dataset['texts'].items():
        result_vocemb, inputs[field] = texts_indexing_fn(texts, vocab_emb)
    return {'vocab_embedding': result_vocemb,
            'inputs': inputs,
            'outputs': to_categorical_tags(raw_dataset)}


def plot_max_seq_len(seq_len, max_seq_len):
    sns.distplot(seq_len, hist=True, kde=True, color='b', label='seq len')
    plt.axvline(x=max_seq_len, color='k', linestyle='--',
                label='max len=%d' % max_seq_len)
    plt.title('seq length in indexed words')
    plt.legend()
    plt.show()


def decide_max_seq_len(indexed_seqs, plot=False):
    '''Returns a suitable value for max_seq_len based on the input to categorise
    It returns the mean + std of the lengths, i.e. a value that
    should cover most of the documents, especially, since texts tend
    to have a powerlog distribution.
    '''
    seq_len = pd.Series([len(seq) for seq in indexed_seqs])
    max_seq_len = np.round(seq_len.mean() + seq_len.std()).astype(int)
    if plot:  # visualize word distribution
        plot_max_seq_len(seq_len, max_seq_len)
    return max_seq_len


def decide_and_plot_max_seq_len(indexed_seqs):
    return decide_max_seq_len(indexed_seqs, plot=True)


def pad_wordid_inputs(indexed_ds, max_seq_len_decider):
    pad_id = len(indexed_ds['vocab_embedding']['itos']) + 1
    padded_input = {}
    for field, seq in indexed_ds['inputs'].items():
        print('deciding max seq len for input field ', field,
              'seq is of type', type(seq))
        max_seq_len = max_seq_len_decider(seq)
        print('max_seq_len', max_seq_len)
        sample_val = seq[0][0]
        if type(sample_val) == int:
            pad_val = pad_id
        elif type(sample_val) == list:
            pad_val = [pad_id for x in range(len(sample_val))]
        else:
            msg = 'X should have either int or list values, but found %s %s'
            raise RuntimeError(msg % (type(sample_val), sample_val))
        padded_input[field] = pad_sequences(
            sequences=seq, maxlen=max_seq_len, padding='post', value=pad_val)
        print('field', field, 'padded input shape', padded_input[field].shape)
    return padded_input


def pad_inputs(indexed_ds, max_seq_len_decider=decide_max_seq_len):
    vocemb_type = indexed_ds['vocab_embedding'].get('type', None)
    if vocemb_type is None or vocemb_type == 'word_embedding':
        return pad_wordid_inputs(indexed_ds, max_seq_len_decider)
    else:
        raise RuntimeError('Unknown vocemb_type ' + vocemb_type)


def pad_and_split_experiment(experiment,
                             max_seq_len_decider=decide_max_seq_len):
    indexed_ds = experiment['indexed_dataset']
    padded_fields = pad_inputs(indexed_ds,
                               max_seq_len_decider=max_seq_len_decider)
    # todo: implement correct splitting for multiple intput fields
    msg = 'Currently only 1 field supported for splitting'
    assert len(list(padded_fields.keys())) == 1, msg
    x_tr = {}
    x_te = {}
    for field, pad_X in padded_fields.items():
        x_tr[field], x_te[field], y_tr, y_te = train_test_split(
            pad_X, indexed_ds['outputs'], test_size=0.1)

    split = {'X_tr': x_tr, 'X_te': x_te, 'y_tr': y_tr, 'y_te': y_te}
    for field, split_X_tr in split['X_tr'].items():
        print('shapes for split X_tr', split_X_tr.shape,
              'split y_tr', split['y_tr'].shape,
              'split X_te', split['X_te'][field].shape,
              'split y_te', split['y_te'].shape)
    return split


def create_word_embedding_input(in_shape, vocab_size, emb_weights=None,
                                hparams={}):
    emb_dim = hparams.get('emb_dim', 100)
    in_len = in_shape[0]
    if emb_weights is not None:
        emb_layer = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                              weights=[emb_weights],
                              input_length=in_len,
                              trainable=hparams.get('emb_trainable', True))
    else:
        emb_layer = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                              input_length=in_len)

    input = Input(shape=in_shape)
    if len(in_shape) == 1:
        model = emb_layer(input)
    elif len(in_shape) == 2:
        def split_ls(x):
            import tensorflow as tf
            return tf.split(x, num_or_size_splits=2, axis=2)
        split = Lambda(split_ls)(input)
        split0 = Reshape((in_len,))(split[0])
        split1 = Reshape((in_len,))(split[1])
        emb_layer0 = emb_layer(split0)
        emb_layer1 = emb_layer(split1)
        model = Concatenate(axis=2)([emb_layer0, emb_layer1])
    else:
        msg = 'Expecting an input shape of 1 or 2 dimensions, but got % '
        raise RuntimeError(msg % in_shape)
    return input, model


def create_final_dense_layers(partial_model, in_dim_hint=None, hparams={}):
    n_layers = hparams.get('final_dense_layers', 1)
    out_classes = hparams.get('output_classes', 2)
    model = partial_model
    in_dim = partial_model.shape[1]
    if type(in_dim) == tf.Dimension:
        in_dim = in_dim.value
    if not type(in_dim) == int:
        print('partial model does not provide correct input dimension,' +
              'using hint instead. partial_model shape:' +
              str(partial_model.shape) + str(type(in_dim)))
        assert in_dim_hint is not None
        print('type of hint:', type(in_dim_hint))
        assert type(in_dim_hint) == int
        in_dim = in_dim_hint
    msg = 'Creating %d dense layers between %s and %s'
    print(msg % (n_layers, in_dim, out_classes))
    for layer in range(n_layers - 1):
        out_dim = int(in_dim/n_layers)
        model = Dense(out_dim, activation='relu')(model)
        in_dim = out_dim
    return Dense(out_classes,
                 activation=hparams.get('out_activation', 'softmax'))(model)


def create_biLSTM(input, embedded_input, hparams={}):
    print('creating biLSTM model with hyper params', hparams)
    print('embedded_input shape', embedded_input.shape)
    model = Dropout(hparams.get('emb_dropout', 0.1))(embedded_input)
    lstm_layers = hparams.get('lstm_layers', 1)
    assert lstm_layers >= 1
    for lstm_layer in range(lstm_layers):
        model = Bidirectional(LSTM(
            units=hparams.get('lstm_dim', 100),
            # intermediate layers return sequences,
            # final layer returns embedding
            return_sequences=False if lstm_layer == lstm_layers - 1 else True,
            recurrent_dropout=hparams.get('lstm_dropout', 0.1)))(model)
    out = create_final_dense_layers(model, hparams=hparams)
    return Model(input, out)


def create_CNN(input, embedded_input, hparams={}):
    print('creating CNN model with hyper params', hparams)
    print('embedded_input shape', embedded_input.shape)
    model = Dropout(hparams.get('emb_dropout', 0.1))(embedded_input)
    print('shape input before 1st conv layer', model.shape)
    for nth, conv_layer in enumerate(hparams['conv_layers']):
        model = Conv1D(conv_layer.get('filters', 128),
                       conv_layer.get('stride', 5), activation="relu")(model)
        model = MaxPooling1D(conv_layer.get('max_pool_over', 5))(model)
        print('shape input after %d th conv layer' % nth, model.shape)
        model = Dropout(hparams.get('cnn_dropout', 0.0))(model)
    print('shape before flatten %s' % model.shape)
    flat_dim = model.shape[1] * model.shape[2]
    model = Flatten()(model)
    print('shape after flatten %s' % model.shape)
    print('expected flat dim %s' % flat_dim)
    out = create_final_dense_layers(
        model, in_dim_hint=flat_dim.value, hparams=hparams)
    return Model(input, out)


def _create_wordemb_model(in_shape, vocab_size, emb_weights=None, hparams={}):
    input, model = create_word_embedding_input(in_shape, vocab_size,
                                               emb_weights, hparams)
    arch = hparams.get('architecture', 'biLSTM')
    if arch == 'biLSTM':
        return create_biLSTM(input, model, hparams=hparams)
    elif arch == 'CNN':
        return create_CNN(input, model, hparams=hparams)
    else:
        raise RuntimeError('Invalid architecture ' + arch)


def single_field(ds_split):
    in_fields = list(ds_split['X_tr'].keys())
    msg = 'Currently only one input field is supported, but found %s'
    assert len(in_fields) == 1, msg % in_fields
    return in_fields[0]


def create_wordemb_model(ds_split, emb_weights, hparams):
    field = single_field(ds_split)
    X = ds_split['X_tr'][field]
    if len(X.shape) == 2:
        in_shape = (X.shape[1], )
    elif len(X.shape) == 3:
        in_shape = (X.shape[1], X.shape[2], )
    else:
        msg = 'Expecting an input of shape (ds_size, seq_len, subtok_len) but found %'
        raise RuntimeError(msg % X.shape)

    return _create_wordemb_model(
        in_shape=in_shape, vocab_size=hparams['voc_size'],
        emb_weights=emb_weights, hparams=hparams)


def train_split(model, ds_split, voc_emb, hparams={}):
    callbacks = []
    if hparams.get('early_stopping', True):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.01, patience=5,
                                       verbose=1)
        callbacks.append(early_stopping)
    return model.fit(ds_split['X_tr'][single_field(ds_split)],
                     ds_split['y_tr'],
                     batch_size=hparams.get('batch_size', 128),
                     epochs=hparams.get('epochs', 8),
                     callbacks=callbacks,
                     validation_split=hparams.get('validation_split', 0.1),
                     verbose=1)


def create_optimizer(hparams={}):
    if 'learning_rate' in hparams:
        return keras.optimizers.Adam(
            lr=hparams.get('learning_rate', 0.001),
            decay=hparams.get('learning_rate_decay', 0.0))
    else:
        return 'adam'


def create_model_and_train_split(ds_split, voc_emb, hparams={}):
    vocemb_type = voc_emb.get('type')
    if vocemb_type is None or vocemb_type == 'word_embedding':
        if 'keras_weights' not in voc_emb:
            voc_emb['keras_weights'] = as_keras_emb_weights(voc_emb)
        model = create_wordemb_model(ds_split, voc_emb['keras_weights'],
                                     hparams)
    else:
        raise RuntimeError('Unknown vocab embedding type ' + vocemb_type)
    model.compile(optimizer=create_optimizer(hparams=hparams),
                  loss="binary_crossentropy", metrics=["accuracy"])
    #print(model.summary())
    history = train_split(model, ds_split, voc_emb, hparams=hparams)
    return model, history


def plot_acc_history(keras_history):
    hist = pd.DataFrame(keras_history.history)
    plt.figure(figsize=(4, 4))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.ylim([0.45, 1.05])
    plt.legend()
    plt.show()


def testModel(inputs, outputs, model):
    fields = list(inputs.keys())
    assert len(fields) == 1, 'Only single input fields supported %s' % inputs
    field = fields[0]
    print('testing input, output shapes', inputs[field].shape, outputs.shape)
    p = model.predict(inputs[field])
    p = np.argmax(p, axis=-1)
    y_argmax = np.argmax(outputs, axis=-1)
    msg = 'fixme: currently assuming binary classification, but your output shape indicates n-ary classification %s '
    assert outputs.shape[1] == 2, msg % outputs.shape
    res = p + y_argmax
    TP = sum(res == 2)
    TN = sum(res == 0)
    diff = p - y_argmax
    FP = sum(diff == 1)
    FN = sum(diff == -1)
    failures = sum((res != 0) & (res != 2))
    acc = (TP + TN)/len(y_argmax)
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Failures': failures,
            'test_size': len(y_argmax),
            'acc': acc}


def freeze_all_but_last_n(model, n=1):
    for layer in model.layers[:-n]:
        layer.trainable = False
    for layer in model.layers[-n:]:
        layer.trainable = True


def load_keras_json_weights(json_graph, h5_weights):
    with open(json_graph, 'r') as graphf:
        model = keras.models.model_from_json(graphf.read())
    model.load_weights(h5_weights)
    return model


def load_keras_clone(original_model, h5_weights):
    model = keras.models.clone_model(original_model)
    model.load_weights(experiment['model']['weights'])
    return model


def load_existing_model(experiment):
    if 'model' not in experiment:
        return None

    if experiment['model']['format'] == 'keras_json_weights':
        model = load_keras_json_weights(
            experiment['model']['graph'], experiment['model']['weights'])
    elif experiment['model']['format'] == 'keras_clone':
        model = load_keras_clone(
            experiment['model']['graph'], experiment['model']['weights'])
    else:
        raise RuntimeError('Unsupported model format',
                               experiment['model']['format'])

    if 'trainable_layers' in experiment['model']:
        freeze_all_but_last_n(model, experiment['model']['trainable_layers'])
        model.compile(
            optimizer=create_optimizer(hparams=experiment['hparams']),
            loss="binary_crossentropy", metrics=["accuracy"])
    return model


def execute_experiment(experiment, iteration):
    indexed_ds = experiment['indexed_dataset']
    hparams = experiment['hparams']
    if 'in_seq_len' in hparams:
        seq_len_decider = lambda seq: hparams['in_seq_len']
    else:
        seq_len_decider = decide_max_seq_len if iteration > 0 else decide_and_plot_max_seq_len
    split = pad_and_split_experiment(
        experiment, max_seq_len_decider=seq_len_decider)
    voc_emb = indexed_ds['vocab_embedding']
    model = load_existing_model(experiment)
    if model is None:
        model, history = create_model_and_train_split(
            split, voc_emb, hparams=hparams)
    else:
        history = train_split(model, split, voc_emb, hparams=hparams)

    plot_acc_history(history)
    test_result = testModel(split['X_te'], split['y_te'], model)
    for param, val in hparams.items():
        test_result[param] = val
    test_result['emb_weights'] = 'no' if voc_emb.get('vecs', None) is None else 'yes'
    return test_result, {'model': model, 'history': history}


def test_result_is_better(result_a, result_b):
    '''Returns True if result_a is better then result_b'''
    if result_b is None:
        return True
    return result_a['acc'] > result_b['acc']


def n_cross_val(experiments, n=5, is_better_fn=test_result_is_better):
    results = []
    best = {'test_result': None,
            'model_hist': None}
    for i in range(n):
        try:
            for exp_id, experiment in experiments.items():
                print('\nExecuting ', exp_id, 'iteration', i, 'of',
                      n, 'with hparams', experiment['hparams'])
                test_result, model_history = experiment['executor'](
                    experiment, i)
                print('test results', exp_id, 'it', i, test_result)
                test_result['n'] = i
                test_result['experiment_id'] = exp_id
                results.append(test_result)
                if is_better_fn(test_result, best['test_result']):
                    best['test_result'] = test_result
                    best['model_hist'] = model_history
                else:
                    # hopefully delete the model to release memory
                    del model_history
            # make sure we clean up after ourselves to avoid running out memory
            gc.collect()
        #except Exception as e:
        #    print('Error executing iteration ', i, '.', type(e), e, e.args, ' attempting to continue')
        finally:
            print('done it', i)
    return pd.DataFrame(results), best


common_hparams = {
    'emb_trainable': True,
    'emb_dropout': 0.2,
    'final_dense_layers': 1,
    'out_activation': 'softmax',
    'epochs': 20,
    'batch_size': 128,
    'validation_split': 0.1,
    'early_stopping': True,
}


biLSTM_hparams = {
    'architecture': 'biLSTM',
    'lstm_dim': 100,
    'lstm_layers': 1,
    'lstm_dropout': 0.1
}


CNN_hparams = {
    'architecture': 'CNN',
    'conv_layers': [
        {'filters': 128, 'stride': 5, 'max_pool_over': 5},
        {'filters': 128, 'stride': 5, 'max_pool_over': 5},
        {'filters': 128, 'stride': 5, 'max_pool_over': 35}
    ],
}


def calc_hparams(indexed_ds):
    result = {}
    voc_emb = indexed_ds['vocab_embedding']
    vocemb_type = voc_emb.get('type', None)
    if vocemb_type is None or vocemb_type == 'word_embedding':
        result['voc_size'] = len(voc_emb['itos']) + 2  # words + pad 'word'
        result['emb_trainable'] = False if voc_emb['vecs'] is None else True
        result['emb_dim'] = voc_emb['dim']
    result['output_classes'] = indexed_ds['outputs'].shape[1]
    return result


def merge_hparams(param_list):
    if len(param_list) == 0:
        raise RuntimeError('param_list must have at least one value')
    if len(param_list) == 1:
        return param_list[0]
    result = dict(param_list[0])
    for add_hparams in param_list[1:]:
        for key, val in add_hparams.items():
            result[key] = val
    return result
