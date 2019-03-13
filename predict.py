
# coding: utf-8

# In[4]:

import os
import pandas as pd
import numpy as np
import pickle
import jieba
import re
from gensim.models.word2vec import Word2Vec
from keras import backend as K
from keras.models import Model
from keras.preprocessing import text as keras_text
from keras.preprocessing import sequence as keras_seq
from keras.layers import Dense, Input, Dropout, Reshape, merge, Embedding, Convolution1D, MaxPooling1D, LSTM, Masking, Highway, SimpleRNN
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

class SaveBestWeights(Callback):
    def __init__(self):
        self.best_acc = -np.Inf
        self.best_val = -np.Inf
        self.weights = None
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        acc = logs.get('acc')
        if current > self.best_val and acc > current:
            self.best_acc = acc
            self.best_val = current
            self.weights = self.model.get_weights()
    def on_train_end(self, logs={}):
        print('train acc: %.2f %%, val acc: %.2f %%' %(self.best_acc*100, self.best_val*100))
        self.model.set_weights(self.weights)

def remove_num(content):
    if content == np.nan:
        return ''
    content = str(content)
    #return content
    new_content = re.sub('[0-9]*\.*[0-9]+','', content)
    new_content = re.sub('（.*）','', new_content)
    new_content = re.sub('\(.*\)','', new_content)
    new_content = new_content.replace('×', '')
    new_content = new_content.replace('*', '')
    new_content = new_content.replace('mm', '')
    new_content = new_content.replace('cm', '')
    new_content = new_content.replace('%', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('-', '')
    new_content = new_content.replace('\r', '')
    new_content = new_content.replace('\n', '')
    
    new_content = new_content.replace(',', '，')
    new_content = new_content.replace('.', '。')
    new_content = new_content.replace(':', '：')
    new_content = new_content.replace(';', '；')
#    if new_content.find('穿刺记录') != -1:
#        new_content = new_content[:new_content.find('穿刺记录')]
    return new_content

def load_data(): 

    data = pd.read_csv('data/提取病理前最后3次超声记录.csv')
    data['sentence'] = list(map(lambda x: remove_num(x), data['超声报告']))
    data['sentence'] = list(map(lambda x: ' '.join(list(jieba.cut(x))), data['sentence']))
    data['label'] = list(map(lambda x: 1 if x=='恶性' else 0, data['病理诊断']))
    data['len'] = list(map(lambda x: len(x.split(' ')), data['sentence']))
    
    tokenizer = keras_text.Tokenizer()
    tokenizer.fit_on_texts(texts=data['sentence'])
    text_seq = tokenizer.texts_to_sequences(data['sentence'])
    maxlen = max([len(text) for text in text_seq])
    nb_classes = max(data['label']) + 1

    text_array = keras_seq.pad_sequences(text_seq, maxlen=maxlen, dtype='int32', padding='post')
    data['text_array'] = [text_array[i] for i in range(text_array.shape[0])]
    data_1 = data[data['序号']==1][['医疗卡号', '病理报告日期', '超声报告', 'text_array', 'label']]
    data_2 = data[data['序号']==2][['医疗卡号', '病理报告日期', '超声报告', 'text_array']]
    data_3 = data[data['序号']==3][['医疗卡号', '病理报告日期', '超声报告', 'text_array']]
    merged_data = pd.merge(data_1, data_2, how='left', on=['医疗卡号', '病理报告日期'], suffixes=('', '_2'))
    merged_data = pd.merge(merged_data, data_3, how='left', on=['医疗卡号', '病理报告日期'], suffixes=('_1', '_3'))
    merged_data['not_nan_1'] = list(map(lambda x: 1 if type(x) == np.ndarray else 0, merged_data['text_array_1']))
    merged_data['not_nan_2'] = list(map(lambda x: 1 if type(x) == np.ndarray else 0, merged_data['text_array_2']))
    merged_data['not_nan_3'] = list(map(lambda x: 1 if type(x) == np.ndarray else 0, merged_data['text_array_3']))
    merged_data['text_array_2'] = list(map(lambda x: x if type(x) == np.ndarray else np.zeros([maxlen], dtype=int), merged_data['text_array_2']))
    merged_data['text_array_3'] = list(map(lambda x: x if type(x) == np.ndarray else np.zeros([maxlen], dtype=int), merged_data['text_array_3']))
    merged_data['超声报告_2'] = list(map(lambda x: x if type(x) == str else '', merged_data['超声报告_2']))
    merged_data['超声报告_3'] = list(map(lambda x: x if type(x) == str else '', merged_data['超声报告_3']))

    vocab_size = len(tokenizer.word_counts) + 1
    vec_path = 'data/vectors.bin'
    word_vecs, vec_size = load_word_vec(vec_path, tokenizer.word_counts)
    add_unknown_words(word_vecs, tokenizer.word_counts)
    W = get_W(word_vecs, tokenizer.word_index, vec_size)
    
    text_array_3 = np.reshape(np.concatenate(list(merged_data['text_array_3']),axis=0),[-1,maxlen])
    text_array_2 = np.reshape(np.concatenate(list(merged_data['text_array_2']),axis=0),[-1,maxlen])
    text_array_1 = np.reshape(np.concatenate(list(merged_data['text_array_1']),axis=0),[-1,maxlen])
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vec = tfidf_vectorizer.fit_transform(merged_data['超声报告_1'])
    
    return maxlen, nb_classes, vocab_size, vec_size, W, merged_data[['超声报告_1', '超声报告_2', '超声报告_3']],        text_array_3, text_array_2, text_array_1,         np.array(merged_data[['not_nan_3', 'not_nan_2', 'not_nan_1']]),         np.array(merged_data['label'], dtype='int32'), tfidf_vec

def load_word_vec(path, vocab):
    model = Word2Vec.load_word2vec_format(path, binary=True)
    word_vecs = {}
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs, model.vector_size

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def get_W(word_vecs, word_index, vec_size):
    W = np.zeros(shape=(len(word_vecs) + 1, vec_size), dtype='float32')
    for word in word_index:
        W[word_index[word]] = word_vecs[word]
    return W

def get_CNN_model(maxlen, vocab_size, vec_size, W, filter_window, filter_size, out_dim, non_static):
    input_text = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=maxlen, 
                          trainable=False)(input_text)
    embedding_dropout = Dropout(.2)(embedding)
    conv = []
    for i in filter_window:
        conv_out = Convolution1D(filter_size, i, border_mode='same',
                                 activation='relu', W_constraint=maxnorm(3), 
                                 input_dim=vec_size, input_length=maxlen)(embedding_dropout)
        conv.append(conv_out)
    merged_conv = merge(conv, mode='concat', concat_axis=2)
    pooling = MaxPooling1D(pool_length=maxlen)(merged_conv)
    reshape = Reshape([len(filter_window)*filter_size])(pooling)
    reshape_dropout = Dropout(.2)(reshape)
    out = Dense(output_dim=1, activation='sigmoid')(reshape)
    model = Model(input=input_text, output=out)
    # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def get_CNN_model_1(maxlen, vocab_size, vec_size, W, filter_window, filter_size, out_dim, non_static):
    input_text = Input(shape=(maxlen, vec_size,))
    input_dropout = Dropout(.2)(input_text)
    conv = []
    for i in filter_window:
        conv_out = Convolution1D(filter_size, i, border_mode='same',
                                 activation='relu', W_constraint=maxnorm(3), 
                                 input_dim=vec_size, input_length=maxlen)(input_dropout)
        conv.append(conv_out)
    merged_conv = merge(conv, mode='concat', concat_axis=2)
    pooling = MaxPooling1D(pool_length=maxlen)(merged_conv)
    reshape = Reshape([len(filter_window)*filter_size])(pooling)
    reshape_dropout = Dropout(.2)(reshape)
    out = Dense(output_dim=1, activation='sigmoid')(reshape)
    model = Model(input=input_text, output=out)
    # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def get_LSTM_model(maxlen, vocab_size, vec_size, W):
    input_text = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=maxlen, 
                          trainable=False)(input_text)
    masking = Masking(mask_value=0.)(embedding)
    lstm = LSTM(output_dim=vec_size, return_sequences=False)(masking)
    out = Dense(output_dim=1, activation='sigmoid')(lstm)
    model = Model(input=input_text, output=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_LSTM_model_1(maxlen, vocab_size, vec_size, W):
    input_text = Input(shape=(maxlen, vec_size, ))
    masking = Masking(mask_value=0.)(embedding)
    lstm = LSTM(output_dim=vec_size, return_sequences=False, recurrent_dropout=.2)(masking)
    lstm_dropout = Dropout(.2)(lstm)
    out = Dense(output_dim=1, activation='sigmoid')(lstm_dropout)
    model = Model(input=input_text, output=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_CNN_3_LSTM_model(maxlen, vocab_size, vec_size, W, filter_window, filter_size, out_dim, non_static):
    input_text = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=maxlen, 
                          trainable=non_static)(input_text)
    conv = []
    for i in filter_window:
        conv_out = Convolution1D(filter_size, i, border_mode='same',
                                 activation='relu', W_constraint=maxnorm(3), 
                                 input_dim=vec_size, input_length=maxlen,
                                 bias=False)(embedding)
        conv.append(conv_out)
    merged_conv = merge(conv, mode='concat', concat_axis=2)
    pooling = MaxPooling1D(pool_length=maxlen)(merged_conv)
    reshape1 = Reshape([len(filter_window)*filter_size])(pooling)
    cnn_model = Model(input=input_text, output=reshape1)
    
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    input_3 = Input(shape=(maxlen,))
    input_4 = Input(shape=(3,300,))
    cnn_out_1 = cnn_model(input_1)
    cnn_out_2 = cnn_model(input_2)
    cnn_out_3 = cnn_model(input_3)
    merged_cnn_out = merge([cnn_out_1, cnn_out_2, cnn_out_3], mode='concat', concat_axis=1)
    reshape2 = Reshape([3, len(filter_window)*filter_size])(merged_cnn_out)
    mul = merge([reshape2, input_4], mode='mul', name='mul')
    masking = Masking(mask_value=0.)(mul)
    lstm = SimpleRNN(output_dim=vec_size, return_sequences=False, unroll=True)(masking)
    dropout2 = Dropout(.5)(lstm)
    dense1 = Dense(output_dim=100, activation='tanh')(dropout2)
    dense2 = Dense(output_dim=100, activation='tanh')(dense1)
    out = Dense(output_dim=out_dim, activation='softmax')(dense2)
    model = Model(input=[input_1, input_2, input_3, input_4], output=out)
    model.compile(optimizer=Adadelta(epsilon=1e-6), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[5]:

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    model_type = 'cnn1'
    non_static = True
    filter_window = [3, 4, 5]
    filter_size = 100
    maxlen, nb_classes, vocab_size, vec_size, W, raw_text, text_array_3, text_array_2, text_array_1, not_nan, text_label, tfidf_vec = load_data()
    
    np.random.seed(9999)
    doc_size = len(text_label)
    shuffle = np.random.permutation(doc_size)
    train_index = shuffle[:int(.8*doc_size)]
    val_index = shuffle[int(.8*doc_size):int(.9*doc_size)]
    test_index = shuffle[int(.9*doc_size):]
    
    train_text_3 = text_array_3[train_index]
    train_text_2 = text_array_2[train_index]
    train_text_1 = text_array_1[train_index]
    train_not_nan = not_nan[train_index]
    train_not_nan = np.repeat(train_not_nan, 300, axis=-1).reshape([-1, 3, 300])
    train_tfidf = tfidf_vec[train_index]

    val_text_3 = text_array_3[val_index]
    val_text_2 = text_array_2[val_index]
    val_text_1 = text_array_1[val_index]
    val_not_nan = not_nan[val_index]
    val_not_nan = np.repeat(val_not_nan, 300, axis=-1).reshape([-1, 3, 300])
    val_tfidf = tfidf_vec[val_index]

    test_text_3 = text_array_3[test_index]
    test_text_2 = text_array_2[test_index]
    test_text_1 = text_array_1[test_index]
    test_not_nan = not_nan[test_index]
    test_not_nan = np.repeat(test_not_nan, 300, axis=-1).reshape([-1, 3, 300])
    test_tfidf = tfidf_vec[test_index]
    
    train_label, val_label, test_label = text_label[train_index], text_label[val_index], text_label[test_index]

    input_text = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=maxlen, trainable=False)(input_text)
    embedding_model = Model(input_text, embedding)
    train_embedding = embedding_model.predict(train_text_1)
    val_embedding = embedding_model.predict(val_text_1)
    test_embedding = embedding_model.predict(test_text_1)

    sbw = SaveBestWeights()
    if model_type == 'cnn':
        model = get_CNN_model(maxlen=maxlen, vocab_size=vocab_size, vec_size=vec_size, 
                              W=W, filter_window = [3, 4, 5], filter_size = 100, 
                              out_dim = nb_classes, non_static=True)
        model.fit(x=train_text_1, y=train_label, batch_size=32, nb_epoch=25,
              validation_data=[val_text_1, val_label], callbacks=[sbw], verbose=2)
        loss, best_acc = model.evaluate(test_text_1, test_label, verbose = 0)
        print ('best acc: %.2f %%' % (best_acc * 100))
    if model_type == 'cnn1':
        model = get_CNN_model_1(maxlen=maxlen, vocab_size=vocab_size, vec_size=vec_size, 
                              W=W, filter_window = [3, 4, 5], filter_size = 100, 
                              out_dim = nb_classes, non_static=True)
        model.fit(x=train_embedding, y=train_label, batch_size=32, nb_epoch=25,
              validation_data=[val_embedding, val_label], callbacks=[sbw], verbose=2)
        loss, best_acc = model.evaluate(test_embedding, test_label, verbose = 0)
        print ('best acc: %.2f %%' % (best_acc * 100))
    elif model_type == 'lstm':
        model = get_LSTM_model(maxlen, vocab_size, vec_size, W)
        model.fit(x=train_text_1, y=train_label, batch_size=32, nb_epoch=25,
              validation_data=[val_text_1, val_label], callbacks=[sbw], verbose=2)
        loss, best_acc = model.evaluate(test_text_1, test_true, verbose = 0)
        print ('best acc: %.2f %%' % (best_acc * 100))
    elif model_type == 'lstm1':
        model = get_LSTM_model_1(maxlen, vocab_size, vec_size, W)
        model.fit(x=train_embedding, y=train_label, batch_size=32, nb_epoch=25,
              validation_data=[val_embedding, val_label], callbacks=[sbw], verbose=2)
        loss, best_acc = model.evaluate(test_embedding, test_true, verbose = 0)
        print ('best acc: %.2f %%' % (best_acc * 100))
    elif model_type == 'cnn3_lstm':
        model = get_CNN_3_LSTM_model(maxlen=maxlen, vocab_size=vocab_size, vec_size=vec_size, 
                                     W=W, filter_window = [3, 4, 5], filter_size = 100, 
                                     out_dim = nb_classes, non_static=True)
        model.fit(x=[train_text_3, train_text_2, train_text_1, train_not_nan], y=train_label, batch_size=64, nb_epoch=25,
              validation_data=[[val_text_3, val_text_2, val_text_1, val_not_nan], val_label], callbacks=[sbw])
        loss, best_acc = model.evaluate([test_text_3, test_text_2, test_text_1, test_not_nan], test_label, verbose = 0)
        print ('best acc: %.2f %%' % (best_acc * 100))
    elif model_type == 'svm':
        model = SVC(kernel='linear')
        model.fit(train_tfidf, train_label)
        train_pred = model.predict(train_tfidf)
        val_pred = model.predict(val_tfidf)
        test_pred = model.predict(test_tfidf)
        train_acc = accuracy_score(train_label, train_pred)
        val_acc = accuracy_score(val_label, val_pred)
        test_acc = accuracy_score(test_label, test_pred)
        print("train acc %.2f %%, val acc %.2f %%, test acc %.2f %%" %(train_acc * 100, val_acc * 100, test_acc * 100))


# In[ ]:



