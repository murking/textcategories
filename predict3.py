# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 20:12:53 2017

@author: shicheng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:10:52 2017

@author: shicheng
"""
import os
import pandas as pd
import numpy as np
import pickle
import jieba
import re
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from gensim.models import KeyedVectors
from keras import backend as k_backend
from keras.models import Model
from keras.preprocessing import text as keras_text
from keras.preprocessing import sequence as keras_seq
from keras.layers import Dense, Input, Dropout, Reshape, merge, Embedding, \
Convolution1D, MaxPooling1D, LSTM, Masking, Highway, TimeDistributed, Bidirectional, AveragePooling1D
from keras.optimizers import Adadelta, Adam, SGD
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from keras import backend as K

class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, **kwargs):
        self.attention_vec = attention_vec
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        assert hasattr(self.attention_vec, '_keras_shape')
        attention_dim = self.attention_vec._keras_shape[1]

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s,
                                   self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = K.tanh(K.dot(h, self.U_a) + attention + self.b_a)
        s = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * s
        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

class SaveBestWeights(Callback):
    def __init__(self):
        self.best_val = -np.Inf
        self.weights = None
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        if current > self.best_val:
            self.best_val = current
            self.weights = self.model.get_weights()
    def on_train_end(self, logs={}):
        self.model.set_weights(self.weights)

def remove_num(content):
    if content == np.nan:
        return ''
    content = str(content)
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
    new_content = new_content.replace(':', '')
    new_content = new_content.replace(';', '；')
    
    new_content = new_content.replace('，', ',')
    #new_content = new_content.replace('。', '')
    return new_content
    
def process_document(doc, sentence_size, tokenizer):
    sentences = re.split('[。]', doc)
    while (len(sentences) < sentence_size):
        sentences.append('')
    sentences = list(map(lambda x: ' '.join(list(jieba.cut(x))),sentences))
    sentences = tokenizer.texts_to_sequences(sentences)
    return sentences

def pad_sentence(sentences, max_len):
    sentences = keras_seq.pad_sequences(sentences, maxlen=max_len, dtype='int32', padding='post')
    return sentences
    
def load_word_vec(path, vocab):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
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
    
def load_data():
    data_path_dic = {'data_path': 'data/3ci.csv', 'vec_path': 'data/vectors.bin'}
    #data_path_dic = {'data_path': '~/text_categorization/data/3ci.csv', 'vec_path': 'data/vectors.bin'}

    data = pd.read_csv(data_path_dic['data_path'], encoding='utf-8')
    data['titlecontent'] = list(map(lambda x: remove_num(x), data['超声报告']))
    data['document'] = list(map(lambda x: ' '.join(list(jieba.cut(x))), data['titlecontent']))
    data['label'] = list(map(lambda x: 1 if x=='恶性' else 0, data['病理诊断']))
    sentence_size = max(list(map(lambda x: len(x.split("。")), data['titlecontent'])))
    tokenizer = keras_text.Tokenizer()
    tokenizer.fit_on_texts(texts=data['document'])
    
    data_1 = data[data['序号']==1][['医疗卡号', '病理报告日期', '超声报告', 'titlecontent', 'label']]
    data_2 = data[data['序号']==2][['医疗卡号', '病理报告日期', '超声报告', 'titlecontent']]
    data_3 = data[data['序号']==3][['医疗卡号', '病理报告日期', '超声报告', 'titlecontent']]
    merged_data = pd.merge(data_1, data_2, how='left', on=['医疗卡号', '病理报告日期'], suffixes=('', '_2'))
    merged_data = pd.merge(merged_data, data_3, how='left', on=['医疗卡号', '病理报告日期'], suffixes=('_1', '_3'))
    merged_data['titlecontent_2'] = list(map(lambda x: x if type(x) == str else '', merged_data['titlecontent_2']))
    merged_data['titlecontent_3'] = list(map(lambda x: x if type(x) == str else '', merged_data['titlecontent_3']))
    
    maxlen = 30
    merged_data['titlecontent_1'] = list(map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_1']))
    #maxlen = max(list(map(lambda x: max(list(map(lambda y: len(y),x))),data['titlecontent_1'])))
    merged_data['titlecontent_1'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_1']))
    merged_data['titlecontent_2'] = list(map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_2']))
    merged_data['titlecontent_2'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_2']))
    merged_data['titlecontent_3'] = list(map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_3']))
    merged_data['titlecontent_3'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_3']))
    
    document_len = max(list(map(lambda x: len(x.split(' ')), data['document'])))
    print(sentence_size)
    
    #tfidf
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vec = tfidf_vectorizer.fit_transform(data['document'])
    
    vocab_size = len(tokenizer.word_counts) + 1
    word_vecs, vec_size = load_word_vec(data_path_dic['vec_path'], tokenizer.word_counts)
    add_unknown_words(word_vecs, tokenizer.word_counts)
    W = get_W(word_vecs, tokenizer.word_index, vec_size)
    sen_array1 = []
    sen_array2 = []
    sen_array3 = []
    for s in merged_data['titlecontent_1']:
        sen_array1.append(s)
    for s in merged_data['titlecontent_2']:
        sen_array2.append(s)
    for s in merged_data['titlecontent_3']:
        sen_array3.append(s)
    return tokenizer, maxlen, vocab_size, vec_size, W, tfidf_vec,\
        np.array(sen_array1), np.array(sen_array2), np.array(sen_array3),\
        np.array(merged_data['label'], dtype='int32')

def CNN_LSTM_model(sen_size, sen_len, vocab_size, vec_size, W):
    filter_window = [1,2,3]
    filter_size = 100
    input_sen = Input(shape=(sen_len,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=sen_len, 
                          trainable=True)(input_sen)
    embedding_dropout = Dropout(.5)(embedding)
    conv = []
    for i in filter_window:
        conv_out = Convolution1D(filter_size, i, border_mode='same',
                                 activation='tanh', W_constraint=maxnorm(3), 
                                 input_dim=vec_size, input_length=sen_size)(embedding_dropout)
        conv.append(conv_out)
    merged_conv = merge(conv, mode='concat', concat_axis=2)
    pooling = MaxPooling1D(pool_length=sen_len)(merged_conv)
    reshape = Reshape([len(filter_window)*filter_size])(pooling)
    cnn_model = Model(input=input_sen, output=reshape)
    
    input_doc = Input(shape=(sen_size, sen_len))
    td = TimeDistributed(cnn_model)(input_doc)
    td_dropout = Dropout(.5)(td)
    lstm2 = LSTM(output_dim=vec_size, use_bias=False, return_sequences=True)(td_dropout)
    pooling2 = AveragePooling1D(pool_length=sen_size)(lstm2)
    reshape2 = Reshape([vec_size])(pooling2)
    out = Dense(output_dim=1, activation='sigmoid')(reshape2)
    model = Model(input=input_doc, output=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def LSTM_LSTM_model(sen_size, sen_len, vocab_size, vec_size, W):
    input_sen = Input(shape=(sen_len,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=sen_len, 
                          trainable=True)(input_sen)
    embedding_dropout = Dropout(.5)(embedding)
    #masking = Masking(mask_value=0.)(embedding)
    lstm1 = LSTM(output_dim=vec_size, return_sequences=True)(embedding_dropout)
    pooling1 = AveragePooling1D(pool_length=sen_len)(lstm1)
    reshape1 = Reshape([vec_size])(pooling1)
    sen_model = Model(input=input_sen, output=reshape1)
    
    input_doc = Input(shape=(sen_size, sen_len))
    td = TimeDistributed(sen_model)(input_doc)
    td_dropout = Dropout(.5)(td)
    lstm2 = LSTM(output_dim=vec_size, return_sequences=True)(td_dropout)
    pooling2 = AveragePooling1D(pool_length=sen_size)(lstm2)
    reshape2 = Reshape([vec_size])(pooling2)
    reshape2_dropout = Dropout(.5)(reshape2)
    out = Dense(output_dim=1, activation='sigmoid')(reshape2_dropout)
    model = Model(input=input_doc, output=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def HSNN_model(sen_size, sen_len, vocab_size, vec_size, W):

    cell_num = 128
    input_sen = Input(shape=(sen_len,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=sen_len, 
                          trainable=True)(input_sen)
    embedding_dropout = Dropout(.5)(embedding)
    masking = Masking(mask_value=0.)(embedding)
    lstm1 = LSTM(units=cell_num, use_bias=True, return_sequences=True)(embedding_dropout)
    pooling1 = AveragePooling1D(pool_size=sen_len)(lstm1)
    reshape1 = Reshape([cell_num])(pooling1)
    sen_model = Model(inputs=input_sen, outputs=reshape1)
    input_doc = Input(shape=(sen_size, sen_len))
    td = TimeDistributed(sen_model)(input_doc)
    td_dropout = Dropout(.5)(td)

    lstm2 = LSTM(units=cell_num, use_bias=True, return_sequences=True)(td_dropout)
    #lstm_attention2 = AttentionLSTM(lstm2,attention_vec=[])
    pooling2 = AveragePooling1D(pool_size=sen_size)(lstm2)
    reshape2 = Reshape([1, cell_num])(pooling2)
    reshape2_dropout = Dropout(.5)(reshape2)
    doc_model = Model(inputs=input_doc, outputs=reshape2_dropout)
    
    input_doc1 = Input(shape=(sen_size, sen_len))
    input_doc2 = Input(shape=(sen_size, sen_len))
    input_doc3 = Input(shape=(sen_size, sen_len))
    doc_out_1 = doc_model(input_doc1)
    doc_out_2 = doc_model(input_doc2)
    doc_out_3 = doc_model(input_doc3)
    merged_doc_out = merge([doc_out_1, doc_out_2, doc_out_3], mode='concat', concat_axis=1)
    do_dropout = Dropout(.5)(merged_doc_out)
    lstm3 = LSTM(units=cell_num, use_bias=True, return_sequences=True)(do_dropout)
    #lstm_attention3 = AttentionLSTM(lstm3,attention_vec=[])
    pooling3 = AveragePooling1D(pool_size=3)(lstm3)
    reshape3 = Reshape([cell_num])(pooling3)
    reshape3_dropout = Dropout(.5)(reshape3)
    out = Dense(units=1, activation='sigmoid')(reshape3_dropout)
    model = Model(inputs=[input_doc1, input_doc2, input_doc3], outputs=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def print_matrics_1(true, pred_prob):
    #precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
    pred = [1 if p >= .5 else 0 for p in pred_prob]
    
    A = sum(list(map(lambda x, y: 1 if x == 1 and y == 1 else 0, true, pred)))
    B = sum(list(map(lambda x, y: 1 if x == 0 and y == 1 else 0, true, pred)))
    C = sum(list(map(lambda x, y: 1 if x == 1 and y == 0 else 0, true, pred)))
    D = sum(list(map(lambda x, y: 1 if x == 0 and y == 0 else 0, true, pred)))
    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    auc = roc_auc_score(true, pred_prob)
    print("A %d, B %d, C %d, D %d, AUC %.4f" %(A, B, C, D, auc))
    print("acc %.2f %%, precision %.2f %%, recall %.2f %%, f1 %.2f %%, auc %.4f"
          %(acc*100, prec*100, recall*100, f1*100, auc))

def save_pic(history,path):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')
    fig.savefig(path)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    model_type = 'lstm2'
    
    token, sen_len, vocab_size, vec_size, W, tfidf_vec, sen1, sen2, sen3, label = load_data()
    sen = sen1
    doc_size = sen.shape[0]
    sen_size = sen.shape[1]

    sbw = SaveBestWeights()
    
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9876)
    train_index, val_test_index = list(sss1.split(label, label))[0]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=9876)
    val_index, test_index = list(sss2.split(label[val_test_index], label[val_test_index]))[0]
    train_sen1, train_sen2, train_sen3 = sen1[train_index], sen2[train_index], sen3[train_index]
    val_sen1, val_sen2, val_sen3 = sen1[val_test_index][val_index], sen2[val_test_index][val_index], sen3[val_test_index][val_index]
    test_sen1, test_sen2, test_sen3 = sen1[val_test_index][test_index], sen2[val_test_index][test_index], sen3[val_test_index][test_index]
    train_label, val_label, test_label = label[train_index], label[val_test_index][val_index], label[val_test_index][test_index]

    if model_type == 'lstm1':
        model = CNN_LSTM_model(sen_size, sen_len, vocab_size, vec_size, W)
        
        history = model.fit(x=train_sen1, y=train_label, batch_size=32, nb_epoch=10,
                      validation_data=[val_sen1, val_label], callbacks=[])
        test_pred = model.predict(test_sen1)
        print_matrics_1(test_label, test_pred)
        save_pic(history,'image/cnnlstm.png')
    elif model_type == 'lstm2':
        model = LSTM_LSTM_model(sen_size, sen_len, vocab_size, vec_size, W)

        history = model.fit(x=train_sen1, y=train_label, batch_size=32, nb_epoch=5,
                      validation_data=[val_sen1, val_label], callbacks=[sbw])
        test_pred = model.predict(test_sen1)
        print_matrics_1(test_label, test_pred)
        save_pic(history,'image/lstmlstm.png')
    elif model_type == 'hsnn':
        with tf.device('/tasl:0'):
            model = HSNN_model(sen_size, sen_len, vocab_size, vec_size, W)
            print(test_sen1.shape)
            print(sen_size,sen_len,vocab_size,vec_size,W)
            model.fit(x=[train_sen3, train_sen2, train_sen1], y=train_label, batch_size=32, nb_epoch=10,
                        validation_data=[[val_sen3, val_sen2, val_sen1], val_label], callbacks=[sbw])
            test_pred = model.predict([test_sen3, test_sen2, test_sen1])
            print_matrics_1(test_label, test_pred)
            #model.save('hrnn_model')
            pickle.dump([token], open("token.p", "wb"))

#%%
'''
pickle.dump(cnn_lstm_pred, open("cnn_lstm_pred.p", "wb"))
pickle.dump(lstm_lstm_pred, open("lstm_lstm_pred.p", "wb"))

'''
