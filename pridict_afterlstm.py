# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 20:12:53 2017

@author: shicheng
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
暂时不要改动，在双层lstm中成功引入
attention 具体位置在句子层之后

'''
import os
import pandas as pd
import numpy as np
import pickle
import jieba
import re
from gensim.models import KeyedVectors
from keras.preprocessing import text as keras_text
from keras.preprocessing import sequence as keras_seq
from keras.layers import Dense, Input, Dropout, Reshape, merge, Embedding, \
    Convolution1D, MaxPooling1D, LSTM, Masking, Highway, TimeDistributed, Bidirectional, AveragePooling1D
from keras.layers.core import *
from keras.optimizers import Adadelta, Adam, SGD
from keras.callbacks import Callback
import tensorflow as tf
from  tensorflow.contrib.seq2seq.python.ops import *
from keras.backend.tensorflow_backend import set_session
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt


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
    new_content = re.sub('[0-9]*\.*[0-9]+', '', content)
    new_content = re.sub('（.*）', '', new_content)
    new_content = re.sub('\(.*\)', '', new_content)
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
    # new_content = new_content.replace('。', '')
    return new_content


def process_document(doc, sentence_size, tokenizer):
    sentences = re.split('[。]', doc)
    while (len(sentences) < sentence_size):
        sentences.append('')
    sentences = list(map(lambda x: ' '.join(list(jieba.cut(x))), sentences))
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
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(word_vecs, word_index, vec_size):
    W = np.zeros(shape=(len(word_vecs) + 1, vec_size), dtype='float32')
    for word in word_index:
        W[word_index[word]] = word_vecs[word]
    return W


def load_data():
    data_path_dic = {'data_path': 'data/乳腺X光报告X光诊断1128.csv', 'vec_path': 'data/vectors.bin'}
    # data_path_dic = {'data_path': '~/text_categorization/data/3ci.csv', 'vec_path': 'data/vectors.bin'}

    data = pd.read_csv(data_path_dic['data_path'], encoding='utf-8')
    data = data.drop_duplicates(['MedCardNo'])

    data['titlecontent'] = list(map(lambda x: remove_num(x), data['titlecontent']))
    data['document'] = list(map(lambda x: ' '.join(list(jieba.cut(x))), data['titlecontent']))
    #data['label'] = list(map(lambda x: 1 if x == '恶性' else 0, data['病理诊断']))
    data['label'] = 0
    for index,row in data.iterrows():
        if row['X光诊断'] == 'BI-RADS0':
            row['label'] = 0
        elif row['X光诊断'] == 'BI-RADS1':
            row['label'] = 1
        elif row['X光诊断'] == 'BI-RADS2':
            row['label'] = 2
        elif row['X光诊断'] == 'BI-RADS3':
            row['label'] = 3
        elif row['X光诊断'] == 'BI-RADS4A':
            row['label'] = 4
        elif row['X光诊断'] == 'BI-RADS4B':
            row['label'] = 5
        elif row['X光诊断'] == 'BI-RADS4C':
            row['label'] = 7
        elif row['X光诊断'] == 'BI-RADS5':
            row['label'] = 8
        elif row['X光诊断'] == 'BI-RADS6':
            row['label'] = 9

    data["序号"] = 1
    sentence_size = max(list(map(lambda x: len(x.split("。")), data['titlecontent'])))
    sentence_toal = sum(list(map(lambda x: len(x.split("。")), data['titlecontent'])))

    tokenizer = keras_text.Tokenizer()
    tokenizer.fit_on_texts(texts=data['document'])

    data_1 = data[data['序号'] == 1][['MedCardNo', 'ReportTime','titlecontent', 'label']]
    data_2 = data[data['序号'] == 2][['MedCardNo', 'ReportTime',  'titlecontent']]
    data_3 = data[data['序号'] == 3][['MedCardNo', 'ReportTime', 'titlecontent']]
    merged_data = pd.merge(data_1, data_2, how='left', on=['MedCardNo', 'ReportTime'], suffixes=('', '_2'))
    merged_data = pd.merge(merged_data, data_3, how='left', on=['MedCardNo', 'ReportTime'], suffixes=('_1', '_3'))
    merged_data['titlecontent_2'] = list(map(lambda x: x if type(x) == str else '', merged_data['titlecontent_2']))
    merged_data['titlecontent_3'] = list(map(lambda x: x if type(x) == str else '', merged_data['titlecontent_3']))
    maxlen = 30
    merged_data['titlecontent_1'] = list(
        map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_1']))
    # maxlen = max(list(map(lambda x: max(list(map(lambda y: len(y),x))),data['titlecontent_1'])))
    merged_data['titlecontent_1'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_1']))
    merged_data['titlecontent_2'] = list(
        map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_2']))
    merged_data['titlecontent_2'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_2']))
    merged_data['titlecontent_3'] = list(
        map(lambda x: process_document(x, sentence_size, tokenizer), merged_data['titlecontent_3']))
    merged_data['titlecontent_3'] = list(map(lambda x: pad_sentence(x, maxlen), merged_data['titlecontent_3']))

    document_len = max(list(map(lambda x: len(x.split(' ')), data['document'])))

    # tfidf
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
    return tokenizer, maxlen, vocab_size, vec_size, W, tfidf_vec, \
           np.array(sen_array1), np.array(sen_array2), np.array(sen_array3), \
           np.array(merged_data['label'], dtype='int32')



def attention_3d_block(inputs,sen_len):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(sen_len, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def lstm_attention_model(sen_size, sen_len, vocab_size, vec_size, W):

    input_sen = Input(shape=(sen_len,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=sen_len,
                          trainable=True)(input_sen)
    embedding_dropout = Dropout(.5)(embedding)
    lstm1 = LSTM(output_dim=vec_size, return_sequences=True)(embedding_dropout)
    pooling1 = AveragePooling1D(pool_length=sen_len)(lstm1)
    reshape1 = Reshape([vec_size])(pooling1)
    sen_model = Model(input=input_sen, output=reshape1)
    input_doc = Input(shape=(sen_size, sen_len))
    td = TimeDistributed(sen_model)(input_doc)
    td_dropout = Dropout(.5)(td)
    lstm2 = LSTM(output_dim=vec_size, return_sequences=True)(td_dropout)
    attention_mul = attention_3d_block(lstm2,sen_size)
    attention_flatten = Flatten()(attention_mul)
    #pooling2 = AveragePooling1D(pool_length=sen_size)(lstm2)
    #reshape2 = Reshape([vec_size])(pooling2)
    reshape2_dropout = Dropout(.5)(attention_flatten)
    out = Dense(output_dim=1, activation='sigmoid')(reshape2_dropout)
    model = Model(input=input_doc, output=out)
    model.compile(optimizer=Adadelta(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

from seq2seq.layers.decoders import AttentionDecoder



def print_matrics_1(true, pred_prob):
    # precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
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
    print("A %d, B %d, C %d, D %d, AUC %.4f" % (A, B, C, D, auc))
    print("acc %.2f %%, precision %.2f %%, recall %.2f %%, f1 %.2f %%, auc %.4f"
          % (acc * 100, prec * 100, recall * 100, f1 * 100, auc))
def show(history):
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
    plt.show()

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model_type = 'model_read'
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
    val_sen1, val_sen2, val_sen3 = sen1[val_test_index][val_index], sen2[val_test_index][val_index], \
                                   sen3[val_test_index][val_index]
    test_sen1, test_sen2, test_sen3 = sen1[val_test_index][test_index], sen2[val_test_index][test_index], \
                                      sen3[val_test_index][test_index]
    train_label, val_label, test_label = label[train_index], label[val_test_index][val_index], label[val_test_index][
        test_index]


    if  model_type == 'att_lstm':
        model = lstm_attention_model(sen_size, sen_len, vocab_size, vec_size, W)

        histort = model.fit(x=train_sen1, y=train_label, batch_size=32, nb_epoch=1,
                  validation_data=[val_sen1, val_label], callbacks=[sbw])
        show(histort)
        test_pred = model.predict(test_sen1)
        model.save('model/oneattention.h5')
        print_matrics_1(test_label, test_pred)
        temm = ['双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体部分退化，呈条索状改变，大致对称。右乳多发点状及斑片状钙化。左乳散在点状钙化。未见明显异常肿块或成簇细小砂粒状钙化。左侧腋下淋巴结显示。',
                '双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体较丰富，呈团片状、结节状密度增高影，边缘膨隆，大致对称。右乳散在环状钙化。左乳内下象限可见成簇粗细不一斑点及斑片状钙化，部分似沿导管分布。未见明显异常肿块。左侧腋下淋巴结显示。']
        rea = model.predict(temm)

        print(rea)
    elif model_type == 'model_read':
        model = load_model('model/oneattention.h5')

        testinput = ['双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体部分退化，呈条索状改变，大致对称。右乳多发点状及斑片状钙化。左乳散在点状钙化。未见明显异常肿块或成簇细小砂粒状钙化。左侧腋下淋巴结显示。',
                     '双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体较丰富，呈团片状、结节状密度增高影，边缘膨隆，大致对称。右乳散在环状钙化。左乳内下象限可见成簇粗细不一斑点及斑片状钙化，部分似沿导管分布。未见明显异常肿块。左侧腋下淋巴结显示。',
                     '双侧乳腺皮肤及乳晕未见明显增厚，乳头无凹陷。皮下脂肪组织结构层次清晰。双侧腺体丰富致密，呈团片状密度增高影，边缘膨隆，大致对称。双乳内可见散在粗大钙化灶。左乳近腋下小结节状影，淋巴结影可能。']
        sentence_size = max(list(map(lambda x: len(x.split("。")), testinput)))
        sentence_toal = sum(list(map(lambda x: len(x.split("。")), testinput)))

        data_document =  list(map(lambda x: ' '.join(list(jieba.cut(x))), testinput))

        tokenizer = keras_text.Tokenizer()
        tokenizer.fit_on_texts(texts=data_document)
        maxlen = 30
        testinput = list(map(lambda x: process_document(x, 14, tokenizer), testinput))
        testinput = list(map(lambda x: pad_sentence(x, maxlen), testinput))
        sen_array1 = []
        for s in testinput:
            sen_array1.append(s)
        for_test = model.predict(np.array(sen_array1))
        for i in for_test:
            print(int(i))
        #print(for_test)
