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
from keras.layers import Dense, Input, Dropout, Reshape, merge, Embedding, \
Convolution1D, MaxPooling1D, LSTM, Masking, Highway, SimpleRNN
from keras.optimizers import Adadelta, Adam
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

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
    
    new_content = new_content.replace(',', '，')
    new_content = new_content.replace('.', '。')
    new_content = new_content.replace(':', '：')
    new_content = new_content.replace(';', '；')
#    if new_content.find('穿刺记录') != -1:
#        new_content = new_content[:new_content.find('穿刺记录')]
    return new_content

def load_data(dataset_name):
    data_folder = {'mr': {'data_path': {'data/mr/rt-polarity.pos': 1, 
                                            'data/mr/rt-polarity.neg': 0},
                              'vec_path': 'D:/GoogleNews-vectors-negative300.bin'},
                       'ultrasound': {'data_path': {'data/ultrasound/malignant.txt': 1, 
                                                    'data/ultrasound/benign.txt': 0},
                                      'vec_path': 'data/ultrasound/vectors.bin'},
                       'new': {'vec_path': 'data/vectors.bin'}}    
    if dataset_name == 'new':
        data = pd.read_csv('data/提取病理前最后3次超声记录.csv')
        data['sentence'] = list(map(lambda x: remove_num(x), data['超声报告']))
        data['sentence'] = list(map(lambda x: ' '.join(list(jieba.cut(x))), data['sentence']))
        data['label'] = list(map(lambda x: 1 if x=='恶性' else 0, data['病理诊断']))
        data['len'] = list(map(lambda x: len(x.split(' ')), data['sentence']))
    else:
        data = pd.DataFrame()
        for path in data_folder[dataset_name]['data_path']:
            tmp = pd.read_csv(path, sep='\t', header=None)
            tmp['sentence'] = tmp[0]
            tmp['label'] = data_folder[dataset_name]['data_path'][path]
            data = pd.concat([data, tmp[['sentence', 'label']]])
        #data['sentence'] = list(map(lambda x: '' if x is np.nan else x,data['sentence']))
    
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
    word_vecs, vec_size = load_word_vec(data_folder[dataset_name]['vec_path'], tokenizer.word_counts)
    add_unknown_words(word_vecs, tokenizer.word_counts)
    W = get_W(word_vecs, tokenizer.word_index, vec_size)
    
    text_array_3 = np.reshape(np.concatenate(list(merged_data['text_array_3']),axis=0),[-1,maxlen])
    text_array_2 = np.reshape(np.concatenate(list(merged_data['text_array_2']),axis=0),[-1,maxlen])
    text_array_1 = np.reshape(np.concatenate(list(merged_data['text_array_1']),axis=0),[-1,maxlen])
    
    return maxlen, nb_classes, vocab_size, vec_size, W, merged_data[['超声报告_1', '超声报告_2', '超声报告_3']],\
        text_array_3, text_array_2, text_array_1, \
        np.array(merged_data[['not_nan_3', 'not_nan_2', 'not_nan_1']]), \
        np.array(merged_data['label'], dtype='int32')

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
                          trainable=non_static)(input_text)
    conv = []
    for i in filter_window:
        conv_out = Convolution1D(filter_size, i, border_mode='same',
                                 activation='relu', W_constraint=maxnorm(3), 
                                 input_dim=vec_size, input_length=maxlen, bias=False)(embedding)
        conv.append(conv_out)
    merged_conv = merge(conv, mode='concat', concat_axis=2)
    pooling = MaxPooling1D(pool_length=maxlen)(merged_conv)
    dropout = Dropout(.5)(pooling)
    reshape = Reshape([len(filter_window)*filter_size])(dropout)
    out = Dense(output_dim=1, activation='sigmoid', W_constraint=maxnorm(3))(reshape)
    model = Model(input=input_text, output=out)
    model.compile(optimizer=Adadelta(epsilon=1e-6), loss='binary_crossentropy',
                  metrics=['accuracy'])
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
#                  metrics=['accuracy'])
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
    
def get_LSTM_model(maxlen, vocab_size, vec_size, W, out_dim):
    input_text = Input(shape=(maxlen,))
    embedding = Embedding(vocab_size, vec_size, weights=[W], input_length=maxlen, 
                          trainable=True)(input_text)
    masking = Masking(mask_value=0.)(embedding)
    lstm = LSTM(output_dim=vec_size, return_sequences=False, unroll=True)(masking)
    dropout = Dropout(.5)(lstm)
    out = Dense(output_dim=out_dim, activation='softmax')(dropout)
    model = Model(input=input_text, output=out)
    model.compile(optimizer=Adadelta(epsilon=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
if __name__ == '__main__':
    
    data_set = 'new'
    model_type = 'cnn'
    non_static = True
    filter_window = [3, 4, 5]
    filter_size = 100
    '''
    file_path = data_set + '.pkl'
    if not os.path.exists(file_path):
        maxlen, nb_classes, vocab_size, vec_size, W, text_seq, text_array, text_label = load_data(data_set)
        pickle.dump([maxlen, nb_classes, vocab_size, vec_size, W, text_seq, text_array, text_label], open(file_path, "wb"))
    else:
        maxlen, nb_classes, vocab_size, vec_size, W, text_seq, text_array, text_label = pickle.load(open(file_path, "rb"))
    '''
    maxlen, nb_classes, vocab_size, vec_size, W, raw_text, text_array_3, text_array_2, text_array_1, not_nan, text_label = load_data(data_set)
    11
    np.random.seed(3435)
    text_cat = to_categorical(text_label, nb_classes)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    acc = []
    fold_nb = 1
    for train_val_index, test_index in skf.split(np.zeros(text_label.shape[0]), text_label):
        print ('epoch: %i' % fold_nb)
        fold_nb += 1
        train_index, val_index, train_label, val_label = train_test_split(train_val_index, 
            text_label[train_val_index], test_size = 0.1, stratify = text_label[train_val_index])
        train_text_3 = text_array_3[train_index]
        train_text_2 = text_array_2[train_index]
        train_text_1 = text_array_1[train_index]
        train_raw_text = raw_text.loc[train_val_index]
        train_not_nan = not_nan[train_index]
        train_not_nan = np.repeat(train_not_nan, 300, axis=-1).reshape([-1, 3, 300])
        
        val_text_3 = text_array_3[val_index]
        val_text_2 = text_array_2[val_index]
        val_text_1 = text_array_1[val_index]
        val_raw_text = raw_text.loc[val_index]
        val_not_nan = not_nan[val_index]
        val_not_nan = np.repeat(val_not_nan, 300, axis=-1).reshape([-1, 3, 300])

        test_text_3 = text_array_3[test_index]
        test_text_2 = text_array_2[test_index]
        test_text_1 = text_array_1[test_index]
        test_raw_text = raw_text.loc[test_index]
        test_not_nan = not_nan[test_index]
        test_not_nan = np.repeat(test_not_nan, 300, axis=-1).reshape([-1, 3, 300])

        train_true, val_true = text_label[train_index], text_label[test_index]
        test_label, test_true = text_cat[test_index], text_label[test_index]


        sbw = SaveBestWeights()
        if model_type == 'cnn':
            model = get_CNN_model(maxlen=maxlen, vocab_size=vocab_size, vec_size=vec_size, 
                                  W=W, filter_window = [3, 4, 5], filter_size = 100, 
                                  out_dim = nb_classes, non_static=True)
            model.fit(x=train_text_1, y=train_label, batch_size=64, nb_epoch=25,
                  validation_data=[val_text_1, val_label], callbacks=[sbw])
            loss, best_acc = model.evaluate(test_text_1, test_true, verbose = 0)
        elif model_type == 'lstm':
            model = get_LSTM_model(maxlen, vocab_size, vec_size, W, nb_classes)
            model.fit(x=train_text_1, y=train_label, batch_size=64, nb_epoch=25,
                  validation_data=[val_text_1, val_label], callbacks=[sbw])
            loss, best_acc = model.evaluate(test_text_1, test_label, verbose = 0)
        elif model_type == 'cnn3_lstm':
            model = get_CNN_3_LSTM_model(maxlen=maxlen, vocab_size=vocab_size, vec_size=vec_size, 
                                         W=W, filter_window = [3, 4, 5], filter_size = 100, 
                                         out_dim = nb_classes, non_static=True)
            model.fit(x=[train_text_3, train_text_2, train_text_1, train_not_nan], y=train_label, batch_size=64, nb_epoch=25,
                  validation_data=[[val_text_3, val_text_2, val_text_1, val_not_nan], val_label], callbacks=[sbw])
            loss, best_acc = model.evaluate([test_text_3, test_text_2, test_text_1, test_not_nan], test_label, verbose = 0)
        '''
        intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer('mul').output)
        intermediate_output = intermediate_layer_model.predict([test_text, test_text, test_text, np.ones([681,3,300])])
        '''
        '''

        pred = model.predict([test_text_3, test_text_2, test_text_1, test_not_nan])
        pred = list(map(lambda x: np.argmax(x), pred))
        
        pred = model.predict(test_text_1)
        pred = list(map(lambda x: np.argmax(x), pred))
        
        precision = precision_score(test_true, pred)
        recall = recall_score(test_true, pred)
        accuracy = accuracy_score(test_true, pred)
        f1 = f1_score(test_true, pred)
        c_matrix = confusion_matrix(test_true, pred)
        accuracy, precision, recall, f1
        
        test_raw_text['true_res'] = test_true
        test_raw_text['cnn_res'] = pred
        test_raw_text['cnn3_lstm_res'] = pred
        test_raw_text.to_csv('test_data.csv')
        test_raw_1 = test_raw_text[list(map(lambda x: True if x=='' else False, test_raw_text['超声报告_2']))]
        test_raw_2 = test_raw_text[list(map(lambda x, y: True if x!='' and y=='' else False, test_raw_text['超声报告_2'], test_raw_text['超声报告_3']))]
        test_raw_3 = test_raw_text[list(map(lambda x: True if x!='' else False, test_raw_text['超声报告_3']))]
        
        sum(list(map(lambda x: x==1, test_raw_1['true_res'])))
        sum(list(map(lambda x: x==0, test_raw_1['true_res'])))
        
        test_true, pred = test_raw_1['true_res'], test_raw_1['cnn_res']
        test_true, pred = test_raw_1['true_res'], test_raw_1['cnn3_lstm_res']
        
        test_true, pred = test_raw_2['true_res'], test_raw_2['cnn_res']
        test_true, pred = test_raw_2['true_res'], test_raw_2['cnn3_lstm_res']
        
        test_true, pred = test_raw_3['true_res'], test_raw_3['cnn_res']
        test_true, pred = test_raw_3['true_res'], test_raw_3['cnn3_lstm_res']
        
        '''
        
        acc.append(best_acc)
        print ('best acc: %.2f %%' % (best_acc * 100))
        print ()
        
    print ('final acc: %.2f %%' % (np.average(acc) * 100))