import re
import jieba
import pickle
import numpy as np
from keras.preprocessing import sequence as keras_seq
from keras.models import load_model


def remove_num(content):
    if content == np.nan:
        return ''
    content = str(content)
    # return content
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

    #    if new_content.find('穿刺记录') != -1:
    #        new_content = new_content[:new_content.find('穿刺记录')]
    return new_content


def load_data(tokenizer, sen):
    sen = remove_num(sen)
    sen = ' '.join(list(jieba.cut(sen)))

    text_seq = tokenizer.texts_to_sequences([sen])

    text_array = keras_seq.pad_sequences(text_seq, maxlen=300, dtype='int32', padding='pre')
    return text_array

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

def load_data_1(tokenizer, sen):
    sen = remove_num(sen)

    sentence_size = 20
    maxlen = 30
    res = process_document(sen, sentence_size, tokenizer)
    res = pad_sentence(res, maxlen)
    return res.reshape([-1,sentence_size,maxlen])


def predict(sen, model_type):
    [tokenizer] = pickle.load(open("tokenizer.p", "rb"))
    text_array = load_data(tokenizer, sen)

    if model_type == 'cnn':
        model = load_model("cnn_model")
    elif model_type == 'lstm':
        model = load_model("lstm_model")
    res = model.predict(text_array)
    return res[0][0]

def predict_hrnn(sen1, sen2, sen3):
    [token] = pickle.load(open("token.p", "rb"))
    text1_array = load_data_1(token, sen1)
    text2_array = load_data_1(token, sen2)
    text3_array = load_data_1(token, sen3)
    #print(text1_array)
    model = load_model("hrnn_model")
    res = model.predict([text1_array, text2_array, text3_array])
    return res[0][0]

if __name__ == '__main__':
    #malignant
    sen_1 = "右侧甲状腺中上极（近腹侧，偏外）可见一个低回声结节，大小约14.3×9.4×10.2mm。内部结构呈实性，回声均匀，形状不规则，边界清晰，边缘光整，无声晕，内部见点状强回声，后方回声伴细钙化引起的衰减，无侧方声影。与包膜接触面积0-25%，未向甲状腺包膜外突出，CDFI示其内血供程度中等，血流模式为混合型。"
    #benign
    sen_2 = "甲状腺:甲状腺左、右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈密集中等回声，回声分布欠均匀。CDFI：未见明显异常血流信号。双侧甲状腺内可见几个不均质回声、低回声及混合性回声,左侧之一约25×20mm，右侧之一约6×3.5mm，形状呈椭圆形，内部回声欠均匀，边界尚清，内部未见明显点状强回声，CDFI：未见明显异常血流信号。双侧甲状旁腺区未见明显占位性病变。双侧颈部未见明显异常肿大淋巴结。"

    #malignant
    msen1 = "甲状腺:甲状腺左、右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈增粗减低回声，回声分布不均匀。CDFI：未见明显异常血流信号。左侧甲状腺内可见一个低回声,之一大小约12×9mm，形状呈欠规则，内部回声欠均匀，边缘光整，边界尚清，内部未见明显点状强回声，后方回声无明显变化，CDFI示其内血供程度低。右侧甲状腺内可见一个低回声,大小约4×3mm，形状呈椭圆形，内部回声欠均匀，边缘光整，边界尚清，内部未见明显点状强回声，后方回声无明显变化，CDFI示其内血供程度低。双侧颈部未见明显异常肿大淋巴结。"
    msen2 = "甲状腺:甲状腺左、右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈增粗减低回声，回声分布不均匀，可见散在片状低回声区。CDFI：未见明显异常血流信号。左侧甲状腺内可见一个低回声,之一大小约12.3×10mm，形状呈欠规则，内部回声欠均匀，边缘光整，边界尚清，内部似见点状强回声，后方回声衰减，CDFI示其内血供程度低。右侧甲状腺内可见一个低回声,大小约5×3.5mm，形状呈椭圆形，内部回声欠均匀，边缘光整，边界尚清，内部未见明显点状强回声，后方回声无明显变化，CDFI示其内血供程度低。双侧甲状旁腺区未见明显占位性病变。双侧颈部未见明显异常肿大淋巴结。"
    msen3 = "左侧甲状腺中部（近腹侧，偏内）可见一个低回声结节，大小约12×9×11mm。内部结构呈实性，回声欠均匀，形状不规则，边界清晰，边缘光整，无声晕，内部无明显钙化强回声，后方回声无明显改变，无侧方声影。与包膜接触面积0-25%，未向甲状腺包膜外突出，CDFI示其内血供程度低。合并甲状腺弥漫性病变。"

    #benign
    bsen1 = "甲状腺:甲状腺左、右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈密集中等回声，回声分布欠均匀。CDFI：未见明显异常血流信号。双侧甲状腺内可见几个不均质回声、低回声及混合性回声,左侧之一约16×10×12mm，右侧之一约3×2×2mm，形状呈椭圆形，内部回声欠均匀，边界尚清，内部未见明显点状强回声，CDFI：未见明显异常血流信号。中部之一呈等回声，边界尚清，周边可见低回声声晕，内部未见明显点状强回声，CDFI：可见血流信号环绕。双侧颈部未见明显异常肿大淋巴结。";
    bsen2 = "甲状腺:甲状腺左、右叶饱满，欠规则，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈密集中等回声，回声分布均匀。CDFI：未见明显异常血流信号。双侧甲状腺内可见几个低回声及混合性回声,右侧之一约4×2×3mm，左侧中部之一约20×12×18mm，形状呈椭圆形，内部回声欠均匀，边界尚清，中部之一呈等回声，边界尚清，周边可见低回声声晕，内部未见明显点状强回声，CDFI：可见血流信号环绕。双侧颈部未见明显异常肿大淋巴结。";
    bsen3 = "甲状腺:甲状腺左叶形态饱满，右叶大小及形态正常，峡部厚度正常，边界清楚，表面光滑、包膜完整，内部呈密集中等回声，回声分布均匀。CDFI：未见明显异常血流信号。右侧甲状腺内可见一个无回声,大小约2×1mm，形状呈椭圆形，内部回声欠均匀，边缘光整，边界尚清，内部未见明显点状强回声，后方回声无明显变化，CDFI示其内血供程度低。左侧甲状腺内可见几个等回声、不均质回声、低回声及混合性回声,中部之一大小约23×14mm，其旁另一大小约11×10mm，形状呈椭圆形，内部回声欠均匀，边缘光整，边界尚清，内部未见明显点状强回声，后方回声无明显变化，CDFI示其内血供程度高。双侧甲状旁腺区未见明显占位性病变。双侧颈部未见明显异常肿大淋巴结。";

    lstm_res = predict(bsen3, 'lstm')
    cnn_res = predict(bsen3, 'cnn')
    hrnn_res = predict_hrnn("", "", "")
    print(cnn_res, lstm_res, hrnn_res)
