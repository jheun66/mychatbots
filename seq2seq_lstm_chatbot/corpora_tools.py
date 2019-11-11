import pickle
import re
from collections import Counter
from nltk.corpus import comtrans

import data_utils

# 토큰 정리 & 소문자로 변환
def clean_sentence(sentence):
    regex_splitter = re.compile("([!?.,:;$\"')( ])")
    clean_words = [re.split(regex_splitter, word.lower()) for word in sentence]
    return [w for words in clean_words for w in words if words if w]


# 처리하기에 긴 문장 필터링, 컴퓨터 성능에 따라 max_len 변경
def filter_sentence_length(sentences_l1, sentences_l2, min_len=0, max_len=20):
    filtered_sentences_l1 = []
    filtered_sentences_l2 = []
    for i in range(len(sentences_l1)):
        if min_len <= len(sentences_l1[i]) <= max_len and \
                                min_len <= len(sentences_l2[i]) <= max_len:
            filtered_sentences_l1.append(sentences_l1[i])
            filtered_sentences_l2.append(sentences_l2[i])
    return filtered_sentences_l1, filtered_sentences_l2


# 단어 사전 만들기, 특수 기호 4개 포함
def create_indexed_dictionary(sentences, dict_size=10000, storage_path=None):
    count_words = Counter()
    dict_words = {}
    opt_dict_size = len(data_utils.OP_DICT_IDS)
    for sen in sentences:
        for word in sen:
            count_words[word] += 1

    dict_words[data_utils._PAD] = data_utils.PAD_ID
    dict_words[data_utils._GO] = data_utils.GO_ID
    dict_words[data_utils._EOS] = data_utils.EOS_ID
    dict_words[data_utils._UNK] = data_utils.UNK_ID

    for idx, item in enumerate(count_words.most_common(dict_size)):
        dict_words[item[0]] = idx + opt_dict_size

    if storage_path:
        pickle.dump(dict_words, open(storage_path, "wb"))
    return dict_words


# token(word)를 tokenID(index)로 변경, 사전에 없는 단어의 경우 unk로 카운트
def sentences_to_indexes(sentences, indexed_dictionary):
    indexed_sentences = []
    not_found_counter = 0
    for sent in sentences:
        idx_sent = []
        for word in sent:
            try:
                idx_sent.append(indexed_dictionary[word])
            except KeyError:
                idx_sent.append(data_utils.UNK_ID)
                not_found_counter += 1
        indexed_sentences.append(idx_sent)

    print('[sentences_to_indexes] Did not find {} words'.format(not_found_counter))
    return indexed_sentences


def extract_max_length(corpora):
    return max([len(sentence) for sentence in corpora])


# 특수 기호를 이용하여 padding과 출력 시퀀스의 시작에 go, 마지막에 eos 추가
def prepare_sentences(sentences_l1, sentences_l2, len_l1, len_l2):
    assert len(sentences_l1) == len(sentences_l2)
    data_set = []
    for i in range(len(sentences_l1)):
        padding_l1 = len_l1 - len(sentences_l1[i])
        pad_sentence_l1 = ([data_utils.PAD_ID]*padding_l1) + sentences_l1[i]

        padding_l2 = len_l2 - len(sentences_l2[i])
        pad_sentence_l2 = [data_utils.GO_ID] + sentences_l2[i] + [data_utils.EOS_ID] + ([data_utils.PAD_ID] * padding_l2)
        data_set.append([pad_sentence_l1, pad_sentence_l2])

    return data_set