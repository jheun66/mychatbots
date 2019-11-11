import time
import math
import sys
import pickle
import glob
import os
import tensorflow as tf
import numpy as np

import corpora_tools as ct
import data_utils

from chatbot_data import retrieve_corpora_from_zip
from sklearn.model_selection import train_test_split

path_q_word2index_dict = "./dictionary/q_word2index_dict.p"
path_a_word2index_dict = "./dictionary/a_word2index_dict.p"
path_q_index2word_dict = "./dictionary/q_index2word_dict.p"
path_a_index2word_dict = "./dictionary/a_index2word_dict.p"
path_data = "./dictionary/data.p"
BATCH_SIZE = 64
TENSORBOARD = './TensorBoard'
model_dir = "./chat/chatbot_model"
MODEL_WEIGHT_FILE = model_dir + "/model.h5"

#model_checkpoints = model_dir + "/chatbot.ckpt"

def build_dataset(use_stored_dictionary=False):
    sen_l1, sen_l2 = retrieve_corpora_from_zip()
    clean_sen_l1 = [ct.clean_sentence(s) for s in sen_l1] #[:50000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
    clean_sen_l2 = [ct.clean_sentence(s) for s in sen_l2] #[:50000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
      
    filt_clean_sen_l1, filt_clean_sen_l2 = ct.filter_sentence_length(clean_sen_l1, clean_sen_l2, max_len=30)

    dirName = "./dictionary"
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    if not use_stored_dictionary:
        q_word2index_dict = ct.create_indexed_dictionary(filt_clean_sen_l1, dict_size=30000, storage_path=path_q_word2index_dict)
        a_word2index_dict = ct.create_indexed_dictionary(filt_clean_sen_l2, dict_size=30000, storage_path=path_a_word2index_dict)
        q_index2word_dict = dict([(idx, word) for word, idx in q_word2index_dict.items()])
        a_index2word_dict = dict([(idx, word) for word, idx in a_word2index_dict.items()])
        pickle.dump(q_index2word_dict, open(path_q_index2word_dict, "wb"))
        pickle.dump(a_index2word_dict, open(path_a_index2word_dict, "wb"))
    else:
        q_word2index_dict = pickle.load(open(path_q_word2index_dict, "rb"))
        a_word2index_dict = pickle.load(open(path_a_word2index_dict, "rb"))
        q_index2word_dict = pickle.load(open(path_q_index2word_dict, "rb"))
        a_index2word_dict = pickle.load(open(path_a_index2word_dict, "rb"))

    q_dic_length = len(q_word2index_dict)
    a_dic_length = len(a_word2index_dict)

    idx_sentences_l1 = ct.sentences_to_indexes(filt_clean_sen_l1, q_word2index_dict)
    idx_sentences_l2 = ct.sentences_to_indexes(filt_clean_sen_l2, a_word2index_dict)

    max_length_l1 = ct.extract_max_length(idx_sentences_l1)
    max_length_l2 = ct.extract_max_length(idx_sentences_l2)
    data_set = ct.prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
    data = ((filt_clean_sen_l1, filt_clean_sen_l2), data_set, (max_length_l1, max_length_l2), (q_dic_length, a_dic_length))
    pickle.dump(data, open(path_data, "wb"))
       
    return (filt_clean_sen_l1, filt_clean_sen_l2), \
           data_set, \
           (max_length_l1, max_length_l2), \
           (q_dic_length, a_dic_length)
           

def build_train_model(dict_l1_length, dict_l2_length):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( dict_l1_length, 200 , mask_zero=True )(encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    decoder_embedding = tf.keras.layers.Embedding( dict_l2_length, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( dict_l2_length , activation=tf.keras.activations.softmax )
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

    file_Name = MODEL_WEIGHT_FILE
    
    '''
    이전과 학습조건이 동일하면
    if os.path.exists(file_Name):
        model.load_weights(MODEL_WEIGHT_FILE)
    '''
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    model.summary()
    
    return model

# onehot_encoding memory error 해결을 위해
def generate_batch(encoder_input_data, decoder_input_data):
    num_batches = len(encoder_input_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = encoder_input_data[start:end]
            decoder_input_data_batch = decoder_input_data[start:end]

            encoder_input = np.array(encoder_input_data_batch)
            decoder_input = np.array(decoder_input_data_batch)

            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, max_sentence_lengths[1]+2, dict_lengths[1]))    # 패딩 크기 추가
            for lineIdx, target_words in enumerate(decoder_input_data_batch):
                for idx, w in enumerate(target_words):
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w] = 1
            yield ([encoder_input, decoder_input], decoder_target_data_batch)


if __name__ == "__main__":
    '''
    file_Name = path_data
    if not os.path.exists(file_Name):
        _, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    else:
        data = pickle.load(open(path_data, "rb"))
        _, data_set, max_sentence_lengths, dict_lengths = data
    '''
    _, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)

    q_word2index_dict = pickle.load(open(path_q_word2index_dict, "rb"))
    a_word2index_dict = pickle.load(open(path_a_word2index_dict, "rb"))
    q_index2word_dict = pickle.load(open(path_q_index2word_dict, "rb"))
    a_index2word_dict = pickle.load(open(path_a_index2word_dict, "rb"))

    model = build_train_model(dict_lengths[0], dict_lengths[1])

    encoder_input_data = [data_set[i][0] for i in range(len(data_set))]
    decoder_input_data = [data_set[i][1] for i in range(len(data_set))]

    X_train, X_test, y_train, y_test = train_test_split(encoder_input_data, decoder_input_data, test_size=0.2, random_state=42)

    train_gen = generate_batch(X_train, y_train)
    test_gen = generate_batch(X_test, y_test)

    train_num_batches = len(X_train) // BATCH_SIZE
    test_num_batches = len(X_test) // BATCH_SIZE

    if not os.path.exists(TENSORBOARD):
        os.makedirs(TENSORBOARD)
        print("make directory")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("make directory")

    # checkpoint와 tensorboard를 위한 디렉토리는 일단 직접 만듬, 나중에 코드 수정
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_WEIGHT_FILE, save_best_only=True)
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD, histogram_freq=0, write_graph=True, write_images=True)
    
    model.fit_generator(generator= train_gen,
                    steps_per_epoch=train_num_batches,
                    epochs=15,
                    verbose=1,
                    validation_data=test_gen,
                    validation_steps=test_num_batches,
                    callbacks=[checkpoint, tbCallBack ])

    model.save_weights(MODEL_WEIGHT_FILE)
    
    
    