import data_utils
import tensorflow as tf
import numpy as np
import corpora_tools as ct
import pickle

path_q_word2index_dict = "./dictionary/q_word2index_dict.p"
path_a_word2index_dict = "./dictionary/a_word2index_dict.p"
path_q_index2word_dict = "./dictionary/q_index2word_dict.p"
path_a_index2word_dict = "./dictionary/a_index2word_dict.p"
path_data = "./dictionary/data.p"
BATCH_SIZE = 128
TENSORBOARD = './TensorBoard'
model_dir = "./chat/chatbot_model"
MODEL_WEIGHT_FILE = model_dir + "/model.h5"

# 한문장만
def sent_to_indexs(sentence, indexed_dictionary):
    idx_sent = []
    for word in sentence:
        try:
            idx_sent.append(indexed_dictionary[word])
        except KeyError:
            idx_sent.append(data_utils.UNK_ID)
    return idx_sent

# 특수 기호를 이용하여 padding과 출력 시퀀스의 시작에 go, 마지막에 eos 추가
def prepare_one_sentence(sentence, len_l1):
    data_set = []
    padding_l1 = len_l1 - len(sentence)
    pad_sentence_l1 = ([data_utils.PAD_ID]*padding_l1) + sentence

    data_set.append([pad_sentence_l1])

    return data_set

def make_inference_models(dict_l1_length, dict_l2_length):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( input_dim=dict_l1_length, output_dim=200 , input_length = 30, mask_zero=True )(encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]
    
    decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    decoder_embedding = tf.keras.layers.Embedding( dict_l2_length, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( dict_l2_length , activation=tf.keras.activations.softmax )
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.load_weights(MODEL_WEIGHT_FILE)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


data = pickle.load(open(path_data, "rb"))
_, data_set, max_sentence_lengths, dict_lengths = data

q_word2index_dict = pickle.load(open(path_q_word2index_dict, "rb"))
a_index2word_dict = pickle.load(open(path_a_index2word_dict, "rb"))

enc_model , dec_model = make_inference_models(dict_lengths[0], dict_lengths[1])

for _ in range(10):
    question = input( 'Enter question : ' )
    question = question.split(" ")
    clean_question = ct.clean_sentence(question)
    test_input_encoder=prepare_one_sentence(sent_to_indexs( clean_question,  q_word2index_dict),30)
    states_values = enc_model.predict( test_input_encoder )
    
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = 1 # GO_id
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )

        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = a_index2word_dict[sampled_word_index]

        if sampled_word_index == 2 or len(decoded_translation.split()) > max_sentence_lengths[1]:
            stop_condition = True
        else:
            decoded_translation += ' {}'.format( sampled_word )
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation )