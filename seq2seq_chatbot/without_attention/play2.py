import tensorflow as tf
import pickle
from model import build_train_model, make_inference_models
from datasets import preprocess_sentence
tf.keras.backend.clear_session()

MODEL_WEIGHT_FILE = "./model.h5"
path_voc_size = "./voc_size.p"
path_tokenizer = "./tokenizer.p"

VOCAB_SIZE = pickle.load(open(path_voc_size, "rb"))
tokenizer = pickle.load(open(path_tokenizer, "rb"))

model = build_train_model(VOCAB_SIZE)

model.load_weights(MODEL_WEIGHT_FILE)

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    
    START_TOKEN = [VOCAB_SIZE - 2]
    END_TOKEN = [VOCAB_SIZE - 1]

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)
    MAX_LENGTH = 40
    
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

enc_model , dec_model = make_inference_models(VOCAB_SIZE)

for _ in range(10):
    question = input( 'Enter question : ' )
    question = preprocess_sentence(question)

    START_TOKEN = [VOCAB_SIZE - 2]
    END_TOKEN = [VOCAB_SIZE - 1]

    question = tf.expand_dims(
        START_TOKEN + tokenizer.encode(question) + END_TOKEN, axis=0)

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