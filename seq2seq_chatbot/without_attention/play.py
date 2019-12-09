import tensorflow as tf
import pickle
from train import build_train_model
from datasets import preprocess_sentence
tf.keras.backend.clear_session()

MODEL_WEIGHT_FILE = "./seq2seq.h5"
path_tokenizer = "./tokenizer.p"


tokenizer = pickle.load(open(path_tokenizer, "rb"))
VOCAB_SIZE = tokenizer.vocab_size + 2

model = build_train_model(VOCAB_SIZE)

model.load_weights(MODEL_WEIGHT_FILE)

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

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

for _ in range(10):
    question = input( 'Enter question : ' )
    predict(question)