import tensorflow as tf
import numpy as np
import pickle
from datasets import tokenize_and_filter

# 파일 기준 경로
path_questions = "./questions.p"
path_answers = "./answers.p"
path_tokenizer = "./tokenizer.p"
MODEL_WEIGHT_FILE = "./seq2seq.h5"

def build_train_model(vocab_size):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( vocab_size, 512 , mask_zero=True )(encoder_inputs)
    _ , state_h , state_c = tf.keras.layers.LSTM( 512 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    decoder_embedding = tf.keras.layers.Embedding( vocab_size, 512 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 512 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( vocab_size , activation=tf.keras.activations.softmax )
    output = decoder_dense ( decoder_outputs )
    
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    model.summary()
    
    return model


# onehot_encoding memory error 해결을 위해
def generate_batch(encoder_input_data, decoder_input_data, BATCH_SIZE):
    num_batches = len(encoder_input_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = encoder_input_data[start:end]
            decoder_input_data_batch = decoder_input_data[start:end]

            encoder_input = np.array(encoder_input_data_batch)
            decoder_input = np.array(decoder_input_data_batch)

            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, 40, VOCAB_SIZE))
            for lineIdx, target_words in enumerate(decoder_input_data_batch):
                for idx, w in enumerate(target_words):
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w] = 1
            yield ([encoder_input, decoder_input], decoder_target_data_batch)


if __name__ == "__main__":
    questions = pickle.load(open(path_questions, "rb"))
    answers = pickle.load(open(path_answers, "rb"))
    tokenizer = pickle.load(open(path_tokenizer, "rb"))

    VOCAB_SIZE = tokenizer.vocab_size + 2
    model = build_train_model(VOCAB_SIZE)
    
    BATCH_SIZE = 64
    train_gen = generate_batch(questions, answers, BATCH_SIZE)

    train_num_batches = len(questions) // BATCH_SIZE

    model.fit_generator(generator= train_gen,
                    steps_per_epoch=train_num_batches,
                    epochs=20,
                    verbose=1)

    model.save_weights(MODEL_WEIGHT_FILE)

