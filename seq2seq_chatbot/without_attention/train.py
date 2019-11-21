import tensorflow as tf
import numpy as np
import pickle
from datasets import tokenize_and_filter
from model import build_train_model, make_inference_models


path_questions = "./questions.p"
path_answers = "./answers.p"
path_voc_size = "./voc_size.p"
MODEL_WEIGHT_FILE = "./model.h5"

BATCH_SIZE = 64

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

            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, 40, VOCAB_SIZE))
            for lineIdx, target_words in enumerate(decoder_input_data_batch):
                for idx, w in enumerate(target_words):
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w] = 1
            yield ([encoder_input, decoder_input], decoder_target_data_batch)


if __name__ == "__main__":

    #questions, answers, VOCAB_SIZE = tokenize_and_filter()

    #pickle.dump(questions, open(path_questions, "wb"))
    #pickle.dump(answers, open(path_answers, "wb"))
    #pickle.dump(VOCAB_SIZE, open(path_voc_size, "wb"))


    questions = pickle.load(open(path_questions, "rb"))
    answers = pickle.load(open(path_answers, "rb"))
    VOCAB_SIZE = pickle.load(open(path_voc_size, "rb"))

    model = build_train_model(VOCAB_SIZE)

    train_gen = generate_batch(questions, answers)

    train_num_batches = len(questions) // BATCH_SIZE

    model.fit_generator(generator= train_gen,
                    steps_per_epoch=train_num_batches,
                    epochs=20,
                    verbose=1)

    model.save_weights(MODEL_WEIGHT_FILE)

