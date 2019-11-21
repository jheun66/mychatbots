import tensorflow as tf

model_dir = "./chat/chatbot_model"
MODEL_WEIGHT_FILE = model_dir + "/model.h5"

def build_train_model(dict_l1_length, dict_l2_length):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( dict_l1_length, 200 , mask_zero=True )(encoder_inputs)
    _ , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    decoder_embedding = tf.keras.layers.Embedding( dict_l2_length, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( dict_l2_length , activation=tf.keras.activations.softmax )
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    model.summary()
    
    return model

def make_inference_models(dict_l1_length, dict_l2_length):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( input_dim=dict_l1_length, output_dim=200 , input_length = 30, mask_zero=True )(encoder_inputs)
    _ , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
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