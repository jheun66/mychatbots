import tensorflow as tf

model_dir = "./chat/chatbot_model"
MODEL_WEIGHT_FILE = model_dir + "/model.h5"

def build_train_model(vocab_size):
    encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    encoder_embedding =  tf.keras.layers.Embedding( vocab_size, 200 , mask_zero=True )(encoder_inputs)
    _ , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype='int32',)
    decoder_embedding = tf.keras.layers.Embedding( vocab_size, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( vocab_size , activation=tf.keras.activations.softmax )
    output = decoder_dense ( decoder_outputs )
    
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    model.summary()
    
    return model