# Aqui vamos a cargar el modelo
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.layers import Input
from keras.utils import CustomObjectScope
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
class Model():
    url = "./tf2-preview_mobilenet_v2_feature_vector_4"
    # url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" 
    # self.input_img = hub.KerasLayer(self.url, input_shape=(224, 224, 3), trainable=False, dtype="float32")
        
    def model(self):
        # Inputs to the model
        self.input_img = layers.Input(
            shape=(self.img_width, self.img_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(self.input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(
            len(self.char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
        )(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[self.input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt)
        self.modelo=model
    def save_model(self):
        self.modelo.save('./model/model.h5')
    def load_model(self):
        with CustomObjectScope({'CTCLayer': CTCLayer}):
            self.modelo = load_model('./model/model.h5')
    def model2(self):
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        mobilnet = hub.KerasLayer(self.url, input_shape=(224, 224, 3), dtype="float32", name="img")
        mobilnet.trainable = False
        self.modelo = tf.keras.Sequential([
            # Capa del modelo preentrenado
            mobilnet,

            # Capa de aplanamiento
            layers.Flatten(),
            # Capa completamente conectada 1
            layers.Dense(128, activation='relu'),
            # Capa de remodelado para agregar una dimensión de secuencia
            layers.Reshape((1, -1)),
            # Capa LSTM para modelar secuencias
            layers.LSTM(256, return_sequences=True), # return_sequences=True para generar una secuencia de salida
            # Capa completamente conectada 2 para la salida
            layers.TimeDistributed(layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax')), # 26 clases para las letras del alfabeto inglés

        ])
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer2(name="ctc_loss")(labels, self.modelo.output)

        # Define the model
        
        self.modelo = keras.models.Model(
            inputs=[self.modelo.input, labels], outputs=output, name="ocr_model_v1"
        )

        self.modelo.summary()
        self.modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    def model3(self):
        mobilnet = hub.KerasLayer(self.url, input_shape=(224, 224, 3), dtype="float32", name="img")
        model = keras.Sequential(
            [
                mobilnet,
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.RepeatVector(10),  # asumimos una longitud máxima de 10 caracteres en el captcha
                layers.LSTM(128, return_sequences=True),
                layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation="softmax"),
            ]
        )
        model.summary()
        self.modelo=model
        self.modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

        
        

class CTCLayer(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
class CTCLayer2(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        # Vamos a recoger el input_length de la capa anterior
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        
        return y_pred



