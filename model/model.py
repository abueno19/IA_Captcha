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
    url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        
        
        
    def model(self):
        # Cargar el modelo MobileNetV2 pre-entrenado desde TensorFlow Hub
        mobilenetv2 = hub.KerasLayer(self.url, input_shape=(224, 224, 3))
        mobilenetv2.trainable = False # Deshabilitar el entrenamiento de las capas de MobileNetV2

        # Inputs del modelo
        self.input_img = layers.Input(shape=(self.img_width, self.img_height, 1), name="image", dtype="float32")
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # Conectar la capa MobileNetV2 como entrada al modelo
        x = mobilenetv2(self.input_img)

        # Reshape para adaptarse a la salida de MobileNetV2
        x = layers.Reshape(target_shape=(-1, 1280))(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Capa de salida
        x = layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

        # Agregar capa CTC para calcular la pérdida CTC en cada paso
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Definir el modelo
        model = keras.models.Model(inputs=[self.input_img, labels], outputs=output, name="ocr_model_v1")

        # Compilar el modelo y devolverlo
        model.compile(optimizer=keras.optimizers.Adam())

        return model
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