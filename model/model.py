# Aqui vamos a cargar el modelo
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.layers import Input
from keras.utils import CustomObjectScope
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.applications import MobileNetV2
class Model():
    url = "./tf2-preview_mobilenet_v2_feature_vector_4"
    # url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    # self.input_img = hub.KerasLayer(self.url, input_shape=(224, 224, 3), trainable=False, dtype="float32")

    def model(self):
        # Inputs to the model
        self.input_img = layers.Input(
            shape=(self.img_width, self.img_height, self.img_channels), name="img_input", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        mobilnet = hub.KerasLayer(self.url, input_shape=(self.img_width, self.img_height, self.img_channels), dtype="float32", name="img")
        x = mobilnet(self.input_img)
        # First conv block
        # x = layers.Conv2D(
        #     32,
        #     (3, 3),
        #     activation="relu",
        #     kernel_initializer="he_normal",
        #     padding="same",
        #     name="Conv1",
        # )(self.input_img)
        # x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # # Second conv block
        # x = layers.Conv2D(
        #     64,
        #     (3, 3),
        #     activation="relu",
        #     kernel_initializer="he_normal",
        #     padding="same",
        #     name="Conv2",
        # )(x)
        # x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        
        x = layers.Reshape((-1,1))(x)
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
        output = layers.Lambda(lambda x: tf.transpose(x, perm=[1, 0, 2]))(x)
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
        model.summary()
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
        self.modelo = keras.Sequential(
            [
                mobilnet,
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.RepeatVector(10),
                layers.Bidirectional(
                    layers.LSTM(128, return_sequences=True, dropout=0.25,activation="tanh")
                    ),
                layers.LSTM(64,  dropout=0.25,activation="tanh"),
                
                # layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax'),
                layers.Reshape((1, -1)),
                layers.TimeDistributed(layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax')), # 26 clases para las letras del alfabeto inglés
            ]
        )
        
        self.modelo.summary()
        # Vamos a compilar el modelo 
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["accuracy"])
    def model4(self):
        mobilnet = hub.KerasLayer(self.url, input_shape=(224, 224, 1), dtype="float32", name="img")
        model = keras.Sequential([
            mobilnet,
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.RepeatVector(10),
            layers.LSTM(128, return_sequences=True, dropout=0.25, activation="tanh"),
            layers.LSTM(64, dropout=0.25, activation="tanh"),
            layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax')
        ])
        inputs = layers.Input(shape=(None, 224, 224, 3))
        x = model(inputs)
        outputs = layers.Lambda(lambda x: tf.keras.backend.ctc_decode(x, input_length=tf.fill([tf.shape(x)[0]], tf.shape(x)[1])[0], greedy=True)[0][0])(x)
        full_model = keras.models.Model(inputs, outputs)
        full_model.summary()
        # Compilamos el modelo con la pérdida de CTC (Connectionist Temporal Classification)
        loss = lambda y_true, y_pred: y_pred
        full_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss)
        return full_model
    def model5(self):
        self.characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.num_to_char = StringLookup(vocabulary=list(self.characters), mask_token=None)
        self.char_to_num = StringLookup(vocabulary=self.num_to_char.get_vocabulary(), mask_token=None, invert=True)
        # Inputs to the model
        self.input_img = layers.Input(
            shape=(self.img_width, self.img_height, self.img_channels), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")
        mobilnet = hub.KerasLayer(self.url, input_shape=(224, 224, 3), dtype="float32", name="img")
        x = mobilnet(self.input_img)
        x = layers.Reshape((1, -1))(x)
        # x = layers.Flatten()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.25, activation="relu")(x)
        x = layers.LSTM(64, dropout=0.25, activation="relu")(x)
        x = layers.Flatten()(x) 
       
        x = layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax')(x)

        # Add CTC layer for calculating CTC loss at each step
        # Add CTC loss layer to the model
        
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[self.input_img, labels], outputs=output, name="ocr_model_v1"
        )

        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=0.001)

        # Compile the model and return
        model.compile(optimizer=opt, loss=lambda y_true, y_pred: y_pred)
        self.model = model
    def modelo6(self):
        # Entrada de imagen
       
        input_img = layers.Input(shape=(self.img_width, self.img_height, self.img_channels), name="input_img")

        # Capa CNN (MobileNet)
        mobilnet = hub.KerasLayer(self.url, input_shape=(224, 224, 3), dtype="float32", name="mobilenet")
        x = mobilnet(input_img)
        
        # Aplanar la salida de la capa KerasLayer
        x = layers.Flatten()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Reshape((1, 128))(x)

        # Capa de entrada de texto
        input_labels = layers.Input(name="input_labels", shape=(None,), dtype="float32")
        
        label_embedding = layers.Embedding(
            input_dim=len(self.char_to_num.get_vocabulary()) + 1, output_dim=128
        )(input_labels)
        label_embedding = layers.Reshape((1, -1))(label_embedding) 

        # Capa RNN (GRU)
        x = layers.Concatenate(axis=1)([x, label_embedding])
        x = layers.GRU(256, return_sequences=True)(x)
        x = layers.GRU(128, return_sequences=True)(x)

        # Capa de salida
        output = layers.TimeDistributed(layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation="softmax"))(x)

        # Modelo final
        
        self.modelo = keras.models.Model(
            inputs=[input_img, input_labels], outputs=output, name="ocr_model"
        )

        self.modelo.summary()
        # Compilar el modelo
        self.modelo.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    def modelo7(self):
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
        )

        # Crear la arquitectura del modelo
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Reshape((-1, 1024))(x)
        

        # Utilizar un enfoque de secuencia a secuencia
        sequence_prediction = layers.LSTM(128, return_sequences=True)(x)
        character_prediction = layers.TimeDistributed(layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax'), name='character_prediction')(sequence_prediction)

        model = keras.models.Model(inputs=inputs, outputs=character_prediction)

        # Utilizar entropía cruzada como función de pérdida
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        self.modelo = model
        self.modelo.summary()

    def modelo8(self):
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
        )
        x = base_model.output
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.LSTM(256)(x)
        output = layers.Dense(len(self.char_to_num.get_vocabulary()) + 1, activation='softmax')(x)

        model = keras.models.Model(inputs=base_model.input, outputs=output)

        
        model.summary()
        self.modelo = model
        self.modelo.compile(optimizer=keras.optimizers.Adam(),
            loss='categorical_crossentropy', metrics=['accuracy'])



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

class CTCLayer3(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, labels, y_pred):
        # Compute the training-time loss value and add it
        
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(labels)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(labels)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(labels, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

class CTCLayer4(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # as a metric to track during training.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        # CTC loss is implemented in a lambda layer
        loss = tf.reduce_mean(
            self.loss_fn(y_true, y_pred, input_length, label_length)
        )
        self.add_loss(loss)

        # Use the mean loss over the batch for logging
        return loss
