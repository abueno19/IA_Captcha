# Aqui vamos a cargar una clase para cargar los datos
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.preprocessing import LabelEncoder


class Date():
    def __init__(self,path,batch_size=16,img_width=224, img_height=224):
        self.path=path
        self.batch_size=batch_size
        self.img_width=img_width
        self.img_width=img_height
        
        
    def load_date(self):
        # print("Dimensiones de las imágenes:", self.images.shape)
        # print("Dimensiones de las etiquetas:", self.labels.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = (
            train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        valid_dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))
        self.valid_dataset = (
            valid_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
  
    def load_date2(self):
        image_names = os.listdir(self.datadir)
        labels = [name.split('.')[0] for name in image_names]
        
        
        # crea un dataframe con las rutas de archivo y las etiquetas
        df = pd.DataFrame({
            'filename': image_names,
            'label': labels
        })
        
        # crea un generador de datos de imagen a partir del dataframe
        # datagen = ImageDataGenerator(rescale=1./255)
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        self.train_dataset = datagen.flow_from_dataframe(
            df,
            directory=self.datadir,
            x_col='filename',
            y_col='label',
            target_size=(224, 224),
            batch_size=32,
            class_mode='input',
            subset='training'
        )
        

        self.valid_dataset = datagen.flow_from_dataframe(
            df,
            directory=self.datadir,
            x_col='filename',
            y_col='label',
            target_size=(224, 224),
            batch_size=32,
            class_mode='input',
            subset='validation'
        )
        # to_categorical(self.train_dataset.labels, num_classes=len(self.train_dataset.class_indices))
        # to_categorical(self.valid_dataset.labels, num_classes=len(self.train_dataset.class_indices))
    def split_data(self, images, labels, train_size=0.9, shuffle=True):
        size = len(images)
        indices = np.arange(size)
        # print("Dimensiones de las imágenes antes de dividir los datos:", images.shape)
        # print("Dimensiones de las etiquetas antes de dividir los datos:", labels.shape)
        if shuffle:
            np.random.shuffle(indices)
        train_samples = int(size * train_size)
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        # print("Dimensiones de los datos de entrenamiento (x_train, y_train):", x_train.shape, y_train.shape)
        # print("Dimensiones de los datos de validación (x_valid, y_valid):", x_valid.shape, y_valid.shape)
        
                
        return x_train, x_valid, y_train, y_valid

    def encode_single_sample(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=self.img_channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        # print("Dimensiones de la imagen después del redimensionamiento:", img.shape)
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # print("Codificación numérica de la etiqueta:", label)

        return {"img_input": img, "label": label}
    def decode_batch_predictions(self,pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text