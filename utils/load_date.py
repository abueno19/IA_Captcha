# Aqui vamos a cargar una clase para cargar los datos
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os

class Date():
    def __init__(self,path,batch_size=16,img_width=224, img_height=224):
        self.path=path
        self.batch_size=batch_size
        self.img_width=img_width
        self.img_width=img_height
        
        
    def load_date(self):
        
        
        # datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     rotation_range = 30,
        #     width_shift_range = 0.25,
        #     height_shift_range = 0.25,
        #     shear_range = 15,
        #     zoom_range = [0.5, 1.5],
        #     validation_split=0.2 #20% para pruebas
        # )
        # self.train_dataset=datagen.flow_from_directory('/home/antonio/mis_repos/Ia2/train/data', target_size=(self.img_width,self.img_height),
        #                                              batch_size=self.batch_size, shuffle=True, subset='training',
        #                                              class_mode='categorical', # Obtener etiquetas categóricas
        #                                              classes=None, # Las clases se obtendrán automáticamente de los nombres de los archivos
        #                                              )
        # self.valid_dataset=datagen.flow_from_directory('/home/antonio/mis_repos/Ia2/train/data', target_size=(self.img_width,self.img_height),
        #                                             batch_size=self.batch_size, shuffle=True, subset='training',
        #                                             class_mode='categorical', # Obtener etiquetas categóricas
        #                                             classes=None, # Las clases se obtendrán automáticamente de los nombres de los archivos
        #                                              )
        # print(self.train_dataset.class_indices)
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = (
            train_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        valid_dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))
        self.valid_dataset = (
            valid_dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    def split_data(self, images, labels, train_size=0.9, shuffle=True):
        size = len(images)
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        train_samples = int(size * train_size)
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        
                
        return x_train, x_valid, y_train, y_valid

    def encode_single_sample(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}
