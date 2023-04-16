# Aqui vamos a cargar una clase para cargar los datos
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow import keras
import os

class Date():
    def __init__(self,path,batch_size=16,img_width=224, img_height=224):
        self.path=path
        self.batch_size=batch_size
        self.img_width=img_width
        self.img_width=img_height
        
        
    def load_date(self):
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
