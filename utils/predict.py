import tensorflow as tf
from tensorflow import keras

class Predict():
    def predict(self,img_path):
        # Vamos a a predecir la imagen usando el modelo que hemos entrenado
        while True:
            threads = self.get_threads()
            # Vamos a comprobar que todos lo hilos estan parados
            if all(not thread.is_alive() for thread in threads):
                break
        
        # Vamos a cargar el modelo
        prediction_model = keras.models.Model(
            self.modelo.get_layer(name="image").input, self.modelo.get_layer(name="dense2").output
        )
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.expand_dims(img, 0)
        # Vamos a predecir
        preds=prediction_model.predict(img)
        
        pred_texts = self.decode_batch_predictions(preds)
        print("La predicción es: ",pred_texts)