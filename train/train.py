# En este archivo vamos a crear la clase que nos premite entrenar el modelo, y guardar el modelo en el formato que queramos, en este caso en formato .h5
import tensorflow as tf
# from keras.models import load_model
# from keras.layers import Input
# from keras.utils import CustomObjectScope
from tensorflow import keras
import datetime
# from tensorflow.keras import layers
# from pathlib import Path

class Train():
    def __init__(self,epoch=50):
        
        self.epochs = epoch
        
        # self.load_model('model.h5')
    def train(self):
        # Vamos a comprobar si todos lo hilos estan parados para poder entrenar el modelo
        while True:
            threads = self.get_threads()
            # Vamos a comprobar que todos lo hilos estan parados
            if all(not thread.is_alive() for thread in threads):
                break
            
        
        # Vamos a entrenar el modelo
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath='best_model_weights.h5', save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        

        self.modelo.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            
        )