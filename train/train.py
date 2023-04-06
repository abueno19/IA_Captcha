# En este archivo vamos a crear la clase que nos premite entrenar el modelo, y guardar el modelo en el formato que queramos, en este caso en formato .h5
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input
from keras.utils import CustomObjectScope
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
class Train():
    def __init__(self,data_dir="/home/antonio/Documentos/train/archive/comprasnet_imagensacerto",api=False , batch_size=16, img_width=200, img_height=50, downsample_factor=4,epoch=50):
        if not api:
            self.model=None
            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.img_width = img_width
            self.img_height = img_height
            self.downsample_factor = downsample_factor
            self.images = sorted(list(map(str, list(self.data_dir.glob("*.png")))))
            self.labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in self.images]
            self.characters = set(char for label in self.labels for char in label)
            self.characters = sorted(list(self.characters))
            self.max_length = max([len(label) for label in self.labels])
            self.labels = [label + " " * (self.max_length - len(label)) for label in self.labels]
            self.char_to_num = layers.StringLookup(vocabulary=list(self.characters), mask_token=None)
            self.num_to_char = layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
            )
            self.x_train, self.x_valid, self.y_train, self.y_valid = self.split_data(np.array(self.images), np.array(self.labels))
            self.epochs = epoch
        else:
            # Vamos a cargar el modelo
            self.load_model('model.h5')
    def train(self):
        # Vamos a entrenar el modelo
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath='best_model_weights.h5', save_best_only=True, save_weights_only=True)
        ]
        self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
        )