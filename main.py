# Este archivo sera el main del proyecto
import tensorflow as tf
# from keras.models import load_model
# from keras.layers import Input
# from keras.utils import CustomObjectScope
# from tensorflow import keras
# from tensorflow.keras import layers
import argparse
import os
import numpy as np
import train.train as train
import utils.load_date as load_date
import model.model as model
import utils.hilos as hilos
from pathlib import Path
from tensorflow.keras import layers




class Main(train.Train , load_date.Date, model.Model, hilos.Hilos):
    def __init__(self) -> None:
        
        self.model_path = None
        self.model_name = None
        self.model_version = None
        self.model_input_shape = None
        self.model_output_shape = None
        self.model_input_name = None
        self.model_output_name = None
        self.model_input_type = None
        self.model_output_type = None
        self.model_input_dtype = None
        self.model_output_dtype = None
        self.img_width= 244
        self.img_height= 244
        self.batch_size= 16
        self.data_dir= Path("/home/antonio/Documentos/train/archive/comprasnet_imagensacerto")
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
        
        
        
        self.start(self.model)
        self.start(self.load_date())
        
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--predict", type=str, default=None, help="predict the image")
    parser.add_argument("--retrain", action="store_true", help="retrain the model")
    args = parser.parse_args()
    Main().train()
    print("hola")
    if args.train:
        Main().train()
        


