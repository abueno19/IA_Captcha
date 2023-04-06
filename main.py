# Este archivo sera el main del proyecto
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input
from keras.utils import CustomObjectScope
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os
import numpy as np
import train.train as train




class Main(train.Train):
    def __init__(self) -> None:
        self.model = None
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
        
        
        




    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true", help="train the model")
        parser.add_argument("--predict", type=str, default=None, help="predict the image")
        parser.add_argument("--retrain", action="store_true", help="retrain the model")
        args = parser.parse_args()

        


