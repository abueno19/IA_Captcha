# Aqui vamos a cargar una clase para cargar los datos
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Date():
    def __init__(self,path,batch_size=16,img_width=224, img_height=224):
        self.path=path
        self.batch_size=batch_size
        self.img_width=img_width
        self.img_width=img_height
        
        
    def load_date(self):
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range = 30,
            width_shift_range = 0.25,
            height_shift_range = 0.25,
            shear_range = 15,
            zoom_range = [0.5, 1.5],
            validation_split=0.2 #20% para pruebas
        )
        data_text=datagen.flow_from_directory('./train/data', target_size=(self.img_width,self.img_height),
                                                     batch_size=self.batch_size, shuffle=True, subset='training',
                                                     class_mode='categorical', # Obtener etiquetas categóricas
                                                     classes=None, # Las clases se obtendrán automáticamente de los nombres de los archivos
                                                     )
        data_prueba=datagen.flow_from_directory('./train/data', target_size=(self.img_width,self.img_height),
                                                    batch_size=self.batch_size, shuffle=True, subset='training',
                                                    class_mode='categorical', # Obtener etiquetas categóricas
                                                    classes=None, # Las clases se obtendrán automáticamente de los nombres de los archivos
                                                     )
        return  data_text,data_prueba