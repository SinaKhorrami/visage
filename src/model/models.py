import os
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import model_from_yaml


class FERGModel96(object):
    """
    FERG-db Model (https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html)

    - 96*96 pixel grayscale images
    """

    def __init__(self, cfg):
        super().__init__()
        self.model_config = cfg['model']
        self.model_file_name = 'ferg96.h5py'
    
    def get_model_structure(self):
        model = Sequential()
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(self.model_config['input_dim'][0], self.model_config['input_dim'][1], 1)
        ))
        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(
            128,
            activation='relu'
        ))
        model.add(Dropout(0.5))
        model.add(Dense(
            len(self.model_config['classes']),
            activation='softmax'
        ))
        model.compile(
            optimizer=Adadelta(),
            loss=categorical_crossentropy,
            metrics=['accuracy']
        )

        yaml_string = model.to_yaml()
        print(yaml_string)

        return model

    def get_trained_model(self):
        model = self.get_model_structure()
        fn = os.path.join(os.path.dirname(__file__), self.model_file_name)
        model.load_weights(fn)

        return model

    def get_face_emotion(self, model, face_image):
        out = np.asarray(face_image.resize((self.model_config['input_dim'][0], self.model_config['input_dim'][1])), dtype='float32')
        out /= 255
        out = out.reshape((1, self.model_config['input_dim'][0], self.model_config['input_dim'][1], 1))
        predicted_class_index = model.predict(out)
        
        return self.model_config['classes'][np.argmax(predicted_class_index)]


class FERGModel(object):
    """
    FERG-db Model (https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html)

    - 96*96 pixel grayscale images
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = None
    
    def get_model_structure(self):
        model = Sequential()
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(self.config['input_dim'][0], self.config['input_dim'][1], 1)
        ))
        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(
            128,
            activation='relu'
        ))
        model.add(Dropout(0.5))
        model.add(Dense(
            len(self.config['classes']),
            activation='softmax'
        ))
        model.compile(
            optimizer=Adadelta(),
            loss=categorical_crossentropy,
            metrics=['accuracy']
        )

        return model

    def load_model_weights(self):
        self.model = self.get_model_structure()
        self.model.load_weights(os.path.join(os.path.dirname(__file__), self.config['file_name']))

    def get_face_emotion(self, face_image):
        out = np.asarray(face_image.resize((self.config['input_dim'][0], self.config['input_dim'][1])), dtype='float32')
        out /= 255
        out = out.reshape((1, self.config['input_dim'][0], self.config['input_dim'][1], 1))
        predicted_class_index = self.model.predict(out)

        return self.config['classes'][np.argmax(predicted_class_index)]
