from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def nvidia_cnn(image_size=(200,100,3), pretrained_weights=None):
    print("Generating UNET-Small Model using settings: ")
    print("\t- image_size= " + str(image_size))

    inputs = Input(image_size)
    norm = BatchNormalization(epsilon=0.001,mode=2, axis=1)(input)
    conv = Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2))(norm)
    conv = Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2))(conv)
    conv = Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2))(conv)
    conv = Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1))(conv)
    conv = Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1))(conv)
    flat = Flatten()(conv)
    dense = Dense(1164, activation='relu')(flat)
    dense = Dense(100, activation='relu')(dense)
    dense = Dense(50, activation='relu')(dense)
    dense = Dense(10, activation='relu')(dense)
    out = Dense(1, activation='tanh')(dense)

    model = Model(input=inputs, output=out)
    adam = Adam(lr=0.0001)
    model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])

    print(model.summary())

    if(pretrained_weights):
        print('Loading Weights from ' + pretrained_weights)
        model.load_weights(pretrained_weights)
