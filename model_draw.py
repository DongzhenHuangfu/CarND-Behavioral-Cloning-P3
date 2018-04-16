from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from keras.utils.visualize_util import plot

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, 
		input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,2,2,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch=5)

plot(model, to_file='modele.png', show_shapes=True)
