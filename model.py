import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
import random

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

del lines[0]

train_samples, validation_samples = train_test_split(lines,test_size=0.3)

def generator(lines, batch_size=32):
	num_samples = len(lines)
	while 1:
		random.shuffle(lines)
		for offset in range(0,num_samples, batch_size):
			batch_samples = lines[offset:offset+batch_size]

			images = []
			measurements = []
			for line in batch_samples:
				correction = 0.3

				source_path_center = line[0]
				source_path_left = line[1]
				source_path_right = line[2]

				filename_center = source_path_center.split('/')[-1]
				filename_left = source_path_left.split('/')[-1]
				filename_right = source_path_right.split('/')[-1]

				current_path_center  = './data/IMG/' + filename_center
				current_path_left = './data/IMG/' + filename_left
				current_path_right = './data/IMG/' + filename_right

				image_center = cv2.imread(current_path_center)
				image_left = cv2.imread(current_path_left)
				image_right = cv2.imread(current_path_right)

				images.extend([image_center, image_left, image_right])

				measurement_center = float(line[3])
				measurements.extend([measurement_center, measurement_center+correction, measurement_center-correction])

			augmented_images, augmented_measurements = [], []
			for image,measurement in zip(images,measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*-1.0)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
									validation_data=validation_generator, \
									nb_val_samples=len(validation_samples), nb_epoch=20, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('./loss.jpg')

model.save('model.h5')
