 #importing the Keras libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Initialing the CNN
classifier = Sequential()

# Adding first Convolutio Layer
classifier.add(Conv2D(32, (3,  3), input_shape=(64, 64, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolution layer
classifier.add(Conv2D(32,( 3,  3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding 3rd Concolution Layer
classifier.add(Conv2D(32, (3,  3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation='softmax'))

# Compiling The CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Part 2 Fittting the CNN to the image
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('mydata/training_set',target_size=(64, 64),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('mydata/test_set',target_size=(64, 64),batch_size=32,class_mode='categorical')

history = classifier.fit_generator(training_set,steps_per_epoch=1422,epochs=15,validation_data=test_set,validation_steps=204)

# Saving the model
classifier.save('Trained_model.h5')

#plotting graph

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

