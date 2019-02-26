from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU

import os

list_dir = os.listdir('dataset')
ignore = 0
if '.DS_Store' in list_dir:
    ignore = 1
    
num_classes = len(list_dir)-ignore

# Arquitetura da CNN
classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size=(3, 3),activation='linear',input_shape=(64,64,3),padding='same'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D((2, 2),padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='linear',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='linear',padding='same'))
classifier.add(LeakyReLU(alpha=0.1))  
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation='linear'))
classifier.add(Dropout(0.5))
classifier.add(LeakyReLU(alpha=0.1))                  
classifier.add(Dense(num_classes, activation='softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Carregando dataset
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset-test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


epochs = 5

history = classifier.fit_generator(
        training_set,
        steps_per_epoch=30//8,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=10//8,
        verbose = 1)


classifier.save('classifier.h5')