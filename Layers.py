import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 32,activation = 'relu'))
classifier.add(Dense(units = 64,activation = 'relu'))
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 256,activation = 'relu'))
classifier.add(Dense(units = 256,activation = 'relu'))
classifier.add(Dense(units = 5,activation = 'softmax'))


classifier.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
classifier.summary()