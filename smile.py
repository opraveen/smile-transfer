import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from wandb.wandb_keras import WandbKerasCallback
import wandb
import smiledataset

run = wandb.init()
config = run.config

config.epochs=10

learning_rate = 0.05
decay_rate = learning_rate / config.epochs
momentum = 1.0
#lr = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
lr = Adam()


# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_X /= 255.0
test_X /= 255.0

model = Sequential()
model.add(Conv2D(64,3,activation="relu", input_shape=(img_rows,img_cols,1),padding='same'))
model.add(Conv2D(96,3,activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(128,3,activation="relu",padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,3,activation="relu",padding='same'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax') )
model.compile(loss='categorical_crossentropy', optimizer=lr, metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y,
    epochs=config.epochs, verbose=1, batch_size=32,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model.save("smile.h5")
