import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_data(batch_size):
    while True:
        y_train = np.random.choice([0,1], (batch_size, 1))
        x_train = np.random.randn(batch_size, 1) *0.01 + (2 * y_train - 1)
        yield x_train, y_train

datagen = get_data(32)

# example data generation.
x,y = next(datagen)
print(x,y)

# build a simple model
model = Sequential([Dense(1, activation = "sigmoid", input_shape = (1,))])
model.compile(loss = "binary_crossentropy", optimizer = "sgd")

# fit a model using data generator
model.fit_generator(datagen, steps_per_epoch = 1000, epochs = 10)

print(model.evaluate_generator(datagen, steps = 1000))
