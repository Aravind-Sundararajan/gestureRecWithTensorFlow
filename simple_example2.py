import tensorflow as tf #importing the tensorflow library
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# you can assign any value to them

training_input = [
    [.1, .2, .3],
    [.2, .4, .6],
    [0, .1, .2],
    [0, -.1, -.2],
    [-.1, -.2, -.3],
]

training_output = [
    [1],
    [1],
    [1],
    [0],
    [0],
]

plt.figure(figsize=(10,10))
for i in range(5):
    plt.plot(training_input[i])
plt.show()


model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=(3,))
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

trained_model = model.fit(training_input, training_output, epochs=1000, verbose=False)
print("Finished training the model")

print(model.predict([[.1,-.1,-.2]]))