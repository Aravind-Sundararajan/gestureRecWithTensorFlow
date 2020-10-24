import tensorflow as tf #importing the tensorflow library
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

print(tf.__version__)

nFrames = 500

# you can assign any value to them

with open('extension_task_data.csv', newline='\n') as f:
    reader = csv.reader(f)
    extension_task_data = [[float(elem) for elem in list] for list in reader]
    
with open('extension_task_time.csv', newline='\n') as f:
    reader = csv.reader(f)
    extension_task_time = [[float(elem) for elem in list] for list in reader]

#print(extension_task_time)

with open('flexion_task_data.csv', newline='\n') as f:
    reader = csv.reader(f)
    flexion_task_data = [[float(elem) for elem in list] for list in reader]
    
with open('flexion_task_time.csv', newline='\n') as f:
    reader = csv.reader(f)
    flexion_task_time = [[float(elem) for elem in list] for list in reader]

#print(flexion_task_time)

training_extension_input = [[time,data] for time,data in zip(extension_task_time, extension_task_data)]
training_flexion_input = [[time,data] for time,data in zip(flexion_task_time, flexion_task_data)]
training_extension_output = [[1] for elem in training_extension_input]
training_flexion_output = [[0] for elem in training_flexion_input]

x_train = np.array(training_extension_input + training_flexion_input)
y_train = np.array(training_extension_output + training_flexion_output)

with open('extension_task_test_data.csv', newline='\n') as f:
    reader = csv.reader(f)
    extension_task_test_data = [[float(elem) for elem in list] for list in reader]
    
with open('extension_task_test_time.csv', newline='\n') as f:
    reader = csv.reader(f)
    extension_task_test_time = [[float(elem) for elem in list] for list in reader]

with open('flexion_task_test_data.csv', newline='\n') as f:
    reader = csv.reader(f)
    flexion_task_test_data = [[float(elem) for elem in list] for list in reader]
    
with open('flexion_task_test_time.csv', newline='\n') as f:
    reader = csv.reader(f)
    flexion_task_test_time = [[float(elem) for elem in list] for list in reader]

with open('random_task_test_data.csv', newline='\n') as f:
    reader = csv.reader(f)
    random_task_test_data = [[float(elem) for elem in list] for list in reader]
    
with open('random_task_test_time.csv', newline='\n') as f:
    reader = csv.reader(f)
    random_task_test_time = [[float(elem) for elem in list] for list in reader]

testing_extension_input = [[time,data] for time,data in zip(extension_task_test_time, extension_task_test_data)]
testing_flexion_input = [[time,data] for time,data in zip(flexion_task_test_time, flexion_task_test_data)]
testing_random_input = [[time,data] for time,data in zip(random_task_test_time, random_task_test_data)]

testing_extension_output = [[1] for elem in testing_extension_input]
testing_flexion_output = [[0] for elem in testing_flexion_input]
    
x_test = np.array(testing_extension_input + testing_flexion_input)
y_test = np.array(testing_extension_output + testing_flexion_output)

print(np.array(x_train).shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, nFrames)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])
print(np.array(x_train[:1]).shape)
predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
model,
tf.keras.layers.Softmax()
])


print("predicting types from the test data set to predictions.csv:")
solutions = np.array(probability_model(x_test))
#for elem in solutions:
#    print(elem)
solutions_strings =[['extension',elem[0]] if elem[0] < .5 else ['flexion',elem[0]] for elem in solutions]
pd.DataFrame(solutions_strings).to_csv("./predictions.csv", header=None, index=None)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

print("predicting types from the random test data set to predictions2.csv:")
projections = probability_model.predict(testing_random_input)
projection_strings =[['extension',elem[0]] if elem[0] < .5 else ['flexion',elem[0]] for elem in projections]
pd.DataFrame(projection_strings).to_csv("./predictions2.csv", header=None, index=None)