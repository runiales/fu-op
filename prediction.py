# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalizing data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Building model
model = tf.keras.models.Sequential() # makes the model sequential
model.add(tf.keras.layers.Flatten()) # converts the images to single pixel rows which serve as input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden layers 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fitting
model.fit(x_train, y_train, epochs=3)

# Testing
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

# Saving the model
model.save('mnist.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

