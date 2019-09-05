# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

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
new_model = tf.keras.models.load_model('mnist.model')

# Asks for a prediction

question = "y"

while question != "n":
    question = input("Do you want to make a prediction? (y/n) ")
    if question == "y":
        predictions = new_model.predict(x_test)
        randomnumber = random.randint(1,10000)
        print("I predict the following image represents this integer:", np.argmax(predictions[randomnumber]))
        print("The correct answer is:", y_test[randomnumber])
        plt.imshow(x_test[randomnumber],cmap=plt.cm.binary)
        plt.show()
    elif question == "n":
        print("Thank you.")
        break
    else:
        print("Please type in 'y' or 'n'.")

