import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = tf.keras.models.load_model('metadata.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
try:
    img = cv2.imread('digits/d8.png')[:, :, 0]
    img = np.invert(np.array([img]))
    print(np.shape(img))
    prediction = model.predict(img)
    print(prediction)
    print(f'Probably {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
except Exception as e:
    print(e)
