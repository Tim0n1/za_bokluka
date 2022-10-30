from model import mnist
from model import np
from model import plt
from model import cv2
from model import tf
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model('metadata.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
try:
    img = cv2.imread('digit recognition/digits/d8.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(prediction)
    print(f'Probably {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
except Exception as e:
    print(e)
