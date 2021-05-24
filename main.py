from flask import Flask, render_template, Response, session, redirect, url_for, request
import cv2, sys

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('index.html', result=get_frame())
    elif request.method == 'GET':
        return render_template('index.html', result='Result . . . ')


def get_frame():
    camera = cv2.VideoCapture(int(sys.argv[1]))
    _, frame = camera.read()
    cv2.imwrite('test.jpg', frame)
    return ai_algo()


def ai_algo():
    # Load the model
    model = tensorflow.keras.models.load_model('converted_keras/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('test.jpg')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    indexMax = np.argmax(prediction)
    dict = {}
    dict.update({0: "paper", 1: "plastic", 2: "glass", 3: "mix", 4: "iron", 5: "organic"})
    out = dict[indexMax]
    print(out)
    return out


if __name__ == '__main__':
    app.run(host='192.168.43.173', debug=True)
