import base64
import cv2
from flask import Flask, request, render_template, jsonify
import numpy as np
import pcd


app = Flask(__name__)

def stringify(img):
    # encode into base 64
    im_b64 = base64.b64encode(img)
    # decode with utf-8
    im_decode = im_b64.decode("utf-8")

    return im_decode

def encode_to_image(img, format):
    # encode to image
    img = cv2.imencode(format, img)
    # get image data
    img = np.array(img[1])

    return img

def bytes_to_numpy(file_bytes):
    #convert string data to numpy array
    im_bytes = np.fromstring(file_bytes, np.uint8)
    
    # convert numpy array to image
    img = cv2.imdecode(im_bytes, cv2.IMREAD_UNCHANGED)

    return img

def read_upload_file(input_file_name):
    # if input_file_name not in request.files:
    #     return 'there is no image uploaded'

    # if(request.files[input_file_name].filename == ''):
    #     return 'there is no image uploaded'
    
    # read binary file
    im_file = request.files[input_file_name].read()

    return im_file

def read_uploaded_file_to_numpy(input_file_name):
    file_bytes = read_upload_file(input_file_name)
    img = bytes_to_numpy(file_bytes)

    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/extract-rgb', methods=['GET', 'POST'])
def extract_rgb():
    if request.method == 'POST':
        result = {}

        try:
            img = read_uploaded_file_to_numpy("im_file")
            rgb_channel = pcd.extractRGB(img)

            img = encode_to_image(img, '.png')
            im_decode = stringify(img)

            result.update({'img': im_decode, 'rgb': rgb_channel, 'status': True})
        except:
            result.update({'status': False, 'msg': 'There is an error occurred'})

        return render_template('extract_rgb.html', data=result)
           

    return render_template('extract_rgb.html')

@app.route('/grayscale', methods=['GET', 'POST'])
def grayscale():
    if request.method == 'POST':
        result = {}
        try:
            img = read_uploaded_file_to_numpy("im_file")

            grayscale = pcd.weightedAverageGrayscale(img)

            img = encode_to_image(img, '.png')
            grayscale = encode_to_image(grayscale, '.png')

            img_decode = stringify(img)
            grayscale_decode = stringify(grayscale)

            result.update({'img': img_decode, 'grayscale': grayscale_decode, 'status': True})
        except:
            result.update({'status': False, 'msg': 'There is an error occurred'})

        return render_template('grayscale.html', data=result)

    return render_template('grayscale.html')





if __name__ == '__main__':
    app.run(debug=True)