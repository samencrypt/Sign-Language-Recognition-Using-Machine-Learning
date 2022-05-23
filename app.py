from flask import Flask, Response,render_template,Response
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import cv2
import time
import numpy as np

app= Flask(__name__)
camera = cv2.VideoCapture(0)
image_x, image_y = 64, 64
classifier = load_model('Trained_model.h5')

def preprocessing(imcrop):
    lower_blue = np.array([0,58,50])
    upper_blue = np.array([30,255,255])
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(imcrop, imcrop, mask=mask)
    cv2.imwrite('mask.png',mask)
    cv2.imwrite('masked.png',result)
    
def predictor():
    test_image = image.load_img('mask.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'
    else :
        return 'None'

def generate_frames():
    img_text = 'None'
    camera = cv2.VideoCapture(0)
    print("on")
    while True:

        success,frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            img = cv2.rectangle(frame,(425,100),(625,300),(88,65,15),2)
            imcrop = img[102:298, 427:623]
            cv2.putText(frame, img_text, (24, 400),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5,(14,14,135),2)
            cv2.imwrite('1.png',imcrop)
            preprocessing(imcrop)
            img_text = predictor()
            ret,buffer = cv2.imencode('.png',frame)
            frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    camera.release()
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')


@app.route('/about')
def about():
    camera.release()
    return render_template('about.html')

@app.route('/contact')
def contact():
    camera.release()
    return render_template('contact.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/close')
def close():
    camera.release()
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)