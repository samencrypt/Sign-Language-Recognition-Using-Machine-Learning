from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
import numpy as np
from camera import VideoCamera
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

def generate_frames(cam):
    while True:

        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
