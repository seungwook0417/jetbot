from flask import Flask, render_template, Response
import traitlets
from jetbot import Robot
import time
import object_follow
import tracking

from jetbot import Camera
from jetbot import bgr8_to_jpeg
import PID
from servoserial import ServoSerial
import ipywidgets.widgets as widgets
from IPython.display import display
import cv2

camera = Camera.instance(width=720, height=720)
robot = Robot()
app = Flask(__name__)

def genframe():
    while True:
        g_camera = Camera.instance(width=720, height=720)
        ####
        # camera = Camera.instance(width=720, height=720)
        # global face_x, face_y, face_w, face_h
        # face_x = face_y = face_w = face_h = 0
        # global target_valuex
        # target_valuex = 2048
        # global target_valuey
        # target_valuey = 2048
        #
        # xservo_pid = PID.PositionalPID(1.9, 0.3, 0.35)
        # yservo_pid = PID.PositionalPID(1.5, 0.2, 0.3)
        #
        # servo_device = ServoSerial()
        # face_image = widgets.Image(format='jpeg', width=300, height=300)
        # face_cascade = cv2.CascadeClassifier('123.xml')
        #
        # frame = camera.value
        # frame = cv2.resize(frame, (300, 300))
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale( gray )
        # if len(faces)>0:
        #     (face_x, face_y, face_w, face_h) = faces[0]
        #     # Mark the detected face
        #     # cv2.rectangle(frame,(face_x,face_y),(face_x+face_h,face_y+face_w),(0,255,0),2)
        #     cv2.rectangle(frame,(face_x+10,face_y),(face_x+face_w-10,face_y+face_h+20),(0,255,0),2)
        #
        #     # Proportion-Integration-Differentiation algorithm
        #     # Input X-axis direction parameter PID control input
        #     xservo_pid.SystemOutput = face_x+face_h/2
        #     xservo_pid.SetStepSignal(150)
        #     xservo_pid.SetInertiaTime(0.01, 0.006)
        #     target_valuex = int(2048 + xservo_pid.SystemOutput)
        #     # Input Y axis direction parameter PID control input
        #     yservo_pid.SystemOutput = face_y+face_w/2
        #     yservo_pid.SetStepSignal(150)
        #     yservo_pid.SetInertiaTime(0.01, 0.006)
        #     target_valuey = int(2048+yservo_pid.SystemOutput)
        #     # Rotate the gimbal to the PID adjustment position
        #     servo_device.Servo_serial_double_control(1, target_valuex, 2, target_valuey)
        # # Real-time return of image data for display
        # face_image.value = bgr8_to_jpeg(frame)

        ####
        frame = g_camera.value
        imgencode = cv2.imencode('.jpg', frame)[1]
        # imgencode = imgencode.tostring()
        # yield '--frame\r\nContent-Type: image/jpeg\r\n\r\n' + imgencode + '\r\n'
        imgencode = imgencode.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + imgencode + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield '--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + '\r\n'

@app.route('/video_feed')
def video_feed():   
    return Response(genframe(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test')
def test():
    object_follow.object_run()
    #tracking.object_run()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)