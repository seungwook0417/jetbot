from flask import Flask, render_template, Response

import part2
import fire
import cv2

camera = cv2.VideoCapture(-1)
app = Flask(__name__)

def genframe():
    while True:
        try:
            ret, frame = camera.read()
            imgencode = cv2.imencode('.jpg', frame)[1]
            # imgencode = imgencode.tostring()
            # yield '--frame\r\nContent-Type: image/jpeg\r\n\r\n' + imgencode + '\r\n'
            imgencode = imgencode.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + imgencode + b'\r\n')
        except:
            pass
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
    #object_follow.object_run()
    #part2.object_run(camera)
    ten.fire(camera)

@app.route('/test2')
def test2():
    while True:
        try:
            ten.fire(camera)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)