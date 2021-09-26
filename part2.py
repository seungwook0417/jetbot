import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import requests
import json
import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import threading

from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os
from keras.models import load_model

import random
import time
execution_path = os.getcwd()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_classifier = cv2.CascadeClassifier('123.xml')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    #print(faces)
    if faces is ():
        return img, []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        #print(roi)
    return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

def detect_from_image(frame):
    print("Runnig!!")
    samplecount = 0
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()
    try:
        file_name_path = 'temp/fire.jpg'
        cv2.imwrite(file_name_path, frame)

        detections = detector.detectObjectsFromImage(input_image=file_name_path,
                                                    output_image_path=os.path.join(execution_path, "1-detected.jpg"),
                                                    minimum_percentage_probability=40)

        #os.remove(file_name_path)
        #os.remove(os.path.join(execution_path, "1-detected.jpg"))
        #print(detections)
        for detection in detections:
            if detection["percentage_probability"] >= 40:
                print("fire")
                alert("fire")
                send_telegram(file_name_path)
                if samplecount == 0:
                    send_kakaotalk("침입자 경보 안내")
                    samplecount +=1
                # send_telegram(file_name_path)
                # alert('fire')
                #print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    except:
        pass
def fire(cap):
    samplecount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        
    
    
        output = cv2.bitwise_and(frame, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)

        file_name_path = 'temp/fire.jpg'
        cv2.imwrite(file_name_path, frame)
        #print("output:", frame)
        if int(no_red) > 20000:
            print ('Fire detected')
            if samplecount == 0:
                send_telegram(file_name_path)
                
            # if samplecount == 0:
            #     send_kakaotalk("침입자 경보 안내")
            #     samplecount +=1
        print(int(no_red))
        print("output:".format(mask))

def execute(cap):
    print("Runnig!!")

    data_path = "models/"
    models_dirs = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    lockcount = 0
    samplecount = 0
    while True:
        try:
            ret, frame = cap.read()
            img, face = face_detector(frame)
            if face != []:
                #print(face)
                # try:
                min_score = 999
                min_score_name = ""
                face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                model = cv2.face.LBPHFaceRecognizer_create()

                for model_name in models_dirs:
                    # print(data_path+model_name)

                    model.read(data_path + model_name)
                    result = model.predict(face)

                    if min_score > result[1]:
                        min_score = result[1]
                        min_score_name = model_name

                if min_score < 500:
                    confidence = int(100 * (1 - (min_score) / 300))
                    print(confidence)
                    display_string = str(confidence) + '% Confidence it is ' + min_score_name
                # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
                # 80 보다 크면 동일 인물로 간주해 UnLocked!
                if confidence > 50:
                    print(min_score_name)
                    #rand = random.randint(0,100000)
                    file_name_path = 'static/unknown1.jpg'
                    cv2.imwrite(file_name_path, frame)
                    #time.sleep('2')
                    #alert("known")
                    send_telegram(file_name_path)
                    if samplecount == 0:
                        send_kakaotalk("침입자 경보 안내")
                        samplecount +=1
                    #os.remove(file_name_path)
                    #alert(min_score_name)
                    # cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    # cv2.imshow('Face Cropper', image)
                else:
                    print("Locked")
                    print(lockcount)
                    lockcount += 1
                    if lockcount == 2:
                        file_name_path = 'static/unknown.jpg'
                        cv2.imwrite(file_name_path, frame)
                        #alert("unknown")
                        send_telegram(file_name_path)
                        lockcount = 0
                    # 80 이하면 타인.. Locked!!!
                    # cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    # cv2.imshow('Face Cropper', image)
        except:
            pass
        #     # 얼굴 검출 안됨
        #     # cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #     # cv2.imshow('Face Cropper', image)
        # if cv2.waitKey(1) == 13:
        #     break
    #cap.release()
    #cv2.destroyAllWindows()

def alert(text):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": "Bearer hIh9YoNLy0w2_1DIXGKsfo40dcNsjlUbR6IoeAorDNQAAAF8EZkC1Q"
    }
    data = {
        "template_object": json.dumps({
            "object_type": "feed",
            "content": {
                    "title": "침입자 알림",
                    "description": text,
                    "image_url": "http://jetbot.seungwook.me/static/DU.png",
                    "image_width": 640,
                    "image_height": 640,
                    "link": {
                    "web_url": "http://jetbot.seungwook.me",
                    "mobile_web_url": "http://jetbot.seungwook.me",
                    "android_execution_params": "contentId=100",
                    "ios_execution_params": "contentId=100"
                }
            },
            "buttons": [
            {
                "title": "CCTV 바로가기",
                "link": {
                    "web_url": "http://jetbot.seungwook.me",
                    "mobile_web_url": "http://jetbot.seungwook.me"
                }
            }]
        })
    }
    response = requests.post(url, headers=headers, data=data)
    print(response.status_code)

def send_kakaotalk(subject):
    headers = {"Authorization": "Bearer hIh9YoNLy0w2_1DIXGKsfo40dcNsjlUbR6IoeAorDNQAAAF8EZkC1Q"}
    friend_id = "hbKEvYuygrWGqpujkKeSo5Ssn7OHv464iLDW"

    send_url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"

    data = {
        'receiver_uuids': '["{}"]'.format(friend_id),
        "template_object": json.dumps({
            "object_type": "text",
            "text": subject,
            "link": {
                "web_url": "http://jetbot.seungwook.me",
                "mobile_web_url": "http://jetbot.seungwook.me"
            },
            "button_title": "CCTV 바로가기"
        })
    }
    response = requests.post(send_url, headers=headers, data=data)
    print(response.status_code)

def send_telegram(photo_path):
    bot = telegram.Bot("1213632651:AAGUS_C7fHNw54bZ92zeqbczXaSkRu0lK74")
    reply_markup =[[InlineKeyboardButton(text="바로가기", url="http://jetbot.seungwook.me")]]
    bot.sendPhoto(chat_id="905298975", photo=open(photo_path, 'rb'), reply_markup=InlineKeyboardMarkup(reply_markup))

def _async_raise(tid, exctype):
  """raises the exception, performs cleanup if needed"""
  tid = ctypes.c_long(tid)
  if not inspect.isclass(exctype):
    exctype = type(exctype)
  res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
  if res == 0:
    raise ValueError("invalid thread id")
  elif res != 1:
    # """if it returns a number greater than one, you're in trouble,
    # and you should call it again with exc=NULL to revert the effect"""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
    raise SystemError("PyThreadState_SetAsyncExc failed")
def stop_thread(thread):
  _async_raise(thread.ident, SystemExit)
'''The function of thread 1, which continuously calls the detection function'''
def test():
    while True:
        execute(cap)

thread1 = threading.Thread(target=test)

def object_run(cap):
    #cap = cv2.VideoCapture(-1)
    #execute(cap)
    execute(cap)
    thread1.setDaemon(False)
    thread1.start()
