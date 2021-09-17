import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import requests
import json
import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os
from keras.models import load_model

execution_path = os.getcwd()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        # print(roi)
    return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


def fire_detector(frame):
    model = load_model('ip.h5')
    predictlist = []
    count = 0
    for i in range(5):
        filename = "temp/test%d.jpg" % count;
        count += 1
        cv2.imwrite(filename, frame)

    for i in range(5):
        filename = 'temp/test' + str(i) + '.jpg'
        img = cv2.resize(img, (320, 240))
        img = np.reshape(img, [1, 320, 240, 3])
        classes = model.predict_classes(img)
        predictlist.append(classes[0][0])
    if (predictlist.count(1) > predictlist.count(0)):
        print('fire')
        file_name_path = 'temp/fire.jpg'
        cv2.imwrite(file_name_path, frame)
        # send_telegram(file_name_path)
        # alert('fire')
    else:
        print('no_fire')


def run():
    cap = cv2.VideoCapture(-1)
    print("Runnig!!")

    data_path = "models/"
    models_dirs = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    while True:
        ret, frame = cap.read()
        # fire_detector(frame)
        img, face = face_detector(frame)
        if face != []:
            # print(face)
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
                display_string = str(confidence) + '% Confidence it is ' + min_score_name
            # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            # 80 보다 크면 동일 인물로 간주해 UnLocked!
            if confidence > 65:
                print(min_score_name)
                # alert(min_score_name)
                # cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # cv2.imshow('Face Cropper', image)
            else:
                print("Locked")
                file_name_path = '/temp/unknown.jpg'
                cv2.imwrite(file_name_path, frame)
                # send_telegram(file_name_path)
                # alert("Locked")
                # 80 이하면 타인.. Locked!!!
                # cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('Face Cropper', image)
            # except:
            #      pass
        else:
            print("검출안됨")
        #     # 얼굴 검출 안됨
        #     # cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #     # cv2.imshow('Face Cropper', image)
        # if cv2.waitKey(1) == 13:
        #     break
    # cap.release()
    # cv2.destroyAllWindows()


def alert(text):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": "Bearer 1G_XtTeTj7zI2oPYy7bHczbGysX_oFgvNvi2wP6kSIc3tZ3P1dM_RO7bIs-gTXt8VypkKgopyNkAAAF723wSQg"
    }
    data = {
        "template_object": json.dumps({
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": "http://192.168.0.43"
            }
        })
    }
    requests.post(url, headers=headers, data=data)


def send_kakaotalk(subject):
    headers = {
        "Authorization": "Bearer 1G_XtTeTj7zI2oPYy7bHczbGysX_oFgvNvi2wP6kSIc3tZ3P1dM_RO7bIs-gTXt8VypkKgopyNkAAAF723wSQg"}
    friend_id = "hbKEvYuygrWGqpujkKeSo5Ssn7OHv464iLDW"

    send_url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"

    data = {
        'receiver_uuids': '["{}"]'.format(friend_id),
        "template_object": json.dumps({
            "object_type": "text",
            "text": subject,
            "link": {
                "web_url": "http://192.168.0.43",
                "mobile_web_url": "http://192.168.0.43"
            },
            "button_title": "바로가기"
        })
    }
    response = requests.post(send_url, headers=headers, data=data)
    response.status_code


def send_telegram(photo_path):
    bot = telegram.Bot("1557538463:AAEy6MWCdtdhVQNKjZD2K67b6xQTW0pdGkU")
    reply_markup = [[InlineKeyboardButton(text="바로가기", url="http://192.168.0.43")]]
    bot.sendPhoto(chat_id="905298975", photo=open(photo_path, 'rb'), reply_markup=InlineKeyboardMarkup(reply_markup))


run()