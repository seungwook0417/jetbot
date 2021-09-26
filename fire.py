import cv2
import numpy as np
import requests
import json
import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def fire(cap):
    samplecount = 0
    while True:
        try:
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
            if int(no_red) > 500:
                print ('Fire detected')
                if samplecount == 0:
                    send_telegram(file_name_path)
                    send_kakaotalk("화재 경보 안내")
                    samplecount +=1
                    
                # if samplecount == 0:
                #     send_kakaotalk("침입자 경보 안내")
                #     samplecount +=1
            print(int(no_red))
            print("output:".format(mask))
        except:
            pass

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