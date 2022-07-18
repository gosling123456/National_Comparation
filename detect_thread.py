# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import  RPi.GPIO as GPIO
import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils
import socket
from concurrent.futures import ThreadPoolExecutor
import threading
from flask import Flask, render_template, Response
import cv2
import random
import pygame
from aip import AipSpeech
import urllib
import webbrowser
import Adafruit_PCA9685
import pyaudio
import wave
import requests
#from sound_voice import get_audio,predict,awake,rand_music,communication,commun
#from video.app import app
pool = ThreadPoolExecutor(max_workers=7, thread_name_prefix='ThreadPool')  
SensorRight = 16
SensorLeft  = 12

PWMA   = 18
AIN1   = 22
AIN2   = 27

PWMB   = 23
BIN1   = 25
BIN2   = 24

BtnPin  = 19
Gpin    = 5
Rpin    = 6
pwm = Adafruit_PCA9685.PCA9685()
def set_servo_angle(channel,angle):
    angle=4096*((angle*11)+500)/20000
    pwm.set_pwm(channel,0,int(angle))
pwm.set_pwm_freq(50)
set_servo_angle(5,85)  #底座舵机
set_servo_angle(4,65)

#智能小车运动函数 
def t_up(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    time.sleep(t_time)
        
def t_stop(t_time):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)
        
def t_down(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)

def t_left(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    time.sleep(t_time)

def t_right(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)
    
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)       # 按物理位置给GPIOs编号
GPIO.setup(Gpin, GPIO.OUT)     # 设置绿色Led引脚模式输出
GPIO.setup(Rpin, GPIO.OUT)     # 设置红色Led引脚模式输出
GPIO.setup(BtnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # 设置输入BtnPin模式，拉高至高电平(3.3V) 
GPIO.setup(SensorRight,GPIO.IN)
GPIO.setup(SensorLeft,GPIO.IN)

GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)
    
L_Motor= GPIO.PWM(PWMA,100)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)
    
count = 0

width = 480
height = 320
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def search_person():
    #t_stop(0.3)
    t_down(80,0.1)
    #t_stop(0.1)
    t_left(80,0.1)
    t_stop(0.1)
    
def cap_read():
    success, frame = cap.read()
    return success, frame
    
def movment(detections,res,counter):
    print(detections,res)
    SR_2 = GPIO.input(SensorRight)
    SL_2 = GPIO.input(SensorLeft)
    print(detections)
    for detection in detections:
        person = detection.categories[0].label
        if person == "person":
                #print(person)
            (left,top,right,bottom)= detection.bounding_box
            #print((x,y,z,w))
            x = (left + right)/2
            y = (top + bottom)/2
            x_p = int(round(x))
            #print(x_p,"*********")
            #print(SL_2)
    x_Lower = 90
    x_Upper = 390
    flag =False
    
    if "person" not in res:
        if counter%5 == 0:
            pool.submit(search_person)
            #pass
        
    #if counter %10 == 0:
    if"person" in res:
        flag = False
        # 判断X方向范围来判断机器人的运动
        if x_p > x_Lower and x_p < x_Upper:
            try:
                #print(SL_2,SR_2)
                if SL_2 == True and SR_2 == True:
                    try:
                        t_up(100,0.3)
                        if counter%40 == 0:
                            print("8s-stop")
                            t_stop(0.2)
                    except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                        GPIO.cleanup()       
                elif SL_2 == True and SR_2 ==False:
                    
                    t_left(80,0.1)
                    t_stop(0.1)
                elif SL_2==False and SR_2 ==True:
                   
                    t_right(80,0.1)
                    t_stop(0.1)
                else:
                    #t_stop(0.1)
                    t_down(70,0.3)
                    t_left(70,0.3)
                    t_stop(0.1)
                
                #t_up(100,0.3)
                #flag = True
                #t_stop(1)
            except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                GPIO.cleanup()               
        elif x_p < x_Lower:
            try:
                if counter%3 == 0:
                    t_right(80,0.1)
                    t_stop(0.1)
                    #flag = False
                    print("leftleftleftleftleftleftleftleftleft")
            except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                GPIO.cleanup()
        elif x_p > x_Upper:
            try:
                if counter%3 == 0:
                    t_left(80,0.1)
                    t_stop(0.1)
                    #flag = False
                    #time.sleep(2)
                    print("rightrightrightrightrightrightright")
            except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                GPIO.cleanup()

    
    

# Start capturing video input from the camera


def run():
  
  model ='efficientdet_lite0.tflite'
  num_threads = 4
  enable_edgetpu = False
# Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()


  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)
  count = 0
  while cap.isOpened():
    
    #conn, addr = s.accept()
    start = time.time()
    success, image = cap_read()
    time.sleep(0.003)
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
      
    counter += 1
    image = cv2.flip(image, 1)
    #pool.submit(server,image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start0 = time.time()
    detections = detector.detect(rgb_image)
    end0 = time.time()
    #print("00000000",start0 - end0)
    
    
   
    image,res = utils.visualize(image, detections,counter)
    if flag:
        pool.submit(movment,detections,res,counter)
    
    
    
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
    end = time.time()
  cap.release()
  cv2.destroyAllWindows()
    



def server():

    addr = ("192.168.137.203", 8885)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(addr)
    s.listen(2)

    print('Waiting connection...')
    print('Accept new connection from {0}'.format(addr))
    sock, addr = s.accept()
    #while True:
        
        
        #pool.submit(deal_data,conn, addr)
    while True:
        ret, frame = cap_read()
        time.sleep(0.07)
        #ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
        #img_encode = cv.imencode('.jpg', frame)[1]
        _, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        reply = sock.recv(1024)
        #print(reply)
        if 'ok' == reply.decode():  # 确认一下服务器get到文件长度和文件名数据
            #i = 0
            data = img_encode
            file_name = "name"
            sock.send('{}'.format(len(data)).encode())
            a = sock.recv(1024)
            #print("a",a.decode())
            go = 0
            total = len(data)
            while go < total:  # 发送文件
                data_to_send = data[go:go + total // 2]
                sock.send(data_to_send)
                go += len(data_to_send)
            c = sock.recv(1024).decode()
            #print("c",c)
    #conn.close()

flag_respond = False
flag = False
flag_music = False        
flag_commun = False
flag_awake = False
flag_talkWithRobot = False
# 自己的APPID AK SK
APP_ID = '26273092'
API_KEY = 'XmOEfSbzqrmmqmptosR52fgp'
SECRET_KEY = 'etEMsii801j5XZGGXxtfpx80nLfeVKSK'


# 准备好APPID、APIKEY、SecretKey
# 创建识别对象
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def play_sound(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()



def get_audio(filepath):
    #filepath = "input.wav" 
    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = filepath
    try:
        p = pyaudio.PyAudio()
    
        # 打开一个新的音频stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*" * 10, "开始录音：请在4秒内输入语音")
        frames = []  # 存放录音数据
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK,exception_on_overflow = False)
            frames.append(data)
        print("*" * 10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()
        # 将音频数据保存到wav文件之中
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        print(e)



# 将上一步录音 通过调用百度api实现语音识别

# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        read_sound = fp.read()
        return read_sound


def predict():
    # 调用百度AI的接口, 识别本地文件
    return client.asr(get_file_content('input.wav'), 'wav', 16000, {
        'dev_pid': 1537,
    })


# 免费、无需注册、只需要发送get请求就可实现聊天的青云客智能机器人，直接调用接口即可。
def talkWithRobot(msg):
    url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'.format(urllib.parse.quote(msg))
    html = requests.get(url)
    return html.json()["content"]


# 随机播放音乐
def rand_music():
    time.sleep(2)
    musil_list = ['小兔子乖乖.mp3', '蓝精灵.mp3', '拔萝卜.mp3', '世上只有妈妈好.mp3', '两只老虎.mp3', '健康歌.mp3', '我有一只小毛驴.mp3', '外婆的澎湖湾.mp3',
                  '鲁冰花.mp3', '爱我你就抱抱我.mp3']
    key = random.randint(0, 9)
    print(musil_list[key])
    play_sound("./music/" + musil_list[key])
    flag_sound = False
    


# 将机器人返回的对话 调用语音合成接口 返回语音
def communication(str_Result):
    res = talkWithRobot(str_Result)
    if '未获取到相关信息' not in res:

        print(res)
        result = client.synthesis(res, 'zh', 1, {
            'per': 4,
            'spd': 3,  # 速度
            'vol': 7  # 音量
        })
        if not isinstance(result, dict):
            with open('test.mp3', 'wb') as f:
                f.write(result)

        # 播放合成后的语音
        play_sound("test.mp3")
#def commun(str_Result):
 #   return str_Result
        

def awake():
    awake_list = ['awake_one.mp3', 'awake_two.mp3', 'awake_three.mp3']

    key = random.randint(0, 2)
    print(awake_list[key])
    play_sound("./sound/" + awake_list[key])
    return awake_list[key]



def voice(str_Result):
    global flag_music
    global flag_respond
    global flag_commun
    global flag
    global flag_awake
    global flag_talkWithRobot
    if "小嘿" in str_Result or "小黑" in str_Result or "小飞" in str_Result:
        flag_commun = True
        flag_respond = True
        flag_awake = True
        print("唤醒")
        awake()
    elif ("说话" in str_Result or "别说了" in str_Result) and flag_awake:
        play_sound("./sound/ok.mp3")
        flag_talkWithRobot = False
        flag_respond = False
    elif "工作" in str_Result and flag_awake:
        print("工作")
        flag = True
        play_sound("./sound/start_work.mp3")
        pool.submit(run)
        
        print("++++++++++++++++++++++++++++++++++++++++++")
        flag_commun = False 
    elif ("休息" in str_Result or "别跑了" in str_Result) and flag_awake:
        flag_commun = True
        play_sound("./sound/stop_work.mp3")
        # global flag
        flag = False
    elif "音乐" in str_Result and flag_awake:
        flag_commun = False
        play_sound("./sound/listen_music.mp3")
        pool.submit(rand_music())
    elif ("暂停" in str_Result or "别放了" in str_Result) and flag_awake:
        
        flag_commun = True
        flag_music = True
        pygame.mixer.music.pause()
        play_sound("./sound/ok.mp3")
    elif ("聊天" in str_Result or "聊聊天" in str_Result) and flag_awake:
        flag_talkWithRobot = True
    elif flag_talkWithRobot:
        communication(str_Result)
def sounds_list():
    in_path = "input.wav"
    while True:
        get_audio(in_path)
        results = predict()  # 语音识别返回的结果是字典格式
        Result = results["result"]  # 将字典中的”result“值取出
        str_Result = "".join(Result)
        print(str_Result)  # 调用识别方法, 并输出# 讲列表格式转换为字符串格式
        voice(str_Result)
        time.sleep(0.04)
 
#pool.submit(run)
app = Flask(__name__)
pool.submit(app)
#pool.submit(run)
pool.submit(server)
pool.submit(sounds_list)





  

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = cap_read()
        time.sleep(0.11)
        #success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='192.168.137.203',port=8888)
    

